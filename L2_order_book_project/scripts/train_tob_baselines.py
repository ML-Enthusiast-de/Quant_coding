#!/usr/bin/env python
"""
FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.

Train baselines on Top-of-Book (TOB) datasets for crypto microstructure.

What this script does:
- Loads TOB dataset parquet(s) produced by build_tob_dataset.py
- Strict chronological split (train/val/test)
- Trains simple baselines:
    1) Dummy (always predict "no move")
    2) Logistic Regression (multiclass -1/0/+1) with class imbalance handling
    3) Ridge Regression (predict forward return), then convert to trading signals with a deadzone
- Evaluates:
    - classification metrics: accuracy, balanced accuracy, macro-F1, confusion matrix
    - regression metrics: IC/corr, MSE
    - simple PnL proxy with spread + taker fee assumptions (per-trade), using mid & bid/ask

No CLI required: edit Config below and run.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Make repo imports work reliably (repo root on sys.path)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Quant_coding
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -----------------------------------------------------------------------------
# CONFIG (edit these)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    product_slug: str = "btcusd"  # matches your dataset filename slug
    sample: str = "1s"            # dataset sampling grid
    horizons: Tuple[str, ...] = ("10s", "30s")  # choose which dataset(s) to train

    # Split settings (strict time split)
    train_frac: float = 0.70
    val_frac: float = 0.15        # test_frac = 1 - train - val

    # PnL proxy settings (VERY simplified)
    taker_fee_bps: float = 5.0    # per side (enter or exit) in basis points (bps)
    use_spread_cost: bool = True  # enter at ask for long, bid for short; exit at mid_fwd

    # Regression -> signal rule: only trade if |predicted fwd return| > deadzone
    # Choose something small relative to your label threshold; tweak later.
    reg_deadzone: float = 0.0     # e.g. 0.00002 ~ 2 bps

    # Logistic -> signal rule: only trade if max prob >= p_min else flat
    cls_p_min: float = 0.40

    # Output
    verbose: bool = True


CFG = Config()


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = REPO_ROOT / "L2_order_book_project"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Utilities: column detection (robust to naming changes)
# -----------------------------------------------------------------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    # Your ALL parquet sometimes has ts as a column; your dataset usually should have ts index.
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    if "ts" in df.columns:
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
        return df

    raise ValueError("Dataset must have a DatetimeIndex or a 'ts' column.")

def label_cols_from_dataset_path(p: Path) -> tuple[str, str]:
    # file name ends with "..._<H>.parquet", e.g. tob_dataset_btcusd_1s_10s.parquet -> H="10s"
    horizon = p.stem.split("_")[-1]
    return f"y_dir3_fwd_{horizon}", f"y_ret_fwd_{horizon}"


def detect_label_column(df: pd.DataFrame) -> str:
    # Prefer a generic 'label' column
    if "label" in df.columns:
        return "label"

    # Otherwise: pick a column that looks like a label
    candidates = [c for c in df.columns if c.lower().startswith("label")]
    if candidates:
        # If multiple, choose the first; you can hardcode later if needed
        return sorted(candidates)[0]

    # As a last resort: common names
    for c in ["y", "target", "cls_label"]:
        if c in df.columns:
            return c

    raise ValueError("Could not find a label column (expected 'label' or 'label_*').")


def detect_fwd_return_column(df: pd.DataFrame) -> Optional[str]:
    # Optional: some datasets store forward return explicitly
    for c in ["fwd_ret", "fwd_return", "target_fwd_ret"]:
        if c in df.columns:
            return c

    # Pattern-based search
    candidates = [c for c in df.columns if "fwd" in c.lower() and "ret" in c.lower()]
    return candidates[0] if candidates else None


def choose_feature_columns(df: pd.DataFrame, label_col: str) -> List[str]:
    # Drop obvious non-features
    drop = {
        label_col,
        "product",
        "raw",
        "exchange_ts",
        "receive_ts",
        "msg_type",
    }

    # Keep only numeric features; ignore strings
    numeric_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    # It's fine if this includes bid/ask/mid/etc. — they are part of TOB feature set.
    if not numeric_cols:
        raise ValueError("No numeric feature columns found.")

    return numeric_cols


# -----------------------------------------------------------------------------
# Strict time split
# -----------------------------------------------------------------------------
def time_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0.0 < train_frac < 1.0) or not (0.0 <= val_frac < 1.0):
        raise ValueError("Bad split fractions.")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    df = df.sort_index()
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    train = df.iloc[:i_train].copy()
    val = df.iloc[i_train:i_val].copy()
    test = df.iloc[i_val:].copy()

    return train, val, test


# -----------------------------------------------------------------------------
# Simple PnL proxy (spread + fee aware, but still simplified)
# -----------------------------------------------------------------------------
def _bps_to_decimal(bps: float) -> float:
    return bps / 10_000.0


def pnl_proxy_from_signals(
    df: pd.DataFrame,
    signal: pd.Series,
    mid_col: str = "mid",
    bid_col: str = "best_bid",
    ask_col: str = "best_ask",
    mid_fwd_col: str = "mid_fwd",
    taker_fee_bps: float = 5.0,
    use_spread_cost: bool = True,
) -> pd.Series:
    """
    Very simplified “one-shot” PnL proxy per timestamp:
    - signal ∈ {-1,0,+1}
    - if long (+1): enter at ask (if spread cost enabled) else mid; exit at future mid
    - if short (-1): enter at bid (if spread cost enabled) else mid; exit at future mid
    - fees charged twice (enter + exit)
    """
    fee = _bps_to_decimal(taker_fee_bps)

    mid = df[mid_col].astype(float)
    bid = df[bid_col].astype(float)
    ask = df[ask_col].astype(float)
    mid_fwd = df[mid_fwd_col].astype(float)

    # entry prices
    if use_spread_cost:
        entry_long = ask
        entry_short = bid
    else:
        entry_long = mid
        entry_short = mid

    sig = signal.astype(int)

    # returns before fees
    r_long = (mid_fwd - entry_long) / entry_long
    r_short = (entry_short - mid_fwd) / entry_short  # profit when price drops

    # apply fees (enter+exit)
    r_long_net = (1.0 + r_long) * (1.0 - fee) * (1.0 - fee) - 1.0
    r_short_net = (1.0 + r_short) * (1.0 - fee) * (1.0 - fee) - 1.0

    ret = pd.Series(0.0, index=df.index)
    ret[sig == 1] = r_long_net[sig == 1]
    ret[sig == -1] = r_short_net[sig == -1]
    ret[sig == 0] = 0.0

    return ret


def sharpe_np(x: np.ndarray, freq: float = 1.0) -> float:
    # freq=1.0 because we’re on 1s grid; later you can annualize if you really want.
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 0.0
    s = x.std()
    if np.isclose(s, 0.0):
        return 0.0
    return np.sqrt(freq) * x.mean() / s


# -----------------------------------------------------------------------------
# Train + evaluate baselines
# -----------------------------------------------------------------------------
def train_and_evaluate_one_dataset(path: Path, cfg: Config) -> Dict[str, object]:
    if cfg.verbose:
        print("=" * 90)
        print(f"[load] {path}")
    df = pd.read_parquet(path)
    df = _ensure_datetime_index(df)

    label_col, yret_col = label_cols_from_dataset_path(path)


    fwd_ret_col = detect_fwd_return_column(df)
    feat_cols = choose_feature_columns(df, label_col)

    # MINIMAL leak guard: drop all target columns from features
    feat_cols = [c for c in feat_cols if not c.startswith("y_")]


    # We need a forward mid to compute PnL proxy in a consistent way.
    # If dataset already has it, use it; otherwise reconstruct from mid + fwd_ret if available.
    if "mid_fwd" not in df.columns:
        if fwd_ret_col is not None:
            # mid_fwd = mid * (1 + fwd_ret)
            if "mid" not in df.columns:
                raise ValueError("Need 'mid' column to reconstruct mid_fwd.")
            df = df.copy()
            df["mid_fwd"] = df["mid"].astype(float) * (1.0 + df[fwd_ret_col].astype(float))
        else:
            # If neither exists, we can’t do the PnL proxy in this script.
            # You can add mid_fwd in build_tob_dataset.py if you want.
            df = df.copy()
            df["mid_fwd"] = np.nan

    # Drop rows where labels/features missing
    df = df.dropna(subset=[label_col] + feat_cols)

    # Split (strict time)
    train_df, val_df, test_df = time_split(df, cfg.train_frac, cfg.val_frac)

    X_train, y_train = train_df[feat_cols].values, train_df[label_col].astype(int).values
    X_val, y_val = val_df[feat_cols].values, val_df[label_col].astype(int).values
    X_test, y_test = test_df[feat_cols].values, test_df[label_col].astype(int).values

    if cfg.verbose:
        print(f"[cols] label={label_col} fwd_ret={fwd_ret_col} features={len(feat_cols)}")
        print(f"[split] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
        print(f"[labels/train] {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"[labels/test ] {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ------------------------
    # Baseline 0: Dummy (always 0 / most frequent)
    # Why: establishes a “do-nothing / no-move” floor.
    # ------------------------
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    yhat_dummy = dummy.predict(X_test)

    # ------------------------
    # Baseline 1: Logistic Regression (multiclass -1/0/+1)
    # Why: simple linear decision boundary; fast; good sanity baseline.
    # We use:
    #   - StandardScaler: stabilizes optimization
    #   - class_weight="balanced": deals with heavy class imbalance (tons of zeros)
    # ------------------------
    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=200,
                    class_weight="balanced",
                    n_jobs=None,
                ),
            ),
        ]
    )
    logreg.fit(X_train, y_train)

    proba_test = logreg.predict_proba(X_test)
    classes = logreg.named_steps["clf"].classes_  # should contain [-1,0,1] but don’t assume order
    yhat_lr = classes[np.argmax(proba_test, axis=1)]

    # “confidence gate” to reduce churn: if max prob < p_min => flat
    maxp = np.max(proba_test, axis=1)
    yhat_lr_gated = yhat_lr.copy()
    yhat_lr_gated[maxp < cfg.cls_p_min] = 0

    # ------------------------
    # Baseline 2: Ridge Regression on forward return (if available)
    # Why: regression is often more stable than 3-class classification.
    # Then you can impose a deadzone to control turnover.
    # ------------------------
    ridge_metrics = {}
    yhat_ridge_sig = None

    if fwd_ret_col is not None:
        y_train_r = train_df[fwd_ret_col].astype(float).values
        y_val_r = val_df[fwd_ret_col].astype(float).values
        y_test_r = test_df[fwd_ret_col].astype(float).values

        ridge = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0)),
            ]
        )
        ridge.fit(X_train, y_train_r)
        pred_r = ridge.predict(X_test)

        # Convert regression predictions -> {-1,0,1} signal with deadzone
        dz = float(cfg.reg_deadzone)
        sig = np.zeros_like(pred_r, dtype=int)
        sig[pred_r > dz] = 1
        sig[pred_r < -dz] = -1
        yhat_ridge_sig = sig

        ridge_metrics = {
            "mse": float(mean_squared_error(y_test_r, pred_r)),
            "ic_corr": float(np.corrcoef(y_test_r, pred_r)[0, 1]) if np.std(pred_r) > 0 else 0.0,
            "deadzone": dz,
        }

    # ------------------------
    # PnL proxy evaluation (if we have mid_fwd + bid/ask)
    # Why: classification metrics alone can lie in microstructure.
    # We want to see if signals are even directionally “tradable” after costs.
    # ------------------------
    can_pnl = all(c in test_df.columns for c in ["best_bid", "best_ask", "mid", "mid_fwd"]) and test_df["mid_fwd"].notna().all()

    def pnl_stats(name: str, sig_arr: np.ndarray) -> Dict[str, float]:
        if not can_pnl:
            return {"trades": float(np.sum(sig_arr != 0))}

        sig_s = pd.Series(sig_arr, index=test_df.index)
        ret = pnl_proxy_from_signals(
            test_df,
            sig_s,
            mid_col="mid",
            bid_col="best_bid",
            ask_col="best_ask",
            mid_fwd_col="mid_fwd",
            taker_fee_bps=cfg.taker_fee_bps,
            use_spread_cost=cfg.use_spread_cost,
        )
        trades = int((sig_arr != 0).sum())
        return {
            "trades": float(trades),
            "mean_ret": float(ret.mean()),
            "sharpe_1s": float(sharpe_np(ret.values, freq=1.0)),
            "hit_rate": float((ret[ret != 0] > 0).mean()) if trades > 0 else 0.0,
        }

    # classification metrics helper
    def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
        return {
            "acc": float(accuracy_score(y_true, y_pred)),
            "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "confusion": confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]).tolist(),
        }

    report = {
        "dataset": str(path.name),
        "rows": int(len(df)),
        "time_start": str(df.index.min()),
        "time_end": str(df.index.max()),
        "label_col": label_col,
        "fwd_ret_col": fwd_ret_col,
        "n_features": int(len(feat_cols)),
        "split": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "config": {
            "taker_fee_bps": cfg.taker_fee_bps,
            "use_spread_cost": cfg.use_spread_cost,
            "cls_p_min": cfg.cls_p_min,
            "reg_deadzone": cfg.reg_deadzone,
        },
        "dummy": {
            "cls": cls_metrics(y_test, yhat_dummy),
            "pnl": pnl_stats("dummy", yhat_dummy),
        },
        "logreg_argmax": {
            "cls": cls_metrics(y_test, yhat_lr),
            "pnl": pnl_stats("logreg_argmax", yhat_lr),
        },
        "logreg_gated": {
            "cls": cls_metrics(y_test, yhat_lr_gated),
            "pnl": pnl_stats("logreg_gated", yhat_lr_gated),
        },
        "ridge": {
            "enabled": bool(fwd_ret_col is not None),
            "reg": ridge_metrics,
            "pnl": pnl_stats("ridge_sig", yhat_ridge_sig) if yhat_ridge_sig is not None else {},
        },
    }

    if cfg.verbose:
        print("-" * 90)
        print("[RESULT] Dummy:", report["dummy"]["cls"], report["dummy"]["pnl"])
        print("[RESULT] LogReg argmax:", report["logreg_argmax"]["cls"], report["logreg_argmax"]["pnl"])
        print("[RESULT] LogReg gated :", report["logreg_gated"]["cls"], report["logreg_gated"]["pnl"])
        if fwd_ret_col is not None:
            print("[RESULT] Ridge:", report["ridge"]["reg"], report["ridge"]["pnl"])
        print("-" * 90)

    return report


def main() -> None:
    # Build dataset paths from config
    paths = []
    for hz in CFG.horizons:
        p = DATASETS_DIR / f"tob_dataset_{CFG.product_slug}_{CFG.sample}_{hz}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")
        paths.append(p)

    all_reports = []
    for p in paths:
        rep = train_and_evaluate_one_dataset(p, CFG)
        all_reports.append(rep)

    # Save a single summary JSON
    out_json = REPORTS_DIR / f"baseline_report_{CFG.product_slug}_{CFG.sample}_{'_'.join(CFG.horizons)}.json"
    out_json.write_text(json.dumps(all_reports, indent=2))
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
