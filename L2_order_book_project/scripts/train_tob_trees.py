#!/usr/bin/env python
"""
FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.

Train cost-aware tree baselines on Top-of-Book datasets (crypto microstructure).

- Strict chronological split (train/val/test)
- EMBARGO around split boundaries to avoid leakage with forward labels
- Models:
  1) HistGradientBoostingClassifier: predict y_dir3_fwd_<H> in {-1,0,1}
  2) HistGradientBoostingRegressor:  predict y_ret_fwd_<H> (forward return)
- Trading rule tuning on VALIDATION:
  - classifier: trade only if max class prob >= p_min else 0
  - regressor: trade only if |pred| >= deadzone else 0
- Evaluation on TEST:
  - classification metrics
  - PnL proxy net of spread + taker fees (simple one-shot)

Outputs:
  L2_order_book_project/data/reports/tree_report_<slug>_<sample>_<H>.json
  L2_order_book_project/models/tree_<slug>_<sample>_<H>.joblib
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


# -----------------------------------------------------------------------------
# Repo path
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Quant_coding
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROJECT_ROOT = REPO_ROOT / "L2_order_book_project"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
MODELS_DIR = REPO_ROOT / "models"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Config (edit here)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    product_slug: str = "btcusd"
    sample: str = "1s"
    horizon: str = "10s"  # choose ONE dataset per run

    # strict time split
    train_frac: float = 0.70
    val_frac: float = 0.15

    # leakage control
    embargo_seconds: int | None = None  # if None -> derived from horizon

    # costs (very simplified proxy)
    taker_fee_bps: float = 5.0
    use_spread_cost: bool = True

    # threshold grids to tune on validation
    p_min_grid: Tuple[float, ...] = (0.34, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60)
    deadzone_grid: Tuple[float, ...] = (0.0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5)

    # tree params (keep simple for now)
    max_depth: int = 3
    learning_rate: float = 0.05
    max_iter: int = 300
    min_samples_leaf: int = 50

    verbose: bool = True


CFG = Config()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
TOB_NONFEATURE_COLS = {
    "product",
    "mid_fwd",  # used for pnl, but also ok as feature if you want — we exclude to avoid leakage-by-construction
}

REQUIRED_PNL_COLS = {"best_bid", "best_ask", "mid", "mid_fwd", "segment_id"}


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "ts" in df.columns:
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
        return df
    raise ValueError("Dataset must have DatetimeIndex or a 'ts' column.")


def _horizon_to_seconds(h: str) -> int:
    # "10s" -> 10, "5s" -> 5
    h = h.strip().lower()
    if not h.endswith("s"):
        raise ValueError(f"Unsupported horizon format: {h} (expected like '10s')")
    return int(h[:-1])


def label_cols_from_path(p: Path) -> Tuple[str, str]:
    hz = p.stem.split("_")[-1]  # ..._10s
    return f"y_dir3_fwd_{hz}", f"y_ret_fwd_{hz}"


def choose_feature_columns(df: pd.DataFrame, label_col: str, yret_col: str) -> List[str]:
    # strictly numeric and exclude label + a few non-features
    drop = {label_col, yret_col} | TOB_NONFEATURE_COLS
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No numeric feature columns found.")
    return cols


def time_split_with_embargo(df: pd.DataFrame, train_frac: float, val_frac: float, embargo_s: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict split by time *plus* embargo to avoid label overlap leakage.
    We drop rows within embargo_s seconds before each boundary.
    """
    df = df.sort_index()
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    t_train_end = df.index[i_train]
    t_val_end = df.index[i_val]

    # base splits
    train = df.iloc[:i_train].copy()
    val = df.iloc[i_train:i_val].copy()
    test = df.iloc[i_val:].copy()

    # embargo: remove rows whose forward window may cross into next split
    # easiest: remove timestamps >= boundary - embargo from the left split
    train = train[train.index < (t_train_end - pd.Timedelta(seconds=embargo_s))]
    val = val[val.index < (t_val_end - pd.Timedelta(seconds=embargo_s))]

    # also ensure val starts after train boundary (already true)
    return train, val, test


def _bps_to_decimal(bps: float) -> float:
    return bps / 10_000.0


def pnl_proxy_from_signals(
    df: pd.DataFrame,
    signal: np.ndarray,
    taker_fee_bps: float,
    use_spread_cost: bool,
) -> pd.Series:
    """
    One-shot proxy:
      - enter at ask/bid (if spread enabled) else mid
      - exit at mid_fwd
      - taker fee charged twice (enter + exit)
    """
    fee = _bps_to_decimal(taker_fee_bps)

    mid = df["mid"].astype(float)
    bid = df["best_bid"].astype(float)
    ask = df["best_ask"].astype(float)
    mid_fwd = df["mid_fwd"].astype(float)

    if use_spread_cost:
        entry_long = ask
        entry_short = bid
    else:
        entry_long = mid
        entry_short = mid

    sig = signal.astype(int)
    r_long = (mid_fwd - entry_long) / entry_long
    r_short = (entry_short - mid_fwd) / entry_short

    r_long_net = (1.0 + r_long) * (1.0 - fee) * (1.0 - fee) - 1.0
    r_short_net = (1.0 + r_short) * (1.0 - fee) * (1.0 - fee) - 1.0

    out = pd.Series(0.0, index=df.index)
    out[sig == 1] = r_long_net[sig == 1]
    out[sig == -1] = r_short_net[sig == -1]
    return out


def sharpe_1s(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return 0.0
    s = x.std()
    if np.isclose(s, 0.0):
        return 0.0
    return x.mean() / s


def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion": confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]).tolist(),
    }


def tune_p_min_on_val(clf, X_val, val_df: pd.DataFrame, p_grid: Tuple[float, ...], taker_fee_bps: float, use_spread_cost: bool) -> Dict[str, float]:
    proba = clf.predict_proba(X_val)
    classes = clf.classes_
    argmax = classes[np.argmax(proba, axis=1)]
    maxp = np.max(proba, axis=1)

    best = {"p_min": float(p_grid[0]), "sharpe": -1e9, "mean_ret": -1e9, "trades": 0.0}
    for pmin in p_grid:
        sig = argmax.copy()
        sig[maxp < pmin] = 0
        ret = pnl_proxy_from_signals(val_df, sig, taker_fee_bps, use_spread_cost)
        trades = float(np.sum(sig != 0))
        s = float(sharpe_1s(ret.values))
        m = float(ret.mean())
        # primary objective: sharpe; tie-breaker: mean_ret
        if (s > best["sharpe"]) or (np.isclose(s, best["sharpe"]) and m > best["mean_ret"]):
            best = {"p_min": float(pmin), "sharpe": s, "mean_ret": m, "trades": trades}
    return best


def tune_deadzone_on_val(reg, X_val, val_df: pd.DataFrame, dz_grid: Tuple[float, ...], taker_fee_bps: float, use_spread_cost: bool) -> Dict[str, float]:
    pred = reg.predict(X_val)
    best = {"deadzone": float(dz_grid[0]), "sharpe": -1e9, "mean_ret": -1e9, "trades": 0.0}
    for dz in dz_grid:
        sig = np.zeros_like(pred, dtype=int)
        sig[pred > dz] = 1
        sig[pred < -dz] = -1
        ret = pnl_proxy_from_signals(val_df, sig, taker_fee_bps, use_spread_cost)
        trades = float(np.sum(sig != 0))
        s = float(sharpe_1s(ret.values))
        m = float(ret.mean())
        if (s > best["sharpe"]) or (np.isclose(s, best["sharpe"]) and m > best["mean_ret"]):
            best = {"deadzone": float(dz), "sharpe": s, "mean_ret": m, "trades": trades}
    return best


# -----------------------------------------------------------------------------
# Main train/eval
# -----------------------------------------------------------------------------
def main() -> None:
    dataset_path = DATASETS_DIR / f"tob_dataset_{CFG.product_slug}_{CFG.sample}_{CFG.horizon}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    print("=" * 98)
    print(f"[load] {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df = _ensure_datetime_index(df)

    label_col, yret_col = label_cols_from_path(dataset_path)

    if label_col not in df.columns or yret_col not in df.columns:
        raise ValueError(f"Expected columns missing. Need: {label_col} and {yret_col}")

    # Ensure mid_fwd exists (your dataset builder should have it; if not, reconstruct from mid*(1+yret))
    if "mid_fwd" not in df.columns:
        if "mid" not in df.columns:
            raise ValueError("Need 'mid' to reconstruct mid_fwd.")
        df = df.copy()
        df["mid_fwd"] = df["mid"].astype(float) * (1.0 + df[yret_col].astype(float))

    # Basic sanity: PnL columns
    for c in REQUIRED_PNL_COLS:
        if c not in df.columns:
            raise ValueError(f"Dataset missing required column for PnL proxy: {c}")

    # Drop NaNs for core columns
    df = df.dropna(subset=[label_col, yret_col, "mid", "best_bid", "best_ask", "mid_fwd"])
    df[label_col] = df[label_col].astype(int)

    feat_cols = choose_feature_columns(df, label_col, yret_col)

    hz_s = _horizon_to_seconds(CFG.horizon)
    embargo_s = CFG.embargo_seconds if CFG.embargo_seconds is not None else hz_s

    train_df, val_df, test_df = time_split_with_embargo(df, CFG.train_frac, CFG.val_frac, embargo_s)

    if CFG.verbose:
        print(f"[cols] label={label_col} yret={yret_col} features={len(feat_cols)} embargo={embargo_s}s")
        print(f"[split] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
        print(f"[labels] train={dict(zip(*np.unique(train_df[label_col].values, return_counts=True)))}")
        print(f"[labels] test ={dict(zip(*np.unique(test_df[label_col].values, return_counts=True)))}")

    X_train = train_df[feat_cols].values
    y_train = train_df[label_col].values
    X_val = val_df[feat_cols].values
    y_val = val_df[label_col].values
    X_test = test_df[feat_cols].values
    y_test = test_df[label_col].values

    # --- classifier ---
    clf = HistGradientBoostingClassifier(
        max_depth=CFG.max_depth,
        learning_rate=CFG.learning_rate,
        max_iter=CFG.max_iter,
        min_samples_leaf=CFG.min_samples_leaf,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # tune p_min on validation using NET pnl proxy
    best_p = tune_p_min_on_val(
        clf, X_val, val_df,
        CFG.p_min_grid,
        CFG.taker_fee_bps,
        CFG.use_spread_cost
    )

    # evaluate classifier on test (with tuned gate)
    proba_test = clf.predict_proba(X_test)
    classes = clf.classes_
    argmax = classes[np.argmax(proba_test, axis=1)]
    maxp = np.max(proba_test, axis=1)
    sig_cls = argmax.copy()
    sig_cls[maxp < best_p["p_min"]] = 0

    cls_rep = {
        "ungated": cls_metrics(y_test, argmax),
        "gated": cls_metrics(y_test, sig_cls),
        "chosen_gate": best_p,
    }
    pnl_cls = pnl_proxy_from_signals(test_df, sig_cls, CFG.taker_fee_bps, CFG.use_spread_cost)

    # --- regressor ---
    reg = HistGradientBoostingRegressor(
        max_depth=CFG.max_depth,
        learning_rate=CFG.learning_rate,
        max_iter=CFG.max_iter,
        min_samples_leaf=CFG.min_samples_leaf,
        random_state=42,
    )
    reg.fit(X_train, train_df[yret_col].astype(float).values)

    best_dz = tune_deadzone_on_val(
        reg, X_val, val_df,
        CFG.deadzone_grid,
        CFG.taker_fee_bps,
        CFG.use_spread_cost
    )

    pred_test = reg.predict(X_test)
    sig_reg = np.zeros_like(pred_test, dtype=int)
    sig_reg[pred_test > best_dz["deadzone"]] = 1
    sig_reg[pred_test < -best_dz["deadzone"]] = -1
    pnl_reg = pnl_proxy_from_signals(test_df, sig_reg, CFG.taker_fee_bps, CFG.use_spread_cost)

    reg_rep = {
        "chosen_deadzone": best_dz,
        "ic_corr": float(np.corrcoef(test_df[yret_col].astype(float).values, pred_test)[0, 1]) if np.std(pred_test) > 0 else 0.0,
        "mse": float(np.mean((test_df[yret_col].astype(float).values - pred_test) ** 2)),
    }

    # summary stats
    def pnl_summary(ret: pd.Series, sig: np.ndarray) -> Dict[str, float]:
        trades = float(np.sum(sig != 0))
        return {
            "trades": trades,
            "mean_ret": float(ret.mean()),
            "sharpe_1s": float(sharpe_1s(ret.values)),
            "hit_rate": float((ret[ret != 0] > 0).mean()) if trades > 0 else 0.0,
        }

    report = {
        "dataset": dataset_path.name,
        "time_start": str(df.index.min()),
        "time_end": str(df.index.max()),
        "label_col": label_col,
        "yret_col": yret_col,
        "n_features": int(len(feat_cols)),
        "features": feat_cols,
        "config": {
            **CFG.__dict__,
            "embargo_seconds_effective": embargo_s,
        },
        "classifier": {
            "metrics": cls_rep,
            "pnl_test": pnl_summary(pnl_cls, sig_cls),
        },
        "regressor": {
            "metrics": reg_rep,
            "pnl_test": pnl_summary(pnl_reg, sig_reg),
        },
    }

    # write report
    out_report = REPORTS_DIR / f"tree_report_{CFG.product_slug}_{CFG.sample}_{CFG.horizon}.json"
    out_report.write_text(json.dumps(report, indent=2))
    print(f"\n[saved] report -> {out_report}")

    # save model bundle (so later we can run “paper inference”)
    out_model = MODELS_DIR / f"tree_{CFG.product_slug}_{CFG.sample}_{CFG.horizon}.joblib"
    bundle = {
        "dataset": dataset_path.name,
        "label_col": label_col,
        "yret_col": yret_col,
        "features": feat_cols,
        "cfg": CFG.__dict__,
        "embargo_seconds_effective": embargo_s,
        "classifier": clf,
        "classifier_p_min": best_p["p_min"],
        "regressor": reg,
        "regressor_deadzone": best_dz["deadzone"],
    }
    joblib.dump(bundle, out_model)
    print(f"[saved] model  -> {out_model}")

    print("\n[TEST] classifier pnl:", report["classifier"]["pnl_test"])
    print("[TEST] regressor  pnl:", report["regressor"]["pnl_test"])
    print("\nDone.")


if __name__ == "__main__":
    main()
