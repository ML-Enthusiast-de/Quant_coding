#!/usr/bin/env python
"""
FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.

Option A (Evaluation Fix) — Focus: 600s horizon

What this script does:
- Loads tob_dataset_<slug>_<sample>_600s.parquet
- Applies a PURGED / EMBARGOED strict time split:
    * train/val/test are chronological
    * BUT we drop the last embargo_seconds of train and val
      to prevent label overlap leakage (forward label uses t+horizon)
- Trains Ridge regression to predict y_ret_fwd_600s
- Converts predictions into trades using a cost-aware threshold:
    trade if |pred| > k * breakeven, direction = sign(pred)
- Chooses k on VAL (with min trade count), reports TEST for:
    (1) taker_realistic: fee=5 bps/side + spread cost
    (2) maker_optimistic: fee=1 bp/side + no spread cost
- Also reports an "oracle" bound using the true y_ret with the same threshold rule.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


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
class Scenario:
    name: str
    fee_bps_per_side: float
    use_spread_cost: bool


@dataclass(frozen=True)
class Config:
    product_slug: str = "btcusd"
    sample: str = "1s"
    horizon: str = "600s"

    # strict split fractions (chronological)
    train_frac: float = 0.70
    val_frac: float = 0.15

    # IMPORTANT: embargo should be ~= horizon seconds to prevent overlap leakage
    embargo_seconds: int = 600

    # threshold search
    k_grid: Tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
    min_trades_val: int = 50

    # model
    ridge_alpha: float = 1.0

    # scenarios
    scenarios: Tuple[Scenario, ...] = (
        Scenario("taker_realistic", fee_bps_per_side=5.0, use_spread_cost=True),
        Scenario("maker_optimistic", fee_bps_per_side=1.0, use_spread_cost=False),
    )

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
# Helpers
# -----------------------------------------------------------------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "ts" in df.columns:
        out = df.copy()
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
        out = out.dropna(subset=["ts"]).set_index("ts").sort_index()
        return out
    raise ValueError("Dataset must have a DatetimeIndex or a 'ts' column.")


def _bps_to_decimal(bps: float) -> float:
    return float(bps) / 10_000.0


def _pick_features(df: pd.DataFrame) -> List[str]:
    """
    Minimal, explicit leak guard:
      - drop any y_* columns
      - drop segment_id (it can leak segmentation/gaps/time)
      - keep numeric only
    """
    feats: List[str] = []
    for c in df.columns:
        if c.startswith("y_"):
            continue
        if c == "segment_id":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    if not feats:
        raise ValueError("No numeric feature columns found after leak-guard.")
    return feats


def _purged_time_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    embargo_seconds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split by row count, then PURGE by time:
      - drop last embargo_seconds of TRAIN
      - drop last embargo_seconds of VAL
    This prevents forward-label overlap across boundaries (t+horizon crossing).
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    df = df.sort_index()
    n = len(df)
    i_train = int(n * train_frac)
    i_val_end = int(n * (train_frac + val_frac))

    train_raw = df.iloc[:i_train].copy()
    val_raw = df.iloc[i_train:i_val_end].copy()
    test = df.iloc[i_val_end:].copy()

    embargo = pd.Timedelta(seconds=int(embargo_seconds))

    def purge_tail(x: pd.DataFrame) -> pd.DataFrame:
        if x.empty:
            return x
        tmax = x.index.max()
        keep_until = tmax - embargo
        return x.loc[x.index <= keep_until].copy()

    train = purge_tail(train_raw)
    val = purge_tail(val_raw)

    return train, val, test


def _ic_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 5 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 0.0
    s = x.std()
    if np.isclose(s, 0.0):
        return 0.0
    return float(x.mean() / s)


def _breakeven_rel(df: pd.DataFrame, fee_bps_per_side: float, use_spread_cost: bool) -> pd.Series:
    """
    Very simple breakeven (relative):
      breakeven ≈ 2*fee + spread_rel (if spread cost enabled)
    This matches what you've already been observing (~0.001... for taker 5bps/side).
    """
    fee = _bps_to_decimal(fee_bps_per_side)
    be = pd.Series(2.0 * fee, index=df.index, dtype=float)
    if use_spread_cost:
        if "spread_rel" in df.columns:
            be = be + df["spread_rel"].astype(float).fillna(0.0)
        else:
            # fallback: spread / mid
            be = be + (df["spread"].astype(float) / df["mid"].astype(float)).fillna(0.0)
    return be


def _mid_fwd_from_yret(df: pd.DataFrame, yret_col: str) -> pd.Series:
    # mid_fwd = mid * (1 + yret)
    return df["mid"].astype(float) * (1.0 + df[yret_col].astype(float))


def _pnl_proxy(
    df: pd.DataFrame,
    signal: np.ndarray,
    yret_col: str,
    fee_bps_per_side: float,
    use_spread_cost: bool,
) -> np.ndarray:
    """
    One-shot PnL proxy:
      - long enters at ask (if spread cost) else mid
      - short enters at bid (if spread cost) else mid
      - exits at mid_fwd derived from y_ret_fwd_H
      - fees charged twice (enter+exit)
    """
    sig = np.asarray(signal, dtype=int)

    mid = df["mid"].astype(float).values
    bid = df["best_bid"].astype(float).values
    ask = df["best_ask"].astype(float).values
    mid_fwd = _mid_fwd_from_yret(df, yret_col).astype(float).values

    fee = _bps_to_decimal(fee_bps_per_side)

    entry_long = ask if use_spread_cost else mid
    entry_short = bid if use_spread_cost else mid

    # gross returns
    r_long = (mid_fwd - entry_long) / entry_long
    r_short = (entry_short - mid_fwd) / entry_short

    # net (two-sided fees)
    r_long_net = (1.0 + r_long) * (1.0 - fee) * (1.0 - fee) - 1.0
    r_short_net = (1.0 + r_short) * (1.0 - fee) * (1.0 - fee) - 1.0

    out = np.zeros_like(mid, dtype=float)
    out[sig == 1] = r_long_net[sig == 1]
    out[sig == -1] = r_short_net[sig == -1]
    out[sig == 0] = 0.0
    return out


def _signals_from_pred(pred: np.ndarray, breakeven: np.ndarray, k: float) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    be = np.asarray(breakeven, dtype=float)
    thr = float(k) * be

    sig = np.zeros_like(pred, dtype=int)
    sig[pred > thr] = 1
    sig[pred < -thr] = -1
    return sig


def _stats_from_pnl(pnl: np.ndarray, signal: np.ndarray) -> Dict[str, float]:
    sig = np.asarray(signal, dtype=int)
    pnl = np.asarray(pnl, dtype=float)

    trades = int(np.sum(sig != 0))
    if trades == 0:
        return {"trades": 0.0, "mean_ret": 0.0, "sharpe": 0.0, "hit_rate": 0.0}

    pnl_trades = pnl[sig != 0]
    hit = float(np.mean(pnl_trades > 0)) if len(pnl_trades) else 0.0
    return {
        "trades": float(trades),
        "mean_ret": float(np.mean(pnl_trades)) if len(pnl_trades) else 0.0,
        "sharpe": float(_sharpe(pnl_trades)),
        "hit_rate": hit,
    }


def _choose_k_on_val(
    val_df: pd.DataFrame,
    pred_val: np.ndarray,
    yret_col: str,
    scenario: Scenario,
    k_grid: Tuple[float, ...],
    min_trades_val: int,
) -> Tuple[float, Dict[str, float]]:
    be = _breakeven_rel(val_df, scenario.fee_bps_per_side, scenario.use_spread_cost).values
    best_k = float(k_grid[0])
    best_stats = {"trades": 0.0, "mean_ret": 0.0, "sharpe": 0.0, "hit_rate": 0.0}
    best_score = -np.inf

    for k in k_grid:
        sig = _signals_from_pred(pred_val, be, float(k))
        pnl = _pnl_proxy(val_df, sig, yret_col, scenario.fee_bps_per_side, scenario.use_spread_cost)
        stats = _stats_from_pnl(pnl, sig)

        if stats["trades"] < float(min_trades_val):
            continue

        score = stats["sharpe"]
        if score > best_score:
            best_score = score
            best_k = float(k)
            best_stats = stats

    # If nothing meets min trades, fall back to k=first with whatever stats
    if best_score == -np.inf:
        k0 = float(k_grid[0])
        sig0 = _signals_from_pred(pred_val, be, k0)
        pnl0 = _pnl_proxy(val_df, sig0, yret_col, scenario.fee_bps_per_side, scenario.use_spread_cost)
        best_k = k0
        best_stats = _stats_from_pnl(pnl0, sig0)

    return best_k, best_stats


def main() -> None:
    slug = CFG.product_slug
    sample = CFG.sample
    hz = CFG.horizon

    path = DATASETS_DIR / f"tob_dataset_{slug}_{sample}_{hz}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    print("=" * 98)
    print(f"[config] slug={slug} sample={sample} horizon={hz}")
    print(f"[config] train_frac={CFG.train_frac} val_frac={CFG.val_frac} embargo_seconds={CFG.embargo_seconds}")
    print(f"[config] k_grid={CFG.k_grid} min_trades_val={CFG.min_trades_val} ridge_alpha={CFG.ridge_alpha}")
    print(f"[config] scenarios: {[s.name for s in CFG.scenarios]}")
    print("=" * 98)

    df = pd.read_parquet(path)
    df = _ensure_datetime_index(df)

    yret_col = f"y_ret_fwd_{hz}"
    if yret_col not in df.columns:
        raise ValueError(f"Expected forward return column '{yret_col}' not found in dataset.")

    # Drop NaNs needed for model + pnl
    required = [yret_col, "mid", "best_bid", "best_ask"]
    df = df.dropna(subset=required).copy()

    feat_cols = _pick_features(df)

    # 1) never allow target columns
    feat_cols = [c for c in feat_cols if not c.startswith("y_")]

    # 2) HARD BLOCK: forward-looking columns must not be features
    LEAK_BLOCKLIST = {"mid_fwd"}  # extend if you add others later
    feat_cols = [c for c in feat_cols if c not in LEAK_BLOCKLIST]

    # 3) sanity asserts (fail fast if leakage sneaks back in)
    assert "mid_fwd" not in feat_cols, "Leak: mid_fwd must not be a feature."


    train_df, val_df, test_df = _purged_time_split(
        df,
        train_frac=CFG.train_frac,
        val_frac=CFG.val_frac,
        embargo_seconds=CFG.embargo_seconds,
    )

    if CFG.verbose:
        print("-" * 98)
        print(f"[load] {path.name}")
        print(f"[rows] total={len(df):,} | train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
        print(f"[time] {df.index.min()} -> {df.index.max()}")
        print(f"[purge] embargo_seconds={CFG.embargo_seconds} (train&val tails removed)")
        print(f"[features] n={len(feat_cols)} (segment_id removed, y_* removed)")

    # Train ridge on y_ret
    X_train = train_df[feat_cols].values
    y_train = train_df[yret_col].astype(float).values

    X_val = val_df[feat_cols].values
    y_val = val_df[yret_col].astype(float).values

    X_test = test_df[feat_cols].values
    y_test = test_df[yret_col].astype(float).values

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(CFG.ridge_alpha))),
        ]
    )
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    ic_test = _ic_corr(y_test, pred_test)
    if CFG.verbose:
        print(f"[IC] test corr(y, pred) = {ic_test:.3f}")

    report: Dict[str, Any] = {
        "dataset": path.name,
        "time_start": str(df.index.min()),
        "time_end": str(df.index.max()),
        "horizon": hz,
        "yret_col": yret_col,
        "n_features": int(len(feat_cols)),
        "features": feat_cols,
        "split": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "config": {
            "train_frac": CFG.train_frac,
            "val_frac": CFG.val_frac,
            "embargo_seconds": CFG.embargo_seconds,
            "k_grid": list(CFG.k_grid),
            "min_trades_val": CFG.min_trades_val,
            "ridge_alpha": CFG.ridge_alpha,
        },
        "ic_test": float(ic_test),
        "scenarios": {},
    }

    for sc in CFG.scenarios:
        # choose k on VAL
        k_star, val_stats = _choose_k_on_val(
            val_df=val_df,
            pred_val=pred_val,
            yret_col=yret_col,
            scenario=sc,
            k_grid=CFG.k_grid,
            min_trades_val=CFG.min_trades_val,
        )

        # evaluate TEST model
        be_test = _breakeven_rel(test_df, sc.fee_bps_per_side, sc.use_spread_cost).values
        sig_test = _signals_from_pred(pred_test, be_test, k_star)
        pnl_test = _pnl_proxy(test_df, sig_test, yret_col, sc.fee_bps_per_side, sc.use_spread_cost)
        test_stats = _stats_from_pnl(pnl_test, sig_test)

        # oracle TEST
        sig_oracle = _signals_from_pred(y_test, be_test, k_star)
        pnl_oracle = _pnl_proxy(test_df, sig_oracle, yret_col, sc.fee_bps_per_side, sc.use_spread_cost)
        oracle_stats = _stats_from_pnl(pnl_oracle, sig_oracle)

        report["scenarios"][sc.name] = {
            "scenario": {"fee_bps_per_side": sc.fee_bps_per_side, "use_spread_cost": sc.use_spread_cost},
            "chosen_on_val": {"k": float(k_star), **{k: float(v) for k, v in val_stats.items()}},
            "test_model": {"k": float(k_star), **{k: float(v) for k, v in test_stats.items()}},
            "test_oracle": {"k": float(k_star), **{k: float(v) for k, v in oracle_stats.items()}},
        }

        print("-" * 98)
        print(f"[{sc.name}] VAL chosen k={k_star:g} | sharpe={val_stats['sharpe']:.3f} "
              f"mean={val_stats['mean_ret']:.3e} trades={int(val_stats['trades'])}")
        print(f"[{sc.name}] TEST model  | sharpe={test_stats['sharpe']:.3f} "
              f"mean={test_stats['mean_ret']:.3e} trades={int(test_stats['trades'])} hit={test_stats['hit_rate']:.3f}")
        print(f"[{sc.name}] TEST oracle | sharpe={oracle_stats['sharpe']:.3f} "
              f"mean={oracle_stats['mean_ret']:.3e} trades={int(oracle_stats['trades'])} hit={oracle_stats['hit_rate']:.3f}")

    out = REPORTS_DIR / f"optionA_purged_costaware_{slug}_{sample}_{hz}.json"
    out.write_text(json.dumps(report, indent=2))
    print("=" * 98)
    print(f"[saved] {out}")
    print("Done.")


if __name__ == "__main__":
    main()
