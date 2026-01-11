#!/usr/bin/env python
"""
FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.

Option A (cost-aware thresholding) @ 600s horizon.

What we do and why:
- Train a regressor on y_ret_fwd_600s (forward mid return).
  Why: regression is smoother than 3-class labels and lets us gate trades by "edge".
- Convert predicted return -> long/short/flat using per-row breakeven thresholds
  that depend on spread and fee assumptions (scenario).
  Why: even a decent predictor loses money if it trades when the move is too small
  to cover costs.
- Strict chronological split + embargo (purge tail of train/val by horizon seconds)
  Why: prevent forward-label overlap leakage across splits.

Outputs:
- JSON report saved to: L2_order_book_project/data/reports/
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor


# -----------------------------------------------------------------------------
# Make repo imports work reliably (repo root on sys.path)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Quant_coding
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROJECT_ROOT = REPO_ROOT / "L2_order_book_project"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    fee_bps_per_side: float
    use_spread_cost: bool  # True = enter at ask/bid, False = enter at mid


@dataclass(frozen=True)
class Config:
    product_slug: str = "btcusd"
    sample: str = "1s"
    horizon: str = "600s"

    train_frac: float = 0.70
    val_frac: float = 0.15

    # Embargo: purge last horizon seconds from train and from val (prevents overlap leakage)
    embargo_seconds: int = 600

    # k grid: multiplier on breakeven threshold (k>1 means "only trade when move is well above costs")
    k_grid: Tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
    min_trades_val: int = 50

    # Model
    max_depth: int = 3
    learning_rate: float = 0.05
    max_iter: int = 300
    min_samples_leaf: int = 50

    verbose: bool = True


CFG = Config()

SCENARIOS: Dict[str, Scenario] = {
    "taker_realistic": Scenario(fee_bps_per_side=5.0, use_spread_cost=True),
    "maker_optimistic": Scenario(fee_bps_per_side=1.0, use_spread_cost=False),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "ts" in df.columns:
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
        return df
    raise ValueError("Dataset must have a DatetimeIndex or a 'ts' column.")


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_index()
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    train = df.iloc[:i_train].copy()
    val = df.iloc[i_train:i_val].copy()
    test = df.iloc[i_val:].copy()
    return train, val, test


def purge_tail_by_time(df: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """
    Remove the last `seconds` of timestamps from df.
    This is the simplest embargo/purge that respects time gaps.
    """
    if len(df) == 0 or seconds <= 0:
        return df
    tmax = df.index.max()
    cutoff = tmax - pd.Timedelta(seconds=seconds)
    return df.loc[df.index <= cutoff].copy()


def _bps_to_decimal(bps: float) -> float:
    return float(bps) / 10_000.0


def choose_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Minimal, explicit feature selection:
    - numeric columns
    - drop segment_id
    - drop any y_* targets
    - drop any *_fwd helper columns (like mid_fwd) to avoid leakage
    """
    cols: List[str] = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c == "segment_id":
            continue
        if c.startswith("y_"):
            continue
        if c.endswith("_fwd") or c == "mid_fwd":
            continue
        cols.append(c)
    if not cols:
        raise ValueError("No numeric feature columns left after exclusions.")
    return cols


def ic_corr(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 10:
        return 0.0
    if np.std(yhat[m]) == 0.0:
        return 0.0
    return float(np.corrcoef(y[m], yhat[m])[0, 1])


def sharpe_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 0.0
    s = x.std()
    if np.isclose(s, 0.0):
        return 0.0
    return float(x.mean() / s)


def pnl_proxy_from_signals(
    df: pd.DataFrame,
    signal: np.ndarray,  # {-1,0,+1}
    scenario: Scenario,
    mid_col: str = "mid",
    bid_col: str = "best_bid",
    ask_col: str = "best_ask",
    mid_fwd_col: str = "mid_fwd",
) -> np.ndarray:
    """
    Simplified per-timestamp PnL proxy (same spirit as your earlier scripts).
    - long: enter at ask if use_spread_cost else mid; exit at mid_fwd
    - short: enter at bid if use_spread_cost else mid; exit at mid_fwd
    - fee applied twice (enter + exit)
    """
    fee = _bps_to_decimal(scenario.fee_bps_per_side)

    mid = df[mid_col].astype(float).values
    bid = df[bid_col].astype(float).values
    ask = df[ask_col].astype(float).values
    mid_fwd = df[mid_fwd_col].astype(float).values

    if scenario.use_spread_cost:
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

    out = np.zeros_like(r_long_net)
    out[sig == 1] = r_long_net[sig == 1]
    out[sig == -1] = r_short_net[sig == -1]
    out[sig == 0] = 0.0
    return out


def compute_required_return_thresholds(df: pd.DataFrame, scenario: Scenario) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-row breakeven thresholds expressed in *mid-return space* (y_ret = mid_fwd/mid - 1).

    We want thresholds for deciding whether to trade:
      - go long if predicted y_ret >  k * req_long
      - go short if predicted y_ret < -k * req_short_mag   (equivalently < req_short which is negative)

    Derivation (net breakeven for long):
      net = (mid_fwd/entry)*(1-fee)^2 - 1
      breakeven: mid_fwd > entry / (1-fee)^2

    Convert to required mid-return:
      req_long = entry / ((1-fee)^2 * mid) - 1

    For short:
      net = (entry/mid_fwd)*(1-fee)^2 - 1
      breakeven: mid_fwd < entry * (1-fee)^2

      req_short = (entry*(1-fee)^2 / mid) - 1  (negative number)
      req_short_mag = -req_short = 1 - entry*(1-fee)^2 / mid
    """
    fee = _bps_to_decimal(scenario.fee_bps_per_side)

    mid = df["mid"].astype(float).values
    bid = df["best_bid"].astype(float).values
    ask = df["best_ask"].astype(float).values

    if scenario.use_spread_cost:
        entry_long = ask
        entry_short = bid
    else:
        entry_long = mid
        entry_short = mid

    denom = (1.0 - fee) ** 2
    # required y_ret (mid_fwd/mid - 1) to breakeven
    req_long = entry_long / (denom * mid) - 1.0                 # positive
    req_short = (entry_short * denom) / mid - 1.0               # negative
    req_short_mag = -req_short                                  # positive magnitude

    # guard: no negatives due to numerical quirks
    req_long = np.maximum(req_long, 0.0)
    req_short_mag = np.maximum(req_short_mag, 0.0)
    return req_long, req_short_mag


def signals_optionA(pred_ret: np.ndarray, req_long: np.ndarray, req_short_mag: np.ndarray, k: float) -> np.ndarray:
    """
    Option A signal rule (cost-aware):
      long  if pred_ret >  k * req_long
      short if pred_ret < -k * req_short_mag
      else flat
    """
    pred_ret = np.asarray(pred_ret, dtype=float)
    sig = np.zeros_like(pred_ret, dtype=int)
    sig[pred_ret > (k * req_long)] = 1
    sig[pred_ret < (-k * req_short_mag)] = -1
    return sig


def oracle_signals(df: pd.DataFrame, scenario: Scenario, k: float) -> np.ndarray:
    """
    Oracle: uses TRUE future mid_fwd to decide if a trade would have been profitable
    under the same scenario and k-threshold gate.
    This is a sanity upper bound (not achievable in real life).
    """
    req_long, req_short_mag = compute_required_return_thresholds(df, scenario)
    # true forward mid-return
    y_true = (df["mid_fwd"].astype(float).values / df["mid"].astype(float).values) - 1.0
    return signals_optionA(y_true, req_long, req_short_mag, k=k)


def eval_one(df: pd.DataFrame, pred_ret: np.ndarray, scenario: Scenario, k: float) -> Dict[str, float]:
    req_long, req_short_mag = compute_required_return_thresholds(df, scenario)
    sig = signals_optionA(pred_ret, req_long, req_short_mag, k=k)
    ret = pnl_proxy_from_signals(df, sig, scenario)
    trades = int((sig != 0).sum())
    hit = float((ret[ret != 0] > 0).mean()) if trades > 0 else 0.0
    return {
        "k": float(k),
        "trades": float(trades),
        "mean_ret": float(np.mean(ret)) if len(ret) else 0.0,
        "sharpe": float(sharpe_np(ret)),
        "hit_rate": float(hit),
    }


def choose_k_on_val(val_df: pd.DataFrame, pred_val: np.ndarray, scenario: Scenario, k_grid: Tuple[float, ...], min_trades: int) -> float:
    best_k = k_grid[0]
    best_score = -1e18

    for k in k_grid:
        m = eval_one(val_df, pred_val, scenario, k=float(k))
        # require some activity; otherwise "no trades" can look artificially safe
        if m["trades"] < float(min_trades):
            continue
        # primary objective: Sharpe (you can swap to mean_ret later)
        score = m["sharpe"]
        if score > best_score:
            best_score = score
            best_k = float(k)

    return best_k


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    dataset_path = DATASETS_DIR / f"tob_dataset_{CFG.product_slug}_{CFG.sample}_{CFG.horizon}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {dataset_path}")

    yret_col = f"y_ret_fwd_{CFG.horizon}"
    label_col = f"y_dir3_fwd_{CFG.horizon}"  # optional; not used here, but useful for debugging

    print("=" * 98)
    print(f"[config] slug={CFG.product_slug} sample={CFG.sample} horizon={CFG.horizon}")
    print(f"[config] train_frac={CFG.train_frac} val_frac={CFG.val_frac} embargo_seconds={CFG.embargo_seconds}")
    print(f"[config] k_grid={CFG.k_grid} min_trades_val={CFG.min_trades_val}")
    print(f"[config] scenarios: {list(SCENARIOS.keys())}")
    print("=" * 98)

    print("-" * 98)
    print(f"[load] {dataset_path.name}")
    df = pd.read_parquet(dataset_path)
    df = _ensure_datetime_index(df)

    # Need these for PnL proxy
    required_cols = ["best_bid", "best_ask", "mid", "mid_fwd", yret_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Drop NaNs in essential columns
    df = df.dropna(subset=required_cols).copy()

    # Features (explicit exclusions)
    feat_cols = choose_feature_columns(df)

    # Split + embargo purge
    train_df, val_df, test_df = time_split(df, CFG.train_frac, CFG.val_frac)
    train_df_p = purge_tail_by_time(train_df, CFG.embargo_seconds)
    val_df_p = purge_tail_by_time(val_df, CFG.embargo_seconds)

    if CFG.verbose:
        print(f"[rows] total={len(df):,} | train={len(train_df_p):,} val={len(val_df_p):,} test={len(test_df):,}")
        print(f"[time] {df.index.min()} -> {df.index.max()}")
        print(f"[purge] embargo_seconds={CFG.embargo_seconds} (train&val tails removed)")
        print(f"[features] n={len(feat_cols)} (segment_id removed, y_* removed, *_fwd removed)")
        if label_col in df.columns:
            vc = df[label_col].value_counts().to_dict()
            print(f"[labels] {vc}")

    # Train regressor on forward return
    X_train = train_df_p[feat_cols].values
    y_train = train_df_p[yret_col].astype(float).values

    X_val = val_df_p[feat_cols].values
    y_val = val_df_p[yret_col].astype(float).values

    X_test = test_df[feat_cols].values
    y_test = test_df[yret_col].astype(float).values

    model = HistGradientBoostingRegressor(
        max_depth=CFG.max_depth,
        learning_rate=CFG.learning_rate,
        max_iter=CFG.max_iter,
        min_samples_leaf=CFG.min_samples_leaf,
        random_state=0,
    )
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    ic_test = ic_corr(y_test, pred_test)
    print(f"[IC] test corr(y, pred) = {ic_test:.3f}")
    print("-" * 98)

    report: Dict[str, object] = {
        "dataset": dataset_path.name,
        "time_start": str(df.index.min()),
        "time_end": str(df.index.max()),
        "horizon": CFG.horizon,
        "yret_col": yret_col,
        "n_features": int(len(feat_cols)),
        "features": feat_cols,
        "split": {
            "train": int(len(train_df_p)),
            "val": int(len(val_df_p)),
            "test": int(len(test_df)),
            "embargo_seconds": int(CFG.embargo_seconds),
        },
        "ic_test": float(ic_test),
        "scenarios": {},
    }

    # For each scenario: choose k on val, then evaluate on test + oracle
    for scen_name, scen in SCENARIOS.items():
        k_star = choose_k_on_val(val_df_p, pred_val, scen, CFG.k_grid, min_trades=CFG.min_trades_val)

        val_metrics = eval_one(val_df_p, pred_val, scen, k=k_star)
        test_model_metrics = eval_one(test_df, pred_test, scen, k=k_star)

        # Oracle upper bound (uses mid_fwd directly)
        oracle_sig = oracle_signals(test_df, scen, k=k_star)
        oracle_ret = pnl_proxy_from_signals(test_df, oracle_sig, scen)
        oracle_trades = int((oracle_sig != 0).sum())
        oracle_hit = float((oracle_ret[oracle_ret != 0] > 0).mean()) if oracle_trades > 0 else 0.0
        test_oracle_metrics = {
            "k": float(k_star),
            "trades": float(oracle_trades),
            "mean_ret": float(np.mean(oracle_ret)) if len(oracle_ret) else 0.0,
            "sharpe": float(sharpe_np(oracle_ret)),
            "hit_rate": float(oracle_hit),
        }

        print(f"[{scen_name}] VAL chosen k={k_star:g} | sharpe={val_metrics['sharpe']:.3f} mean={val_metrics['mean_ret']:.3e} trades={int(val_metrics['trades'])}")
        print(f"[{scen_name}] TEST model  | sharpe={test_model_metrics['sharpe']:.3f} mean={test_model_metrics['mean_ret']:.3e} trades={int(test_model_metrics['trades'])} hit={test_model_metrics['hit_rate']:.3f}")
        print(f"[{scen_name}] TEST oracle | sharpe={test_oracle_metrics['sharpe']:.3f} mean={test_oracle_metrics['mean_ret']:.3e} trades={int(test_oracle_metrics['trades'])} hit={test_oracle_metrics['hit_rate']:.3f}")
        print("-" * 98)

        report["scenarios"][scen_name] = {
            "scenario": {
                "fee_bps_per_side": float(scen.fee_bps_per_side),
                "use_spread_cost": bool(scen.use_spread_cost),
            },
            "chosen_on_val": val_metrics,
            "test_model": test_model_metrics,
            "test_oracle": test_oracle_metrics,
        }

    out_path = REPORTS_DIR / f"optionA_tree_costaware_{CFG.product_slug}_{CFG.sample}_{CFG.horizon}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print("=" * 98)
    print(f"[saved] {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
