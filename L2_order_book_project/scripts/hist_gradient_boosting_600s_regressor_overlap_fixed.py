#!/usr/bin/env python
"""
FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.

Option A (cost-aware thresholding) with PROPER non-overlapping trade simulation.

Why this exists:
- The naive "one-shot PnL per timestamp" method is NOT valid for long horizons (e.g. 600s),
  because it implicitly opens overlapping positions every second.
- This script simulates actual trades:
    enter -> hold horizon -> exit -> next entry allowed after exit

Pipeline:
- Load dataset (tob_dataset_<slug>_<sample>_<horizon>.parquet)
- Strict time split + embargo around boundaries
- Train regressor on forward return y_ret_fwd_<horizon> (default Ridge)
- Option A signal: trade if |pred_ret| > k * breakeven(row)
- Simulate non-overlapping trades for:
    - model predictions
    - oracle (true y_ret)
- Select k on validation; evaluate on test
- Save report JSON

No CLI: edit Config and run.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------------
# Repo root on sys.path (so relative imports work elsewhere if needed)
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Quant_coding
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class CostScenario:
    name: str
    fee_bps_per_side: float
    use_spread_cost: bool


@dataclass(frozen=True)
class Config:
    product_slug: str = "btcusd"
    sample: str = "1s"
    horizon: str = "600s"  # <-- focus here

    train_frac: float = 0.70
    val_frac: float = 0.15

    # Embargo around split boundaries in seconds (for horizon overlap)
    embargo_seconds: int = 600  # for 600s horizon

    # Option A k grid: threshold = k * breakeven(row)
    k_grid: Tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
    min_trades_val: int = 50

    # Model
    ridge_alpha: float = 1.0

    # Scenarios to compare
    scenarios: Tuple[CostScenario, ...] = (
        CostScenario("taker_realistic", fee_bps_per_side=5.0, use_spread_cost=True),
        CostScenario("maker_optimistic", fee_bps_per_side=1.0, use_spread_cost=False),
    )

    verbose: bool = True


CFG = Config()

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = REPO_ROOT / "L2_order_book_project"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "ts" in df.columns:
        out = df.copy()
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
        out = out.dropna(subset=["ts"]).set_index("ts").sort_index()
        return out
    raise ValueError("Expected a DatetimeIndex or a 'ts' column.")


def _horizon_seconds(h: str) -> int:
    # supports "10s", "60s", "300s", "600s"
    if not h.endswith("s"):
        raise ValueError(f"Unsupported horizon format: {h} (expected like '600s')")
    return int(h[:-1])


def label_cols(horizon: str) -> Tuple[str, str]:
    return f"y_dir3_fwd_{horizon}", f"y_ret_fwd_{horizon}"


def choose_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Minimal, no-magic feature selection:
    - keep numeric columns
    - drop leakage/targets: y_*
    - drop forward-looking cols: *_fwd
    - drop segment_id (not a real feature; also can leak gaps)
    """
    drop_prefixes = ("y_",)
    drop_suffixes = ("_fwd",)

    cols = []
    for c in df.columns:
        if c == "segment_id":
            continue
        if any(c.startswith(p) for p in drop_prefixes):
            continue
        if any(c.endswith(s) for s in drop_suffixes):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    if not cols:
        raise ValueError("No numeric feature columns left after filtering.")
    return cols


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_index()
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    train = df.iloc[:i_train].copy()
    val = df.iloc[i_train:i_val].copy()
    test = df.iloc[i_val:].copy()
    return train, val, test


def apply_embargo(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    embargo_seconds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Purge windows around the split boundaries to prevent horizon overlap leakage.

    Boundary1 = max(train.ts)
      - drop last embargo_seconds from train
      - drop first embargo_seconds from val

    Boundary2 = max(val.ts)
      - drop last embargo_seconds from val
      - drop first embargo_seconds from test
    """
    if embargo_seconds <= 0:
        return train, val, test

    b1 = train.index.max()
    b2 = val.index.max()
    td = pd.Timedelta(seconds=int(embargo_seconds))

    train2 = train.loc[train.index <= (b1 - td)]
    val2 = val.loc[(val.index >= (b1 + td)) & (val.index <= (b2 - td))]
    test2 = test.loc[test.index >= (b2 + td)]

    return train2, val2, test2


def bps_to_dec(bps: float) -> float:
    return float(bps) / 10_000.0


def breakeven_move_rel(
    df: pd.DataFrame,
    fee_bps_per_side: float,
    use_spread_cost: bool,
    mid_col: str = "mid",
    bid_col: str = "best_bid",
    ask_col: str = "best_ask",
) -> pd.Series:
    """
    Breakeven REQUIRED absolute move (relative) to cover:
    - fees: enter + exit => 2 * fee
    - half-spread penalty if crossing spread (approx): (ask-bid)/(2*mid)

    This is intentionally simple and stable.
    """
    fee = bps_to_dec(fee_bps_per_side)
    out = pd.Series(2.0 * fee, index=df.index, dtype="float64")

    if use_spread_cost:
        mid = df[mid_col].astype(float)
        spr = (df[ask_col].astype(float) - df[bid_col].astype(float)).clip(lower=0.0)
        half_spread_rel = (spr / (2.0 * mid)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = out + half_spread_rel

    return out


def trade_returns_for_entries(
    df: pd.DataFrame,
    side: np.ndarray,  # +1 long, -1 short, 0 flat
    fee_bps_per_side: float,
    use_spread_cost: bool,
    bid_col: str = "best_bid",
    ask_col: str = "best_ask",
    mid_col: str = "mid",
    mid_fwd_col: str = "mid_fwd",
) -> np.ndarray:
    """
    Compute net return per entry row (one trade that exits at mid_fwd).
    Uses multiplicative fee model (enter + exit).
    """
    fee = bps_to_dec(fee_bps_per_side)

    bid = df[bid_col].astype(float).to_numpy()
    ask = df[ask_col].astype(float).to_numpy()
    mid = df[mid_col].astype(float).to_numpy()
    mid_fwd = df[mid_fwd_col].astype(float).to_numpy()

    if use_spread_cost:
        entry_long = ask
        entry_short = bid
    else:
        entry_long = mid
        entry_short = mid

    ret = np.zeros(len(df), dtype="float64")

    long_mask = (side == 1)
    short_mask = (side == -1)

    # long: (exit/entry)*(1-fee)^2 - 1
    if long_mask.any():
        entry = entry_long[long_mask]
        exitp = mid_fwd[long_mask]
        gross = exitp / entry
        net = gross * (1.0 - fee) * (1.0 - fee) - 1.0
        ret[long_mask] = net

    # short: (entry/exit)*(1-fee)^2 - 1
    if short_mask.any():
        entry = entry_short[short_mask]
        exitp = mid_fwd[short_mask]
        gross = entry / exitp
        net = gross * (1.0 - fee) * (1.0 - fee) - 1.0
        ret[short_mask] = net

    return ret


def sharpe_trade_level(x: np.ndarray) -> float:
    """
    Sharpe over trades (not per-second). Assumes trades are iid-ish.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    s = x.std()
    if np.isclose(s, 0.0):
        return 0.0
    return float(np.sqrt(len(x)) * x.mean() / s)


def simulate_nonoverlap_trades(
    df: pd.DataFrame,
    pred_ret: np.ndarray,
    horizon_seconds: int,
    k: float,
    scenario: CostScenario,
    yret_true: Optional[np.ndarray] = None,
    mode: str = "model",  # "model" or "oracle"
) -> Dict[str, float]:
    """
    Non-overlapping trade simulation:
    - At each time t (row), compute entry signal:
        model: sign(pred_ret) if |pred_ret| > k*breakeven else 0
        oracle: sign(yret_true) if |yret_true| > k*breakeven else 0
    - If enter, hold horizon_seconds; next entry only allowed after exit.
    - Exit uses mid_fwd at entry row (dataset provides it).

    Returns trade-level stats.
    """
    if mode not in ("model", "oracle"):
        raise ValueError("mode must be 'model' or 'oracle'")

    # must have mid_fwd to evaluate returns
    if "mid_fwd" not in df.columns:
        raise ValueError("Dataset needs 'mid_fwd' column for trade simulation.")

    be = breakeven_move_rel(df, scenario.fee_bps_per_side, scenario.use_spread_cost).to_numpy()
    be = np.asarray(be, dtype="float64")

    if mode == "oracle":
        if yret_true is None:
            raise ValueError("oracle mode requires yret_true")
        raw = np.asarray(yret_true, dtype="float64")
    else:
        raw = np.asarray(pred_ret, dtype="float64")

    # entry side from threshold
    side = np.zeros(len(df), dtype="int8")
    thr = float(k) * be
    side[raw > thr] = 1
    side[raw < -thr] = -1

    # Precompute per-row trade return IF we entered there
    per_entry_ret = trade_returns_for_entries(
        df,
        side=side,
        fee_bps_per_side=scenario.fee_bps_per_side,
        use_spread_cost=scenario.use_spread_cost,
    )

    # Non-overlap: if we enter at i, skip to first index >= t_i + horizon
    idx = df.index.values  # datetime64[ns, UTC]
    horizon_td = np.timedelta64(int(horizon_seconds), "s")

    trade_rets = []
    trade_hits = 0

    i = 0
    n = len(df)
    while i < n:
        if side[i] == 0:
            i += 1
            continue

        # must have a valid mid_fwd at this entry
        if not np.isfinite(per_entry_ret[i]):
            i += 1
            continue

        trade_rets.append(per_entry_ret[i])
        if per_entry_ret[i] > 0:
            trade_hits += 1

        exit_time = idx[i] + horizon_td
        # jump to the first row with ts >= exit_time
        i = int(np.searchsorted(idx, exit_time, side="left"))

    trade_rets = np.asarray(trade_rets, dtype="float64")
    trades = int(len(trade_rets))

    return {
        "k": float(k),
        "trades": float(trades),
        "mean_ret": float(trade_rets.mean()) if trades > 0 else 0.0,
        "sharpe": float(sharpe_trade_level(trade_rets)) if trades > 1 else 0.0,
        "hit_rate": float(trade_hits / trades) if trades > 0 else 0.0,
    }


def fit_ridge_predict(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: List[str], yret_col: str, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=float(alpha))),
        ]
    )

    Xtr = train_df[feat_cols].to_numpy()
    ytr = train_df[yret_col].astype(float).to_numpy()
    model.fit(Xtr, ytr)

    Xv = val_df[feat_cols].to_numpy()
    Xt = test_df[feat_cols].to_numpy()

    pred_tr = model.predict(Xtr)
    pred_val = model.predict(Xv)
    pred_test = model.predict(Xt)

    return pred_tr, pred_val, pred_test


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    horizon_sec = _horizon_seconds(CFG.horizon)
    ydir_col, yret_col = label_cols(CFG.horizon)

    dataset_path = DATASETS_DIR / f"tob_dataset_{CFG.product_slug}_{CFG.sample}_{CFG.horizon}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    print("=" * 98)
    print(f"[config] slug={CFG.product_slug} sample={CFG.sample} horizon={CFG.horizon}")
    print(f"[config] train_frac={CFG.train_frac} val_frac={CFG.val_frac} embargo_seconds={CFG.embargo_seconds}")
    print(f"[config] k_grid={CFG.k_grid} min_trades_val={CFG.min_trades_val} ridge_alpha={CFG.ridge_alpha}")
    print(f"[config] scenarios: {[s.name for s in CFG.scenarios]}")
    print("=" * 98)

    print("-" * 98)
    print(f"[load] {dataset_path.name}")
    df = pd.read_parquet(dataset_path)
    df = _ensure_datetime_index(df)

    # Must have required columns
    required = ["best_bid", "best_ask", "mid", "mid_fwd", yret_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # drop rows with NaNs in yret/midfwd upfront
    df = df.dropna(subset=[yret_col, "mid_fwd", "best_bid", "best_ask", "mid"]).copy()

    train, val, test = time_split(df, CFG.train_frac, CFG.val_frac)
    train, val, test = apply_embargo(train, val, test, CFG.embargo_seconds)

    print(f"[rows] total={len(df):,} | train={len(train):,} val={len(val):,} test={len(test):,}")
    print(f"[time] {df.index.min()} -> {df.index.max()}")
    print(f"[purge] embargo_seconds={CFG.embargo_seconds}")

    feat_cols = choose_feature_columns(df)
    print(f"[features] n={len(feat_cols)} (segment_id removed, y_* removed, *_fwd removed)")

    # Train regression model
    pred_tr, pred_val, pred_test = fit_ridge_predict(train, val, test, feat_cols, yret_col, CFG.ridge_alpha)

    # IC on test
    ic_test = corr_safe(test[yret_col].astype(float).to_numpy(), pred_test)
    print(f"[IC] test corr(y, pred) = {ic_test:.3f}")

    # For oracle sim we need yret arrays
    yret_val = val[yret_col].astype(float).to_numpy()
    yret_test = test[yret_col].astype(float).to_numpy()

    report: Dict[str, object] = {
        "dataset": dataset_path.name,
        "time_start": str(df.index.min()),
        "time_end": str(df.index.max()),
        "yret_col": yret_col,
        "horizon_seconds": horizon_sec,
        "split": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "ic_test": ic_test,
        "features": feat_cols,
        "config": {
            "product_slug": CFG.product_slug,
            "sample": CFG.sample,
            "horizon": CFG.horizon,
            "train_frac": CFG.train_frac,
            "val_frac": CFG.val_frac,
            "embargo_seconds": CFG.embargo_seconds,
            "k_grid": list(CFG.k_grid),
            "min_trades_val": CFG.min_trades_val,
            "ridge_alpha": CFG.ridge_alpha,
            "scenarios": [s.__dict__ for s in CFG.scenarios],
        },
        "scenarios": {},
    }

    print("-" * 98)

    for scen in CFG.scenarios:
        # Choose k on VAL by trade-level sharpe, respecting min_trades_val
        best = None
        best_key = (-np.inf, -np.inf)  # (sharpe, mean_ret)

        for k in CFG.k_grid:
            stats = simulate_nonoverlap_trades(
                df=val,
                pred_ret=pred_val,
                horizon_seconds=horizon_sec,
                k=k,
                scenario=scen,
                mode="model",
            )
            trades = int(stats["trades"])
            if trades < CFG.min_trades_val:
                continue

            key = (float(stats["sharpe"]), float(stats["mean_ret"]))
            if key > best_key:
                best_key = key
                best = stats

        # fallback if none meets min trades: choose the k with max sharpe anyway
        if best is None:
            fallback_best = None
            fallback_key = (-np.inf, -np.inf)
            for k in CFG.k_grid:
                stats = simulate_nonoverlap_trades(
                    df=val,
                    pred_ret=pred_val,
                    horizon_seconds=horizon_sec,
                    k=k,
                    scenario=scen,
                    mode="model",
                )
                key = (float(stats["sharpe"]), float(stats["mean_ret"]))
                if key > fallback_key:
                    fallback_key = key
                    fallback_best = stats
            best = fallback_best

        assert best is not None
        chosen_k = float(best["k"])

        # Evaluate on TEST for model + oracle
        test_model = simulate_nonoverlap_trades(
            df=test,
            pred_ret=pred_test,
            horizon_seconds=horizon_sec,
            k=chosen_k,
            scenario=scen,
            mode="model",
        )
        test_oracle = simulate_nonoverlap_trades(
            df=test,
            pred_ret=pred_test,  # unused
            horizon_seconds=horizon_sec,
            k=chosen_k,
            scenario=scen,
            yret_true=yret_test,
            mode="oracle",
        )

        print(f"[{scen.name}] VAL chosen k={chosen_k:g} | sharpe={best['sharpe']:.3f} mean={best['mean_ret']:.3e} trades={int(best['trades'])}")
        print(f"[{scen.name}] TEST model  | sharpe={test_model['sharpe']:.3f} mean={test_model['mean_ret']:.3e} trades={int(test_model['trades'])} hit={test_model['hit_rate']:.3f}")
        print(f"[{scen.name}] TEST oracle | sharpe={test_oracle['sharpe']:.3f} mean={test_oracle['mean_ret']:.3e} trades={int(test_oracle['trades'])} hit={test_oracle['hit_rate']:.3f}")
        print("-" * 98)

        report["scenarios"][scen.name] = {
            "scenario": scen.__dict__,
            "chosen_on_val": best,
            "test_model": test_model,
            "test_oracle": test_oracle,
        }

    out_path = REPORTS_DIR / f"optionA_nonoverlap_trade_sim_{CFG.product_slug}_{CFG.sample}_{CFG.horizon}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print("=" * 98)
    print(f"[saved] {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
