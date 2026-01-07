#!/usr/bin/env python
"""
FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.

Cost-aware baselines for Top-of-Book (TOB) datasets (crypto microstructure).

Goal:
- Compare SHORT horizons vs LONG horizons under a realistic constraint:
  "Only trade when predicted move is large enough to beat costs."
- Evaluate with a horizon-hold backtest:
  enter at time t, hold exactly H seconds, realize y_ret_fwd_H, then next decision.

What this script does:
1) Load datasets: tob_dataset_<slug>_<sample>_<H>.parquet
2) Strict time split: train / val / test (chronological)
3) Train Ridge regression to predict y_ret_fwd_<H>
4) Gate trades using a cost-aware threshold:
     breakeven ~ spread_rel + 2 * fee
     trade if |pred_ret| > k * breakeven
   Choose k by maximizing VAL net Sharpe.
5) Evaluate on TEST.
6) Print an ORACLE baseline (uses true future return to decide); shows feasibility given costs.

No CLI: edit Config below and run.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Ensure repo root on sys.path (works from VSCode / debugpy)
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Quant_coding
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------
# CONFIG (edit these)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    product_slug: str = "btcusd"
    sample: str = "1s"

    # Compare short vs long here:
    horizons: Tuple[str, ...] = ("10s", "30s", "60s", "120s", "300s", "600s")

    # Strict time split (by row order, chronological index)
    train_frac: float = 0.70
    val_frac: float = 0.15

    # Costs (very simplified, but consistent with your breakeven work)
    taker_fee_bps: float = 5.0          # per side
    use_spread_cost: bool = True        # include spread_rel in breakeven/cost model

    # Gate multiplier grid: trade only if |pred| > k * breakeven
    k_grid: Tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0)

    # Ridge model
    ridge_alpha: float = 1.0

    # Safety: drop target cols from features
    drop_target_prefix: str = "y_"

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
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
        return df
    raise ValueError("Dataset must have DatetimeIndex or a 'ts' column.")


def _parse_horizon_seconds(h: str) -> int:
    h = h.strip().lower()
    if h.endswith("s"):
        return int(float(h[:-1]))
    if h.endswith("m"):
        return int(float(h[:-1]) * 60)
    raise ValueError(f"Unsupported horizon format: {h} (use like '10s', '300s')")


def _bps_to_dec(bps: float) -> float:
    return float(bps) / 10_000.0


def label_cols_for_horizon(horizon: str) -> tuple[str, str]:
    # matches your dataset naming
    return f"y_dir3_fwd_{horizon}", f"y_ret_fwd_{horizon}"


def choose_feature_columns(df: pd.DataFrame, label_col: str, yret_col: str, drop_prefix: str) -> List[str]:
    # numeric-only, drop targets and obvious non-features
    drop = {label_col, yret_col, "mid_fwd"}  # mid_fwd is derived target-ish for PnL calcs
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in drop]
    # minimal leak guard: remove anything starting with y_
    numeric_cols = [c for c in numeric_cols if not str(c).startswith(drop_prefix)]
    if not numeric_cols:
        raise ValueError("No numeric feature columns after filtering.")
    return numeric_cols


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_index()
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train].copy(), df.iloc[i_train:i_val].copy(), df.iloc[i_val:].copy()


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return 0.0
    s = x.std()
    if np.isclose(s, 0.0):
        return 0.0
    return float(x.mean() / s * np.sqrt(len(x)))  # per-trade sharpe-ish


def horizon_hold_backtest(
    df: pd.DataFrame,
    pred_ret: np.ndarray,
    y_ret: np.ndarray,
    horizon_s: int,
    taker_fee_bps: float,
    use_spread_cost: bool,
    k: float,
    mode: str,
) -> Dict[str, float]:
    """
    Non-overlapping horizon-hold:
      - decide at t
      - realize y_ret_fwd_H at t
      - skip ahead to t + H seconds for next decision

    mode:
      - "model": use pred_ret for decisions
      - "oracle": use y_ret (true future return) for decisions (upper bound)
    """
    if len(df) == 0:
        return {"trades": 0.0, "mean_ret": 0.0, "sharpe": 0.0, "hit_rate": 0.0, "avg_cost": 0.0}

    fee = _bps_to_dec(taker_fee_bps)
    idx = df.index

    # breakeven proxy in "relative return" terms
    if "spread_rel" in df.columns:
        spread_rel = df["spread_rel"].astype(float).to_numpy()
    else:
        # fallback: spread / mid
        spread_rel = (df["spread"].astype(float) / df["mid"].astype(float)).to_numpy()

    breakeven = (spread_rel if use_spread_cost else 0.0) + 2.0 * fee  # enter+exit fee

    # decision variable
    if mode == "model":
        decision = pred_ret
    elif mode == "oracle":
        decision = y_ret
    else:
        raise ValueError("mode must be 'model' or 'oracle'")

    trade_rets: List[float] = []
    trade_costs: List[float] = []

    i = 0
    n = len(df)

    while i < n:
        # skip invalid label rows
        if not np.isfinite(y_ret[i]) or not np.isfinite(breakeven[i]) or not np.isfinite(decision[i]):
            i += 1
            continue

        thr = float(k) * float(breakeven[i])
        sig = 0
        if decision[i] > thr:
            sig = 1
        elif decision[i] < -thr:
            sig = -1

        if sig == 0:
            i += 1
            continue

        # net return proxy for a round-trip taker trade:
        # profit ~= sig * y_ret - (spread_rel + 2*fee)
        cost_i = float(breakeven[i])
        net = float(sig) * float(y_ret[i]) - cost_i

        trade_rets.append(net)
        trade_costs.append(cost_i)

        # advance time by horizon (non-overlapping)
        next_ts = idx[i] + pd.Timedelta(seconds=horizon_s)
        j = i + 1
        while j < n and idx[j] < next_ts:
            j += 1
        i = j

    if len(trade_rets) == 0:
        return {"trades": 0.0, "mean_ret": 0.0, "sharpe": 0.0, "hit_rate": 0.0, "avg_cost": float(np.nanmean(breakeven))}

    tr = np.asarray(trade_rets, dtype=float)
    hit = float(np.mean(tr > 0.0))
    return {
        "trades": float(len(tr)),
        "mean_ret": float(tr.mean()),
        "sharpe": float(sharpe(tr)),
        "hit_rate": hit,
        "avg_cost": float(np.mean(trade_costs)),
    }


def run_one_horizon(path: Path, horizon: str, cfg: Config) -> Dict[str, object]:
    print("-" * 90)
    print(f"[load] {path.name}")

    df = pd.read_parquet(path)
    df = _ensure_datetime_index(df)

    label_col, yret_col = label_cols_for_horizon(horizon)

    # sanity: must have required columns
    needed = {label_col, yret_col, "mid", "spread", "segment_id"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # build mid_fwd if not present (for later extensions; not needed for this backtest)
    if "mid_fwd" not in df.columns:
        df = df.copy()
        df["mid_fwd"] = df["mid"].astype(float) * (1.0 + df[yret_col].astype(float))

    # features
    feat_cols = choose_feature_columns(df, label_col, yret_col, cfg.drop_target_prefix)

    # drop NaNs
    df = df.dropna(subset=feat_cols + [yret_col])

    # split
    train_df, val_df, test_df = time_split(df, cfg.train_frac, cfg.val_frac)

    X_train = train_df[feat_cols].to_numpy()
    y_train_ret = train_df[yret_col].astype(float).to_numpy()

    X_val = val_df[feat_cols].to_numpy()
    y_val_ret = val_df[yret_col].astype(float).to_numpy()

    X_test = test_df[feat_cols].to_numpy()
    y_test_ret = test_df[yret_col].astype(float).to_numpy()

    if cfg.verbose:
        print(f"[rows] {len(df):,} | train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
        print(f"[time] {df.index.min()} -> {df.index.max()}")
        # label distribution (dir3)
        counts = train_df[label_col].value_counts().to_dict() if label_col in train_df.columns else {}
        print(f"[train labels] {counts}")
        print(f"[features] n={len(feat_cols)}")

    # model: Ridge on forward return
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=cfg.ridge_alpha))])
    model.fit(X_train, y_train_ret)

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    horizon_s = _parse_horizon_seconds(horizon)

    # tune k on VAL
    best = {"k": None, "sharpe": -1e18, "mean_ret": 0.0, "trades": 0.0}
    val_stats_by_k = {}

    for k in cfg.k_grid:
        st = horizon_hold_backtest(
            val_df, pred_val, y_val_ret, horizon_s,
            taker_fee_bps=cfg.taker_fee_bps,
            use_spread_cost=cfg.use_spread_cost,
            k=k, mode="model",
        )
        val_stats_by_k[str(k)] = st
        if st["sharpe"] > best["sharpe"]:
            best = {"k": k, "sharpe": st["sharpe"], "mean_ret": st["mean_ret"], "trades": st["trades"]}

    k_star = float(best["k"]) if best["k"] is not None else float(cfg.k_grid[len(cfg.k_grid)//2])

    # test eval @ chosen k
    test_model = horizon_hold_backtest(
        test_df, pred_test, y_test_ret, horizon_s,
        taker_fee_bps=cfg.taker_fee_bps,
        use_spread_cost=cfg.use_spread_cost,
        k=k_star, mode="model",
    )

    # oracle (upper bound) on TEST using true future return
    test_oracle = horizon_hold_backtest(
        test_df, pred_test, y_test_ret, horizon_s,
        taker_fee_bps=cfg.taker_fee_bps,
        use_spread_cost=cfg.use_spread_cost,
        k=k_star, mode="oracle",
    )

    # basic correlation / IC on test (not cost-aware)
    ic = float(np.corrcoef(y_test_ret, pred_test)[0, 1]) if np.std(pred_test) > 0 else 0.0

    if cfg.verbose:
        print(f"[VAL] chosen k={k_star} | sharpe={best['sharpe']:.3f} mean_ret={best['mean_ret']:.3e} trades={best['trades']}")
        print(f"[TEST:model]  trades={test_model['trades']:.0f} sharpe={test_model['sharpe']:.3f} mean_ret={test_model['mean_ret']:.3e} hit={test_model['hit_rate']:.3f}")
        print(f"[TEST:oracle] trades={test_oracle['trades']:.0f} sharpe={test_oracle['sharpe']:.3f} mean_ret={test_oracle['mean_ret']:.3e} hit={test_oracle['hit_rate']:.3f}")
        print(f"[TEST] IC(corr)={ic:.3f}")

    return {
        "dataset": path.name,
        "horizon": horizon,
        "horizon_seconds": horizon_s,
        "time_start": str(df.index.min()),
        "time_end": str(df.index.max()),
        "n_rows": int(len(df)),
        "n_features": int(len(feat_cols)),
        "features": feat_cols,
        "config": {
            "taker_fee_bps": cfg.taker_fee_bps,
            "use_spread_cost": cfg.use_spread_cost,
            "k_grid": list(cfg.k_grid),
            "ridge_alpha": cfg.ridge_alpha,
            "train_frac": cfg.train_frac,
            "val_frac": cfg.val_frac,
        },
        "val": {
            "by_k": val_stats_by_k,
            "chosen": best,
        },
        "test": {
            "ic_corr": ic,
            "model": test_model,
            "oracle": test_oracle,
            "k": k_star,
        },
    }


def main() -> None:
    print("=" * 98)
    print(f"[config] slug={CFG.product_slug} sample={CFG.sample} horizons={CFG.horizons}")
    print(f"[config] taker_fee_bps={CFG.taker_fee_bps} use_spread_cost={CFG.use_spread_cost} k_grid={CFG.k_grid}")
    print("=" * 98)

    reports: List[Dict[str, object]] = []

    for h in CFG.horizons:
        p = DATASETS_DIR / f"tob_dataset_{CFG.product_slug}_{CFG.sample}_{h}.parquet"
        if not p.exists():
            print(f"[skip] missing: {p.name}")
            continue
        rep = run_one_horizon(p, h, CFG)
        reports.append(rep)

    if not reports:
        raise SystemExit("No datasets found. Build datasets first (build_tob_dataset.py).")

    out = REPORTS_DIR / f"costaware_horizon_report_{CFG.product_slug}_{CFG.sample}.json"
    out.write_text(json.dumps(reports, indent=2))
    print("\n" + "=" * 98)
    print(f"[saved] {out}")
    print("Done.")


if __name__ == "__main__":
    main()
