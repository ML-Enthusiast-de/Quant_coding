#!/usr/bin/env python
"""
FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.

Evaluate cost-aware tradability across horizons under different execution assumptions.

This script:
- Loads tob_dataset_*.parquet files from data/datasets
- Strict time split (train/val/test)
- Trains Ridge regression to predict forward return y_ret_fwd_<H>
- Converts predictions to trades using a COST-AWARE gate:
      trade if |pred_ret| > k * breakeven_ret
  where breakeven_ret is computed per-row from spread + fees.

- Compares 2 scenarios:
    1) taker_realistic: fee=5 bps per side, pay spread (enter at ask/bid, exit at mid_fwd)
    2) maker_optimistic: fee=1 bp per side, no spread cost (enter at mid, exit at mid_fwd)
       (NOTE: optimistic proxy; real maker needs fill/queue model)

- Picks best k on VALIDATION Sharpe (optionally requiring min trades)
- Evaluates on TEST and also computes an ORACLE feasibility benchmark:
    uses true forward return as prediction (upper bound under the same cost model)

Output:
  L2_order_book_project/data/reports/cost_scenario_report_<slug>_<sample>.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Any

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
    use_spread_cost: bool  # True: enter at ask/bid; False: enter at mid (optimistic proxy)


@dataclass(frozen=True)
class Config:
    product_slug: str = "btcusd"
    sample: str = "1s"
    horizons: Tuple[str, ...] = ("10s", "30s", "60s", "120s", "300s", "600s")

    train_frac: float = 0.70
    val_frac: float = 0.15

    # IMPORTANT: k grid starts at 1.0 (otherwise you allow trades that can’t breakeven)
    k_grid: Tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)

    # optional: avoid selecting k that yields almost no trades
    min_trades_val: int = 10

    # model
    ridge_alpha: float = 1.0

    # logging
    verbose: bool = True

    scenarios: Tuple[Scenario, ...] = (
        Scenario(name="taker_realistic", fee_bps_per_side=5.0, use_spread_cost=True),
        Scenario(name="maker_optimistic", fee_bps_per_side=1.0, use_spread_cost=False),
    )


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
TO_DROP_FROM_FEATURES_PREFIXES = ("y_",)  # drop all target columns


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "ts" in df.columns:
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
        return df
    raise ValueError("Dataset must have a DatetimeIndex or a 'ts' column.")


def label_cols_from_dataset_path(p: Path) -> tuple[str, str]:
    horizon = p.stem.split("_")[-1]  # ..._10s, ..._600s
    return f"y_dir3_fwd_{horizon}", f"y_ret_fwd_{horizon}"


def choose_feature_columns(df: pd.DataFrame, label_col: str, yret_col: str) -> List[str]:
    drop = {label_col, yret_col, "mid_fwd"}  # mid_fwd is derived from target; avoid leakage
    numeric_cols = []
    for c in df.columns:
        if c in drop:
            continue
        if any(c.startswith(pref) for pref in TO_DROP_FROM_FEATURES_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    if not numeric_cols:
        raise ValueError("No numeric feature columns found.")
    return numeric_cols


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    train = df.iloc[:i_train].copy()
    val = df.iloc[i_train:i_val].copy()
    test = df.iloc[i_val:].copy()
    return train, val, test


def _bps_to_decimal(bps: float) -> float:
    return bps / 10_000.0


def breakeven_return(df: pd.DataFrame, fee_bps_per_side: float, use_spread_cost: bool) -> pd.Series:
    """
    Compute per-row breakeven return (relative) consistent with our PnL proxy.

    If use_spread_cost=True (taker-style, enter at ask/bid and exit at mid_fwd):
      You effectively pay ~half-spread on entry (and none on exit if you mark out at mid_fwd).
      So spread cost term is (spread / (2*mid)).

    Plus round-trip fee: 2 * fee_per_side.

    If use_spread_cost=False:
      spread term = 0 (optimistic “mid fill” proxy).
    """
    fee_rt = 2.0 * _bps_to_decimal(fee_bps_per_side)
    if not use_spread_cost:
        return pd.Series(fee_rt, index=df.index, dtype="float64")

    mid = df["mid"].astype(float)
    spread = (df["best_ask"].astype(float) - df["best_bid"].astype(float)).clip(lower=0.0)

    # half-spread relative to mid
    half_spread_rel = (spread / (2.0 * mid)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (half_spread_rel + fee_rt).astype("float64")


def pnl_proxy_one_shot(
    df: pd.DataFrame,
    signal: np.ndarray,
    fee_bps_per_side: float,
    use_spread_cost: bool,
) -> pd.Series:
    """
    One-shot PnL proxy per timestamp:
      signal ∈ {-1, 0, +1}
      if long: enter at ask (or mid if spread_cost=False); exit at mid_fwd
      if short: enter at bid (or mid); exit at mid_fwd
      fees charged twice (enter + exit)
    """
    fee = _bps_to_decimal(fee_bps_per_side)

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

    ret = pd.Series(0.0, index=df.index)
    ret[sig == 1] = r_long_net[sig == 1]
    ret[sig == -1] = r_short_net[sig == -1]
    return ret


def sharpe_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 0.0
    s = x.std()
    if np.isclose(s, 0.0):
        return 0.0
    return float(x.mean() / s)


def ic_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 5:
        return 0.0
    if np.std(y_pred) <= 1e-12 or np.std(y_true) <= 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def signals_from_pred(pred_ret: np.ndarray, be: np.ndarray, k: float) -> np.ndarray:
    thr = k * be
    sig = np.zeros_like(pred_ret, dtype=int)
    sig[pred_ret > thr] = 1
    sig[pred_ret < -thr] = -1
    return sig


def eval_one_split(df_split: pd.DataFrame, pred_ret: np.ndarray, scen: Scenario, k: float) -> Dict[str, float]:
    be = breakeven_return(df_split, scen.fee_bps_per_side, scen.use_spread_cost).values
    sig = signals_from_pred(pred_ret, be, k)

    ret = pnl_proxy_one_shot(df_split, sig, scen.fee_bps_per_side, scen.use_spread_cost)

    trades = int((sig != 0).sum())
    hit = float((ret[ret != 0] > 0).mean()) if trades > 0 else 0.0

    return {
        "k": float(k),
        "trades": float(trades),
        "mean_ret": float(ret.mean()),
        "sharpe": float(sharpe_np(ret.values)),
        "hit_rate": hit,
    }


def eval_oracle(df_split: pd.DataFrame, yret: np.ndarray, scen: Scenario, k: float) -> Dict[str, float]:
    be = breakeven_return(df_split, scen.fee_bps_per_side, scen.use_spread_cost).values
    sig = signals_from_pred(yret, be, k)

    ret = pnl_proxy_one_shot(df_split, sig, scen.fee_bps_per_side, scen.use_spread_cost)

    trades = int((sig != 0).sum())
    hit = float((ret[ret != 0] > 0).mean()) if trades > 0 else 0.0

    return {
        "k": float(k),
        "trades": float(trades),
        "mean_ret": float(ret.mean()),
        "sharpe": float(sharpe_np(ret.values)),
        "hit_rate": hit,
    }


# -----------------------------------------------------------------------------
# Main per-dataset evaluation
# -----------------------------------------------------------------------------
def run_one_dataset(path: Path, cfg: Config) -> Dict[str, Any]:
    print("-" * 98)
    print(f"[load] {path.name}")
    df = pd.read_parquet(path)
    df = _ensure_datetime_index(df)

    label_col, yret_col = label_cols_from_dataset_path(path)

    required = {"mid", "best_bid", "best_ask", label_col, yret_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    # Make mid_fwd for PnL (derived from mid + yret) — this is NOT used as feature.
    df = df.copy()
    df["mid_fwd"] = df["mid"].astype(float) * (1.0 + df[yret_col].astype(float))

    feat_cols = choose_feature_columns(df, label_col=label_col, yret_col=yret_col)

    # Drop NaNs
    df = df.dropna(subset=feat_cols + [yret_col, "mid_fwd", "best_bid", "best_ask", "mid"])

    train_df, val_df, test_df = time_split(df, cfg.train_frac, cfg.val_frac)

    X_train = train_df[feat_cols].values
    X_val = val_df[feat_cols].values
    X_test = test_df[feat_cols].values

    y_train = train_df[yret_col].astype(float).values
    y_val = val_df[yret_col].astype(float).values
    y_test = test_df[yret_col].astype(float).values

    # Baseline model: Ridge regression (predict forward return)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=cfg.ridge_alpha)),
        ]
    )
    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    ds_rep: Dict[str, Any] = {
        "dataset": path.name,
        "time_start": str(df.index.min()),
        "time_end": str(df.index.max()),
        "yret_col": yret_col,
        "n_features": int(len(feat_cols)),
        "features": feat_cols,
        "split": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "ic_test": ic_corr(y_test, pred_test),
        "scenarios": {},
    }

    if cfg.verbose:
        print(f"[rows] {len(df):,} | train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
        print(f"[time] {ds_rep['time_start']} -> {ds_rep['time_end']}")
        print(f"[features] n={len(feat_cols)} | IC(test)={ds_rep['ic_test']:.3f}")

    for scen in cfg.scenarios:
        # Choose k on VAL: maximize Sharpe, but enforce min_trades_val
        best = None
        for k in cfg.k_grid:
            m = eval_one_split(val_df, pred_val, scen, k)
            if m["trades"] < cfg.min_trades_val:
                continue
            if (best is None) or (m["sharpe"] > best["sharpe"]):
                best = m

        if best is None:
            # fallback: choose the smallest k and accept that trades may be 0
            k0 = cfg.k_grid[0]
            best = eval_one_split(val_df, pred_val, scen, k0)

        chosen_k = float(best["k"])

        test_m = eval_one_split(test_df, pred_test, scen, chosen_k)
        oracle_m = eval_oracle(test_df, y_test, scen, chosen_k)

        ds_rep["scenarios"][scen.name] = {
            "scenario": {"fee_bps_per_side": scen.fee_bps_per_side, "use_spread_cost": scen.use_spread_cost},
            "chosen_on_val": best,
            "test_model": test_m,
            "test_oracle": oracle_m,
        }

        if cfg.verbose:
            print(
                f"[{scen.name}] VAL chosen k={chosen_k:g} | sharpe={best['sharpe']:.3f} "
                f"mean={best['mean_ret']:.3e} trades={int(best['trades'])}"
            )
            print(
                f"[{scen.name}] TEST model  trades={int(test_m['trades'])} sharpe={test_m['sharpe']:.3f} "
                f"mean={test_m['mean_ret']:.3e} hit={test_m['hit_rate']:.3f}"
            )
            print(
                f"[{scen.name}] TEST oracle trades={int(oracle_m['trades'])} sharpe={oracle_m['sharpe']:.3f} "
                f"mean={oracle_m['mean_ret']:.3e} hit={oracle_m['hit_rate']:.3f}"
            )

    return ds_rep


def main() -> None:
    print("=" * 98)
    print(f"[config] slug={CFG.product_slug} sample={CFG.sample} horizons={CFG.horizons}")
    print(f"[config] k_grid={CFG.k_grid} min_trades_val={CFG.min_trades_val} ridge_alpha={CFG.ridge_alpha}")
    print("[config] scenarios:")
    for s in CFG.scenarios:
        print(f"  - {s.name}: fee_bps_per_side={s.fee_bps_per_side} use_spread_cost={s.use_spread_cost}")
    print("=" * 98)

    reports: List[Dict[str, Any]] = []

    for hz in CFG.horizons:
        p = DATASETS_DIR / f"tob_dataset_{CFG.product_slug}_{CFG.sample}_{hz}.parquet"
        if not p.exists():
            print(f"[skip] missing {p.name}")
            continue
        rep = run_one_dataset(p, CFG)
        reports.append(rep)

    out = REPORTS_DIR / f"cost_scenario_report_{CFG.product_slug}_{CFG.sample}.json"
    out.write_text(json.dumps(reports, indent=2))
    print("\n" + "=" * 98)
    print(f"[saved] {out}")
    print("Done.")


if __name__ == "__main__":
    main()
