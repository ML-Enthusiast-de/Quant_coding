#!/usr/bin/env python
"""
FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.

Diagnose whether short-horizon TOB targets are large enough to overcome
(simple) spread + taker-fee costs.

Reads datasets from:
  L2_order_book_project/data/datasets/tob_dataset_<slug>_<sample>_<horizon>.parquet

Prints:
- label class balance (-1/0/+1)
- forward return distribution (abs & signed)
- spread distribution (abs & relative)
- estimated breakeven move under the same assumptions as pnl_proxy:
    enter by crossing spread (ask for long / bid for short), exit at future mid,
    plus taker fees for entry+exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# CONFIG (edit these, no CLI)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    product_slug: str = "btcusd"
    sample: str = "1s"
    horizons: Tuple[str, ...] = ("10s", "30s")

    # Cost assumptions (match your training script)
    taker_fee_bps: float = 5.0       # per side
    use_spread_cost: bool = True     # enter at bid/ask vs enter at mid

    # Output control
    quantiles: Tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99)


CFG = Config()


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "L2_order_book_project"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"


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


def label_cols_from_dataset_path(p: Path) -> tuple[str, str]:
    # filename ends with "..._<H>.parquet"
    horizon = p.stem.split("_")[-1]
    return f"y_dir3_fwd_{horizon}", f"y_ret_fwd_{horizon}"


def _bps_to_decimal(bps: float) -> float:
    return bps / 10_000.0


def _q_stats(x: pd.Series, qs: Tuple[float, ...]) -> dict:
    x = x.dropna().astype(float)
    if len(x) == 0:
        return {}
    return {f"q{int(100*q)}": float(x.quantile(q)) for q in qs}


def main() -> None:
    fee = _bps_to_decimal(CFG.taker_fee_bps)

    print("=" * 90)
    print(f"[config] slug={CFG.product_slug} sample={CFG.sample} horizons={CFG.horizons}")
    print(f"[config] taker_fee_bps={CFG.taker_fee_bps} use_spread_cost={CFG.use_spread_cost}")
    print("=" * 90)

    for hz in CFG.horizons:
        path = DATASETS_DIR / f"tob_dataset_{CFG.product_slug}_{CFG.sample}_{hz}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        print("\n" + "-" * 90)
        print(f"[load] {path.name}")
        df = pd.read_parquet(path)
        df = _ensure_datetime_index(df)

        ydir_col, yret_col = label_cols_from_dataset_path(path)

        need_cols = ["mid", "best_bid", "best_ask", ydir_col, yret_col]
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")

        # Basic stats
        ydir = df[ydir_col].astype(int)
        yret = df[yret_col].astype(float)

        counts = ydir.value_counts().to_dict()
        n = len(df)

        spread = (df["best_ask"].astype(float) - df["best_bid"].astype(float))
        mid = df["mid"].astype(float)
        spread_rel = spread / mid

        # Cost proxy to “exit at future mid”
        # long entry at ask => cost component ≈ (ask-mid)/mid = half_spread/mid
        # short entry at bid => cost component ≈ (mid-bid)/mid = half_spread/mid
        # + taker fees twice (enter+exit)
        half_spread_rel = (spread / 2.0) / mid

        if CFG.use_spread_cost:
            breakeven_rel = half_spread_rel + 2.0 * fee
        else:
            breakeven_rel = 2.0 * fee

        abs_move = yret.abs()
        move_vs_cost = abs_move / breakeven_rel.replace(0.0, np.nan)

        print(f"[rows] {n:,}  | time: {df.index.min()} -> {df.index.max()}")
        print(f"[labels] counts={counts}  | frac0={counts.get(0,0)/n:.3f}")

        print("\n[forward return] (relative, per horizon)")
        print("  abs(y_ret)  :", _q_stats(abs_move, CFG.quantiles))
        print("  y_ret (raw) :", _q_stats(yret, CFG.quantiles))

        print("\n[spread]")
        print("  spread abs  :", _q_stats(spread, CFG.quantiles))
        print("  spread rel  :", _q_stats(spread_rel, CFG.quantiles))

        print("\n[cost model] breakeven move (relative)")
        print("  breakeven   :", _q_stats(breakeven_rel, CFG.quantiles))

        print("\n[signal-to-cost] abs(move) / breakeven")
        print("  ratio       :", _q_stats(move_vs_cost, CFG.quantiles))

        # A very interpretable headline:
        # How often is |move| > breakeven?
        beat = (abs_move > breakeven_rel).mean()
        print(f"\n[headline] P(|move| > breakeven) = {beat:.3%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
