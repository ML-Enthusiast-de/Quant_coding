#!/usr/bin/env python
"""
FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.

Build longer-horizon TOB datasets by reusing an existing (short-horizon) dataset
as the feature base, then recomputing labels/targets for longer horizons.

Why this script exists:
- Your build_tob_dataset.py already engineered features (returns/rolling stats/etc.).
- For Option B, we keep FEATURES fixed and only change the HORIZON TARGET/LABEL.
- This isolates the effect of horizon length.

Outputs:
  L2_order_book_project/data/datasets/tob_dataset_<slug>_<sample>_<H>.parquet

Each output contains:
- same feature columns as the base dataset
- y_ret_fwd_<H> : forward mid return over horizon H
- y_dir3_fwd_<H>: 3-class label in {-1,0,1} via deadzone
- mid_fwd       : forward mid price (for PnL proxy)

No CLI needed: edit Config and run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Make imports work reliably (repo root on sys.path)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Quant_coding
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -----------------------------------------------------------------------------
# CONFIG (edit these)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    product_slug: str = "btcusd"
    sample: str = "1s"

    # Use an existing dataset as "feature base"
    # (must exist): tob_dataset_btcusd_1s_10s.parquet for example
    base_horizon: str = "10s"

    # New horizons to generate (Option B: longer horizons)
    new_horizons: tuple[str, ...] = ("60s", "120s", "300s", "600s")

    # 3-class deadzone in RELATIVE return units
    # Keep it explicit; you can tune later.
    label_deadzone: float = 5e-05

    verbose: bool = True


CFG = Config()


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = REPO_ROOT / "L2_order_book_project"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


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
    raise ValueError("Dataset must have DatetimeIndex or 'ts' column.")


def _label_cols(horizon: str) -> tuple[str, str]:
    return f"y_dir3_fwd_{horizon}", f"y_ret_fwd_{horizon}"


def _horizon_steps(sample: str, horizon: str) -> int:
    s_td = pd.Timedelta(sample)
    h_td = pd.Timedelta(horizon)
    steps_f = h_td / s_td
    steps = int(round(float(steps_f)))
    # sanity: require integer alignment
    if not np.isclose(steps_f, steps, rtol=0, atol=1e-9):
        raise ValueError(f"Horizon {horizon} is not an integer multiple of sample {sample}.")
    if steps <= 0:
        raise ValueError("Computed non-positive steps.")
    return steps


def _compute_forward_mid_by_segment(mid: pd.Series, seg: pd.Series, steps: int) -> pd.Series:
    # shift within segment only
    return mid.groupby(seg).shift(-steps)


def _make_dir3_label(y_ret: pd.Series, deadzone: float) -> pd.Series:
    y = y_ret.to_numpy(dtype=float)
    lab = np.zeros_like(y, dtype=int)
    lab[y > deadzone] = 1
    lab[y < -deadzone] = -1
    return pd.Series(lab, index=y_ret.index, dtype="int8")


def _drop_existing_targets(df: pd.DataFrame) -> pd.DataFrame:
    # remove any old label/target columns + mid_fwd from the base
    drop = [c for c in df.columns if c.startswith("y_dir3_fwd_") or c.startswith("y_ret_fwd_")]
    if "mid_fwd" in df.columns:
        drop.append("mid_fwd")
    if drop:
        df = df.drop(columns=drop)
    return df


def main() -> None:
    base_path = DATASETS_DIR / f"tob_dataset_{CFG.product_slug}_{CFG.sample}_{CFG.base_horizon}.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_path}")

    if CFG.verbose:
        print("=" * 90)
        print(f"[load base] {base_path}")
    base = pd.read_parquet(base_path)
    base = _ensure_datetime_index(base)

    required = {"segment_id", "mid"}
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f"Base dataset missing required columns: {sorted(missing)}")

    # Keep the features, drop old targets
    feat_df = _drop_existing_targets(base)

    # For each new horizon, compute new targets/labels and save new dataset file
    for hz in CFG.new_horizons:
        steps = _horizon_steps(CFG.sample, hz)
        if CFG.verbose:
            print("-" * 90)
            print(f"[build] horizon={hz}  (steps={steps} on sample={CFG.sample})")

        df = feat_df.copy()

        mid = df["mid"].astype(float)
        seg = df["segment_id"]

        mid_fwd = _compute_forward_mid_by_segment(mid, seg, steps)
        y_ret = (mid_fwd / mid) - 1.0

        ydir_col, yret_col = _label_cols(hz)
        df["mid_fwd"] = mid_fwd.astype(float)
        df[yret_col] = y_ret.astype(float)
        df[ydir_col] = _make_dir3_label(df[yret_col], CFG.label_deadzone)

        # drop rows without forward value (near segment ends)
        df = df.dropna(subset=["mid_fwd", yret_col, ydir_col])

        out_path = DATASETS_DIR / f"tob_dataset_{CFG.product_slug}_{CFG.sample}_{hz}.parquet"
        df.to_parquet(out_path)

        counts = df[ydir_col].value_counts().to_dict()
        frac0 = counts.get(0, 0) / max(1, len(df))

        if CFG.verbose:
            print(f"[save] {out_path}")
            print(f"[rows] {len(df):,}  label_counts={counts}  frac0={frac0:.3f}")

    if CFG.verbose:
        print("\nDone. Next: train baselines across short+long horizons to compare.")


if __name__ == "__main__":
    main()
