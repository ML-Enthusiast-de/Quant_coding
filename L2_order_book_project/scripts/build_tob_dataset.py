#!/usr/bin/env python
"""
FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.

Build a model-ready dataset from reconstructed Top-of-Book (TOB) data.

Input (preferred):
  L2_order_book_project/data/processed/tob_<product>_ALL_<sample>.parquet

Fallback input:
  L2_order_book_project/data/processed/runs/tob_<product>_..._<sample>.parquet (all runs concatenated)

Output:
  L2_order_book_project/data/datasets/tob_dataset_<product>_<sample>_<horizon>.parquet

What it does:
- loads TOB
- runs sanity checks
- builds microstructure features (TOB-only baseline)
- builds forward labels (regression + 3-class direction) WITHIN segment_id
- writes one dataset per horizon

No CLI required. Edit CFG below.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Make imports/paths robust
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Quant_coding
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROJECT_ROOT = REPO_ROOT / "L2_order_book_project"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROCESSED_DIR / "runs"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# CONFIG (edit these; no CLI)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    product: str = "BTC-USD"
    sample: str = "1s"                 # must match your TOB resample grid
    prefer_all_file: bool = True       # use .../processed/tob_<slug>_ALL_<sample>.parquet if present
    horizons_s: tuple[int, ...] = (1, 5, 10, 30)  # forward horizons for labels

    # label deadzone to avoid micro-noise: 1bp = 0.0001
    # (for 3-class direction label: up/down/flat)
    deadzone_bps: float = 1.0

    # rolling windows in seconds (since sample is 1s, these are row windows)
    win_short: int = 5
    win_med: int = 30
    win_long: int = 120

    verbose: bool = True


CFG = Config()


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def _product_slug(product: str) -> str:
    return product.replace("-", "").lower()


def _ensure_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df is indexed by a UTC DatetimeIndex named 'ts'.
    Accepts either:
      - ts already as DatetimeIndex
      - ts as a column (string or datetime)
    """
    if isinstance(df.index, pd.DatetimeIndex):
        # make sure tz-aware UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = df.index.name or "ts"
        return df

    # if ts is a column, use it
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.loc[~ts.isna()].copy()
        df["ts"] = ts.loc[~ts.isna()].values
        df = df.set_index("ts").sort_index()
        return df

    raise ValueError("TOB must have a DatetimeIndex or a 'ts' column.")


def load_tob(cfg: Config) -> pd.DataFrame:
    slug = _product_slug(cfg.product)

    all_path = PROCESSED_DIR / f"tob_{slug}_ALL_{cfg.sample}.parquet"
    if cfg.prefer_all_file and all_path.exists():
        df = pd.read_parquet(all_path)
        df = _ensure_ts_index(df)
        if cfg.verbose:
            print(f"[load] using ALL file: {all_path}")
        return df

    # fallback: concat runs
    run_files = sorted(RUNS_DIR.glob(f"tob_{slug}_*_{cfg.sample}.parquet"))
    if not run_files:
        raise FileNotFoundError(
            f"No TOB files found. Expected ALL: {all_path} or runs in {RUNS_DIR}"
        )
    if cfg.verbose:
        print(f"[load] ALL not found -> concatenating {len(run_files)} run files from {RUNS_DIR}")

    dfs = []
    for p in run_files:
        d = pd.read_parquet(p)
        d = _ensure_ts_index(d)
        dfs.append(d)

    df = pd.concat(dfs, axis=0).sort_index()
    return df



# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
def sanity_report(tob: pd.DataFrame) -> None:
    # index should be datetime
    if not isinstance(tob.index, pd.DatetimeIndex):
        raise ValueError("TOB dataframe index must be a DatetimeIndex (ts).")

    required = {"segment_id", "best_bid", "best_ask", "bid_sz", "ask_sz", "spread", "mid", "microprice", "imbalance_1"}
    missing = required - set(tob.columns)
    if missing:
        raise ValueError(f"TOB missing required columns: {sorted(missing)}")

    # basic checks
    n = len(tob)
    bad_spread = int((tob["spread"] < 0).sum())
    crossed = int((tob["best_bid"] >= tob["best_ask"]).sum())
    zero_sizes = int(((tob["bid_sz"] <= 0) | (tob["ask_sz"] <= 0)).sum())

    print("=" * 80)
    print("[sanity] rows:", f"{n:,}")
    print("[sanity] time range:", tob.index.min(), "->", tob.index.max())
    print("[sanity] segments:", int(tob["segment_id"].nunique()))
    print("[sanity] negative spread rows:", bad_spread)
    print("[sanity] crossed book rows (bid>=ask):", crossed)
    print("[sanity] non-positive sizes rows:", zero_sizes)

    # quick summary stats
    print("[sanity] spread (abs) median:", float(tob["spread"].median()))
    print("[sanity] spread (rel) median:", float((tob["spread"] / tob["mid"]).median()))
    print("=" * 80)


# -----------------------------------------------------------------------------
# Feature engineering (TOB-only baseline)
# -----------------------------------------------------------------------------
def _within_segment(group: pd.DataFrame, fn):
    # helper: apply fn per segment, preserves index
    return group.groupby("segment_id", group_keys=False).apply(fn)


def add_features(tob: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = tob.copy()

    # --- basic rel features ---
    df["spread_rel"] = df["spread"] / df["mid"]
    df["micro_minus_mid"] = (df["microprice"] - df["mid"]) / df["mid"]

    # --- mid returns (within segment) ---
    def _rets(g: pd.DataFrame) -> pd.DataFrame:
        mid = g["mid"]
        out = pd.DataFrame(index=g.index)
        out["mid_ret_1"] = mid.pct_change(1)
        out["mid_ret_5"] = mid.pct_change(5)
        out["mid_ret_10"] = mid.pct_change(10)
        return out

    rets = _within_segment(df, _rets)
    df = df.join(rets)

    # --- imbalance dynamics ---
    def _imb(g: pd.DataFrame) -> pd.DataFrame:
        imb = g["imbalance_1"]
        out = pd.DataFrame(index=g.index)
        out["imb_chg_1"] = imb.diff(1)
        out["imb_rolling_mean_short"] = imb.rolling(cfg.win_short, min_periods=cfg.win_short).mean()
        out["imb_rolling_mean_med"] = imb.rolling(cfg.win_med, min_periods=cfg.win_med).mean()
        return out

    imb = _within_segment(df, _imb)
    df = df.join(imb)

    # --- volatility proxy on mid returns ---
    def _vol(g: pd.DataFrame) -> pd.DataFrame:
        r1 = g["mid"].pct_change(1)
        out = pd.DataFrame(index=g.index)
        out["vol_short"] = r1.rolling(cfg.win_short, min_periods=cfg.win_short).std()
        out["vol_med"] = r1.rolling(cfg.win_med, min_periods=cfg.win_med).std()
        out["vol_long"] = r1.rolling(cfg.win_long, min_periods=cfg.win_long).std()
        return out

    vol = _within_segment(df, _vol)
    df = df.join(vol)

    # --- event intensity proxy (how often TOB changed) ---
    # You already reduced rows upstream; on a 1s grid we approximate intensity by changes in mid/spread
    def _intensity(g: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=g.index)
        mid_changed = (g["mid"].diff() != 0).astype(float)
        spr_changed = (g["spread"].diff() != 0).astype(float)
        out["mid_change_flag"] = mid_changed
        out["spread_change_flag"] = spr_changed
        out["change_rate_short"] = (mid_changed + spr_changed).rolling(cfg.win_short, min_periods=cfg.win_short).mean()
        out["change_rate_med"] = (mid_changed + spr_changed).rolling(cfg.win_med, min_periods=cfg.win_med).mean()
        return out

    inten = _within_segment(df, _intensity)
    df = df.join(inten)

    return df


# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
def add_labels(df: pd.DataFrame, horizon_s: int, cfg: Config) -> pd.DataFrame:
    out = df.copy()

    # forward mid return within segment
    def _fwd(g: pd.DataFrame) -> pd.Series:
        return g["mid"].shift(-horizon_s) / g["mid"] - 1.0

    fwd_ret = out.groupby("segment_id", group_keys=False).apply(_fwd)
    out[f"y_ret_fwd_{horizon_s}s"] = fwd_ret

    # 3-class direction label with deadzone (bps)
    eps = cfg.deadzone_bps * 1e-4
    y = out[f"y_ret_fwd_{horizon_s}s"]
    out[f"y_dir3_fwd_{horizon_s}s"] = np.where(y > eps, 1, np.where(y < -eps, -1, 0)).astype("int8")

    return out


# -----------------------------------------------------------------------------
# Build + save dataset per horizon
# -----------------------------------------------------------------------------
def build_and_save(cfg: Config) -> None:
    tob = load_tob(cfg)
    tob = tob.sort_index()

    sanity_report(tob)

    df = add_features(tob, cfg)

    # core columns we always keep (useful for debugging)
    core_cols = [
        "segment_id",
        "best_bid", "best_ask", "bid_sz", "ask_sz",
        "spread", "mid", "microprice", "imbalance_1",
        "spread_rel", "micro_minus_mid",
    ]

    feature_cols = [
        # returns
        "mid_ret_1", "mid_ret_5", "mid_ret_10",
        # imbalance
        "imb_chg_1", "imb_rolling_mean_short", "imb_rolling_mean_med",
        # vol
        "vol_short", "vol_med", "vol_long",
        # intensity-ish
        "mid_change_flag", "spread_change_flag", "change_rate_short", "change_rate_med",
    ]

    # ensure we only reference existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    slug = _product_slug(cfg.product)

    for h in cfg.horizons_s:
        d = add_labels(df, h, cfg)

        label_cols = [f"y_ret_fwd_{h}s", f"y_dir3_fwd_{h}s"]

        # drop rows with NaNs in features/labels (rolling + forward shift)
        keep_cols = core_cols + feature_cols + label_cols
        keep_cols = [c for c in keep_cols if c in d.columns]

        ds = d[keep_cols].dropna().copy()

        out_path = DATASETS_DIR / f"tob_dataset_{slug}_{cfg.sample}_{h}s.parquet"
        ds.to_parquet(out_path, index=True)

        if cfg.verbose:
            counts = ds[label_cols[1]].value_counts().to_dict()
            print(f"[save] {out_path} rows={len(ds):,} label_counts={counts}")

    print("\nDone. Next step: train baseline models with strict time-split evaluation.")


if __name__ == "__main__":
    build_and_save(CFG)
