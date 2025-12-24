#!/usr/bin/env python
"""
FOR RESEARCH PURPOSE ONLY. NOT FOR PRODUCTION USE.

Inference script: SP500 cross-sectional tree model (21d horizon)

- Loads the trained "live" model bundle from models/sp500_tree_cs_21d_live.pkl
- Downloads / loads recent SP500 prices
- Builds features for the LAST date only
- Outputs top N tickers to LONG and bottom N to SHORT (net of costs)
- Logs full-universe predictions + side/rank to parquet
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------
# Setup project root + imports from src
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loading_cross import load_sp500_adj_close
from src.signals_cross import CROSS_FEATURES  # (optional) kept for reference / consistency


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
LOG_PATH = Path(PROJECT_ROOT) / "data" / "paper_trade" / "sp500_cs_predictions.parquet"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def log_predictions(
    as_of_date: pd.Timestamp,
    df_long: pd.DataFrame,
    df_short: pd.DataFrame,
    df_all: pd.DataFrame,
    config: dict,
    log_path: Path = LOG_PATH,
) -> None:
    """
    df_long, df_short: subsets of df_all, indexed by ticker
    df_all: full-universe predictions, indexed by ticker, must contain:
        [pred_fwd_21, pred_fwd_21_net, pred_daily_net, edge_vs_eqw_daily]
    config: dict with model_name, model_version, lookahead, q_live, cost_bps, etc.
    """

    required_cols = {"pred_fwd_21", "pred_fwd_21_net", "pred_daily_net", "edge_vs_eqw_daily"}
    missing = required_cols - set(df_all.columns)
    if missing:
        raise ValueError(f"log_predictions: df_all missing columns: {sorted(missing)}")

    # ranks over full universe (1 = best)
    rank = df_all["pred_daily_net"].rank(ascending=False, method="first")
    df_all = df_all.assign(rank=rank)

    # side = long/short/flat based on membership
    side = pd.Series("flat", index=df_all.index, dtype="object")
    side.loc[df_long.index] = "long"
    side.loc[df_short.index] = "short"

    log_df = pd.DataFrame(
        {
            "as_of_date": as_of_date,
            "ticker": df_all.index,
            "lookahead": config["lookahead"],
            "side": side.values,
            "rank": df_all["rank"].values,
            "pred_fwd_21": df_all["pred_fwd_21"].values,
            "pred_fwd_21_net": df_all["pred_fwd_21_net"].values,
            "pred_daily_net": df_all["pred_daily_net"].values,
            "edge_vs_eqw_daily": df_all["edge_vs_eqw_daily"].values,
            "model_name": config.get("model_name", "sp500_tree_cs_21d"),
            "model_version": config.get("model_version", "unknown"),
            "q_live": config.get("q_live", np.nan),
            "cost_bps": config.get("cost_bps", np.nan),
            "train_start": config.get("train_start", None),
            "train_end": config.get("train_end", None),
        }
    )

    # append (or create) parquet log, de-dup on (as_of_date, ticker, model_version)
    if log_path.exists():
        old = pd.read_parquet(log_path)
        key_cols = ["as_of_date", "ticker", "model_version"]
        old_idx = old.set_index(key_cols).index
        new_idx = log_df.set_index(key_cols).index
        old = old[~old_idx.isin(new_idx)]
        combined = pd.concat([old, log_df], ignore_index=True)
    else:
        combined = log_df

    combined.to_parquet(log_path)
    print(f"Logged {len(log_df)} predictions to {log_path}")


# ---------------------------------------------------------------------
# Helper: build features for the last date only
# ---------------------------------------------------------------------
def build_live_feature_matrix(
    prices: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """
    prices: DataFrame (date x ticker) of adjusted close, sorted by date.

    Returns:
        as_of_date: last date in prices
        X_live_df:  DataFrame index=ticker, columns=feature_names
    """
    prices = prices.sort_index()
    last_date = prices.index[-1]

    # Daily returns
    ret_1  = prices.pct_change(1)
    ret_5  = prices.pct_change(5)
    ret_10 = prices.pct_change(10)
    ret_21 = prices.pct_change(21)

    # Rolling vol
    vol_10 = ret_1.rolling(10).std()
    vol_20 = ret_1.rolling(20).std()
    vol_60 = ret_1.rolling(60).std()

    # Moving averages
    ma10  = prices.rolling(10).mean()
    ma20  = prices.rolling(20).mean()
    ma50  = prices.rolling(50).mean()
    ma200 = prices.rolling(200).mean()

    last_price = prices.loc[last_date]

    feats = pd.DataFrame(index=prices.columns)

    # --- Core return features ---
    if "ret_1" in feature_names:
        feats["ret_1"] = ret_1.loc[last_date]
    if "ret_5" in feature_names:
        feats["ret_5"] = ret_5.loc[last_date]
    if "ret_10" in feature_names:
        feats["ret_10"] = ret_10.loc[last_date]
    if "ret_21" in feature_names:
        feats["ret_21"] = ret_21.loc[last_date]

    # --- Volatility features ---
    if "vol_10" in feature_names:
        feats["vol_10"] = vol_10.loc[last_date]
    if "vol_20" in feature_names:
        feats["vol_20"] = vol_20.loc[last_date]
    if "vol_60" in feature_names:
        feats["vol_60"] = vol_60.loc[last_date]
    if "vol_ratio_10_60" in feature_names:
        feats["vol_ratio_10_60"] = feats["vol_10"] / feats["vol_60"]

    # --- Moving-average distances ---
    if "ma10_rel" in feature_names:
        feats["ma10_rel"] = ma10.loc[last_date] / last_price - 1.0
    if "ma20_rel" in feature_names:
        feats["ma20_rel"] = ma20.loc[last_date] / last_price - 1.0
    if "ma50_rel" in feature_names:
        feats["ma50_rel"] = ma50.loc[last_date] / last_price - 1.0
    if "ma200_rel" in feature_names:
        feats["ma200_rel"] = ma200.loc[last_date] / last_price - 1.0

    # --- Simple trend flags ---
    if "trend_up_50" in feature_names:
        feats["trend_up_50"] = (last_price > ma50.loc[last_date]).astype(float)
    if "trend_up_200" in feature_names:
        feats["trend_up_200"] = (last_price > ma200.loc[last_date]).astype(float)

    # --- Calendar feature: day-of-week ---
    if "dow" in feature_names:
        feats["dow"] = float(last_date.dayofweek)  # 0=Mon, ..., 4=Fri

    # Keep only the columns the model expects, in the right order
    feats = feats[feature_names]

    # Drop tickers with any NaNs (too little history etc.)
    feats = feats.dropna(how="any")

    return last_date, feats


# ---------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------
def run_inference(
    model_path: Path | None = None,
    n_long: int = 20,
    n_short: int = 20,
    price_start: str = "2015-01-01",
    force_download: bool = False,
    do_log: bool = True,
):
    """
    Load model bundle and produce long/short picks for the last date.

    price_start:
        Earliest date to load prices from. We just need enough to compute moving averages
        (up to 200d) and lookahead returns.
    """
    if model_path is None:
        model_path = Path(PROJECT_ROOT) / "models" / "sp500_tree_cs_21d_live.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    q_live = bundle.get("q_live", np.nan)
    lookahead = int(bundle["lookahead"])
    cost_bps = float(bundle["cost_bps"])  # expected to be decimal (e.g. 0.001 = 10 bps)
    feature_names = bundle["features"]

    # --- Load recent price data ---
    prices = load_sp500_adj_close(start=price_start, force_download=force_download)

    # We only need the last ~ (200 + lookahead) days to compute features
    min_rows = 200 + lookahead + 5
    if len(prices) > min_rows:
        prices = prices.iloc[-min_rows:]

    # --- Build features for last date ---
    as_of_date, X_live_df = build_live_feature_matrix(prices, feature_names)
    if X_live_df.empty:
        raise RuntimeError("No valid tickers after feature construction (all NaNs?).")

    tickers_live = X_live_df.index.to_numpy()
    X_live = X_live_df.values

    # --- Model predictions: forward returns over lookahead ---
    pred_fwd = model.predict(X_live)

    # Apply round-trip transaction cost assumption for the horizon
    pred_fwd_net = (1.0 + pred_fwd) * (1.0 - cost_bps) - 1.0

    # Convert to daily net return for comparability
    pred_daily_net = (1.0 + pred_fwd_net) ** (1.0 / lookahead) - 1.0

    # Equal-weight baseline over predicted universe
    eqw_fwd = float(np.mean(pred_fwd))
    eqw_fwd_net = (1.0 + eqw_fwd) * (1.0 - cost_bps) - 1.0
    eqw_daily = (1.0 + eqw_fwd_net) ** (1.0 / lookahead) - 1.0

    edge_vs_eqw = pred_daily_net - eqw_daily

    df_out = pd.DataFrame(
        {
            "ticker": tickers_live,
            "pred_fwd_21": pred_fwd,
            "pred_fwd_21_net": pred_fwd_net,
            "pred_daily_net": pred_daily_net,
            "edge_vs_eqw_daily": edge_vs_eqw,
        }
    ).set_index("ticker")

    df_sorted = df_out.sort_values("pred_daily_net", ascending=False)
    top_long = df_sorted.head(n_long).copy()
    top_short = df_sorted.tail(n_short).sort_values("pred_daily_net", ascending=True).copy()

    # --- Pretty print ---
    print("=" * 80)
    print(f"Paper-trade recommendations as of {as_of_date.date()}  |  horizon: {lookahead} days")
    print(f"Model trained on: {bundle.get('train_start', 'unknown')} – {bundle.get('train_end', 'unknown')}")
    print(f"Hyperparams (Optuna): {bundle.get('optuna_best_params', {})}")
    print(f"q_live (typical fraction long/short): {q_live}")
    print(f"Assumed round-trip cost per horizon: {cost_bps * 1e4:.1f} bps")
    print(f"Equal-weight baseline (predicted daily net): {eqw_daily:.6f}")
    print("=" * 80)
    print()

    print(f"Top {n_long} tickers to LONG (net of costs):")
    print(top_long.round(6))
    print()

    print(f"Bottom {n_short} tickers to SHORT (net of costs):")
    print(top_short.round(6))
    print()

    # --- Log predictions ---
    if do_log:
        config = {
            "model_name": bundle.get("model_name", "sp500_tree_cs_21d"),
            "model_version": bundle.get("model_version", model_path.stem),
            "lookahead": lookahead,
            "q_live": q_live,
            "cost_bps": cost_bps,
            "train_start": bundle.get("train_start", None),
            "train_end": bundle.get("train_end", None),
        }
        log_predictions(as_of_date, top_long, top_short, df_sorted, config)

    return {
        "as_of_date": as_of_date,
        "eqw_daily_net": eqw_daily,
        "top_long": top_long,
        "top_short": top_short,
        "raw": df_sorted,
    }


if __name__ == "__main__":
    run_inference()
