# src/signals_cross.py
from __future__ import annotations

import numpy as np
import pandas as pd

# Default cross-sectional feature set
CROSS_FEATURES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_21",
    "vol_10",
    "vol_20",
    "vol_60",
    "ma20_rel",
    "ma50_rel",
    "ma200_rel",
    "dow",
]

DEFAULT_TARGET_COL = "target_fwd_21"


def make_cross_sectional_signals(
    prices: pd.DataFrame,
    lookahead: int = 21,
) -> pd.DataFrame:
    """
    Build cross-sectional signals on a price panel.

    prices: DataFrame with
        index: DatetimeIndex (sorted)
        columns: tickers

    Returns a long-format DataFrame with MultiIndex (date, ticker)
    and columns = features + target_fwd_<lookahead>.

    All features use ONLY past information relative to each date.
    The target is the forward <lookahead>-day return.
    """
    prices = prices.sort_index()
    prices.index.name = "date"
    prices.columns.name = "ticker"

    # --- Time-series returns (per stock) ---
    ret_1 = prices.pct_change(1)
    ret_5 = prices.pct_change(5)
    ret_10 = prices.pct_change(10)
    ret_21 = prices.pct_change(21)

    # --- Rolling volatility of daily returns ---
    daily_ret = ret_1
    vol_10 = daily_ret.rolling(10).std()
    vol_20 = daily_ret.rolling(20).std()
    vol_60 = daily_ret.rolling(60).std()

    # --- Moving-average relative distances ---
    ma20 = prices.rolling(20).mean()
    ma50 = prices.rolling(50).mean()
    ma200 = prices.rolling(200).mean()

    ma20_rel = ma20 / prices - 1.0
    ma50_rel = ma50 / prices - 1.0
    ma200_rel = ma200 / prices - 1.0

    # --- Forward lookahead return (target) ---
    # target_fwd_L(t) = (P_{t+L} - P_t) / P_t
    fwd_ret = prices.shift(-lookahead) / prices - 1.0
    target_col = f"target_fwd_{lookahead}"

    # Stack into long format
    def stack(df: pd.DataFrame) -> pd.Series:
        df.index.name = "date"
        df.columns.name = "ticker"
        return df.stack(dropna=False)

    data = {
        "ret_1": stack(ret_1),
        "ret_5": stack(ret_5),
        "ret_10": stack(ret_10),
        "ret_21": stack(ret_21),
        "vol_10": stack(vol_10),
        "vol_20": stack(vol_20),
        "vol_60": stack(vol_60),
        "ma20_rel": stack(ma20_rel),
        "ma50_rel": stack(ma50_rel),
        "ma200_rel": stack(ma200_rel),
        target_col: stack(fwd_ret),
    }

    signals_df = pd.DataFrame(data)

    # Add calendar feature: day-of-week (0=Mon, 4=Fri)
    idx_dates = signals_df.index.get_level_values("date")
    signals_df["dow"] = idx_dates.dayofweek.astype(np.int8)

    # Drop rows with any NaNs (from rolling windows or forward shift)
    signals_df = signals_df.dropna()

    return signals_df


def build_cross_sectional_matrix(
    signals_df: pd.DataFrame,
    feature_names: list[str] | None = None,
    target_col: str | None = None,
):
    """
    Turn a long-format signal dataframe into X, y, dates, tickers.

    signals_df: index = MultiIndex(date, ticker)
    """
    if feature_names is None:
        feature_names = CROSS_FEATURES

    if target_col is None:
        # assume 21-day target by default
        target_cols = [c for c in signals_df.columns if c.startswith("target_fwd_")]
        if not target_cols:
            raise ValueError("No target_fwd_* column found in signals_df.")
        if len(target_cols) > 1:
            raise ValueError(
                f"Multiple target_fwd_* columns found: {target_cols}. "
                "Please pass target_col explicitly."
            )
        target_col = target_cols[0]

    X = signals_df[feature_names].values.astype(np.float32)
    y = signals_df[target_col].values.astype(np.float32)

    dates = signals_df.index.get_level_values("date")
    tickers = signals_df.index.get_level_values("ticker")

    return X, y, dates, tickers
