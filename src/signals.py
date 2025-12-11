# src/signals.py

import numpy as np
import pandas as pd


def make_basic_signals(prices: pd.Series) -> pd.DataFrame:
    """
    Build a basic set of signals from a price series.
    Each row t uses only info up to time t.
    Target is NEXT-day return.
    """
    # Ensure prices are sorted and have a proper DatetimeIndex
    prices = prices.sort_index().copy()
    prices.index = pd.to_datetime(prices.index)

    df = pd.DataFrame(index=prices.index)
    df["price"] = prices

    # --- Returns ---
    # Daily return (t-1 -> t)
    ret_1 = prices.pct_change(1)
    df["ret_1"] = ret_1

    # Past 5 / 10 / 21 day returns
    df["ret_5"] = prices.pct_change(5)
    df["ret_10"] = prices.pct_change(10)
    df["ret_21"] = prices.pct_change(21)   # ~1 trading month

    # --- Volatility ---
    # Original short-horizon vol on daily returns
    df["vol_10"] = ret_1.rolling(10).std()
    df["vol_20"] = ret_1.rolling(20).std()

    # Longer-horizon vol on log-returns
    log_ret = np.log(prices / prices.shift(1))
    df["vol_60"] = log_ret.rolling(60).std()
    df["vol_ratio_10_60"] = df["vol_10"] / (df["vol_60"] + 1e-8)

    # --- Moving-average relative distances (trend / overextension) ---
    ma_10 = prices.rolling(10).mean()
    ma_20 = prices.rolling(20).mean()
    ma_50 = prices.rolling(50).mean()
    ma_200 = prices.rolling(200).mean()

    # positive -> price below MA
    df["ma10_rel"] = ma_10 / prices - 1.0
    df["ma20_rel"] = ma_20 / prices - 1.0
    df["ma50_rel"] = ma_50 / prices - 1.0
    df["ma200_rel"] = ma_200 / prices - 1.0

    # --- Simple trend regime flags ---
    df["trend_up_50"] = (prices > ma_50).astype(np.float32)
    df["trend_up_200"] = (prices > ma_200).astype(np.float32)

    # --- Calendar feature ---
    df["dow"] = df.index.dayofweek.astype(np.int8)  # 0=Mon, 4=Fri

    # --- Target: next-day return (t -> t+1) ---
    df["target_ret_1"] = ret_1.shift(-1)

    # Drop rows where any feature/target is NaN (from rolling & shift)
    df = df.dropna()

    return df


# Default set of features / signals we use
DEFAULT_FEATURES = [
    # returns
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_21",

    # volatility
    "vol_10",
    "vol_20",
    "vol_60",
    "vol_ratio_10_60",

    # moving-average relative distances
    "ma10_rel",
    "ma20_rel",
    "ma50_rel",
    "ma200_rel",

    # simple regime flags
    "trend_up_50",
    "trend_up_200",

    # calendar
    "dow",
]



def build_sequence_dataset(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Index,
    seq_len: int = 30,
):
    """
    Build a (num_samples, seq_len, num_features) dataset for sequence models.

    For each i >= seq_len-1, we create:
        X_seq[i] = X[i-seq_len+1 : i+1]   (uses history up to day i)
        y_seq[i] = y[i]                   (target: next-day return after day i)

    dates_seq[i] = dates[i]
    """
    assert len(X) == len(y) == len(dates)

    X_seqs = []
    y_seqs = []
    date_seqs = []

    for i in range(seq_len - 1, len(X)):
        X_seqs.append(X[i - seq_len + 1 : i + 1])
        y_seqs.append(y[i])
        date_seqs.append(dates[i])

    X_seqs = np.stack(X_seqs).astype(np.float32)
    y_seqs = np.array(y_seqs, dtype=np.float32)
    date_seqs = pd.Index(date_seqs)

    return X_seqs, y_seqs, date_seqs




def build_feature_matrix(
    df: pd.DataFrame,
    feature_names: list[str] | None = None,
):
    """
    Turn a signal dataframe into X (features), y (target), and dates.
    You can control which signals to use via feature_names.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURES

    X = df[feature_names].values.astype(np.float32)
    y = df["target_ret_1"].values.astype(np.float32)
    dates = df.index

    return X, y, dates
