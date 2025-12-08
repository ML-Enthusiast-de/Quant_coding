# src/signals.py
import numpy as np
import pandas as pd


def make_basic_signals(prices: pd.Series) -> pd.DataFrame:
    """
    Build a basic set of signals from a price series.
    Each row t uses only info up to time t.
    Target is NEXT-day return.
    """
    prices = prices.sort_index()
    df = pd.DataFrame(index=prices.index)
    df["price"] = prices

    # Daily return (t-1 -> t)
    ret_1 = prices.pct_change(1)
    df["ret_1"] = ret_1

    # Past 5 / 10 day returns
    df["ret_5"] = prices.pct_change(5)
    df["ret_10"] = prices.pct_change(10)

    # Rolling volatility of daily returns
    df["vol_10"] = ret_1.rolling(10).std()
    df["vol_20"] = ret_1.rolling(20).std()

    # Moving-average relative distances
    ma_10 = prices.rolling(10).mean()
    ma_50 = prices.rolling(50).mean()
    df["ma10_rel"] = ma_10 / prices - 1.0   # positive -> price below MA
    df["ma50_rel"] = ma_50 / prices - 1.0

    # --- Target: next-day return (t -> t+1) ---
    df["target_ret_1"] = ret_1.shift(-1)

    # Drop rows where any feature/target is NaN (from rolling & shift)
    df = df.dropna()

    return df

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


# Default set of features / signals we use
DEFAULT_FEATURES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "vol_10",
    "vol_20",
    "ma10_rel",
    "ma50_rel",
]


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
