# src/data_loading.py
from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")


def download_daily_prices(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance and save it as CSV in data/.
    Returns the dataframe.
    """
    DATA_DIR.mkdir(exist_ok=True)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,  # adjust for splits/dividends
        progress=False,
    )

    # Make sure the index has a name (used as CSV column header)
    df.index.name = "date"

    csv_path = DATA_DIR / f"{ticker}.csv"
    df.to_csv(csv_path)

    print(f"Saved {ticker} data to {csv_path}")
    return df


def load_daily_close(ticker: str) -> pd.Series:
    """
    Load adjusted close prices from cached CSV in data/.
    Be robust to different column names for the date/index
    and ensure we return a clean float Series.
    """
    csv_path = DATA_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Call download_daily_prices('{ticker}') first."
        )

    # Parse the FIRST column as dates and use it as index,
    # regardless of its header name ("date", "Date", or empty).
    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
    df.index.name = "date"

    if "Adj Close" in df.columns:
        s = df["Adj Close"]
    elif "Close" in df.columns:
        s = df["Close"]
    else:
        # fallback: use last column if neither is found
        s = df.iloc[:, -1]

    # Force numeric dtype, drop anything non-numeric just in case
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = ticker

    return s