from __future__ import annotations

from pathlib import Path
import pandas as pd
import yfinance as yf

# Always anchor paths at the project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)






def load_sp500_universe(csv_path: Path | None = None) -> list[str]:
    """
    Load S&P 500 ticker universe from a CSV with a 'symbol' column.
    """
    if csv_path is None:
        csv_path = DATA_DIR / "sp500_symbols.csv"

    df = pd.read_csv(csv_path)
    if "symbol" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'symbol' column.")
    tickers = df["symbol"].dropna().astype(str).str.upper().tolist()
    return tickers


def _to_yf_symbol(sym: str) -> str:
    # yfinance uses - instead of . for some tickers, e.g. BRK-B, BF-B
    return sym.replace(".", "-")


def download_sp500_adj_close(
    start: str = "2000-01-01",
    end: str | None = None,
    csv_path: Path | None = None,
    outfile: Path | None = None,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Download adjusted close prices for the S&P 500 universe via yfinance,
    in batches, and cache to a parquet file.

    Returns a DataFrame:
        index: DatetimeIndex (trading days)
        columns: original ticker symbols from the CSV
    """
    if outfile is None:
        outfile = DATA_DIR / "sp500_adj_close.parquet"

    tickers_orig = load_sp500_universe(csv_path)
    print(f"Universe size from CSV: {len(tickers_orig)}")

    # Build mapping from original symbol -> yf symbol
    yf_symbols = {sym: _to_yf_symbol(sym) for sym in tickers_orig}

    all_close = []
    for i in range(0, len(tickers_orig), batch_size):
        batch_orig = tickers_orig[i : i + batch_size]
        batch_yf = [yf_symbols[sym] for sym in batch_orig]

        print(f"Downloading batch {i // batch_size + 1} "
              f"({len(batch_yf)} tickers): {batch_orig[:5]} ...")

        data = yf.download(
            tickers=batch_yf,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )

        if data.empty:
            print("  -> Empty data for this batch, skipping.")
            continue

        # yfinance multi-output: MultiIndex with ('Adj Close', 'AAPL'), ...
        if isinstance(data.columns, pd.MultiIndex):
            lvl0 = data.columns.get_level_values(0)
            if "Adj Close" in lvl0:
                close = data["Adj Close"]
            elif "Close" in lvl0:
                close = data["Close"]
            else:
                print("  -> No 'Adj Close' or 'Close' in columns, skipping.")
                continue
        else:
            close = data

        # Drop days where this batch has all NaNs
        close = close.dropna(how="all")

        # Map yfinance tickers back to original symbols where possible
        rename_map = {}
        for orig in batch_orig:
            yf_sym = yf_symbols[orig]
            if yf_sym in close.columns:
                rename_map[yf_sym] = orig

        close = close.rename(columns=rename_map)

        # Keep only columns we successfully mapped back
        close = close.loc[:, list(rename_map.values())]

        if close.empty:
            print("  -> No valid columns after mapping, skipping.")
            continue

        print("  -> Batch close shape:", close.shape)
        all_close.append(close)

    if not all_close:
        raise RuntimeError("No price data downloaded for any batch.")

    # Align on index & concatenate columns
    full_close = pd.concat(all_close, axis=1)
    # Drop days where *all* stocks in the full universe are NaN
    full_close = full_close.sort_index().dropna(how="all")
    # Remove duplicated columns if any
    full_close = full_close.loc[:, ~full_close.columns.duplicated()]

    print("Final price panel shape:", full_close.shape)
    full_close.to_parquet(outfile)
    print(f"Saved S&P 500 adj close to {outfile}")
    return full_close



def load_sp500_adj_close(
    start: str = "2000-01-01",
    end: str | None = None,
    csv_path: Path | None = None,
    outfile: Path | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load S&P 500 adjusted close prices from cache, or download if missing.

    You can pass force_download=True to refresh the file.
    """
    if outfile is None:
        outfile = DATA_DIR / "sp500_adj_close.parquet"

    if force_download or not outfile.exists():
        return download_sp500_adj_close(start=start, end=end, csv_path=csv_path, outfile=outfile)

    prices = pd.read_parquet(outfile)

    # If user requested a later start date, filter
    if start is not None:
        prices = prices.loc[prices.index >= pd.to_datetime(start)]
    if end is not None:
        prices = prices.loc[prices.index <= pd.to_datetime(end)]

    # Ensure columns are upper-case tickers
    prices.columns = [str(c).upper() for c in prices.columns]
    return prices
