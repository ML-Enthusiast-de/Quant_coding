import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Wiring / imports
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading_cross import load_sp500_adj_close
from src.backtest import (
    equity_curve_from_returns,
    cagr,
    annualized_vol,
    sharpe_ratio,
    max_drawdown,
)

PRED_PATH = PROJECT_ROOT / "data" / "paper_trade" / "sp500_cs_predictions.parquet"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def load_predictions(path: Path = PRED_PATH):
    """Load logged paper-trade predictions."""
    if not path.exists():
        raise FileNotFoundError(f"No prediction log found at {path}")

    df = pd.read_parquet(path)

    if "as_of_date" not in df.columns:
        raise ValueError("Prediction file must contain an 'as_of_date' column.")
    if "ticker" not in df.columns:
        raise ValueError("Prediction file must contain a 'ticker' column.")
    if "side" not in df.columns:
        raise ValueError("Prediction file must contain a 'side' column.")
    if "lookahead" not in df.columns:
        raise ValueError("Prediction file must contain a 'lookahead' column.")

    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df = df.rename(columns={"as_of_date": "date"})  # align with signals/backtest style

    lookahead = int(df["lookahead"].iloc[0])

    if "cost_bps" in df.columns:
        cost_bps = float(df["cost_bps"].iloc[0])
    else:
        cost_bps = 0.0

    return df, lookahead, cost_bps


def compute_realized_forward_returns(prices: pd.DataFrame, lookahead: int) -> pd.DataFrame:
    """
    prices: DataFrame indexed by date, columns=tickers, values=adj close.
    Returns long-form DataFrame [date, ticker, realized_fwd] where
    realized_fwd is the lookahead-day forward return from that date.
    """
    prices = prices.sort_index()
    fwd_prices = prices.shift(-lookahead)
    fwd_ret = fwd_prices / prices - 1.0
    fwd_ret.index.name = "date"

    realized = (
        fwd_ret.stack()
        .rename("realized_fwd")
        .reset_index()  # columns: date, ticker, realized_fwd
    )
    return realized


def build_daily_portfolio_returns(
    merged: pd.DataFrame,
    lookahead: int,
    cost_bps: float,
):
    """
    merged: predictions joined with realized_fwd, columns at least:
        date, ticker, side, realized_fwd

    Returns three Series of daily returns:
        eqw, long_only, long_short
    """

    def to_daily(R):
        return (1.0 + R) ** (1.0 / lookahead) - 1.0

    def _per_date(group: pd.DataFrame) -> pd.Series:
        g = group.dropna(subset=["realized_fwd"])
        if g.empty:
            return pd.Series({"eqw": 0.0, "long": 0.0, "long_short": 0.0})

        # Equal-weight benchmark: all tickers we predicted on that date
        eqw_21 = g["realized_fwd"].mean()

        g_long = g[g["side"] == "long"]
        g_short = g[g["side"] == "short"]

        # If we somehow have no longs/shorts for a day, fall back to eqw for that leg
        long_21 = g_long["realized_fwd"].mean() if not g_long.empty else eqw_21
        short_21 = g_short["realized_fwd"].mean() if not g_short.empty else eqw_21

        # Long-short return before costs: long top bucket, short bottom bucket
        long_short_21 = long_21 - short_21

        # Apply transaction costs on the 21d horizon
        # long-only: 1 round-trip -> cost_bps
        # long-short: 2 round-trips (long + short) -> 2 * cost_bps
        if cost_bps > 0.0:
            long_21 = (1.0 + long_21) * (1.0 - cost_bps) - 1.0
            long_short_21 = (1.0 + long_short_21) * (1.0 - 2.0 * cost_bps) - 1.0

        eqw_d = to_daily(eqw_21)
        long_d = to_daily(long_21)
        long_short_d = to_daily(long_short_21)

        return pd.Series({"eqw": eqw_d, "long": long_d, "long_short": long_short_d})

    daily = merged.groupby("date").apply(_per_date)

    eqw = daily["eqw"].astype(float)
    long = daily["long"].astype(float)
    long_short = daily["long_short"].astype(float)

    return eqw, long, long_short


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    # 1) Load predictions
    preds, lookahead, cost_bps = load_predictions()

    # 2) Pull just enough price history to cover prediction horizon
    start = preds["date"].min() - pd.Timedelta(days=5)
    end = preds["date"].max() + pd.Timedelta(days=lookahead * 2)

    prices = load_sp500_adj_close(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )

    # 3) Compute realized forward returns per (date, ticker)
    realized = compute_realized_forward_returns(prices, lookahead)

    # 4) Join predictions with realized returns
    merged = preds.merge(realized, on=["date", "ticker"], how="left")

    # 5) Build daily portfolio returns based on recorded "side"
    eqw, long, long_short = build_daily_portfolio_returns(
        merged, lookahead=lookahead, cost_bps=cost_bps
    )

    # Drop any NaNs (e.g. last dates where we don't yet have 21d future data)
    eqw = eqw.replace([np.inf, -np.inf], np.nan).dropna()
    long = long.replace([np.inf, -np.inf], np.nan).dropna()
    long_short = long_short.replace([np.inf, -np.inf], np.nan).dropna()

    # 6) Equity curves
    eqw_eq = equity_curve_from_returns(eqw)
    long_eq = equity_curve_from_returns(long)
    long_short_eq = equity_curve_from_returns(long_short)

    print("=== Paper-trade evaluation (net of costs) ===")
    print(f"# unique dates with predictions: {merged['date'].nunique()}")
    print(f"Lookahead: {lookahead} trading days")
    print(f"Assumed round-trip cost: {cost_bps * 1e4:.1f} bps")

    results = {
        "eqw_cagr": cagr(eqw_eq),
        "eqw_vol": annualized_vol(eqw),
        "eqw_sharpe": sharpe_ratio(eqw),
        "eqw_max_dd": max_drawdown(eqw_eq),

        "long_cagr": cagr(long_eq),
        "long_vol": annualized_vol(long),
        "long_sharpe": sharpe_ratio(long),
        "long_max_dd": max_drawdown(long_eq),

        "ls_cagr": cagr(long_short_eq),
        "ls_vol": annualized_vol(long_short),
        "ls_sharpe": sharpe_ratio(long_short),
        "ls_max_dd": max_drawdown(long_short_eq),
    }

    print("\nSummary metrics (net):")
    for k, v in results.items():
        print(f"{k:>12}: {v: .4f}")

    # 7) Plots
    plt.figure(figsize=(10, 4))
    eqw_eq.plot(label="Equal-weight benchmark")
    long_eq.plot(label="Paper-trade long-only (net)")
    plt.legend()
    plt.title("Paper-trade: long-only vs equal-weight (net of costs)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    long_short_eq.plot(label="Paper-trade long-short (net)")
    plt.legend()
    plt.title("Paper-trade: long-short strategy (net of costs)")
    plt.tight_layout()
    plt.show()

    # 8) (Optional) calibration: predicted vs realized forward returns
    if "pred_fwd_21_net_long" in merged.columns:
        sample = merged.dropna(subset=["realized_fwd"]).copy()
        if not sample.empty:
            sample = sample.sample(min(len(sample), 5000), random_state=0)

            plt.figure(figsize=(5, 5))
            plt.scatter(
                sample["pred_fwd_21_net_long"],
                sample["realized_fwd"],
                alpha=0.3,
            )
            plt.xlabel("Predicted 21d net return (long)")
            plt.ylabel("Realized 21d gross return")
            plt.title("Prediction vs outcome (sample)")
            plt.axhline(0, linewidth=1)
            plt.axvline(0, linewidth=1)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
