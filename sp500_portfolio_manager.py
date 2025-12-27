#!/usr/bin/env python
"""
FOR RESEARCH PURPOSE ONLY. NOT FOR PRODUCTION USE.

Simple portfolio manager on top of the SP500 cross-sectional tree model.

- Reads current holdings from data/portfolio/current_portfolio.csv (you must create this file)
- Asks you to confirm the CSV is up to date (shares / tickers)
- Updates current_price using latest SP500 prices
- Calls run_inference() from sp500_cs_inference_live.py to get today's signals
- Joins signals with holdings and suggests:
    - which names to SELL / REDUCE
    - which names to KEEP
    - which new names to BUY (limited number)
- Takes transaction costs into account (via model bundle cost_bps)
- Avoids suggesting a huge number of trades (hard caps on buys/sells)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------
# Project setup & imports
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sp500_cs_inference_live import run_inference
from src.data_loading_cross import load_sp500_adj_close

# Paths
PORTFOLIO_CSV = Path(PROJECT_ROOT) / "data" / "portfolio" / "current_portfolio.csv"
MODEL_PATH = Path(PROJECT_ROOT) / "models" / "sp500_tree_cs_21d_live.pkl"

# Trade constraints
MAX_BUY_TRADES = 5
MAX_SELL_TRADES = 5


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_model_config(model_path: Path = MODEL_PATH) -> dict:
    """
    Load lookahead + cost_bps from the live model bundle so that
    trade thresholds are consistent with the research notebook.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {model_path}")

    bundle = joblib.load(model_path)
    lookahead = int(bundle["lookahead"])
    cost_bps = float(bundle["cost_bps"])  # round-trip cost per horizon

    return {
        "lookahead": lookahead,
        "cost_bps": cost_bps,
    }


def load_portfolio(path: Path = PORTFOLIO_CSV) -> pd.DataFrame:
    """
    Load current portfolio from CSV.

    Expected columns:
        ticker, shares, buy_price, current_price

    Returns DataFrame with 'ticker' uppercased.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Portfolio CSV not found at {path}.\n"
            "Create it with columns: ticker,shares,buy_price,current_price"
        )

    df = pd.read_csv(path)
    required = {"ticker", "shares", "buy_price", "current_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Portfolio CSV missing columns: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["shares"] = df["shares"].astype(float)
    df["buy_price"] = df["buy_price"].astype(float)
    df["current_price"] = df["current_price"].astype(float)

    return df


def update_current_prices(
    portfolio_df: pd.DataFrame,
    price_start: str = "2015-01-01",
) -> tuple[pd.DataFrame, pd.Timestamp, list[str]]:
    """
    Update current_price in the portfolio using latest SP500 prices.

    Returns:
        updated_portfolio_df
        last_date used for prices
        list of tickers that were not found in the SP500 price panel
    """
    tickers = portfolio_df["ticker"].unique().tolist()

    prices = load_sp500_adj_close(start=price_start, force_download=False)
    prices = prices.sort_index()
    last_date = prices.index[-1]

    # Ticketers that exist in SP500 panel
    spx_tickers = [t for t in tickers if t in prices.columns]
    missing = [t for t in tickers if t not in prices.columns]

    if spx_tickers:
        last_prices = prices.loc[last_date, spx_tickers]

        df = portfolio_df.set_index("ticker")
        df.loc[spx_tickers, "current_price"] = last_prices.values
        portfolio_df = df.reset_index()

    return portfolio_df, last_date, missing


def compute_portfolio_metrics(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
        position_value
        pnl_abs
        pnl_pct
    """
    portfolio = portfolio_df.copy()
    portfolio["position_value"] = portfolio["shares"] * portfolio["current_price"]
    portfolio["pnl_abs"] = (portfolio["current_price"] - portfolio["buy_price"]) * portfolio["shares"]
    portfolio["pnl_pct"] = (portfolio["current_price"] / portfolio["buy_price"] - 1.0).replace([np.inf, -np.inf], np.nan)
    return portfolio


def join_portfolio_with_signals(
    portfolio_df: pd.DataFrame,
    signals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join current holdings with model signals.

    signals_df is the 'raw' DataFrame returned by run_inference(),
    indexed by ticker and containing:
        pred_fwd_21, pred_fwd_21_net, pred_daily_net, edge_vs_eqw_daily
    """
    signals = signals_df.copy()
    if "rank" not in signals.columns:
        signals["rank"] = signals["pred_daily_net"].rank(ascending=False, method="first")

    portfolio = portfolio_df.copy()
    portfolio = portfolio.set_index("ticker")
    portfolio = portfolio.join(signals, how="left")

    return portfolio.reset_index()


def suggest_trades(
    portfolio_with_signals: pd.DataFrame,
    signals_df: pd.DataFrame,
    lookahead: int,
    cost_bps: float,
) -> dict:
    """
    Core decision logic.

    - Uses edge_vs_eqw_daily and rank to suggest limited buy/sell lists.
    - Takes transaction cost into account by requiring the expected
      edge over the horizon to exceed cost_bps (up or down) for trades.
    """
    df = portfolio_with_signals.copy()
    df = df.set_index("ticker")

    # Derived metrics
    df["horizon_edge_vs_eqw"] = df["edge_vs_eqw_daily"] * lookahead

    universe_size = len(signals_df)
    # e.g. bottom 30% of universe is considered "bad region"
    sell_rank_cutoff = int(0.7 * universe_size)

    # --- SELL candidates: held names with clearly negative edge or very bad rank ---
    sell_mask = (
        df["pred_daily_net"].notna()
        & (
            (df["horizon_edge_vs_eqw"] < -cost_bps)  # expected underperformance beyond costs
            | (df["rank"] > sell_rank_cutoff)        # deep in tail of ranking
        )
    )
    sell_candidates = df[sell_mask].copy()
    sell_candidates = sell_candidates.sort_values(
        ["horizon_edge_vs_eqw", "rank"]
    )  # worst first

    sell_candidates = sell_candidates.head(MAX_SELL_TRADES)

    # --- BUY candidates: strong signals not currently held ---
    held_tickers = set(df.index)
    signals = signals_df.copy()
    if "rank" not in signals.columns:
        signals["rank"] = signals["pred_daily_net"].rank(ascending=False, method="first")

    signals["horizon_edge_vs_eqw"] = signals["edge_vs_eqw_daily"] * lookahead

    buy_mask = (
        ~signals.index.isin(held_tickers)
        & (signals["horizon_edge_vs_eqw"] > cost_bps)  # expected outperformance beyond cost
    )

    buy_candidates = signals[buy_mask].copy()
    buy_candidates = buy_candidates.sort_values("pred_daily_net", ascending=False)
    buy_candidates = buy_candidates.head(MAX_BUY_TRADES)

    # Names that are held and have positive edge: "KEEP"
    keep_mask = (
        df["pred_daily_net"].notna()
        & (df["horizon_edge_vs_eqw"] >= -cost_bps)  # not clearly negative
    )
    keep_candidates = df[keep_mask].copy()
    keep_candidates = keep_candidates.sort_values("rank")

    return {
        "sell": sell_candidates.reset_index(),
        "buy": buy_candidates.reset_index(),
        "keep": keep_candidates.reset_index(),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("=== SP500 Portfolio Manager (Research Only) ===")
    print(f"Looking for portfolio CSV at: {PORTFOLIO_CSV}")
    print()

    # 1) Ask user if CSV is up to date
    ans = input(
        "Have you updated current_portfolio.csv with your latest positions (shares / tickers)? [y/n]: "
    ).strip().lower()

    if ans not in ("y", "yes"):
        print("Please update the CSV and rerun this script.")
        return

    # 2) Load portfolio
    portfolio_df = load_portfolio(PORTFOLIO_CSV)
    print(f"Loaded {len(portfolio_df)} positions from CSV.")
    print()

    # 3) Update current prices from SP500 data
    portfolio_df, price_date, missing = update_current_prices(portfolio_df)
    print(f"Updated current_price using SP500 data as of {price_date.date()}.")
    if missing:
        print("Warning: the following tickers are not in the SP500 panel and have not been updated:")
        print(", ".join(missing))
        print("They will be kept in the portfolio but have no model signals.")
    print()

    # 4) Compute portfolio-level metrics
    portfolio_df = compute_portfolio_metrics(portfolio_df)
    total_value = portfolio_df["position_value"].sum()
    total_pnl = portfolio_df["pnl_abs"].sum()
    total_pnl_pct = total_pnl / (total_value - total_pnl) if total_value != total_pnl else np.nan

    print("Current portfolio snapshot:")
    print(portfolio_df.sort_values("position_value", ascending=False).round(4))
    print()
    print(f"Total portfolio value: {total_value:,.2f}")
    print(f"Unrealized P&L:       {total_pnl:,.2f} ({total_pnl_pct * 100:.2f}% approx)")
    print()

    # 5) Get model config (lookahead & cost) from bundle
    cfg = load_model_config(MODEL_PATH)
    lookahead = cfg["lookahead"]
    cost_bps = cfg["cost_bps"]

    print(f"Model config: lookahead={lookahead} days, round-trip cost={cost_bps * 1e4:.1f} bps")
    print()

    # 6) Run inference to get today's signals
    print("Running model inference to get today's signals...")
    inf_result = run_inference(do_log=True)  # you can set do_log=False if you don't want extra logging
    as_of_date = inf_result["as_of_date"]
    signals = inf_result["raw"]  # full universe sorted by pred_daily_net

    print(f"Got signals for as-of date {as_of_date.date()} for {len(signals)} SP500 names.")
    print()

    # 7) Join portfolio with signals and suggest trades
    portfolio_with_signals = join_portfolio_with_signals(portfolio_df, signals)
    trade_suggestions = suggest_trades(
        portfolio_with_signals,
        signals_df=signals,
        lookahead=lookahead,
        cost_bps=cost_bps,
    )

    sell_df = trade_suggestions["sell"]
    buy_df = trade_suggestions["buy"]
    keep_df = trade_suggestions["keep"]

    # 8) Print suggestions
    print("=" * 80)
    print("TRADE SUGGESTIONS (RESEARCH ONLY)")
    print("=" * 80)
    print()

    # SELL
    if not sell_df.empty:
        print(f"Suggested SELLS / REDUCTIONS (max {MAX_SELL_TRADES} shown):")
        cols = [
            "ticker",
            "shares",
            "current_price",
            "pnl_pct",
            "pred_daily_net",
            "edge_vs_eqw_daily",
            "rank",
        ]
        cols = [c for c in cols if c in sell_df.columns]
        print(
            sell_df[cols]
            .sort_values("edge_vs_eqw_daily")
            .round(6)
            .to_string(index=False)
        )
    else:
        print("No strong SELL candidates based on model and cost thresholds.")

    print()
    # BUY
    if not buy_df.empty:
        print(f"Suggested NEW BUYS (max {MAX_BUY_TRADES} shown):")
        cols = [
            "ticker",
            "pred_daily_net",
            "edge_vs_eqw_daily",
            "rank",
        ]
        cols = [c for c in cols if c in buy_df.columns]
        print(
            buy_df[cols]
            .sort_values("pred_daily_net", ascending=False)
            .round(6)
            .to_string(index=False)
        )
    else:
        print("No strong NEW BUY candidates beyond cost thresholds.")

    print()
    # KEEP
    print("Names currently held with non-negative edge (suggest KEEP / REVIEW):")
    cols = [
        "ticker",
        "shares",
        "current_price",
        "pnl_pct",
        "pred_daily_net",
        "edge_vs_eqw_daily",
        "rank",
    ]
    cols = [c for c in cols if c in keep_df.columns]
    if not keep_df.empty:
        print(
            keep_df[cols]
            .sort_values("rank")
            .head(20)  # don't spam too much
            .round(6)
            .to_string(index=False)
        )
    else:
        print("No holdings with positive model edge (this is unusual, check signals).")

    print()
    print("Note: This is a research tool. You still decide if/what to trade.")
    print("      Trade counts are capped to avoid 20+ trades/day.")


if __name__ == "__main__":
    main()
