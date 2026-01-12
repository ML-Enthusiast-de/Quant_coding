> 🚨 **DISCLAIMER – RESEARCH ONLY, NOT FINANCIAL ADVICE** 🚨  
> This repository is strictly for **research and educational purposes**.  
> It is **not** investment advice, does **not** constitute a trading system, and is **not** intended for live trading with real money.  
> Use at your own risk.

# QuantCoding – ML Trading Experiments (SPY & S&P 500)

This repo is my personal sandbox for **quantitative trading research** in Python.

---

## 📌 New: Options Surface Research (SPY IV Surface Lab)

A separate, dedicated options research track lives here:

➡️ **`Options/README.md`**

It focuses on building **daily SPY implied volatility surfaces (EOD)** from option chains, applying data hygiene + no-arbitrage diagnostics, fitting a smooth surface, and researching **surface dynamics** (ATM level, skew, term structure) with hedged PnL proxies.

> Go to: **Options/README.md** for setup, data expectations, and pipeline details.

---

There are currently three main “tracks”:

1. **Single-asset time-series** on SPY (daily data, sequence models)  
2. **Cross-sectional S&P 500** stock-picking (panel data, momentum baseline → ML rankers + portfolio helper)
3. **L2 Order Book Microstructure** → see L2_order_book_project/README.md

Pipeline in both cases:

> data → signal engineering → ML models → backtests → risk/return metrics → paper-trade signals

The main goal is **research & learning**, *not* production trading or financial advice.

---

## 1. SPY Time-Series Project

### Data & Signals

- Daily SPY data downloaded via `yfinance` and cached as CSV (`data/SPY.csv`).
- Signal generation in `src/signals.py`, including:
  - multi-horizon returns: `ret_1`, `ret_5`, `ret_10`, `ret_21`
  - volatility features: `vol_10`, `vol_20`, `vol_60`, `vol_ratio_10_60`
  - moving-average distances: `ma10_rel`, `ma20_rel`, `ma50_rel`, `ma200_rel`
  - simple regime flags: `trend_up_50`, `trend_up_200`
  - calendar feature: day-of-week (`dow`)
- Target for the SPY sequence models: **next-day return** (`target_ret_1`).

### Models

All models are trained on a **chronological train/test split**.  
Inside the training window, I use an inner train/validation split and **Optuna** for hyperparameter tuning.

- `src/models_tree.py`  
  - Tree-based regressor (`HistGradientBoostingRegressor`).  
  - Tuned on **validation Sharpe**, including a **quantile `q`** that controls when the strategy is long (ranking-based threshold).

- `src/models_lstm.py`  
  - LSTM **regressor** (PyTorch) on sequences of signals.  
  - Trained on next-day returns, then used with a **ranking strategy**:
    - compute predicted returns for each day,
    - go long only on the top-`q` fraction of days (Sharpe-tuned via Optuna).

- `src/models_lstm_class.py`  
  - LSTM **classifier** for next-day direction (up/down).  
  - Tuned on validation **log-loss**.  
  - Serves mostly as a **control/negative result**: shows how a complex model can collapse to “always up” when the directional signal is weak.

- `src/models_tcn.py`  
  - TCN **regressor** (Temporal Convolutional Network, PyTorch) on sequences.  
  - Uses causal, dilated 1D convolutions with residual blocks.  
  - Same ranking-style trading rule as the LSTM regressor (long on top-`q` predicted returns), with `q` and model hyperparameters tuned via Optuna.

- `src/models_transformer.py`  
  - Lightweight **time-series Transformer regressor** for SPY.  
  - Uses positional encodings + self-attention over the last `seq_len` days.  
  - Also plugged into the same ranking framework and Optuna tuning.

### Backtesting & Metrics

Backtesting utilities live in `src/backtest.py`:

- Equity curves for:
  - Buy & Hold (SPY)
  - Tree strategy (Sharpe-tuned ranking)
  - LSTM classifier strategy (directional, prob threshold)
  - LSTM regressor strategy (ranking)
  - TCN regressor strategy (ranking)
  - Transformer regressor strategy (ranking)
- Metrics per strategy:
  - CAGR
  - annualized volatility
  - Sharpe ratio
  - max drawdown
  - fraction of days in the market (long ratio)
  - number of trades (position changes)

Main SPY notebooks:

- `notebooks/01_download_and_explore.ipynb`
- `notebooks/02_SPY_tree_baseline.ipynb`
- `notebooks/03_SPY_lstm_tree_tcn_transformer.ipynb`

---

## 2. Cross-Sectional S&P 500 Project

### Data & Universe

- S&P 500 constituents scraped from Slickcharts and stored in:
  - `data/sp500_symbols.csv` (columns incl. `symbol`, optionally `sector`)
- Adjusted close prices for the full S&P 500 universe:
  - downloaded in batches via `yfinance` and cached as  
    `data/sp500_adj_close.parquet`
- Loader logic in `src/data_loading_cross.py`:
  - handles ticker quirks (e.g. `BRK.B` ↔ `BRK-B`) and batching
  - returns a `(dates × tickers)` price panel.

### Cross-Sectional Signals

Defined in `src/signals_cross.py` as a MultiIndex panel `(date, ticker)`:

- past returns: `ret_1`, `ret_5`, `ret_21`, etc.
- volatility & simple risk measures
- moving-average gaps & simple regime flags
- **forward 21-day return** as target: `target_fwd_21`
- optional `sector` column (from `sp500_symbols.csv`) for **sector-neutral** experiments

Flattened to `X, y, dates, tickers` via `build_cross_sectional_matrix`, which is used as input to the ML rankers.

### Baseline: Cross-Sectional Momentum

`notebooks/04_cross_sectional_sp500.ipynb`:

- Build a **time-based train / val / test split** on dates.
- Implement classic **cross-sectional momentum**:
  - each date, rank stocks by past 21-day return (`ret_21`),
  - long-only: equal-weight top decile,
  - long-short: long top decile, short bottom decile,
  - benchmark: equal-weight all stocks.
- Convert 21-day forward returns to daily-equivalent and evaluate with:
  - CAGR, volatility, Sharpe, max drawdown.

### Tree & XGB Rankers + Robustness (with costs)

`notebooks/05_sp500_cs_tree_robustness.ipynb`:

- Add **cross-sectional ML rankers** that use the full feature set to predict 21-day forward returns:
  - `HistGradientBoostingRegressor` (tree),
  - `XGBRegressor` (XGBoost).
- For each test window:
  - build train/validation sets using only data *before* the test period,
  - use **Optuna** to tune:
    - model hyperparameters (depth, learning rate, iterations, etc.),
    - the trading quantile `q` (size of long/short buckets),
    - optimizing **validation long-short Sharpe (net of costs)**.
- Include simple **transaction-cost modeling**:
  - per-21-day **round-trip cost** (e.g. 10 bps) applied to both momentum and ML portfolios,
  - report **net-of-cost** returns and Sharpe.
- Evaluate across multiple non-overlapping test windows (e.g. 2005–2009, 2010–2014, 2015–2019, 2020–2024) to check **regime robustness**.
- Report per-window metrics:
  - momentum vs tree vs XGB, long-only and long-short,
  - Sharpe differences (`tree – momentum`, `xgb – momentum`),
  - optional **sector-neutral** variants when sector data is available.

At the end of `05_sp500_cs_tree_robustness.ipynb`, a final **cross-sectional tree model** is tuned on a global train/validation split and trained once on all historical data.  
This model is saved as a bundle:

- `models/sp500_tree_cs_21d_live.pkl`

including:

- the trained `HistGradientBoostingRegressor`
- the feature list (`CROSS_FEATURES`)
- lookahead horizon
- cost assumptions
- tuned live quantile `q_live`
- basic training metadata

---

## 3. Paper-Trade / Live Inference

The **live inference** entry point for the cross-sectional model is:

- `scripts/sp500_cs_inference_live.py`

Again: this is **paper trading only**, not a live execution system.

---

## 4. Experimental Portfolio Manager

There is an experimental **portfolio manager** that sits on top of the live cross-sectional model and your own holdings:

- Script: `scripts/portfolio_manager.py` (imports `run_inference` from `sp500_cs_inference_live.py`)
- Portfolio file (user-owned, gitignored):  
  `data/portfolio/current_portfolio.csv`
