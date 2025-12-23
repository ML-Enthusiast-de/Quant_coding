# QuantCoding – ML Trading Experiments (SPY & S&P 500)

This repo is my personal sandbox for **quantitative trading research** in Python.

There are currently two main “tracks”:

1. **Single-asset time-series** on SPY (daily data, sequence models)
2. **Cross-sectional S&P 500** stock-picking (panel data, momentum baseline → ML rankers)

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

The main SPY experiments, comparisons, and plots live in:

- `notebooks/01_download_and_explore.ipynb`
- `notebooks/02_SPY_tree_baseline.ipynb`
- `notebooks/03_SPY_lstm_tree_tcn_transformer.ipynb`

At the end of the SPY notebook I build a small **performance report table** summarizing these metrics for all strategies side by side, plus a summary of the **best Optuna hyperparameters** and validation objectives.

> These signals ignore transaction costs/slippage and are for **research only** – not financial advice.

---

## 2. Cross-Sectional S&P 500 Project

### Data & Universe

- S&P 500 constituents scraped from Slickcharts and stored in:
  - `data/sp500_symbols.csv` (column: `symbol`)
- Adjusted close prices for the full S&P 500 universe:
  - downloaded in batches via `yfinance` and cached as  
    `data/sp500_adj_close.parquet`
- Loader logic in `src/data_loading_cross.py`:
  - handles ticker quirks (e.g. `BRK.B` ↔ `BRK-B`) and batching
  - returns a `(dates × tickers)` price panel.

### Cross-Sectional Signals

Defined in `src/signals_cross.py` as a MultiIndex panel `(date, symbol)`:

- past returns: `ret_1`, `ret_5`, `ret_21`, etc.
- volatility & simple risk measures
- moving-average gaps & simple regime flags
- **forward 21-day return** as target: `target_fwd_21`

Flattened to `X, y, dates, tickers` via `build_cross_sectional_matrix`, which is used as input to the ML rankers.

### Baseline & Tree Ranker (with costs)

Main notebooks:

- `notebooks/04_cross_sectional_sp500.ipynb`
  - Build a **time-based train / val / test split** on dates.
  - Implement classic **cross-sectional momentum**:
    - each date, rank stocks by past 21-day return (`ret_21`),
    - long-only: equal-weight top decile,
    - long-short: long top decile, short bottom decile,
    - benchmark: equal-weight all stocks.
  - Convert 21-day forward returns to daily-equivalent and evaluate with:
    - CAGR, volatility, Sharpe, max drawdown.

- `notebooks/05_sp500_cs_tree_robustness.ipynb`
  - Add a **cross-sectional HistGradientBoostingRegressor** that uses the full feature set to predict 21-day forward returns.
  - Use **Optuna** to tune both:
    - tree hyperparameters (depth, learning rate, iterations, leaf size),
    - the trading quantile `q` (size of the long/short buckets),
    - optimizing **validation long-short Sharpe**, with data strictly before the test window.
  - Include simple **transaction-cost modeling**:
    - per-21-day **round-trip cost** (e.g. 10 bps) applied to both momentum and tree portfolios,
    - report **net-of-cost** returns and Sharpe.
  - Run the full train/val/tune/test procedure across multiple non-overlapping test windows (e.g. 2005–2009, 2010–2014, …) to check **robustness over regimes**.
  - Summarize per-window metrics:
    - momentum vs tree, long-only and long-short,
    - Sharpe differences (`tree – momentum`) to see where the ML ranker consistently adds value.

### Paper-Trade / Live Inference

- A lightweight **inference script** (e.g. `scripts/sp500_cs_live_inference.py`) that:
  - loads the saved cross-sectional tree model and its config (lookahead, `q_live`, cost assumptions),
  - pulls the latest S&P 500 prices,
  - rebuilds only the **most recent feature panel** (no need to reprocess the full history),
  - computes predicted 21-day forward returns,
  - applies the **net-of-cost** ranking rule and prints:
    - top *N* tickers to **long**,
    - bottom *N* tickers to **short**,
    - with an estimated daily edge vs. equal-weight benchmark.

This “paper trading” mode is meant for **offline experimentation only**, not live trading.

---

## Repo Structure (simplified)

```text
data/
  SPY.csv                     # SPY daily prices (yfinance cache)
  sp500_symbols.csv           # S&P 500 universe
  sp500_adj_close.parquet     # S&P 500 adj close panel

notebooks/
  01_download_and_explore.ipynb             # basic SPY data + signals exploration
  02_SPY_tree_baseline.ipynb                # SPY tree baseline & backtests
  03_SPY_lstm_tree_tcn_transformer.ipynb    # SPY sequence models (LSTM/TCN/Transformer)
  04_cross_sectional_sp500.ipynb            # S&P 500 cross-sectional momentum & panel prep
  05_sp500_cs_tree_robustness.ipynb         # S&P 500 tree vs momentum, robustness + costs

src/
  __init__.py
  data_loading.py           # SPY yfinance download + CSV cache
  data_loading_cross.py     # S&P 500 universe & panel download
  signals.py                # SPY feature engineering & targets
  signals_cross.py          # cross-sectional S&P 500 signals & targets
  models_tree.py            # tree regressor + helpers (SPY)
  models_lstm.py            # LSTM regressor (SPY sequence)
  models_lstm_class.py      # LSTM classifier (SPY direction baseline)
  models_tcn.py             # TCN regressor (SPY sequence)
  models_transformer.py     # Transformer regressor (SPY sequence)
  backtest.py               # equity curves & risk/return metrics

scripts/
  sp500_cs_live_inference.py  # load saved cross-sectional tree + config, output top-N long/short (paper-trade)

configs/
  best_params_spy.json        # Optuna best hyperparameters (tree, LSTM reg, LSTM cls, TCN, Transformer)
  sp500_cs_tree_live.json     # config for S&P 500 live cross-sectional tree (lookahead, q_live, costs, etc.)

models/
  sp500_cs_tree_live.joblib   # serialized S&P 500 cross-sectional tree model

README.md
requirements.txt
