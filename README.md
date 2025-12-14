# QuantCoding – ML Trading Experiments on SPY

This repo is my personal sandbox for **quantitative trading research** in Python.  
I focus on a single liquid instrument (SPY) and build an end-to-end pipeline:

> data → signal engineering → ML models → backtests → risk/return metrics → paper-trade signals

The main goal is **research & learning**, *not* production trading or financial advice.

---

## What’s inside

### Data & Signals

- Daily SPY data downloaded via `yfinance` and cached as CSV.
- Signal generation in `src/signals.py`, including:
  - multi-horizon returns: `ret_1`, `ret_5`, `ret_10`, `ret_21`
  - volatility features: `vol_10`, `vol_20`, `vol_60`, `vol_ratio_10_60`
  - moving-average distances: `ma10_rel`, `ma20_rel`, `ma50_rel`, `ma200_rel`
  - simple regime flags: `trend_up_50`, `trend_up_200`
  - calendar feature: day-of-week (`dow`)
- Targets are **next-day returns** (`target_ret_1`).

---

## Models

All models are trained on a **chronological train/test split**.  
Inside the training window, I use an inner train/validation split and **Optuna** for hyperparameter tuning.

- `src/models_tree.py`  
  - Tree-based regressor (`HistGradientBoostingRegressor`).  
  - Tuned on **validation Sharpe**, including a **quantile `q`** that controls when the strategy is long (ranking-based threshold).

- `src/models_lstm.py`  
  - LSTM **regressor** (PyTorch) on sequences of signals.  
  - Trained on next-day returns, then used with a **ranking strategy**:
    - compute predicted returns,
    - go long only on the top-`q` fraction of days (Sharpe-tuned via Optuna).

- `src/models_lstm_class.py`  
  - LSTM **classifier** for next-day direction (up/down).  
  - Tuned on validation **log-loss**.  
  - Mostly serves as a negative/control result: demonstrates how a complex model can still collapse to “always up” on weak directional signal.

- `src/models_tcn.py`  
  - TCN **regressor** (Temporal Convolutional Network, PyTorch) on sequences.  
  - Uses causal, dilated 1D convolutions with residual blocks.  
  - Same ranking-style trading rule as the LSTM regressor (long on top-`q` predicted returns), with `q` and model hyperparameters tuned on validation **Sharpe**.

---

## Backtesting & Metrics

Backtesting utilities live in `src/backtest.py`:

- Equity curves for:
  - Buy & Hold (SPY)
  - Tree strategy (Sharpe-tuned ranking)
  - LSTM classifier strategy (directional, prob threshold)
  - LSTM regressor strategy (ranking)
  - TCN regressor strategy (ranking)
- Metrics per strategy:
  - CAGR
  - annualized volatility
  - Sharpe ratio
  - max drawdown
  - fraction of days in the market (long ratio)
  - number of trades (position changes)

At the end of the main notebook I build a small **performance report table** summarizing these metrics for all strategies side by side, plus a summary of the **best Optuna hyperparameters** and validation objectives for each model (tree, LSTM reg, TCN reg, LSTM cls).


> These signals are for **research only**, ignore costs/slippage, and are **not financial advice**.

Main experiments, comparisons, and plots live in the notebooks (e.g. `03_*.ipynb`).

---

## Repo Structure (simplified)

```text
src/
  data_loading.py       # yfinance download + CSV cache
  signals.py            # feature engineering & targets
  models_tree.py        # tree regressor + evaluation helpers
  models_lstm.py        # LSTM regressor (sequence model)
  models_lstm_class.py  # LSTM classifier (directional baseline)
  models_tcn.py         # TCN regressor (temporal convolutional network)
  backtest.py           # equity curves & risk/return metrics

notebooks/
  01_*.ipynb            # basic data + signals exploration
  02_*.ipynb            # tree baseline & first backtests
  03_*.ipynb            # LSTM & TCN models, Optuna tuning, comparisons, reports

configs/
  best_params_spy.json  # Optuna best hyperparameters (tree, LSTM reg, LSTM cls, TCN reg)
