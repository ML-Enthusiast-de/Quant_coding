# QuantCoding – ML Trading Experiments on SPY

This repo is my personal sandbox for **quantitative trading research** in Python.  
I focus on a single liquid instrument (SPY) and build an end-to-end pipeline:

> data → signal engineering → ML models → backtests → risk/return metrics

The main goal is *research & learning*, not production trading.

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

### Models

All models are trained on a **chronological train/test split** with an inner
train/validation split for hyperparameter tuning via **Optuna**.

- `src/models_tree.py`  
  - Tree-based regressor (`HistGradientBoostingRegressor`)  
  - Tuned on validation Sharpe, with a **quantile-based trading threshold**.

- `src/models_lstm.py`  
  - LSTM **regressor** (PyTorch) on sequences of signals.  
  - Used with a **ranking strategy**: long only on top-q predicted returns.

- `src/models_lstm_class.py`  
  - LSTM **classifier** for next-day direction (up/down).  
  - Mostly serves as a negative result: shows how a complex model can collapse
    to “always up” when the signal is weak.

### Backtesting & Metrics

Backtesting utilities live in `src/backtest.py`:

- Equity curves for:
  - Buy & Hold (SPY)
  - Tree strategy
  - LSTM classifier strategy
  - LSTM regressor strategy
- Metrics:
  - CAGR
  - annualized volatility
  - Sharpe ratio
  - max drawdown
  - (plus simple stats like fraction of days in the market, number of trades)

Main experiments and plots are in the notebooks (e.g. `03_*.ipynb`).

---

## Repo Structure (simplified)

```text
src/
  data_loading.py      # yfinance download + CSV cache
  signals.py           # feature engineering & targets
  models_tree.py       # tree regressor + evaluation helpers
  models_lstm.py       # LSTM regressor
  models_lstm_class.py # LSTM classifier
  backtest.py          # equity curves & metrics

notebooks/
  01_*.ipynb           # basic data + signals exploration
  02_*.ipynb           # tree baseline & backtests
  03_*.ipynb           # LSTM models, Optuna tuning, comparisons

configs/
  best_params_spy.json # Optuna best hyperparameters for each model
