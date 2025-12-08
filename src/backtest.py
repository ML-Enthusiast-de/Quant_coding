# src/backtest.py
import numpy as np
import pandas as pd


def equity_curve_from_returns(
    returns: pd.Series | np.ndarray,
    start_value: float = 1.0,
) -> pd.Series:
    returns = pd.Series(returns)
    equity = (1.0 + returns).cumprod() * start_value
    equity.index = returns.index
    return equity


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    total_periods = len(equity)
    if total_periods == 0:
        return np.nan
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    return (end_val / start_val) ** (periods_per_year / total_periods) - 1.0


def annualized_vol(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
) -> float:
    returns = pd.Series(returns)
    return float(returns.std()) * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    returns = pd.Series(returns)
    if returns.std() == 0:
        return np.nan
    mean_ret = float(returns.mean())
    ann_ret = (1.0 + mean_ret) ** periods_per_year - 1.0
    ann_vol = annualized_vol(returns, periods_per_year)
    return (ann_ret - rf) / ann_vol


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdowns = equity / running_max - 1.0
    return float(drawdowns.min())  # negative
