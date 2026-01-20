from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================
HERE = Path(__file__).resolve()
OPTIONS_DIR = HERE.parents[1]
DATA_DIR = OPTIONS_DIR / "data" / "processed"

FACTORS_PATH = DATA_DIR / "surface_factors.parquet"

OUT_DAILY_PATH = DATA_DIR / "pnl_proxy_constmat_daily.parquet"
OUT_WINDOW_SUMMARY_PATH = DATA_DIR / "pnl_proxy_constmat_windows.csv"
OUT_OVERALL_SUMMARY_PATH = DATA_DIR / "pnl_proxy_constmat_overall.csv"


# =========================================================
# Config
# =========================================================
# Pick which factors to evaluate (high coverage ones)
FACTOR_LIST = [
    # 30/60/90d closest series (high coverage)
    "bf25_30d_closest",
    "bf25_60d_closest",
    "rr25_30d_closest",
    "rr25_60d_closest",
    "atm_30d_closest",
    "atm_60d_closest",
    "term_slope_atm_closest",
    "term_slope_rr25_closest",
]

# Transaction costs in "factor units" (IV units)
COST_PER_TRADE = 0.0015
COST_PER_UNIT_TURNOVER = 0.0010

MIN_AR1_OBS = 60
EPS = 1e-12


# =========================================================
# Rolling windows
# =========================================================
@dataclass
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_windows(dates: pd.DatetimeIndex) -> tuple[list[Window], tuple[int, int, int]]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    n = len(dates)

    # sensible defaults for EOD research
    # (works even if you only have ~200-300 days)
    min_train = max(120, int(round(0.60 * n)))
    test_len = max(60, int(round(0.20 * n)))
    step = test_len  # non-overlapping

    if n < min_train + test_len:
        # fallback to single split
        min_train = max(120, int(round(0.60 * n)))
        test_len = n - min_train
        step = test_len

    if test_len < 30:
        return [], (min_train, test_len, step)

    windows = []
    i0 = min_train
    while True:
        j = i0 + test_len - 1
        if j >= n:
            break
        windows.append(
            Window(
                train_start=dates[0],
                train_end=dates[i0 - 1],
                test_start=dates[i0],
                test_end=dates[j],
            )
        )
        i0 += step
        if i0 >= n:
            break

    return windows, (min_train, test_len, step)


# =========================================================
# AR(1) on daily changes
# =========================================================
def ar1_fit(dx: np.ndarray) -> tuple[float, float]:
    x = dx[:-1]
    y = dx[1:]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < MIN_AR1_OBS:
        return np.nan, np.nan
    x2, y2 = x[m], y[m]
    X = np.column_stack([np.ones_like(x2), x2])
    beta, *_ = np.linalg.lstsq(X, y2, rcond=None)
    return float(beta[0]), float(beta[1])


def ar1_predict(dx_last: float, a: float, b: float) -> float:
    if not (np.isfinite(dx_last) and np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return float(a + b * dx_last)


def pos_from_pred(pred: float) -> float:
    if not np.isfinite(pred):
        return 0.0
    return 1.0 if pred > 0 else (-1.0 if pred < 0 else 0.0)


# =========================================================
# Backtest
# =========================================================
def backtest_factor(df: pd.DataFrame, factor_col: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    s = pd.to_numeric(df[factor_col], errors="coerce").dropna()
    if len(s) < 250:
        return pd.DataFrame(), pd.DataFrame(), {"status": "too_few_obs", "n": int(len(s))}

    # proxy "return" = change in factor (constant-maturity series)
    dx = s.diff().dropna()
    dates = dx.index

    windows, (min_train, test_len, step) = make_windows(dates)
    meta = {
        "factor": factor_col,
        "n_obs": int(len(dx)),
        "min_train": int(min_train),
        "test_len": int(test_len),
        "step": int(step),
        "n_windows": int(len(windows)),
        "status": "ok" if windows else "single_split",
    }

    if not windows:
        # single split
        split = max(120, int(round(0.60 * len(dates))))
        if split >= len(dates) - 30:
            return pd.DataFrame(), pd.DataFrame(), {"status": "too_few_after_split", "n": int(len(dx))}
        windows = [Window(dates[0], dates[split - 1], dates[split], dates[-1])]

    daily_rows = []
    win_rows = []

    for w_id, w in enumerate(windows, start=1):
        train_mask = (dates >= w.train_start) & (dates <= w.train_end)
        test_mask = (dates >= w.test_start) & (dates <= w.test_end)

        dx_train = dx.loc[train_mask].to_numpy(dtype=float)
        a, b = ar1_fit(dx_train)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue

        test_dates = dates[test_mask]
        if len(test_dates) == 0:
            continue

        # position decided from forecast of next-day change, applied with 1-day delay
        pos = pd.Series(index=test_dates, data=0.0, dtype=float)
        for d in test_dates:
            pred = ar1_predict(float(dx.loc[d]), a, b)
            pos.loc[d] = pos_from_pred(pred)

        pos_trade = pos.shift(1).fillna(0.0)

        # PnL proxy: position * realized change
        pnl = pos_trade * dx.loc[test_dates]

        # costs
        turnover = pos_trade.diff().abs().fillna(0.0)
        cost = COST_PER_TRADE * (turnover > 0).astype(float) + COST_PER_UNIT_TURNOVER * turnover
        pnl_net = pnl - cost
        equity = pnl_net.cumsum()

        # window metrics
        pn = pnl_net.to_numpy(dtype=float)
        mu = float(np.mean(pn))
        sd = float(np.std(pn, ddof=1) + EPS)
        sharpe = (mu / sd) * math.sqrt(252.0) if len(pn) > 30 else np.nan

        eq = equity.to_numpy(dtype=float)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        maxdd = float(dd.min()) if len(dd) else np.nan

        tfrac = float(np.mean(turnover.to_numpy(dtype=float) > 0))

        daily_rows.append(
            pd.DataFrame(
                {
                    "date": test_dates,
                    "factor": factor_col,
                    "window_id": w_id,
                    "train_start": w.train_start,
                    "train_end": w.train_end,
                    "test_start": w.test_start,
                    "test_end": w.test_end,
                    "pos": pos_trade.values,
                    "dx_realized": dx.loc[test_dates].values,
                    "turnover": turnover.values,
                    "cost": cost.values,
                    "pnl": pnl_net.values,
                    "equity": equity.values,
                    "ar1_a": a,
                    "ar1_b": b,
                }
            )
        )

        win_rows.append(
            {
                "factor": factor_col,
                "window_id": w_id,
                "train_start": w.train_start.date(),
                "train_end": w.train_end.date(),
                "test_start": w.test_start.date(),
                "test_end": w.test_end.date(),
                "n_days": int(len(pn)),
                "mean_pnl": mu,
                "vol_pnl": sd,
                "sharpe_proxy": sharpe,
                "max_drawdown_proxy": maxdd,
                "trade_day_frac": tfrac,
            }
        )

    daily_out = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    win_out = pd.DataFrame(win_rows)
    return daily_out, win_out, meta


def overall_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    out = []
    for factor, g in daily.groupby("factor"):
        pnl = pd.to_numeric(g["pnl"], errors="coerce").to_numpy(dtype=float)
        pnl = pnl[np.isfinite(pnl)]
        if len(pnl) < 60:
            continue
        mu = float(np.mean(pnl))
        sd = float(np.std(pnl, ddof=1) + EPS)
        sharpe = (mu / sd) * math.sqrt(252.0)

        eq = pd.to_numeric(g["equity"], errors="coerce").to_numpy(dtype=float)
        eq = eq[np.isfinite(eq)]
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        maxdd = float(dd.min()) if len(dd) else np.nan

        tfrac = float(np.mean(pd.to_numeric(g["turnover"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0))

        out.append(
            {
                "factor": factor,
                "n_days": int(len(pnl)),
                "mean_pnl": mu,
                "vol_pnl": sd,
                "sharpe_proxy": sharpe,
                "max_drawdown_proxy": maxdd,
                "trade_day_frac": tfrac,
            }
        )
    return pd.DataFrame(out).sort_values("sharpe_proxy", ascending=False)


def main():
    if not FACTORS_PATH.exists():
        raise FileNotFoundError(f"Missing: {FACTORS_PATH}")

    df = pd.read_parquet(FACTORS_PATH)
    df["quote_date"] = pd.to_datetime(df["quote_date"], errors="coerce")
    df = df.dropna(subset=["quote_date"]).sort_values("quote_date").set_index("quote_date")

    print(f"Loaded surface factors: days={len(df):,}")

    daily_all = []
    win_all = []
    metas = []

    for fac in FACTOR_LIST:
        if fac not in df.columns:
            print(f"Skip missing factor: {fac}")
            continue

        daily, win, meta = backtest_factor(df, fac)
        metas.append(meta)

        if meta.get("status") != "ok" and meta.get("status") != "single_split":
            print(f"{fac}: {meta}")
            continue

        print(f"{fac}: n_obs={meta['n_obs']} windows={meta['n_windows']} train={meta['min_train']} test={meta['test_len']}")

        if not daily.empty:
            daily_all.append(daily)
        if not win.empty:
            win_all.append(win)

    if not daily_all:
        raise RuntimeError("No backtests produced. Likely factor coverage is too low for selected columns.")

    daily_out = pd.concat(daily_all, ignore_index=True)
    win_out = pd.concat(win_all, ignore_index=True) if win_all else pd.DataFrame()
    overall = overall_summary(daily_out)

    OUT_DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily_out.to_parquet(OUT_DAILY_PATH, index=False)
    win_out.to_csv(OUT_WINDOW_SUMMARY_PATH, index=False)
    overall.to_csv(OUT_OVERALL_SUMMARY_PATH, index=False)

    print("\n=== Const-maturity proxy backtest done ===")
    print(f"Wrote daily:   {OUT_DAILY_PATH}")
    print(f"Wrote windows: {OUT_WINDOW_SUMMARY_PATH}")
    print(f"Wrote overall: {OUT_OVERALL_SUMMARY_PATH}")

    print("\nOverall (top 10 by Sharpe_proxy):")
    print(overall.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
