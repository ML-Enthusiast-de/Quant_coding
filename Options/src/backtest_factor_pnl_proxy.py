# Options/src/backtest_factor_pnl_proxy.py
from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


# =========================================================
# Paths
# =========================================================
HERE = Path(__file__).resolve()
OPTIONS_DIR = HERE.parents[1]
DATA_DIR = OPTIONS_DIR / "data" / "processed"

SMILE_GRID_PATH = DATA_DIR / "surface_smile_grid.parquet"
FACTORS_PATH = DATA_DIR / "surface_factors.parquet"

OUT_DAILY_PATH = DATA_DIR / "pnl_proxy_results_rolling.parquet"
OUT_WINDOW_SUMMARY_PATH = DATA_DIR / "pnl_proxy_window_summary.csv"
OUT_OVERALL_SUMMARY_PATH = DATA_DIR / "pnl_proxy_overall_summary.csv"


# =========================================================
# Config
# =========================================================
DELTA_TARGET = 0.25

GAP_THRESHOLDS_DAYS = {30: 20.0, 60: 30.0, 90: 45.0}

# Proxy costs (IV units)
COST_PER_TRADE_IV = 0.0015
COST_PER_UNIT_TURNOVER_IV = 0.0010

# AR(1)
MIN_AR1_OBS = 40
EPS = 1e-12

# k search window
K_SEARCH_MIN = -0.9
K_SEARCH_MAX = 0.9


# =========================================================
# BS forward-delta mapping
# =========================================================
def d1_from_k_sigma_T(k: float, sigma: float, T: float) -> float:
    return (-k + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T) + 1e-12)


def call_forward_delta_from_k(k: float, sigma: float, T: float) -> float:
    return float(norm.cdf(d1_from_k_sigma_T(k, sigma, T)))


def put_forward_delta_from_k(k: float, sigma: float, T: float) -> float:
    return float(norm.cdf(d1_from_k_sigma_T(k, sigma, T)) - 1.0)


# =========================================================
# Smile helpers
# =========================================================
def interp_at_k(df_exp: pd.DataFrame, k0: float) -> float:
    x = df_exp["k"].to_numpy(dtype=float)
    y = df_exp["iv_fit"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.nan
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    if k0 < x[0] or k0 > x[-1]:
        return np.nan
    return float(np.interp(k0, x, y))


def solve_k_for_delta(df_exp: pd.DataFrame, T: float, target: float, is_call: bool) -> float:
    def f(k: float) -> float:
        iv = interp_at_k(df_exp, k)
        if not np.isfinite(iv) or iv <= 0:
            return np.nan
        d = call_forward_delta_from_k(k, iv, T) if is_call else put_forward_delta_from_k(k, iv, T)
        return d - target

    ks = np.linspace(K_SEARCH_MIN, K_SEARCH_MAX, 61)
    vals = np.array([f(float(kk)) for kk in ks], dtype=float)
    m = np.isfinite(vals)
    ks, vals = ks[m], vals[m]
    if len(ks) < 8:
        return np.nan

    for i in range(len(ks) - 1):
        a, b = float(ks[i]), float(ks[i + 1])
        fa, fb = float(vals[i]), float(vals[i + 1])
        if np.isfinite(fa) and abs(fa) < 1e-10:
            return a
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            try:
                return float(brentq(lambda z: f(z), a, b, maxiter=80))
            except Exception:
                return np.nan
    return np.nan


# =========================================================
# AR(1)
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


def ar1_predict_next(dx_last: float, a: float, b: float) -> float:
    if not (np.isfinite(dx_last) and np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return float(a + b * dx_last)


def signal_from_pred(pred: float) -> float:
    if not np.isfinite(pred):
        return np.nan
    return 1.0 if pred > 0 else (-1.0 if pred < 0 else 0.0)


# =========================================================
# Build node IV series from smile grid (closest maturity)
# =========================================================
def build_node_iv_series(smile: pd.DataFrame, factors: pd.DataFrame, tenor_days: int, gap_thr_days: float) -> pd.DataFrame:
    target_T = tenor_days / 365.0
    gap_col = f"gap_days_{tenor_days}d"

    f = factors[["quote_date", gap_col]].copy()
    f["quote_date"] = pd.to_datetime(f["quote_date"], errors="coerce")
    f[gap_col] = pd.to_numeric(f[gap_col], errors="coerce")
    ok_days = set(f.loc[f[gap_col].notna() & (f[gap_col] <= gap_thr_days), "quote_date"].dt.date)

    s = smile.copy()
    s["quote_date"] = pd.to_datetime(s["quote_date"], errors="coerce").dt.date
    s["T"] = pd.to_numeric(s["T"], errors="coerce")
    s["k"] = pd.to_numeric(s["k"], errors="coerce")
    s["iv_fit"] = pd.to_numeric(s["iv_fit"], errors="coerce")
    s = s.dropna(subset=["quote_date", "expiry", "T", "k", "iv_fit"])
    s = s[(s["T"] > 0) & (s["iv_fit"] > 0)]

    rows = []
    for qd, g_day in s.groupby("quote_date", sort=True):
        if qd not in ok_days:
            continue

        exp_Ts = g_day.groupby("expiry")["T"].median()
        if exp_Ts.empty:
            continue

        exp = exp_Ts.index[np.argmin(np.abs(exp_Ts.to_numpy(dtype=float) - target_T))]
        T_used = float(exp_Ts.loc[exp])
        gap_days = float(abs(T_used - target_T) * 365.0)

        g_exp = g_day[g_day["expiry"] == exp]
        atm = interp_at_k(g_exp, 0.0)

        k_c25 = solve_k_for_delta(g_exp, T_used, target=DELTA_TARGET, is_call=True)
        k_p25 = solve_k_for_delta(g_exp, T_used, target=-DELTA_TARGET, is_call=False)

        c25 = interp_at_k(g_exp, k_c25) if np.isfinite(k_c25) else np.nan
        p25 = interp_at_k(g_exp, k_p25) if np.isfinite(k_p25) else np.nan

        rows.append(
            {
                "quote_date": pd.to_datetime(qd),
                "tenor_days": tenor_days,
                "T_used": T_used,
                "gap_days": gap_days,
                "atm_iv": atm,
                "c25_iv": c25,
                "p25_iv": p25,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.dropna(subset=["quote_date"]).sort_values("quote_date").reset_index(drop=True)


def compute_proxy_returns(node_iv: pd.DataFrame, kind: str) -> pd.Series:
    df = node_iv.copy().sort_values("quote_date")
    df = df.dropna(subset=["atm_iv", "c25_iv", "p25_iv"])
    if df.empty:
        return pd.Series(dtype=float)

    df = df.set_index("quote_date")
    d_atm = df["atm_iv"].diff()
    d_c25 = df["c25_iv"].diff()
    d_p25 = df["p25_iv"].diff()

    if kind == "BF25":
        ret = 0.5 * (d_p25 + d_c25) - d_atm
    elif kind == "RR25":
        ret = (d_p25 - d_c25)
    else:
        raise ValueError("kind must be BF25 or RR25")

    ret = pd.to_numeric(ret, errors="coerce")
    return ret.dropna()


# =========================================================
# Rolling windows (adaptive for small samples)
# =========================================================
@dataclass
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_adaptive_windows(dates: pd.DatetimeIndex) -> tuple[list[Window], tuple[int, int, int]]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    n = len(dates)

    # adaptive defaults:
    # - train ~60% of data but at least 40 days
    # - test ~25% of data but at least 20 days
    # - step = test_len (non-overlapping)
    min_train = max(40, int(round(0.60 * n)))
    test_len = max(20, int(round(0.25 * n)))
    step = test_len

    if n < min_train + test_len:
        # fallback: 60/40 split single window
        min_train = max(40, int(round(0.60 * n)))
        test_len = n - min_train
        step = test_len

    if test_len < 10:
        return [], (min_train, test_len, step)

    windows = []
    i0 = min_train
    while True:
        test_start = dates[i0]
        test_end_idx = i0 + test_len - 1
        if test_end_idx >= n:
            break
        windows.append(
            Window(
                train_start=dates[0],
                train_end=dates[i0 - 1],
                test_start=test_start,
                test_end=dates[test_end_idx],
            )
        )
        i0 += step
        if i0 >= n:
            break

    return windows, (min_train, test_len, step)


# =========================================================
# Backtest
# =========================================================
def backtest_rolling_ar1(factor: pd.Series, ret_proxy: pd.Series, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fac = pd.to_numeric(factor, errors="coerce").astype(float).dropna()
    rp = pd.to_numeric(ret_proxy, errors="coerce").astype(float).dropna()

    common = fac.index.intersection(rp.index)
    fac = fac.reindex(common)
    rp = rp.reindex(common)
    n = len(common)

    if n < 60:
        print(f"  {label}: too few common dates after alignment: n={n}")
        return pd.DataFrame(), pd.DataFrame()

    dx = fac.diff()

    windows, (min_train, test_len, step) = make_adaptive_windows(common)
    print(f"  {label}: aligned_dates={n}, windows={len(windows)} | min_train={min_train}, test_len={test_len}, step={step}")

    if not windows:
        # No rolling windows possible -> do one expanding split (train first 60%, test rest)
        split = max(40, int(round(0.60 * n)))
        w = Window(common[0], common[split - 1], common[split], common[-1])
        windows = [w]

    daily_rows = []
    window_rows = []

    for w_idx, w in enumerate(windows, start=1):
        train_mask = (common >= w.train_start) & (common <= w.train_end)
        test_mask = (common >= w.test_start) & (common <= w.test_end)

        dx_train = dx.loc[train_mask].to_numpy(dtype=float)
        a, b = ar1_fit(dx_train)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue

        test_dates = common[test_mask]
        if len(test_dates) == 0:
            continue

        pos = pd.Series(index=test_dates, data=0.0, dtype=float)
        for d in test_dates:
            dx_last = float(dx.loc[d]) if np.isfinite(dx.loc[d]) else np.nan
            pred = ar1_predict_next(dx_last, a, b)
            pos.loc[d] = signal_from_pred(pred)

        pos_trade = pos.shift(1).fillna(0.0)
        rp_test = rp.loc[test_dates]

        pnl = pos_trade * rp_test
        turnover = pos_trade.diff().abs().fillna(0.0)
        cost = COST_PER_TRADE_IV * (turnover > 0).astype(float) + COST_PER_UNIT_TURNOVER_IV * turnover
        pnl_net = pnl - cost
        equity = pnl_net.cumsum()

        daily_rows.append(
            pd.DataFrame(
                {
                    "date": test_dates,
                    "label": label,
                    "window_id": w_idx,
                    "train_start": w.train_start,
                    "train_end": w.train_end,
                    "test_start": w.test_start,
                    "test_end": w.test_end,
                    "pos": pos_trade.values,
                    "ret_proxy": rp_test.values,
                    "turnover": turnover.values,
                    "cost": cost.values,
                    "pnl": pnl_net.values,
                    "equity": equity.values,
                }
            )
        )

        pn = pnl_net.to_numpy(dtype=float)
        mu = float(np.mean(pn))
        sd = float(np.std(pn, ddof=1) + EPS)
        sharpe = (mu / sd) * math.sqrt(252.0) if len(pn) > 10 else np.nan

        eq = equity.to_numpy(dtype=float)
        peak = np.maximum.accumulate(eq) if len(eq) else np.array([])
        dd = eq - peak if len(eq) else np.array([])
        maxdd = float(dd.min()) if len(dd) else np.nan

        t_rate = float(np.mean(turnover.to_numpy(dtype=float) > 0)) if len(turnover) else np.nan

        window_rows.append(
            {
                "label": label,
                "window_id": w_idx,
                "train_start": w.train_start.date(),
                "train_end": w.train_end.date(),
                "test_start": w.test_start.date(),
                "test_end": w.test_end.date(),
                "n_days": int(len(pn)),
                "mean_pnl": mu,
                "vol_pnl": sd,
                "sharpe_proxy": sharpe,
                "max_drawdown_proxy": maxdd,
                "trade_day_frac": t_rate,
                "ar1_a": a,
                "ar1_b": b,
            }
        )

    daily_out = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    windows_out = pd.DataFrame(window_rows)
    return daily_out, windows_out


def overall_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()

    out = []
    for label, g in daily.groupby("label"):
        pnl = pd.to_numeric(g["pnl"], errors="coerce").to_numpy(dtype=float)
        pnl = pnl[np.isfinite(pnl)]
        if len(pnl) < 30:
            continue
        mu = float(np.mean(pnl))
        sd = float(np.std(pnl, ddof=1) + EPS)
        sharpe = (mu / sd) * math.sqrt(252.0)

        eq = pd.to_numeric(g["equity"], errors="coerce").to_numpy(dtype=float)
        eq = eq[np.isfinite(eq)]
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        maxdd = float(dd.min()) if len(dd) else np.nan

        t_rate = float(np.mean(pd.to_numeric(g["turnover"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0))

        out.append(
            {
                "label": label,
                "n_days": int(len(pnl)),
                "mean_pnl": mu,
                "vol_pnl": sd,
                "sharpe_proxy": sharpe,
                "max_drawdown_proxy": maxdd,
                "trade_day_frac": t_rate,
            }
        )
    return pd.DataFrame(out).sort_values("label")


# =========================================================
# Main
# =========================================================
def main():
    if not SMILE_GRID_PATH.exists():
        raise FileNotFoundError(f"Missing: {SMILE_GRID_PATH}")
    if not FACTORS_PATH.exists():
        raise FileNotFoundError(f"Missing: {FACTORS_PATH}")

    smile = pd.read_parquet(SMILE_GRID_PATH)
    factors = pd.read_parquet(FACTORS_PATH)

    factors = factors.copy()
    factors["quote_date"] = pd.to_datetime(factors["quote_date"], errors="coerce")
    factors = factors.dropna(subset=["quote_date"]).sort_values("quote_date").reset_index(drop=True)
    factors = factors.set_index("quote_date")

    daily_all = []
    window_all = []

    for tenor in [30, 60]:
        gap_thr = GAP_THRESHOLDS_DAYS[tenor]

        node_iv = build_node_iv_series(smile, factors.reset_index(), tenor_days=tenor, gap_thr_days=gap_thr)
        if node_iv.empty:
            print(f"Tenor {tenor}d: no node IV series after gap filtering.")
            continue

        rp_bf = compute_proxy_returns(node_iv, "BF25")
        rp_rr = compute_proxy_returns(node_iv, "RR25")

        bf_col = f"bf25_{tenor}d_closest"
        rr_col = f"rr25_{tenor}d_closest"
        if bf_col not in factors.columns or rr_col not in factors.columns:
            print(f"Tenor {tenor}d: missing factor columns. Skipping.")
            continue

        label = f"BF25_{tenor}D_gap<={gap_thr:g}"
        daily, win = backtest_rolling_ar1(factors[bf_col], rp_bf, label=label)
        if not daily.empty:
            daily_all.append(daily)
            window_all.append(win)

        label = f"RR25_{tenor}D_gap<={gap_thr:g}"
        daily, win = backtest_rolling_ar1(factors[rr_col], rp_rr, label=label)
        if not daily.empty:
            daily_all.append(daily)
            window_all.append(win)

    if not daily_all:
        raise RuntimeError(
            "No backtests produced. Likely data overlap is too small. "
            "Try loosening GAP thresholds or compute node IV via maturity interpolation."
        )

    daily_out = pd.concat(daily_all, ignore_index=True)
    windows_out = pd.concat(window_all, ignore_index=True) if window_all else pd.DataFrame()
    overall = overall_summary(daily_out)

    OUT_DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily_out.to_parquet(OUT_DAILY_PATH, index=False)
    windows_out.to_csv(OUT_WINDOW_SUMMARY_PATH, index=False)
    overall.to_csv(OUT_OVERALL_SUMMARY_PATH, index=False)

    print("\n=== Proxy backtest done ===")
    print(f"Wrote daily:   {OUT_DAILY_PATH}")
    print(f"Wrote windows: {OUT_WINDOW_SUMMARY_PATH}")
    print(f"Wrote overall: {OUT_OVERALL_SUMMARY_PATH}")

    if not overall.empty:
        print("\nOverall summary:")
        print(overall.to_string(index=False))

    if not windows_out.empty:
        print("\nWindow summary (best/worst Sharpe per label):")
        for label, g in windows_out.groupby("label"):
            g2 = g.dropna(subset=["sharpe_proxy"]).sort_values("sharpe_proxy")
            if len(g2) == 0:
                continue
            worst = g2.iloc[0]
            best = g2.iloc[-1]
            print(f"\n{label}")
            print(f"  worst window_id={int(worst['window_id'])} test={worst['test_start']}..{worst['test_end']} sharpe={worst['sharpe_proxy']:.2f}")
            print(f"  best  window_id={int(best['window_id'])} test={best['test_start']}..{best['test_end']} sharpe={best['sharpe_proxy']:.2f}")


if __name__ == "__main__":
    main()
