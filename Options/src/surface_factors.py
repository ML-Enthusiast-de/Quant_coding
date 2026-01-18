from __future__ import annotations

import math
from pathlib import Path

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
OUT_PATH = DATA_DIR / "surface_factors.parquet"

# =========================================================
# Config
# =========================================================
TENORS_DAYS = [30, 60, 90]
TENORS = [d / 365.0 for d in TENORS_DAYS]
DELTA_TARGET = 0.25

# k solve search window
K_SEARCH_MIN = -0.9
K_SEARCH_MAX = 0.9
EPS = 1e-8


# -----------------------------
# Black–Scholes forward-delta mapping
# -----------------------------
def d1_from_k_sigma_T(k: float, sigma: float, T: float) -> float:
    return (-k + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T) + EPS)


def call_forward_delta_from_k(k: float, sigma: float, T: float) -> float:
    return norm.cdf(d1_from_k_sigma_T(k, sigma, T))


def put_forward_delta_from_k(k: float, sigma: float, T: float) -> float:
    return norm.cdf(d1_from_k_sigma_T(k, sigma, T)) - 1.0


# -----------------------------
# Interpolation helpers
# -----------------------------
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
                return float(brentq(lambda z: f(z), a, b, maxiter=100))
            except Exception:
                return np.nan
    return np.nan


def interp_in_T(Ts: np.ndarray, ys: np.ndarray, T0: float) -> float:
    m = np.isfinite(Ts) & np.isfinite(ys)
    Ts, ys = Ts[m], ys[m]
    if len(Ts) < 2:
        return np.nan

    idx = np.argsort(Ts)
    Ts, ys = Ts[idx], ys[idx]

    if T0 < Ts[0] or T0 > Ts[-1]:
        return np.nan

    return float(np.interp(T0, Ts, ys))


def closest_in_T(Ts: np.ndarray, ys: np.ndarray, T0: float) -> tuple[float, float, float]:
    """
    Always pick nearest maturity (no tolerance).
    Returns (y_closest, T_used, gap_days)
    """
    m = np.isfinite(Ts) & np.isfinite(ys)
    Ts2, ys2 = Ts[m], ys[m]
    if len(Ts2) == 0:
        return np.nan, np.nan, np.nan

    idx = int(np.argmin(np.abs(Ts2 - T0)))
    T_used = float(Ts2[idx])
    gap_days = float(abs(T_used - T0) * 365.0)
    return float(ys2[idx]), T_used, gap_days


# =========================================================
# Main
# =========================================================
def main():
    if not SMILE_GRID_PATH.exists():
        raise FileNotFoundError(f"Missing: {SMILE_GRID_PATH}")

    df = pd.read_parquet(SMILE_GRID_PATH)

    need = ["quote_date", "expiry", "T", "k", "iv_fit"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {SMILE_GRID_PATH.name}: {missing}\nFound: {list(df.columns)}")

    df = df.copy()
    df["T"] = pd.to_numeric(df["T"], errors="coerce")
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df["iv_fit"] = pd.to_numeric(df["iv_fit"], errors="coerce")
    df = df.dropna(subset=["quote_date", "expiry", "T", "k", "iv_fit"])
    df = df[(df["T"] > 0) & (df["iv_fit"] > 0)]

    print(f"Loaded fitted smile grid: {len(df):,} rows")
    print("Computing factors (exact interpolation + closest maturity with gap tracking)...")

    out_rows = []

    for qd, g_day in df.groupby("quote_date", sort=True):
        exp_rows = []

        for ex, g_exp in g_day.groupby("expiry", sort=True):
            T = float(np.nanmedian(g_exp["T"]))
            if not np.isfinite(T) or T <= 0:
                continue

            atm = interp_at_k(g_exp, 0.0)

            k_c25 = solve_k_for_delta(g_exp, T, target=DELTA_TARGET, is_call=True)
            k_p25 = solve_k_for_delta(g_exp, T, target=-DELTA_TARGET, is_call=False)

            iv_c25 = interp_at_k(g_exp, k_c25) if np.isfinite(k_c25) else np.nan
            iv_p25 = interp_at_k(g_exp, k_p25) if np.isfinite(k_p25) else np.nan

            rr = (iv_p25 - iv_c25) if np.isfinite(iv_p25) and np.isfinite(iv_c25) else np.nan
            bf = (0.5 * (iv_p25 + iv_c25) - atm) if np.isfinite(iv_p25) and np.isfinite(iv_c25) and np.isfinite(atm) else np.nan

            exp_rows.append({"T": T, "atm_iv": atm, "rr_25": rr, "bf_25": bf})

        if not exp_rows:
            continue

        exp_df = pd.DataFrame(exp_rows)
        Ts = exp_df["T"].to_numpy(dtype=float)

        day_out = {"quote_date": qd}

        # Exact tenors (interpolation)
        for d, T0 in zip(TENORS_DAYS, TENORS):
            day_out[f"atm_{d}d_exact"] = interp_in_T(Ts, exp_df["atm_iv"].to_numpy(dtype=float), T0)
            day_out[f"rr25_{d}d_exact"] = interp_in_T(Ts, exp_df["rr_25"].to_numpy(dtype=float), T0)
            day_out[f"bf25_{d}d_exact"] = interp_in_T(Ts, exp_df["bf_25"].to_numpy(dtype=float), T0)

        # Closest maturity factors + tracking
        for d, T0 in zip(TENORS_DAYS, TENORS):
            v, T_used, gap = closest_in_T(Ts, exp_df["atm_iv"].to_numpy(dtype=float), T0)
            day_out[f"atm_{d}d_closest"] = v
            day_out[f"T_used_{d}d"] = T_used
            day_out[f"gap_days_{d}d"] = gap

            v, T_used, gap = closest_in_T(Ts, exp_df["rr_25"].to_numpy(dtype=float), T0)
            day_out[f"rr25_{d}d_closest"] = v

            v, T_used, gap = closest_in_T(Ts, exp_df["bf_25"].to_numpy(dtype=float), T0)
            day_out[f"bf25_{d}d_closest"] = v

        # Term slopes short-long
        d_short, d_long = TENORS_DAYS[0], TENORS_DAYS[-1]

        # exact slopes
        a_s = day_out.get(f"atm_{d_short}d_exact", np.nan)
        a_l = day_out.get(f"atm_{d_long}d_exact", np.nan)
        day_out["term_slope_atm_exact"] = (a_s - a_l) if np.isfinite(a_s) and np.isfinite(a_l) else np.nan

        r_s = day_out.get(f"rr25_{d_short}d_exact", np.nan)
        r_l = day_out.get(f"rr25_{d_long}d_exact", np.nan)
        day_out["term_slope_rr25_exact"] = (r_s - r_l) if np.isfinite(r_s) and np.isfinite(r_l) else np.nan

        # closest slopes
        a_s = day_out.get(f"atm_{d_short}d_closest", np.nan)
        a_l = day_out.get(f"atm_{d_long}d_closest", np.nan)
        day_out["term_slope_atm_closest"] = (a_s - a_l) if np.isfinite(a_s) and np.isfinite(a_l) else np.nan

        r_s = day_out.get(f"rr25_{d_short}d_closest", np.nan)
        r_l = day_out.get(f"rr25_{d_long}d_closest", np.nan)
        day_out["term_slope_rr25_closest"] = (r_s - r_l) if np.isfinite(r_s) and np.isfinite(r_l) else np.nan

        out_rows.append(day_out)

    out = pd.DataFrame(out_rows).sort_values("quote_date").reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print("\nDone.")
    print(f"Wrote: {OUT_PATH}")
    print(f"Days: {len(out):,}")

    cols = [c for c in out.columns if c != "quote_date"]
    coverage = out[cols].notna().mean().sort_values(ascending=True)

    print("\nCoverage (fraction non-NaN) — lowest 12:")
    print(coverage.head(12).to_string())

    print("\nCoverage (fraction non-NaN) — highest 12:")
    print(coverage.tail(12).to_string())

    # show maturity gap stats for closest ATM
    gap_cols = [f"gap_days_{d}d" for d in TENORS_DAYS]
    print("\nClosest-maturity gap days (ATM) summary:")
    for c in gap_cols:
        s = out[c].dropna()
        if len(s) == 0:
            print(f"- {c}: no data")
        else:
            print(f"- {c}: median={s.median():.1f}, p90={s.quantile(0.9):.1f}, max={s.max():.1f}")


if __name__ == "__main__":
    main()
