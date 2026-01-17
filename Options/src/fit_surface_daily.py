from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


# =========================================================
# Paths
# =========================================================
HERE = Path(__file__).resolve()
OPTIONS_DIR = HERE.parents[1]
DATA_DIR = OPTIONS_DIR / "data" / "processed"

IV_PATH = DATA_DIR / "spy_iv_points.parquet"
OUT_PATH = DATA_DIR / "surface_smile_grid.parquet"

# =========================================================
# Config
# =========================================================
# k = log(K/F) grid to evaluate fitted smiles on
K_MIN, K_MAX = -0.6, 0.6
K_GRID_N = 61
K_GRID = np.linspace(K_MIN, K_MAX, K_GRID_N)

# Minimum points required to fit a spline for a given expiry
MIN_POINTS_PER_EXPIRY = 10

# Weighting: tighter spreads => more weight
# weight = 1 / max(rel_spread, REL_SPREAD_FLOOR)
REL_SPREAD_FLOOR = 0.02

# Spline smoothing factor. Larger => smoother curve.
# We'll set it per group proportional to number of points, but keep sane bounds.
S_MULT = 0.5


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def fit_smile_spline(k: np.ndarray, iv: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Fit a weighted cubic spline IV(k) and evaluate it on K_GRID.
    Returns: (iv_fit_grid, diagnostics)
    """
    # sort by k
    idx = np.argsort(k)
    k = k[idx]
    iv = iv[idx]
    w = w[idx]

    # drop non-finite
    m = np.isfinite(k) & np.isfinite(iv) & np.isfinite(w) & (w > 0)
    k, iv, w = k[m], iv[m], w[m]

    # need enough unique k values
    if len(k) < MIN_POINTS_PER_EXPIRY or len(np.unique(k)) < 6:
        return np.full_like(K_GRID, np.nan, dtype=float), {"ok": False, "reason": "too_few_points"}

    # choose smoothing
    # heuristic: s ~ S_MULT * n * var(iv)
    n = len(iv)
    var = float(np.nanvar(iv))
    s = S_MULT * n * max(var, 1e-6)

    try:
        spl = UnivariateSpline(k, iv, w=w, k=3, s=s)
        iv_fit = spl(K_GRID)

        # Diagnostics on in-sample points
        iv_pred = spl(k)
        rmse = float(np.sqrt(np.nanmean((iv_pred - iv) ** 2)))
        mae = float(np.nanmean(np.abs(iv_pred - iv)))

        return iv_fit.astype(float), {
            "ok": True,
            "n_points": int(n),
            "rmse": rmse,
            "mae": mae,
            "s": float(s),
            "k_min": float(np.min(k)),
            "k_max": float(np.max(k)),
        }
    except Exception as e:
        return np.full_like(K_GRID, np.nan, dtype=float), {"ok": False, "reason": f"spline_fail:{type(e).__name__}"}


def main():
    if not IV_PATH.exists():
        raise FileNotFoundError(f"Missing: {IV_PATH}")

    df = pd.read_parquet(IV_PATH)

    need = ["quote_date", "expiry", "T", "log_moneyness", "iv", "rel_spread"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {IV_PATH.name}: {missing}\nFound: {list(df.columns)}")

    # Basic hygiene filters
    df = df.copy()
    df["T"] = pd.to_numeric(df["T"], errors="coerce")
    df["k"] = pd.to_numeric(df["log_moneyness"], errors="coerce")
    df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
    df["rel_spread"] = pd.to_numeric(df["rel_spread"], errors="coerce")

    df = df.dropna(subset=["quote_date", "expiry", "T", "k", "iv", "rel_spread"])
    df = df[(df["T"] > 0) & (df["iv"] > 0) & np.isfinite(df["iv"])]

    # weights: tighter spreads matter more
    df["w"] = 1.0 / np.maximum(df["rel_spread"].to_numpy(dtype=float), REL_SPREAD_FLOOR)

    # Fit per (date, expiry)
    rows = []
    grouped = df.groupby(["quote_date", "expiry"], sort=True)
    total = len(grouped)
    print(f"Loaded IV points: {len(df):,} rows")
    print(f"Fitting smiles for: {total:,} (quote_date x expiry)")

    for i, ((qd, ex), g) in enumerate(grouped, start=1):
        k = g["k"].to_numpy(dtype=float)
        iv = g["iv"].to_numpy(dtype=float)
        w = g["w"].to_numpy(dtype=float)

        iv_fit_grid, diag = fit_smile_spline(k, iv, w)

        out = {
            "quote_date": qd,
            "expiry": ex,
            "T_median": float(np.nanmedian(g["T"])),
            "iv_fit_grid": iv_fit_grid,
            "k_grid": K_GRID,  # store once per row; parquet handles list-like columns ok
        }
        out.update(diag)
        rows.append(out)

        if i % 500 == 0:
            print(f"  fitted {i:,}/{total:,} smiles...")

    out_df = pd.DataFrame(rows)

    # Explode the grid columns to a tidy long table:
    # quote_date, expiry, T, k, iv_fit
    ok = out_df[out_df["ok"] == True].copy()

    # If nothing fits, fail loudly
    if len(ok) == 0:
        raise RuntimeError("No smiles were successfully fitted. Check filters / data quality.")

    # explode arrays into rows
    ok = ok[["quote_date", "expiry", "T_median", "n_points", "rmse", "mae", "s", "k_grid", "iv_fit_grid"]].copy()
    ok = ok.explode(["k_grid", "iv_fit_grid"], ignore_index=True)
    ok = ok.rename(columns={"T_median": "T", "k_grid": "k", "iv_fit_grid": "iv_fit"})

    # cast
    ok["T"] = pd.to_numeric(ok["T"], errors="coerce")
    ok["k"] = pd.to_numeric(ok["k"], errors="coerce")
    ok["iv_fit"] = pd.to_numeric(ok["iv_fit"], errors="coerce")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ok.to_parquet(OUT_PATH, index=False)

    print("\nDone.")
    print(f"Wrote: {OUT_PATH}")
    print(f"Rows (grid): {len(ok):,}")
    print("Median RMSE:", float(np.nanmedian(ok["rmse"])))


if __name__ == "__main__":
    main()
