from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================
HERE = Path(__file__).resolve()
OPTIONS_DIR = HERE.parents[1]
DATA_DIR = OPTIONS_DIR / "data" / "processed"

FACTORS_PATH = DATA_DIR / "surface_factors.parquet"
OUT_CSV = DATA_DIR / "factor_forecast_report.csv"
OUT_PARQUET = DATA_DIR / "factor_forecast_report.parquet"


# =========================================================
# Config: maturity gap filters (days)
# =========================================================
GAP_THRESHOLDS_DAYS = {
    30: 20.0,
    60: 30.0,
    90: 45.0,
}


# =========================================================
# Helpers
# =========================================================
def rmse(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(x * x)))


def mae(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(np.abs(x)))


def sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Sign accuracy only where the prediction actually has a direction.
    If the model predicts 0-change (like RW), sign accuracy is undefined -> NaN.
    """
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return np.nan

    yt = np.sign(y_true[m])
    yp = np.sign(y_pred[m])

    # ignore zero predictions (no directional call)
    nzp = yp != 0
    if nzp.sum() == 0:
        return np.nan

    return float(np.mean(yt[nzp] == yp[nzp]))



def infer_gap_col_and_threshold(factor: str) -> tuple[str | None, float | None]:
    """
    For factors like atm_30d_closest / rr25_60d_closest / bf25_90d_closest:
    use gap_days_30d / gap_days_60d / gap_days_90d and thresholds.
    For term_slope_*_closest: use the LONG tenor (90d) by default.
    For *_exact: no gap filtering (returns None).
    """
    if factor.endswith("_exact"):
        return None, None

    # closest tenor factors
    m = re.search(r"_(30|60|90)d_closest$", factor)
    if m:
        d = int(m.group(1))
        return f"gap_days_{d}d", GAP_THRESHOLDS_DAYS[d]

    # term slopes: pick the long end (90d) as "quality" proxy
    if factor.endswith("_closest") and factor.startswith("term_slope_"):
        return "gap_days_90d", GAP_THRESHOLDS_DAYS[90]

    return None, None


def ewma_forecast(dx: np.ndarray, span: int = 20) -> np.ndarray:
    """
    Predict next change with EWMA of past changes.
    yhat[t] uses info up to t (i.e., EWMA of dx up to t), predicts dx[t+1].
    """
    s = pd.Series(dx)
    ew = s.ewm(span=span, adjust=False).mean().to_numpy()
    return ew


def ar1_forecast(dx: np.ndarray) -> np.ndarray:
    """
    Fit: dx[t+1] = a + b*dx[t] using OLS on available pairs.
    Return predictions for each t (predicting t+1).
    """
    x = dx[:-1]
    y = dx[1:]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 25:
        return np.full_like(dx, np.nan, dtype=float)

    x2 = x[m]
    y2 = y[m]
    X = np.column_stack([np.ones_like(x2), x2])
    beta, *_ = np.linalg.lstsq(X, y2, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    # predictions aligned to dx index: pred[t] predicts dx[t+1]
    pred = np.full_like(dx, np.nan, dtype=float)
    pred[:-1] = a + b * dx[:-1]
    return pred


def evaluate_one_series(dates: np.ndarray, x: np.ndarray) -> tuple[int, dict]:
    """
    Build next-day change prediction:
      y[t] = dx[t+1] = x[t+1] - x[t]
      Predictions use only info at or before t.

    Returns n_obs and metrics dict.
    """
    # sort just in case
    idx = np.argsort(dates)
    x = x[idx]

    # dx[t] = x[t] - x[t-1]
    dx = np.diff(x)

    # align predictions to y = dx[t] where t corresponds to "next-day change"
    # We'll define:
    #   y = dx (length N-1) where y[i] corresponds to change from i -> i+1
    y = dx.copy()

    # RW predicts yhat = 0
    yhat_rw = np.zeros_like(y)

    # EWMA predicts yhat[i] = EWMA(dx up to i-1) (since predicting dx at i)
    # easiest: compute EWMA on y, shift by 1
    yhat_ew = ewma_forecast(y, span=20)
    yhat_ew = np.roll(yhat_ew, 1)
    yhat_ew[0] = np.nan

    # AR1 predicts y[i] from y[i-1] (i.e., dy relationship)
    yhat_ar_full = ar1_forecast(y)
    yhat_ar = np.roll(yhat_ar_full, 1)
    yhat_ar[0] = np.nan

    m = np.isfinite(y) & np.isfinite(yhat_rw)  # same mask base
    n_obs = int(m.sum())

    err_rw = yhat_rw - y
    err_ew = yhat_ew - y
    err_ar = yhat_ar - y

    metrics = {
        "rw_rmse": rmse(err_rw),
        "rw_mae": mae(err_rw),
        "rw_signacc": sign_accuracy(y, yhat_rw),
        "ewma_rmse": rmse(err_ew),
        "ewma_mae": mae(err_ew),
        "ewma_signacc": sign_accuracy(y, yhat_ew),
        "ar1_rmse": rmse(err_ar),
        "ar1_mae": mae(err_ar),
        "ar1_signacc": sign_accuracy(y, yhat_ar),
    }
    return n_obs, metrics


def print_top_improvers(report: pd.DataFrame, block: str, model: str, top_k: int = 10) -> None:
    """
    model in {"ewma", "ar1"}.
    Ranks by rmse improvement vs RW (positive is good).
    """
    dfb = report[report["block"] == block].copy()
    if dfb.empty:
        return

    dfb["rw_rmse"] = pd.to_numeric(dfb["rw_rmse"], errors="coerce")
    dfb[f"{model}_rmse"] = pd.to_numeric(dfb[f"{model}_rmse"], errors="coerce")
    dfb["rmse_improve"] = dfb["rw_rmse"] - dfb[f"{model}_rmse"]
    dfb = dfb.replace([np.inf, -np.inf], np.nan).dropna(subset=["rmse_improve"])
    dfb = dfb.sort_values("rmse_improve", ascending=False)

    print(f"\nTop {top_k} factors where {model.upper()} beats RW most (by RMSE) — block={block}:")
    show = dfb.head(top_k)[["factor", "n_obs", "rw_rmse", f"{model}_rmse", "rmse_improve", "rw_signacc", f"{model}_signacc"]]
    print(show.to_string(index=False))


# =========================================================
# Main
# =========================================================
def main():
    if not FACTORS_PATH.exists():
        raise FileNotFoundError(f"Missing: {FACTORS_PATH}")

    df = pd.read_parquet(FACTORS_PATH).copy()
    if "quote_date" not in df.columns:
        raise ValueError("surface_factors.parquet must contain 'quote_date' column")

    df["quote_date"] = pd.to_datetime(df["quote_date"], errors="coerce")
    df = df.dropna(subset=["quote_date"]).sort_values("quote_date").reset_index(drop=True)

    # factors to evaluate: keep numeric columns, skip metadata
    skip_prefixes = ("gap_days_", "T_used_")
    factor_cols = []
    for c in df.columns:
        if c == "quote_date":
            continue
        if any(c.startswith(p) for p in skip_prefixes):
            continue
        # only evaluate numeric-like cols
        if pd.api.types.is_numeric_dtype(df[c]):
            factor_cols.append(c)

    if not factor_cols:
        raise ValueError("No numeric factor columns found to evaluate.")

    rows = []

    # Evaluate ALL_DAYS + GAP_FILTERED (where applicable)
    for factor in factor_cols:
        x_all = pd.to_numeric(df[factor], errors="coerce").to_numpy(dtype=float)
        dates = df["quote_date"].to_numpy()

        # --------------------------
        # ALL_DAYS block
        # --------------------------
        m_all = np.isfinite(x_all)
        if m_all.sum() >= 60:
            n_obs, metrics = evaluate_one_series(dates[m_all], x_all[m_all])
            rows.append({"block": "ALL_DAYS", "factor": factor, "n_obs": n_obs, **metrics})

        # --------------------------
        # GAP_FILTERED block
        # --------------------------
        gap_col, thr = infer_gap_col_and_threshold(factor)
        if gap_col is not None and gap_col in df.columns and thr is not None:
            gap = pd.to_numeric(df[gap_col], errors="coerce").to_numpy(dtype=float)
            m_gap = np.isfinite(x_all) & np.isfinite(gap) & (gap <= thr)
            if m_gap.sum() >= 60:
                n_obs, metrics = evaluate_one_series(dates[m_gap], x_all[m_gap])
                rows.append({"block": f"GAP_FILTERED({gap_col}<= {thr:g})", "factor": factor, "n_obs": n_obs, **metrics})

    report = pd.DataFrame(rows)
    if report.empty:
        raise ValueError("No evaluations were produced. Check factor coverage / NaNs.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT_CSV, index=False)
    report.to_parquet(OUT_PARQUET, index=False)

    print("\n=== Forecast baseline report (next-day factor changes) ===")
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_PARQUET}")
    print(f"Rows: {len(report):,}  |  Factors evaluated: {report['factor'].nunique():,}")

    # Print a compact view for ALL_DAYS
    all_view = report[report["block"] == "ALL_DAYS"].sort_values("factor")
    if not all_view.empty:
        print("\nSample (ALL_DAYS) — first 12 rows:")
        print(all_view.head(12).to_string(index=False))

    # Print top improvers for each block that exists
    blocks = list(report["block"].unique())
    for b in blocks:
        print_top_improvers(report, block=b, model="ewma", top_k=10)
        print_top_improvers(report, block=b, model="ar1", top_k=10)


if __name__ == "__main__":
    main()
