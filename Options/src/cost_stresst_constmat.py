from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# Paths (relative)
# =========================================================
HERE = Path(__file__).resolve()
OPTIONS_DIR = HERE.parents[1]
DATA_DIR = OPTIONS_DIR / "data" / "processed"

DAILY_PATH = DATA_DIR / "pnl_proxy_constmat_daily.parquet"
WINDOWS_PATH = DATA_DIR / "pnl_proxy_constmat_windows.csv"

OUT_WINDOWS_STRESS = DATA_DIR / "pnl_proxy_constmat_windows_cost_stress.csv"
OUT_OVERALL_STRESS = DATA_DIR / "pnl_proxy_constmat_overall_cost_stress.csv"

# =========================================================
# Stress settings
# =========================================================
# We apply extra costs on top of whatever the daily file currently contains.
# If you want to ignore original costs and fully recompute, set IGNORE_EXISTING_COST=True.
IGNORE_EXISTING_COST = False

# Multipliers to test
COST_MULTIPLIERS = [1, 5, 10]

# If your daily file doesn't have a 'cost' column,
# we compute a simple proxy:
#   cost = COST_PER_TRADE * 1{pos changes} + COST_PER_UNIT_TURNOVER * abs(pos - pos_prev)
COST_PER_TRADE = 0.002
COST_PER_UNIT_TURNOVER = 0.001

# =========================================================
# Helpers
# =========================================================
def infer_date_col(df: pd.DataFrame) -> str:
    for c in ["date", "quote_date", "QUOTE_DATE"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a date column. Columns: {list(df.columns)}")


def sharpe_annualized(x: np.ndarray, periods_per_year: int = 252) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    mu = np.mean(x)
    sd = np.std(x, ddof=1)
    if sd <= 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * mu / sd)


def max_drawdown(cum: np.ndarray) -> float:
    cum = np.asarray(cum, dtype=float)
    if len(cum) == 0:
        return np.nan
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.min(dd))


def compute_cost_from_positions(df: pd.DataFrame, pos_col: str = "pos") -> pd.Series:
    pos = pd.to_numeric(df[pos_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pos_prev = np.roll(pos, 1)
    pos_prev[0] = 0.0
    dpos = pos - pos_prev
    traded = (np.abs(dpos) > 1e-12).astype(float)
    turnover = np.abs(dpos)
    cost = COST_PER_TRADE * traded + COST_PER_UNIT_TURNOVER * turnover
    return pd.Series(cost, index=df.index)


def window_summaries(daily: pd.DataFrame, windows: pd.DataFrame, pnl_col: str) -> pd.DataFrame:
    # windows has factor, window_id, train/test start/end
    out = []
    date_col = infer_date_col(daily)
    daily = daily.copy()
    daily[date_col] = pd.to_datetime(daily[date_col]).dt.date

    for r in windows.itertuples(index=False):
        f = getattr(r, "factor")
        w = int(getattr(r, "window_id"))
        test_start = pd.to_datetime(getattr(r, "test_start")).date()
        test_end = pd.to_datetime(getattr(r, "test_end")).date()

        d = daily[(daily["factor"] == f) & (daily[date_col] >= test_start) & (daily[date_col] <= test_end)].copy()
        x = pd.to_numeric(d[pnl_col], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]

        if len(x) == 0:
            continue

        cum = np.cumsum(x)
        out.append({
            "factor": f,
            "window_id": w,
            "test_start": str(test_start),
            "test_end": str(test_end),
            "n_days": int(len(x)),
            "mean_pnl": float(np.mean(x)),
            "vol_pnl": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "sharpe_proxy": sharpe_annualized(x),
            "max_drawdown_proxy": max_drawdown(cum),
            "trade_day_frac": float(np.mean(np.abs(np.diff(pd.to_numeric(d["pos"], errors="coerce").fillna(0.0).to_numpy(dtype=float))) > 1e-12)) if "pos" in d.columns and len(d) > 1 else np.nan
        })

    return pd.DataFrame(out).sort_values(["factor", "window_id"]).reset_index(drop=True)


def overall_summaries(daily: pd.DataFrame, pnl_col: str) -> pd.DataFrame:
    date_col = infer_date_col(daily)
    daily = daily.copy()
    daily[date_col] = pd.to_datetime(daily[date_col]).dt.date

    out = []
    for f, g in daily.groupby("factor", sort=True):
        x = pd.to_numeric(g[pnl_col], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 5:
            continue
        cum = np.cumsum(x)
        pos = pd.to_numeric(g["pos"], errors="coerce").fillna(0.0).to_numpy(dtype=float) if "pos" in g.columns else None
        trade_frac = float(np.mean(np.abs(np.diff(pos)) > 1e-12)) if pos is not None and len(pos) > 1 else np.nan

        out.append({
            "factor": f,
            "n_days": int(len(x)),
            "mean_pnl": float(np.mean(x)),
            "vol_pnl": float(np.std(x, ddof=1)),
            "sharpe_proxy": sharpe_annualized(x),
            "max_drawdown_proxy": max_drawdown(cum),
            "trade_day_frac": trade_frac,
        })

    return pd.DataFrame(out).sort_values("sharpe_proxy", ascending=False).reset_index(drop=True)


# =========================================================
# Main
# =========================================================
def main():
    if not DAILY_PATH.exists():
        raise FileNotFoundError(f"Missing daily results: {DAILY_PATH}")
    if not WINDOWS_PATH.exists():
        raise FileNotFoundError(f"Missing windows file: {WINDOWS_PATH}")

    daily = pd.read_parquet(DAILY_PATH)
    windows = pd.read_csv(WINDOWS_PATH)

    # Basic required fields
    for c in ["factor", "pos"]:
        if c not in daily.columns:
            raise ValueError(f"Daily file missing required column '{c}'. Found: {list(daily.columns)}")

    # We need the underlying "gross" pnl driver. Prefer explicit columns.
    # Common patterns:
    # - dx + pos  -> gross_pnl = pos * dx
    # - gross_pnl already present
    if "gross_pnl" in daily.columns:
        gross = pd.to_numeric(daily["gross_pnl"], errors="coerce")
    elif "dx" in daily.columns:
        gross = pd.to_numeric(daily["pos"], errors="coerce") * pd.to_numeric(daily["dx"], errors="coerce")
        daily["gross_pnl"] = gross
    elif "pnl" in daily.columns:
        # We can still stress by adding extra cost on top of pnl if needed,
        # but better if we can reconstruct gross.
        gross = None
    else:
        raise ValueError(
            "Daily file must contain either ('gross_pnl') or ('dx') or ('pnl'). "
            f"Found columns: {list(daily.columns)}"
        )

    # Existing cost if present
    if "cost" in daily.columns:
        base_cost = pd.to_numeric(daily["cost"], errors="coerce").fillna(0.0)
    else:
        base_cost = None

    # If no cost column, compute a proxy cost from positions
    if base_cost is None:
        daily["cost"] = compute_cost_from_positions(daily, pos_col="pos")
        base_cost = daily["cost"]

    # If no gross info but pnl exists, we can only "stress add" extra cost
    if gross is None:
        daily["pnl_base"] = pd.to_numeric(daily["pnl"], errors="coerce")
        if daily["pnl_base"].isna().all():
            raise ValueError("Daily file has 'pnl' but it's all NaN.")
        # derive an implied gross = pnl + cost (best effort)
        daily["gross_pnl"] = daily["pnl_base"] + base_cost
    else:
        daily["pnl_base"] = daily["gross_pnl"] - base_cost

    # Now apply cost multipliers
    all_win = []
    all_overall = []

    for m in COST_MULTIPLIERS:
        if IGNORE_EXISTING_COST:
            # Use *only* stressed proxy costs derived from positions
            stressed_cost = m * compute_cost_from_positions(daily, pos_col="pos")
        else:
            # Scale the existing cost that was used (or computed)
            stressed_cost = m * base_cost

        pnl_col = f"pnl_costx{m}"
        daily[pnl_col] = daily["gross_pnl"] - stressed_cost

        wsum = window_summaries(daily, windows, pnl_col=pnl_col)
        wsum["cost_multiplier"] = m
        all_win.append(wsum)

        osum = overall_summaries(daily, pnl_col=pnl_col)
        osum["cost_multiplier"] = m
        all_overall.append(osum)

    win_df = pd.concat(all_win, ignore_index=True) if all_win else pd.DataFrame()
    overall_df = pd.concat(all_overall, ignore_index=True) if all_overall else pd.DataFrame()

    win_df.to_csv(OUT_WINDOWS_STRESS, index=False)
    overall_df.to_csv(OUT_OVERALL_STRESS, index=False)

    print("\n=== Cost stress done ===")
    print("Wrote windows:", OUT_WINDOWS_STRESS)
    print("Wrote overall:", OUT_OVERALL_STRESS)

    # Show a compact before/after view for each factor (window-level)
    if len(win_df) > 0:
        pivot = win_df.pivot_table(
            index=["factor", "window_id"],
            columns="cost_multiplier",
            values="sharpe_proxy",
            aggfunc="first",
        ).reset_index()
        pivot = pivot.rename(columns={m: f"sharpe_costx{m}" for m in COST_MULTIPLIERS})
        print("\nSharpe by window (cost stress):")
        print(pivot.to_string(index=False))

    # Show overall top 10 at each multiplier
    if len(overall_df) > 0:
        for m in COST_MULTIPLIERS:
            sub = overall_df[overall_df["cost_multiplier"] == m].sort_values("sharpe_proxy", ascending=False).head(10)
            print(f"\nTop 10 overall by Sharpe (cost x{m}):")
            print(sub[["factor", "n_days", "sharpe_proxy", "mean_pnl", "vol_pnl", "trade_day_frac"]].to_string(index=False))


if __name__ == "__main__":
    main()
