from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# Paths (relative)
# =========================================================
HERE = Path(__file__).resolve()
OPTIONS_DIR = HERE.parents[1]                 # .../Quant_coding/Options
DATA_DIR = OPTIONS_DIR / "data" / "processed"

# Preferred input (const-maturity run), fallback to rolling run
IN_PRIMARY = DATA_DIR / "pnl_proxy_constmat_daily.parquet"
IN_FALLBACK = DATA_DIR / "pnl_proxy_results_rolling.parquet"

OUT_DAILY = DATA_DIR / "cost_attribution_daily.parquet"
OUT_SUMMARY = DATA_DIR / "cost_attribution_summary.csv"

# cost multipliers to report (if base costs exist)
COST_MULTS = [1, 5, 10]


# =========================================================
# Helpers
# =========================================================
def pick_first_existing(cols: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def ensure_datetime(s: pd.Series) -> pd.Series:
    # accept date, datetime, string
    out = pd.to_datetime(s, errors="coerce")
    return out


def sharpe_like(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 10:
        return np.nan
    v = float(x.std(ddof=0))
    if v <= 0:
        return np.nan
    return float(x.mean() / v) * np.sqrt(252.0)


# =========================================================
# Main
# =========================================================
def main():
    # 1) Load
    if IN_PRIMARY.exists():
        in_path = IN_PRIMARY
    elif IN_FALLBACK.exists():
        in_path = IN_FALLBACK
    else:
        raise FileNotFoundError(
            f"Could not find input parquet.\n"
            f"Tried:\n- {IN_PRIMARY}\n- {IN_FALLBACK}"
        )

    df = pd.read_parquet(in_path)
    if len(df) == 0:
        raise RuntimeError(f"Loaded empty dataset: {in_path}")

    cols = list(df.columns)
    print(f"Loaded: {in_path}")
    print(f"Rows: {len(df):,} | Cols: {len(cols)}")

    # 2) Identify key columns
    date_col = pick_first_existing(cols, ["quote_date", "date", "as_of_date", "dt"])
    if date_col is None:
        raise ValueError(f"No date column found. Have: {cols}")

    # Strategy id column (factor/label)
    factor_col = pick_first_existing(cols, ["factor", "label", "strategy", "name"])
    if factor_col is None:
        raise ValueError(f"No factor/label column found. Have: {cols}")

    # Optional: window_id (if rolling windows are present)
    window_col = pick_first_existing(cols, ["window_id", "fold", "split_id"])

    # Position column (needed for turnover)
    pos_col = pick_first_existing(cols, ["pos", "position", "signal_pos", "w", "weight"])
    if pos_col is None:
        print("\nWARNING: No position column found -> turnover/trade-day metrics limited.")
        print("Looked for: pos/position/signal_pos/w/weight\n")

    # PnL / cost columns (we will use whatever exists)
    gross_col = pick_first_existing(cols, ["gross_pnl", "pnl_gross", "pnl_before_cost", "pnl_raw"])
    cost_col = pick_first_existing(cols, ["cost", "tx_cost", "cost_pnl", "pnl_cost"])
    net_col = pick_first_existing(cols, ["net_pnl", "pnl_net", "pnl_after_cost", "pnl"])

    # If we have gross and cost, net can be derived
    if gross_col is not None and cost_col is not None and net_col is None:
        df["net_pnl"] = pd.to_numeric(df[gross_col], errors="coerce") - pd.to_numeric(df[cost_col], errors="coerce")
        net_col = "net_pnl"

    # If we only have "pnl", treat it as net (but we can't decompose)
    can_decompose = (gross_col is not None and cost_col is not None)

    print("\nDetected columns:")
    print(f"- date_col:   {date_col}")
    print(f"- factor_col: {factor_col}")
    print(f"- window_col: {window_col}")
    print(f"- pos_col:    {pos_col}")
    print(f"- gross_col:  {gross_col}")
    print(f"- cost_col:   {cost_col}")
    print(f"- net_col:    {net_col}")
    print(f"- decompose:  {can_decompose}")

    # 3) Normalize types
    df = df.copy()
    df[date_col] = ensure_datetime(df[date_col])
    df = df.dropna(subset=[date_col, factor_col]).copy()

    # Make sure numeric
    if pos_col is not None:
        df[pos_col] = pd.to_numeric(df[pos_col], errors="coerce")

    if gross_col is not None:
        df[gross_col] = pd.to_numeric(df[gross_col], errors="coerce")
    if cost_col is not None:
        df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")
    if net_col is not None:
        df[net_col] = pd.to_numeric(df[net_col], errors="coerce")

    # 4) Create grouping keys
    group_cols = [factor_col]
    if window_col is not None:
        group_cols.append(window_col)

    # 5) Build daily attribution table
    out_daily_rows = []
    for key, g in df.groupby(group_cols, sort=True):
        g = g.sort_values(date_col).copy()

        # turnover metrics
        if pos_col is not None:
            pos = g[pos_col]
            dpos = (pos - pos.shift(1)).abs()
            g["pos_change_abs"] = dpos
            g["trade_day"] = (dpos > 0).astype(int)
        else:
            g["pos_change_abs"] = np.nan
            g["trade_day"] = np.nan

        # pnl metrics
        if can_decompose:
            g["gross_pnl"] = g[gross_col]
            g["base_cost"] = g[cost_col]
            g["net_pnl_x1"] = g["gross_pnl"] - 1.0 * g["base_cost"]
            g["net_pnl_x5"] = g["gross_pnl"] - 5.0 * g["base_cost"]
            g["net_pnl_x10"] = g["gross_pnl"] - 10.0 * g["base_cost"]
        else:
            # best-effort: treat net_col as net pnl, no cost reconstruction
            if net_col is not None:
                g["net_pnl_x1"] = g[net_col]
            else:
                g["net_pnl_x1"] = np.nan

        # attach factor + window back as columns
        if isinstance(key, tuple):
            for c, v in zip(group_cols, key):
                g[c] = v
        else:
            g[group_cols[0]] = key

        out_daily_rows.append(g)

    out_daily = pd.concat(out_daily_rows, ignore_index=True)
    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)
    out_daily.to_parquet(OUT_DAILY, index=False)

    # 6) Summary table
    summary_rows = []
    for key, g in out_daily.groupby(group_cols, sort=True):
        g = g.sort_values(date_col)

        # turnover
        turnover = np.nan
        trade_frac = np.nan
        if pos_col is not None:
            turnover = float(pd.to_numeric(g["pos_change_abs"], errors="coerce").fillna(0.0).sum() / max(len(g), 1))
            trade_frac = float(pd.to_numeric(g["trade_day"], errors="coerce").dropna().mean()) if g["trade_day"].notna().any() else np.nan

        row = {}
        if isinstance(key, tuple):
            for c, v in zip(group_cols, key):
                row[c] = v
        else:
            row[group_cols[0]] = key

        row["n_days"] = int(len(g))
        row["turnover_per_day"] = turnover
        row["trade_day_frac"] = trade_frac

        # gross/cost/net if possible
        if can_decompose:
            row["gross_mean"] = float(g["gross_pnl"].mean())
            row["gross_vol"] = float(g["gross_pnl"].std(ddof=0))
            row["gross_sharpe"] = sharpe_like(g["gross_pnl"])

            row["base_cost_mean"] = float(g["base_cost"].mean())
            row["base_cost_sum"] = float(g["base_cost"].sum())
            gross_sum = float(g["gross_pnl"].sum())
            row["cost_share_of_gross"] = (row["base_cost_sum"] / gross_sum) if abs(gross_sum) > 1e-12 else np.nan

            for m in COST_MULTS:
                col = f"net_pnl_x{m}"
                if col in g.columns:
                    row[f"net_mean_x{m}"] = float(g[col].mean())
                    row[f"net_vol_x{m}"] = float(g[col].std(ddof=0))
                    row[f"net_sharpe_x{m}"] = sharpe_like(g[col])
        else:
            # only net x1 is meaningful
            if "net_pnl_x1" in g.columns:
                row["net_mean_x1"] = float(g["net_pnl_x1"].mean())
                row["net_vol_x1"] = float(g["net_pnl_x1"].std(ddof=0))
                row["net_sharpe_x1"] = sharpe_like(g["net_pnl_x1"])

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        ["net_sharpe_x1"] if "net_sharpe_x1" in summary.columns else ["n_days"],
        ascending=False,
        na_position="last",
    )

    summary.to_csv(OUT_SUMMARY, index=False)

    # 7) Print compact report
    print("\n=== Turnover + cost attribution report ===")
    print(f"Wrote daily:   {OUT_DAILY}")
    print(f"Wrote summary: {OUT_SUMMARY}")
    print(f"Strategies: {len(summary):,}")

    show_cols = [c for c in [
        factor_col, window_col, "n_days", "turnover_per_day", "trade_day_frac",
        "gross_sharpe", "net_sharpe_x1", "net_sharpe_x5", "net_sharpe_x10",
        "cost_share_of_gross"
    ] if c is not None and c in summary.columns]

    if show_cols:
        print("\nTop 12 (by net_sharpe_x1 if available):")
        print(summary[show_cols].head(12).to_string(index=False))

    if not can_decompose:
        print(
            "\nNOTE: Could not fully decompose gross vs cost vs net.\n"
            "To enable full cost attribution, your daily backtest parquet should include BOTH:\n"
            "- a gross pnl column (e.g. gross_pnl)\n"
            "- a cost column (e.g. cost)\n"
            "Then this script will automatically produce net PnL for cost x1/x5/x10.\n"
        )


if __name__ == "__main__":
    main()
