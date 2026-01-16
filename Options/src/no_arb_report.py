from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# Relative paths
# =========================================================
HERE = Path(__file__).resolve()
OPTIONS_DIR = HERE.parents[1]          # .../Quant_coding/Options
DATA_DIR = OPTIONS_DIR / "data" / "processed"

CHAIN_PATH = DATA_DIR / "spy_chain_clean.parquet"
OUT_PATH = DATA_DIR / "no_arb_summary.parquet"

# Tolerances (prices are noisy; EOD snapshots too)
# Monotonicity: allow tiny increases due to rounding/noise
MONO_TOL_ABS = 1e-4

# Convexity: use an absolute tol plus spread-based tol for local noise
CONV_TOL_ABS = 1e-4
CONV_TOL_SPREAD_MULT = 0.5  # allow 0.5 * (sum of 3 local spreads)

MIN_STRIKES = 8  # need enough points for meaningful checks


def load_calls(chain_path: Path) -> pd.DataFrame:
    """
    Load cleaned chain and keep only call rows + minimal columns needed for no-arb checks.
    """
    df = pd.read_parquet(chain_path)

    need = ["quote_date", "expiry", "strike", "option_type", "mid", "bid", "ask", "T", "S", "F", "DF"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {chain_path.name}: {missing}\nFound: {list(df.columns)}")

    calls = df[df["option_type"] == "C"].copy()
    calls = calls.dropna(subset=["quote_date", "expiry", "strike", "mid"])

    calls["strike"] = pd.to_numeric(calls["strike"], errors="coerce")
    calls["mid"] = pd.to_numeric(calls["mid"], errors="coerce")
    calls["bid"] = pd.to_numeric(calls["bid"], errors="coerce")
    calls["ask"] = pd.to_numeric(calls["ask"], errors="coerce")

    calls = calls.dropna(subset=["strike", "mid"])
    calls = calls[calls["strike"] > 0]

    # spread diagnostics
    calls["spread"] = calls["ask"] - calls["bid"]
    calls.loc[~np.isfinite(calls["spread"]), "spread"] = np.nan

    return calls


def monotonicity_violations(strikes: np.ndarray, prices: np.ndarray, tol_abs: float) -> tuple[int, int]:
    """
    Adjacent monotonicity check:
      if K increases, call should not increase:
        C_i >= C_{i+1} - tol
    Returns: (violations, checks)
    """
    n = len(strikes)
    if n < 2:
        return 0, 0

    diffs = prices[:-1] - prices[1:]  # should be >= 0
    viol = int(np.sum(diffs < -tol_abs))
    return viol, int(len(diffs))


def convexity_violations_uneven(
    strikes: np.ndarray,
    prices: np.ndarray,
    spreads: np.ndarray | None,
    tol_abs: float,
    tol_spread_mult: float,
) -> tuple[int, int]:
    """
    Convexity / butterfly no-arbitrage check for an UNEVEN strike grid.

    For K1<K2<K3, convexity implies:
      C2 <= w1*C1 + w3*C3
    where:
      w1 = (K3-K2)/(K3-K1)
      w3 = (K2-K1)/(K3-K1)

    We mark a violation if:
      C2 - (w1*C1 + w3*C3) > tol
    with tol = tol_abs + tol_spread_mult * (spread1+spread2+spread3) (if spreads are available).
    """
    n = len(strikes)
    if n < 3:
        return 0, 0

    viol = 0
    checks = 0

    for i in range(1, n - 1):
        K1, K2, K3 = strikes[i - 1], strikes[i], strikes[i + 1]
        C1, C2, C3 = prices[i - 1], prices[i], prices[i + 1]

        if not (np.isfinite(K1) and np.isfinite(K2) and np.isfinite(K3)):
            continue
        if not (np.isfinite(C1) and np.isfinite(C2) and np.isfinite(C3)):
            continue
        if K3 <= K1:
            continue

        w1 = (K3 - K2) / (K3 - K1)
        w3 = (K2 - K1) / (K3 - K1)

        rhs = w1 * C1 + w3 * C3
        lhs = C2

        tol = tol_abs
        if spreads is not None and len(spreads) == n:
            s_local = spreads[i - 1] + spreads[i] + spreads[i + 1]
            if np.isfinite(s_local):
                tol = tol + tol_spread_mult * s_local

        if (lhs - rhs) > tol:
            viol += 1

        checks += 1

    return int(viol), int(checks)


def analyze_one_chain(g: pd.DataFrame) -> dict:
    """
    Analyze one (quote_date, expiry) call chain and return metrics dict.
    """
    g = g.sort_values("strike")

    K_raw = g["strike"].to_numpy(dtype=float)
    C_raw = g["mid"].to_numpy(dtype=float)
    sp_raw = g["spread"].to_numpy(dtype=float) if "spread" in g.columns else None

    # Deduplicate strikes (keep first occurrence)
    _, idx = np.unique(K_raw, return_index=True)
    K = K_raw[idx]
    C = C_raw[idx]
    sp = sp_raw[idx] if sp_raw is not None else None

    out = {
        "n_strikes": int(len(K)),
        "mono_viol": 0,
        "mono_checks": 0,
        "conv_viol": 0,
        "conv_checks": 0,
        "mono_viol_frac": np.nan,
        "conv_viol_frac": np.nan,
        "spread_mean": float(np.nanmean(g["spread"])) if "spread" in g.columns else np.nan,
        "spread_median": float(np.nanmedian(g["spread"])) if "spread" in g.columns else np.nan,
        "T_median": float(np.nanmedian(g["T"])) if "T" in g.columns else np.nan,
        "S_last": float(g["S"].dropna().iloc[0]) if g["S"].notna().any() else np.nan,
        "F_last": float(g["F"].dropna().iloc[0]) if g["F"].notna().any() else np.nan,
        "DF_last": float(g["DF"].dropna().iloc[0]) if g["DF"].notna().any() else np.nan,
    }

    if len(K) < MIN_STRIKES:
        return out

    mono_v, mono_n = monotonicity_violations(K, C, MONO_TOL_ABS)
    conv_v, conv_n = convexity_violations_uneven(
        K, C, sp, tol_abs=CONV_TOL_ABS, tol_spread_mult=CONV_TOL_SPREAD_MULT
    )

    out["mono_viol"] = mono_v
    out["mono_checks"] = mono_n
    out["conv_viol"] = conv_v
    out["conv_checks"] = conv_n
    out["mono_viol_frac"] = (mono_v / mono_n) if mono_n > 0 else np.nan
    out["conv_viol_frac"] = (conv_v / conv_n) if conv_n > 0 else np.nan

    return out


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not CHAIN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {CHAIN_PATH}")

    calls = load_calls(CHAIN_PATH)
    grouped = calls.groupby(["quote_date", "expiry"], sort=True)

    total = len(grouped)
    print(f"Loaded calls: {len(calls):,} rows")
    print(f"Chains to check: {total:,} (quote_date x expiry)")

    rows = []
    for i, ((qd, ex), g) in enumerate(grouped, start=1):
        r = analyze_one_chain(g)
        r["quote_date"] = qd
        r["expiry"] = ex
        rows.append(r)

        if i % 500 == 0:
            print(f"  checked {i:,}/{total:,} chains...")

    out = pd.DataFrame(rows)
    out["dte_median"] = out["T_median"] * 365.0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    valid = out[out["n_strikes"] >= MIN_STRIKES].copy()

    print("\n=== No-arbitrage report (calls) ===")
    print(f"Output: {OUT_PATH}")
    print(f"Chains with >= {MIN_STRIKES} strikes: {len(valid):,}/{len(out):,}")

    if len(valid) > 0:
        mono_any = (valid["mono_viol"] > 0).mean()
        conv_any = (valid["conv_viol"] > 0).mean()

        print(f"% chains with ANY monotonicity violation: {100*mono_any:.2f}%")
        print(f"% chains with ANY convexity violation:     {100*conv_any:.2f}%")

        print(f"Median monotonicity violation frac: {100*np.nanmedian(valid['mono_viol_frac']):.3f}%")
        print(f"Median convexity violation frac:     {100*np.nanmedian(valid['conv_viol_frac']):.3f}%")

        top = valid.sort_values(["conv_viol_frac", "mono_viol_frac"], ascending=False).head(10)
        show_cols = [
            "quote_date", "expiry", "n_strikes", "dte_median",
            "mono_viol_frac", "conv_viol_frac", "spread_median"
        ]
        print("\nTop 10 worst chains (by convexity viol frac):")
        print(top[show_cols].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
