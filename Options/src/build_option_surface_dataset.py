import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


# =========================================================
# Relative paths (based on this file's location)
# =========================================================
HERE = Path(__file__).resolve()
OPTIONS_DIR = HERE.parents[1]          # .../Quant_coding/Options
ROOT_DIR = OPTIONS_DIR.parents[0]      # .../Quant_coding

CSV_PATH = ROOT_DIR / "data" / "archive" / "spy_2020_2022.csv"
OUT_DIR = OPTIONS_DIR / "data" / "processed"

# Cleaning knobs
MAX_REL_SPREAD = 0.30     # drop if (ask-bid)/mid > this
MIN_MID = 0.01            # drop tiny mids
LOG_MONEYNESS_CLIP = 0.8  # keep if |log(K/F)| <= this


# -----------------------------
# Black–Scholes on forward (F, DF)
# -----------------------------
def _d1_d2(F, K, sigma, T):
    if sigma <= 0 or T <= 0 or F <= 0 or K <= 0:
        return np.nan, np.nan
    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    return d1, d2


def bs_forward_price(F, K, sigma, T, DF, is_call):
    d1, d2 = _d1_d2(F, K, sigma, T)
    if not np.isfinite(d1) or not np.isfinite(d2):
        return np.nan
    if is_call:
        return DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def no_arb_bounds(F, K, DF, is_call):
    if is_call:
        return DF * max(F - K, 0.0), DF * F
    return DF * max(K - F, 0.0), DF * K


def implied_vol(price, F, K, T, DF, is_call):
    if price <= 0 or F <= 0 or K <= 0 or T <= 0 or DF <= 0:
        return np.nan

    lb, ub = no_arb_bounds(F, K, DF, is_call)
    if price < lb - 1e-10 or price > ub + 1e-10:
        return np.nan

    if abs(price - lb) < 1e-10:
        return 1e-6  # essentially intrinsic

    def f(sig):
        return bs_forward_price(F, K, sig, T, DF, is_call) - price

    try:
        return brentq(f, 1e-6, 5.0, maxiter=200)
    except Exception:
        return np.nan


def forward_delta(F, K, sigma, T, is_call):
    d1, _ = _d1_d2(F, K, sigma, T)
    if not np.isfinite(d1):
        return np.nan
    return norm.cdf(d1) if is_call else (norm.cdf(d1) - 1.0)


# -----------------------------
# Put-call parity regression per (date, expiry):
# C - P = a + b*K  => DF = -b, F = a/DF
# -----------------------------
def estimate_forward_df(strikes, call_mids, put_mids):
    m = (
        np.isfinite(strikes)
        & np.isfinite(call_mids)
        & np.isfinite(put_mids)
        & (strikes > 0)
        & (call_mids > 0)
        & (put_mids > 0)
    )
    K = strikes[m].astype(float)
    y = (call_mids[m] - put_mids[m]).astype(float)

    if len(K) < 8:
        return np.nan, np.nan, int(len(K))

    X = np.column_stack([np.ones_like(K), K])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    DF = -b
    if not np.isfinite(DF) or DF <= 0:
        return np.nan, np.nan, int(len(K))

    F = a / DF
    if not np.isfinite(F) or F <= 0:
        return np.nan, np.nan, int(len(K))

    return F, DF, int(len(K))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your CSV headers look like ' [QUOTE_DATE]' with spaces + brackets.
    This turns them into clean names like 'QUOTE_DATE'.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()              # remove leading/trailing spaces
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
    )
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at:\n{CSV_PATH}")

    # low_memory=False avoids chunked dtype guessing (and many DtypeWarnings)
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = normalize_columns(df)

    required = [
        "QUOTE_DATE", "EXPIRE_DATE", "DTE", "UNDERLYING_LAST", "STRIKE",
        "C_BID", "C_ASK", "P_BID", "P_ASK",
        "C_VOLUME", "P_VOLUME"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    # Parse dates + basics
    df["quote_date"] = pd.to_datetime(df["QUOTE_DATE"]).dt.date
    df["expiry"] = pd.to_datetime(df["EXPIRE_DATE"]).dt.date
    df["strike"] = pd.to_numeric(df["STRIKE"], errors="coerce")
    df["S"] = pd.to_numeric(df["UNDERLYING_LAST"], errors="coerce")
    df["DTE"] = pd.to_numeric(df["DTE"], errors="coerce")
    df["T"] = df["DTE"] / 365.0

    # Long-form chain (call + put rows)
    calls = pd.DataFrame({
        "quote_date": df["quote_date"],
        "expiry": df["expiry"],
        "T": df["T"],
        "strike": df["strike"],
        "option_type": "C",
        "bid": pd.to_numeric(df["C_BID"], errors="coerce"),
        "ask": pd.to_numeric(df["C_ASK"], errors="coerce"),
        "volume": pd.to_numeric(df["C_VOLUME"], errors="coerce"),
        "S": df["S"],
    })

    puts = pd.DataFrame({
        "quote_date": df["quote_date"],
        "expiry": df["expiry"],
        "T": df["T"],
        "strike": df["strike"],
        "option_type": "P",
        "bid": pd.to_numeric(df["P_BID"], errors="coerce"),
        "ask": pd.to_numeric(df["P_ASK"], errors="coerce"),
        "volume": pd.to_numeric(df["P_VOLUME"], errors="coerce"),
        "S": df["S"],
    })

    chain = pd.concat([calls, puts], ignore_index=True)
    chain = chain.dropna(subset=["quote_date", "expiry", "T", "strike", "bid", "ask", "S"])
    chain = chain[chain["T"] > 0]

    # Clean quotes
    chain = chain[(chain["bid"] > 0) & (chain["ask"] > 0) & (chain["ask"] >= chain["bid"])].copy()
    chain["mid"] = 0.5 * (chain["bid"] + chain["ask"])
    chain["spread"] = chain["ask"] - chain["bid"]
    chain = chain[chain["mid"] >= MIN_MID]
    chain["rel_spread"] = chain["spread"] / chain["mid"]
    chain = chain[chain["rel_spread"] <= MAX_REL_SPREAD].reset_index(drop=True)

    # Estimate F, DF per (quote_date, expiry) via parity
    piv = chain.pivot_table(
        index=["quote_date", "expiry", "T", "strike", "S"],
        columns="option_type",
        values="mid",
        aggfunc="first",
    ).reset_index()

    if "C" not in piv.columns:
        piv["C"] = np.nan
    if "P" not in piv.columns:
        piv["P"] = np.nan

    fdf_rows = []
    for (qd, ex), g in piv.groupby(["quote_date", "expiry"], sort=False):
        K = g["strike"].to_numpy()
        Cmid = g["C"].to_numpy()
        Pmid = g["P"].to_numpy()
        F, DF_est, n = estimate_forward_df(K, Cmid, Pmid)

        # fallback
        S0 = float(g["S"].dropna().iloc[0]) if g["S"].notna().any() else np.nan
        if not np.isfinite(F):
            F = S0
        if not np.isfinite(DF_est):
            DF_est = 1.0

        fdf_rows.append({"quote_date": qd, "expiry": ex, "F": F, "DF": DF_est, "parity_pairs": n})

    fdf = pd.DataFrame(fdf_rows)
    chain = chain.merge(fdf, on=["quote_date", "expiry"], how="left")

    # Save cleaned chain
    chain_path = OUT_DIR / "spy_chain_clean.parquet"
    chain.to_parquet(chain_path, index=False)

    # Compute IV points
    iv_df = chain.dropna(subset=["F", "DF"]).copy()
    iv_df["log_moneyness"] = np.log(iv_df["strike"] / iv_df["F"])
    iv_df = iv_df[np.isfinite(iv_df["log_moneyness"])]
    iv_df = iv_df[iv_df["log_moneyness"].between(-LOG_MONEYNESS_CLIP, LOG_MONEYNESS_CLIP)]

    iv_list = []
    delta_list = []
    for r in iv_df.itertuples(index=False):
        is_call = (r.option_type == "C")
        iv = implied_vol(float(r.mid), float(r.F), float(r.strike), float(r.T), float(r.DF), is_call)
        iv_list.append(iv)
        delta_list.append(forward_delta(float(r.F), float(r.strike), float(iv), float(r.T), is_call) if np.isfinite(iv) else np.nan)

    iv_df["iv"] = np.array(iv_list, dtype=float)
    iv_df["delta_fwd"] = np.array(delta_list, dtype=float)
    iv_df = iv_df[np.isfinite(iv_df["iv"])].reset_index(drop=True)

    iv_df["moneyness"] = iv_df["strike"] / iv_df["F"]

    keep = [
        "quote_date", "expiry", "T",
        "strike", "moneyness", "log_moneyness",
        "option_type",
        "bid", "ask", "mid", "spread", "rel_spread",
        "volume",
        "S", "F", "DF", "parity_pairs",
        "iv", "delta_fwd",
    ]
    iv_out = iv_df[keep].copy()

    iv_path = OUT_DIR / "spy_iv_points.parquet"
    iv_out.to_parquet(iv_path, index=False)

    print("Done.")
    print("Read:", CSV_PATH)
    print("Wrote:", chain_path)
    print("Wrote:", iv_path)
    print(f"Clean chain rows: {len(chain):,}")
    print(f"IV points rows:  {len(iv_out):,}")
    print("Median parity_pairs:", int(np.nanmedian(chain["parity_pairs"])))


if __name__ == "__main__":
    main()

