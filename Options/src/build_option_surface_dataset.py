import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


# =========================================================
# Relative paths
# =========================================================
HERE = Path(__file__).resolve()
OPTIONS_DIR = HERE.parents[1]          # .../Quant_coding/Options
ROOT_DIR = OPTIONS_DIR.parents[0]      # .../Quant_coding

CSV_PATH = ROOT_DIR / "data" / "archive" / "spy_2020_2022.csv"
OUT_DIR = OPTIONS_DIR / "data" / "processed"

RAW_MONTH_DIR = OUT_DIR / "raw_month"          # stage 1 output
CHAIN_CHUNKS_DIR = OUT_DIR / "chain_chunks"    # stage 2 output
IV_CHUNKS_DIR = OUT_DIR / "iv_chunks"          # stage 2 output
CHECKPOINT_PATH = OUT_DIR / "checkpoint.json"

# Cleaning knobs
MAX_REL_SPREAD = 0.30
MIN_MID = 0.01
LOG_MONEYNESS_CLIP = 0.8

# Streaming knobs
CSV_CHUNKSIZE = 250_000

# Output flush knobs (processed)
FLUSH_CHAIN_ROWS = 30_000
FLUSH_IV_ROWS = 20_000

# progress inside IV loop
PROGRESS_EVERY = 50_000


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
        return 1e-6

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
# Put-call parity regression per (date, expiry)
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


# -----------------------------
# Helpers
# -----------------------------
def normalize_columns_index(cols: pd.Index) -> pd.Index:
    return (
        cols.astype(str)
        .str.strip()
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
    )


def get_raw_usecols_for_normalized(csv_path: Path, want_norm: list[str]) -> list[str]:
    raw_cols = pd.read_csv(csv_path, nrows=0).columns
    norm_cols = normalize_columns_index(raw_cols)

    norm_to_raw = {}
    for raw, norm in zip(raw_cols, norm_cols):
        if norm in norm_to_raw and norm_to_raw[norm] != raw:
            raise ValueError(f"Duplicate normalized column name '{norm}' from raw columns.")
        norm_to_raw[norm] = raw

    missing = [c for c in want_norm if c not in norm_to_raw]
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {missing}\n"
            f"Available normalized columns: {sorted(norm_to_raw.keys())}"
        )

    return [norm_to_raw[c] for c in want_norm]


def atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "raw_shard_done": False,
        "raw_parts_next": {},           # month -> next raw part index
        "processed_months_done": [],    # months fully processed
        "month_next_part": {},          # month -> next processed output part index
    }


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(CHECKPOINT_PATH)


def next_index_for(state: dict, key_dict: str, month: str, fallback_dir: Path, glob_pat: str) -> int:
    if month in state[key_dict]:
        return int(state[key_dict][month])

    existing = sorted(fallback_dir.glob(glob_pat.format(month=month)))
    if not existing:
        idx = 1
    else:
        last = existing[-1].stem
        part_str = last.split("_part")[-1]
        idx = int(part_str) + 1

    state[key_dict][month] = idx
    return idx


def _to_float_series(s: pd.Series) -> pd.Series:
    """
    Robust numeric parse for messy columns:
    - handles commas "1,234"
    - handles blanks / NA
    - returns float64
    """
    # ensure string for replace, but keep NaN
    s2 = s.astype(str)
    s2 = s2.str.replace(",", "", regex=False)
    s2 = s2.replace({"nan": np.nan, "None": np.nan, "": np.nan, "NA": np.nan, "N/A": np.nan, "null": np.nan})
    return pd.to_numeric(s2, errors="coerce").astype("float64")


def coerce_raw_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 1 raw shards:
    - store date columns as string
    - store ALL numeric columns as float64 (even volume/DTE) to avoid Int casting issues
    """
    df = df.copy()

    for c in ["QUOTE_DATE", "EXPIRE_DATE"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    num_cols = [
        "DTE", "UNDERLYING_LAST", "STRIKE",
        "C_BID", "C_ASK", "P_BID", "P_ASK",
        "C_VOLUME", "P_VOLUME",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = _to_float_series(df[c])

    return df


# -----------------------------
# Stage 2: per-day pipeline (within a month)
# -----------------------------
def build_long_chain_for_day(df_day: pd.DataFrame) -> pd.DataFrame:
    df = df_day.copy()
    df["quote_date"] = pd.to_datetime(df["QUOTE_DATE"], errors="coerce").dt.date
    df["expiry"] = pd.to_datetime(df["EXPIRE_DATE"], errors="coerce").dt.date
    df["strike"] = pd.to_numeric(df["STRIKE"], errors="coerce")
    df["S"] = pd.to_numeric(df["UNDERLYING_LAST"], errors="coerce")
    df["DTE"] = pd.to_numeric(df["DTE"], errors="coerce")
    df["T"] = df["DTE"] / 365.0

    calls = pd.DataFrame({
        "quote_date": df["quote_date"],
        "expiry": df["expiry"],
        "T": df["T"],
        "strike": df["strike"],
        "option_type": "C",
        "bid": pd.to_numeric(df["C_BID"], errors="coerce"),
        "ask": pd.to_numeric(df["C_ASK"], errors="coerce"),
        "volume": pd.to_numeric(df.get("C_VOLUME"), errors="coerce"),
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
        "volume": pd.to_numeric(df.get("P_VOLUME"), errors="coerce"),
        "S": df["S"],
    })

    chain = pd.concat([calls, puts], ignore_index=True)
    chain = chain.dropna(subset=["quote_date", "expiry", "T", "strike", "bid", "ask", "S"])
    chain = chain[chain["T"] > 0].reset_index(drop=True)
    return chain


def clean_chain(chain: pd.DataFrame) -> pd.DataFrame:
    df = chain.copy()
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])].copy()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread"] = df["ask"] - df["bid"]
    df = df[df["mid"] >= MIN_MID]
    df["rel_spread"] = df["spread"] / df["mid"]
    df = df[df["rel_spread"] <= MAX_REL_SPREAD].reset_index(drop=True)
    return df


def add_forward_df(chain: pd.DataFrame) -> pd.DataFrame:
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

        S0 = float(g["S"].dropna().iloc[0]) if g["S"].notna().any() else np.nan
        if not np.isfinite(F):
            F = S0
        if not np.isfinite(DF_est):
            DF_est = 1.0

        fdf_rows.append({"quote_date": qd, "expiry": ex, "F": F, "DF": DF_est, "parity_pairs": n})

    fdf = pd.DataFrame(fdf_rows)
    return chain.merge(fdf, on=["quote_date", "expiry"], how="left")


def compute_iv_points(chain: pd.DataFrame, date_str: str) -> pd.DataFrame:
    iv_df = chain.dropna(subset=["F", "DF"]).copy()
    iv_df["log_moneyness"] = np.log(iv_df["strike"] / iv_df["F"])
    iv_df = iv_df[np.isfinite(iv_df["log_moneyness"])]
    iv_df = iv_df[iv_df["log_moneyness"].between(-LOG_MONEYNESS_CLIP, LOG_MONEYNESS_CLIP)].reset_index(drop=True)

    iv_list = np.empty(len(iv_df), dtype=float)
    delta_list = np.empty(len(iv_df), dtype=float)
    iv_list[:] = np.nan
    delta_list[:] = np.nan

    for i, r in enumerate(iv_df.itertuples(index=False)):
        if (i + 1) % PROGRESS_EVERY == 0:
            print(f"    {date_str}: solved {i+1:,}/{len(iv_df):,} IVs...")

        is_call = (r.option_type == "C")
        iv = implied_vol(float(r.mid), float(r.F), float(r.strike), float(r.T), float(r.DF), is_call)
        iv_list[i] = iv
        if np.isfinite(iv):
            delta_list[i] = forward_delta(float(r.F), float(r.strike), float(iv), float(r.T), is_call)

    iv_df["iv"] = iv_list
    iv_df["delta_fwd"] = delta_list
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
    return iv_df[keep].copy()


def flush_processed_month(month_key: str, chain_buf_list: list[pd.DataFrame], iv_buf_list: list[pd.DataFrame], state: dict) -> None:
    if not chain_buf_list and not iv_buf_list:
        return

    buf_chain = pd.concat(chain_buf_list, ignore_index=True) if chain_buf_list else pd.DataFrame()
    buf_iv = pd.concat(iv_buf_list, ignore_index=True) if iv_buf_list else pd.DataFrame()

    part = next_index_for(
        state=state,
        key_dict="month_next_part",
        month=month_key,
        fallback_dir=IV_CHUNKS_DIR,
        glob_pat="iv_{month}_part*.parquet",
    )

    chain_path = CHAIN_CHUNKS_DIR / f"chain_{month_key}_part{part:04d}.parquet"
    iv_path = IV_CHUNKS_DIR / f"iv_{month_key}_part{part:04d}.parquet"

    if len(buf_chain) > 0:
        atomic_to_parquet(buf_chain, chain_path)
    if len(buf_iv) > 0:
        atomic_to_parquet(buf_iv, iv_path)

    state["month_next_part"][month_key] = part + 1
    save_checkpoint(state)
    print(f"  Flushed processed month={month_key} part={part:04d}: chain_rows={len(buf_chain):,}, iv_rows={len(buf_iv):,}")


# =========================================================
# Stage 1: shard raw CSV -> monthly raw parquet parts (order independent, safe dtypes)
# =========================================================
def stage1_shard_raw_csv(state: dict) -> None:
    if state.get("raw_shard_done", False):
        print("Stage 1: raw sharding already done. Skipping.")
        return

    RAW_MONTH_DIR.mkdir(parents=True, exist_ok=True)

    want_norm = [
        "QUOTE_DATE", "EXPIRE_DATE", "DTE", "UNDERLYING_LAST", "STRIKE",
        "C_BID", "C_ASK", "P_BID", "P_ASK",
        "C_VOLUME", "P_VOLUME",
    ]
    raw_usecols = get_raw_usecols_for_normalized(CSV_PATH, want_norm)

    reader = pd.read_csv(
        CSV_PATH,
        usecols=raw_usecols,
        chunksize=CSV_CHUNKSIZE,
        low_memory=False,
    )

    for chunk_idx, chunk in enumerate(reader, start=1):
        chunk.columns = normalize_columns_index(chunk.columns)

        qd = pd.to_datetime(chunk["QUOTE_DATE"], errors="coerce")
        chunk = chunk.assign(_qd=qd).dropna(subset=["_qd"])
        chunk = chunk.assign(_month=chunk["_qd"].astype(str).str.slice(0, 7))

        for month, g in chunk.groupby("_month", sort=False):
            g2 = g.drop(columns=["_qd", "_month"])
            g2 = coerce_raw_dtypes(g2)

            part = next_index_for(
                state=state,
                key_dict="raw_parts_next",
                month=month,
                fallback_dir=RAW_MONTH_DIR,
                glob_pat="spy_raw_{month}_part*.parquet",
            )
            out_path = RAW_MONTH_DIR / f"spy_raw_{month}_part{part:04d}.parquet"
            atomic_to_parquet(g2, out_path)

            state["raw_parts_next"][month] = part + 1

        if chunk_idx % 5 == 0:
            save_checkpoint(state)
            print(f"Stage1: wrote raw parts through CSV chunk {chunk_idx}")

    state["raw_shard_done"] = True
    save_checkpoint(state)
    print("Stage 1 done: raw sharding complete.")


# =========================================================
# Stage 2: process monthly raw shards -> chain/iv chunks
# =========================================================
def stage2_process_months(state: dict) -> None:
    CHAIN_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    IV_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_MONTH_DIR.exists():
        raise FileNotFoundError(f"Raw month directory not found: {RAW_MONTH_DIR}")

    done = set(state.get("processed_months_done", []))

    raw_parts = sorted(RAW_MONTH_DIR.glob("spy_raw_????-??_part*.parquet"))
    months = sorted({p.name.split("_")[2] for p in raw_parts})

    if not months:
        raise FileNotFoundError(f"No raw shard parts found in {RAW_MONTH_DIR}")

    print(f"Stage2: found {len(months)} months to process.")

    for month in months:
        if month in done:
            print(f"Stage2: month {month} already done. Skipping.")
            continue

        print(f"\nStage2: processing month {month} ...")
        parts = sorted(RAW_MONTH_DIR.glob(f"spy_raw_{month}_part*.parquet"))
        df_month = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
        df_month.columns = normalize_columns_index(df_month.columns)

        qd = pd.to_datetime(df_month["QUOTE_DATE"], errors="coerce").dt.date
        df_month = df_month.assign(_quote_date=qd).dropna(subset=["_quote_date"])

        chain_buf_list: list[pd.DataFrame] = []
        iv_buf_list: list[pd.DataFrame] = []
        chain_rows = 0
        iv_rows = 0

        for day, df_day in df_month.groupby("_quote_date", sort=True):
            date_str = day.isoformat()

            chain = build_long_chain_for_day(df_day.drop(columns=["_quote_date"]))
            chain = clean_chain(chain)
            chain = add_forward_df(chain)
            iv_out = compute_iv_points(chain, date_str)

            chain_buf_list.append(chain)
            iv_buf_list.append(iv_out)
            chain_rows += len(chain)
            iv_rows += len(iv_out)

            if chain_rows >= FLUSH_CHAIN_ROWS or iv_rows >= FLUSH_IV_ROWS:
                flush_processed_month(month, chain_buf_list, iv_buf_list, state)
                chain_buf_list = []
                iv_buf_list = []
                chain_rows = 0
                iv_rows = 0

        flush_processed_month(month, chain_buf_list, iv_buf_list, state)

        state.setdefault("processed_months_done", [])
        state["processed_months_done"].append(month)
        save_checkpoint(state)
        done.add(month)

        print(f"Stage2: month {month} done.")

    print("\nStage 2 done: all months processed.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at:\n{CSV_PATH}")

    state = load_checkpoint()
    print("Checkpoint loaded from:", CHECKPOINT_PATH)

    print("\n=== Stage 1: Shard raw CSV by month (order independent) ===")
    stage1_shard_raw_csv(state)

    print("\n=== Stage 2: Build clean chain + IV points from monthly shards ===")
    stage2_process_months(state)

    print("\nDone.")
    print("Outputs:")
    print(f"- Raw shards: {RAW_MONTH_DIR}")
    print(f"- Chain chunks: {CHAIN_CHUNKS_DIR}")
    print(f"- IV chunks: {IV_CHUNKS_DIR}")
    print(f"- Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
