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

CHAIN_CHUNKS_DIR = OUT_DIR / "chain_chunks"
IV_CHUNKS_DIR = OUT_DIR / "iv_chunks"
CHECKPOINT_PATH = OUT_DIR / "checkpoint.json"

# Cleaning knobs
MAX_REL_SPREAD = 0.30
MIN_MID = 0.01
LOG_MONEYNESS_CLIP = 0.8

# Streaming knobs
CHUNKSIZE = 250_000  # raw CSV rows per pandas chunk
# Flush buffers when they reach these sizes (rows in long-form)
FLUSH_CHAIN_ROWS = 300_000
FLUSH_IV_ROWS = 200_000

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
# Helpers: columns / checkpoint / IO
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
        raise ValueError(f"Missing required columns after normalization: {missing}\n"
                         f"Available normalized columns: {sorted(norm_to_raw.keys())}")

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
        "last_saved_date": None,   # ISO date string
        "month_next_part": {},     # e.g., {"2020-01": 3}
    }


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(CHECKPOINT_PATH)


def month_key_from_date(d: pd.Timestamp | str) -> str:
    # expects date-like; output "YYYY-MM"
    s = str(d)
    return s[:7]


def next_part_index(month_key: str, state: dict) -> int:
    # Prefer checkpoint; otherwise infer from existing files
    if month_key in state["month_next_part"]:
        return int(state["month_next_part"][month_key])

    existing = sorted(IV_CHUNKS_DIR.glob(f"iv_{month_key}_part*.parquet"))
    if not existing:
        idx = 1
    else:
        # iv_YYYY-MM_part0007.parquet
        last = existing[-1].stem
        part_str = last.split("_part")[-1]
        idx = int(part_str) + 1

    state["month_next_part"][month_key] = idx
    return idx


# -----------------------------
# Core pipeline per day (keeps parity groups intact)
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


# -----------------------------
# Chunked buffering + flushing (MONTH parts)
# -----------------------------
def flush_month_buffers(month_key: str, buf_chain: pd.DataFrame, buf_iv: pd.DataFrame, state: dict) -> None:
    """
    Writes one chain chunk + one iv chunk for a given month, increments part index.
    """
    if len(buf_chain) == 0 and len(buf_iv) == 0:
        return

    part = next_part_index(month_key, state)
    chain_path = CHAIN_CHUNKS_DIR / f"chain_{month_key}_part{part:04d}.parquet"
    iv_path = IV_CHUNKS_DIR / f"iv_{month_key}_part{part:04d}.parquet"

    if len(buf_chain) > 0:
        atomic_to_parquet(buf_chain, chain_path)
    if len(buf_iv) > 0:
        atomic_to_parquet(buf_iv, iv_path)

    state["month_next_part"][month_key] = part + 1
    print(f"  Flushed month={month_key} part={part:04d}: chain_rows={len(buf_chain):,}, iv_rows={len(buf_iv):,}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHAIN_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    IV_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at:\n{CSV_PATH}")

    state = load_checkpoint()
    last_saved_date = state.get("last_saved_date", None)
    print("Checkpoint last_saved_date:", last_saved_date)

    # We read only needed columns; header normalization is handled
    want_norm = [
        "QUOTE_DATE", "EXPIRE_DATE", "DTE", "UNDERLYING_LAST", "STRIKE",
        "C_BID", "C_ASK", "P_BID", "P_ASK",
        "C_VOLUME", "P_VOLUME",
    ]
    raw_usecols = get_raw_usecols_for_normalized(CSV_PATH, want_norm)

    carry = None  # carry last incomplete day between CSV chunks

    # Buffers keyed by month (we keep only a few months at a time because CSV is usually sorted by date)
    month_chain_buf = {}
    month_iv_buf = {}
    month_chain_rows = {}
    month_iv_rows = {}

    processed_days = 0
    skipped_days = 0

    reader = pd.read_csv(
        CSV_PATH,
        usecols=raw_usecols,
        chunksize=CHUNKSIZE,
        low_memory=False,
    )

    for chunk_idx, chunk in enumerate(reader, start=1):
        # normalize headers
        chunk.columns = normalize_columns_index(chunk.columns)

        # optional fast skip: if we already saved up to a date, skip those rows early
        qd = pd.to_datetime(chunk["QUOTE_DATE"], errors="coerce").dt.date
        chunk = chunk.assign(_quote_date=qd).dropna(subset=["_quote_date"])

        if last_saved_date is not None:
            # skip anything <= last_saved_date
            chunk = chunk[chunk["_quote_date"] > pd.to_datetime(last_saved_date).date()]

        if len(chunk) == 0:
            print(f"Chunk {chunk_idx}: all rows skipped by checkpoint/date filter.")
            continue

        # attach carry
        if carry is not None and len(carry) > 0:
            carry.columns = normalize_columns_index(carry.columns)
            chunk = pd.concat([carry, chunk.drop(columns=["_quote_date"])], ignore_index=True)
        else:
            chunk = chunk.drop(columns=["_quote_date"])

        # determine day boundaries (assumes CSV is ordered by QUOTE_DATE)
        qd2 = pd.to_datetime(chunk["QUOTE_DATE"], errors="coerce").dt.date
        chunk = chunk.assign(_quote_date=qd2).dropna(subset=["_quote_date"])

        last_day = chunk["_quote_date"].iloc[-1]
        carry = chunk[chunk["_quote_date"] == last_day].drop(columns=["_quote_date"])
        complete = chunk[chunk["_quote_date"] != last_day].drop(columns=["_quote_date"])

        # process complete days in this chunk
        if len(complete) > 0:
            for day, df_day in complete.groupby(pd.to_datetime(complete["QUOTE_DATE"], errors="coerce").dt.date, sort=False):
                date_str = day.isoformat()
                print(f"Processing day {date_str} ...")

                # day pipeline
                chain = build_long_chain_for_day(df_day)
                chain = clean_chain(chain)
                chain = add_forward_df(chain)
                iv_out = compute_iv_points(chain, date_str)

                mkey = month_key_from_date(date_str)

                # init month buffers
                if mkey not in month_chain_buf:
                    month_chain_buf[mkey] = []
                    month_iv_buf[mkey] = []
                    month_chain_rows[mkey] = 0
                    month_iv_rows[mkey] = 0

                month_chain_buf[mkey].append(chain)
                month_iv_buf[mkey].append(iv_out)
                month_chain_rows[mkey] += len(chain)
                month_iv_rows[mkey] += len(iv_out)

                processed_days += 1

                # flush if large enough
                if month_chain_rows[mkey] >= FLUSH_CHAIN_ROWS or month_iv_rows[mkey] >= FLUSH_IV_ROWS:
                    buf_chain = pd.concat(month_chain_buf[mkey], ignore_index=True) if month_chain_rows[mkey] > 0 else pd.DataFrame()
                    buf_iv = pd.concat(month_iv_buf[mkey], ignore_index=True) if month_iv_rows[mkey] > 0 else pd.DataFrame()

                    flush_month_buffers(mkey, buf_chain, buf_iv, state)

                    # reset buffers for that month
                    month_chain_buf[mkey] = []
                    month_iv_buf[mkey] = []
                    month_chain_rows[mkey] = 0
                    month_iv_rows[mkey] = 0

                    # only update checkpoint once we actually wrote something
                    state["last_saved_date"] = date_str
                    last_saved_date = date_str
                    save_checkpoint(state)

            print(f"After chunk {chunk_idx}: processed_days={processed_days}, skipped_days={skipped_days}")

        else:
            print(f"Chunk {chunk_idx}: no complete day yet (carry={last_day.isoformat()}).")

    # process final carry
    if carry is not None and len(carry) > 0:
        day = pd.to_datetime(carry["QUOTE_DATE"], errors="coerce").dt.date.iloc[0]
        date_str = day.isoformat()
        if last_saved_date is None or day > pd.to_datetime(last_saved_date).date():
            print(f"Processing final carry day {date_str} ...")

            chain = build_long_chain_for_day(carry)
            chain = clean_chain(chain)
            chain = add_forward_df(chain)
            iv_out = compute_iv_points(chain, date_str)

            mkey = month_key_from_date(date_str)
            if mkey not in month_chain_buf:
                month_chain_buf[mkey] = []
                month_iv_buf[mkey] = []
                month_chain_rows[mkey] = 0
                month_iv_rows[mkey] = 0

            month_chain_buf[mkey].append(chain)
            month_iv_buf[mkey].append(iv_out)
            month_chain_rows[mkey] += len(chain)
            month_iv_rows[mkey] += len(iv_out)

            processed_days += 1

    # flush all remaining month buffers at end
    for mkey in sorted(month_chain_buf.keys()):
        buf_chain = pd.concat(month_chain_buf[mkey], ignore_index=True) if month_chain_rows[mkey] > 0 else pd.DataFrame()
        buf_iv = pd.concat(month_iv_buf[mkey], ignore_index=True) if month_iv_rows[mkey] > 0 else pd.DataFrame()

        if len(buf_chain) == 0 and len(buf_iv) == 0:
            continue

        flush_month_buffers(mkey, buf_chain, buf_iv, state)

    # update checkpoint with the latest date we processed (best-effort)
    # (we only know reliably dates that got flushed; that’s okay)
    save_checkpoint(state)

    print("\nDone.")
    print("Chunk outputs:")
    print(f"- {CHAIN_CHUNKS_DIR}")
    print(f"- {IV_CHUNKS_DIR}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
