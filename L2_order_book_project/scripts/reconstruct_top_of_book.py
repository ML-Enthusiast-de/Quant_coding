#!/usr/bin/env python
"""
FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.

Reconstruct Top-of-Book time series from ALL Coinbase Advanced Trade L2 WS runs.

- Scans: L2_order_book_project/data/raw/coinbase_l2_<product>_YYYYMMDD_HHMMSS/
- For each run folder:
    - loads all part_*.parquet
    - reconstructs Top-of-Book via L2Book
    - resamples to CFG.sample
    - writes per-run output to: L2_order_book_project/data/processed/runs/

Optional:
- Also writes ONE combined file over all runs (streaming write if pyarrow is installed).

FIX INCLUDED:
- Enforces a stable schema across streamed chunks (segment_id stays int64)
- Strips Arrow schema metadata to avoid pandas-metadata mismatch across chunks
"""

from __future__ import annotations

import json
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

# Optional streaming ParquetWriter for combined output
try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False


# -----------------------------------------------------------------------------
# Make imports work reliably (repo root on sys.path)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Quant_coding
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from L2_order_book_project.src.l2_book import L2Book  # noqa: E402


# -----------------------------------------------------------------------------
# CONFIG (edit these, no CLI needed)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    product: str = "BTC-USD"
    sample: str = "1s"               # "250ms", "1s", "5s", ...
    gap_seconds: int = 10            # gap > this => new segment, require new snapshot
    prefer_exchange_ts: bool = True  # True: exchange_ts first, else receive_ts
    verbose: bool = True

    process_all_runs: bool = True    # True: iterate ALL run dirs for this product
    write_combined: bool = True      # True: also create one combined top-of-book file


CFG = Config()


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = REPO_ROOT / "L2_order_book_project"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROCESSED_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
TOB_COLS = [
    "segment_id",
    "best_bid",
    "best_ask",
    "bid_sz",
    "ask_sz",
    "spread",
    "mid",
    "microprice",
    "imbalance_1",
]


def _product_slug(product: str) -> str:
    return product.replace("-", "").lower()


def _safe_to_ts_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _choose_row_ts(df: pd.DataFrame, prefer_exchange_ts: bool) -> pd.Series:
    ex = df["exchange_ts"] if "exchange_ts" in df.columns else pd.Series([None] * len(df))
    rx = df["receive_ts"] if "receive_ts" in df.columns else pd.Series([None] * len(df))
    ex_ts = _safe_to_ts_series(ex)
    rx_ts = _safe_to_ts_series(rx)
    return ex_ts.fillna(rx_ts) if prefer_exchange_ts else rx_ts.fillna(ex_ts)



def _tob_same(prev: tuple[float, float, float, float], curr: tuple[float, float, float, float]) -> bool:
    # price usually matches exactly; sizes may wobble, so allow tiny tolerance
    return (
        math.isclose(prev[0], curr[0], rel_tol=0.0, abs_tol=1e-12) and  # best_bid
        math.isclose(prev[1], curr[1], rel_tol=0.0, abs_tol=1e-12) and  # best_ask
        math.isclose(prev[2], curr[2], rel_tol=0.0, abs_tol=1e-12) and  # bid_sz
        math.isclose(prev[3], curr[3], rel_tol=0.0, abs_tol=1e-12)      # ask_sz
    )


def list_run_dirs(product: str) -> list[Path]:
    """
    Return ALL run folders like:
      coinbase_l2_btcusd_YYYYMMDD_HHMMSS
    sorted by name timestamp (which sorts chronologically).
    """
    slug = _product_slug(product)
    dirs = [p for p in RAW_DIR.glob(f"coinbase_l2_{slug}_*") if p.is_dir()]
    dirs.sort(key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"No run folders found for product={product} under {RAW_DIR}")
    return dirs


def list_part_files(run_dir: Path) -> list[Path]:
    parts = sorted(run_dir.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No part_*.parquet found in {run_dir}")
    return parts


def iter_l2_updates(raw_msg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """
    Expected Coinbase Advanced Trade schema (commonly):
      {"channel":"l2_data","timestamp":...,"sequence_num":...,
       "events":[{"type":"snapshot"|"update","product_id":...,"updates":[...]}]}
    """
    if raw_msg.get("channel") != "l2_data":
        return

    msg_ts = raw_msg.get("timestamp")
    seq = raw_msg.get("sequence_num")

    for ev in raw_msg.get("events", []) or []:
        ev_type = ev.get("type")  # "snapshot" or "update"
        product = ev.get("product_id")
        for u in ev.get("updates", []) or []:
            yield {
                "msg_ts": msg_ts,
                "sequence_num": seq,
                "event_type": ev_type,
                "product_id": product,
                "side": u.get("side"),
                "event_time": u.get("event_time"),
                "price_level": u.get("price_level"),
                "new_quantity": u.get("new_quantity"),
            }


def _enforce_tob_dtypes(tob: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure stable dtypes (critical for streaming ParquetWriter):
      - segment_id must be int64 (resample/ffill can turn it into float)
      - numeric columns float64
    """
    tob2 = tob.copy()

    # segment_id: fill any NaNs (shouldn't happen, but safe) and force int64
    if "segment_id" in tob2.columns:
        tob2["segment_id"] = tob2["segment_id"].ffill().fillna(0)
        # If it became float like 1.0, 2.0, cast safely:
        tob2["segment_id"] = tob2["segment_id"].astype("int64")

    # force all other TOB columns to float64 if present
    for c in TOB_COLS:
        if c == "segment_id":
            continue
        if c in tob2.columns:
            tob2[c] = pd.to_numeric(tob2[c], errors="coerce").astype("float64")

    return tob2


def reconstruct_one_run(run_dir: Path, cfg: Config, segment_offset: int = 0) -> pd.DataFrame:
    """
    Reconstruct Top-of-Book from one run folder.
    Returns resampled DataFrame (index=ts).
    segment_offset lets us make segment_ids unique across runs when combining.
    """
    parts = list_part_files(run_dir)

    last_tob: tuple[float, float, float, float] | None = None


    book = L2Book()
    have_snapshot = False
    segment_id = int(segment_offset)
    last_ts: pd.Timestamp | None = None

    out_rows: list[dict[str, Any]] = []

    total_msgs = 0
    total_updates = 0
    total_applied = 0

    for i, part_path in enumerate(parts):
        df = pd.read_parquet(part_path)
        if "raw" not in df.columns:
            if cfg.verbose:
                print(f"[warn] {run_dir.name}/{part_path.name}: missing 'raw' -> skipped")
            continue

        ts = _choose_row_ts(df, cfg.prefer_exchange_ts)
        df = df.assign(ts=ts).dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

        if cfg.verbose:
            print(f"  [part {i+1}/{len(parts)}] {part_path.name}: rows={len(df):,}")

        for r in df.itertuples(index=False):
            total_msgs += 1
            ts_msg: pd.Timestamp = r.ts

            # gap detection: new segment => require fresh snapshot
            if last_ts is not None and (ts_msg - last_ts).total_seconds() > cfg.gap_seconds:
                segment_id += 1
                have_snapshot = False
            last_ts = ts_msg

            raw_str = getattr(r, "raw", None)
            if not raw_str:
                continue

            try:
                raw = json.loads(raw_str)
            except Exception:
                continue

            any_updates = False
            for u in iter_l2_updates(raw):
                any_updates = True
                total_updates += 1

                if u["event_type"] == "snapshot" and not have_snapshot:
                    book.clear()
                    have_snapshot = True

                if not have_snapshot:
                    continue

                side = u["side"]
                price_level = u["price_level"]
                new_qty = u["new_quantity"]
                if side is None or price_level is None or new_qty is None:
                    continue

                book.apply_update(side, price_level, new_qty)
                total_applied += 1

            if not any_updates or not have_snapshot:
                continue

            bb, ba, bb_sz, ba_sz = book.best_bid_ask()
            if bb is None or ba is None:
                continue

            spread = ba - bb
            mid = (ba + bb) / 2.0
            micro = (
                (bb * ba_sz + ba * bb_sz) / (bb_sz + ba_sz)
                if (bb_sz is not None and ba_sz is not None and (bb_sz + ba_sz) > 0)
                else mid
            )
            imb1 = (
                (bb_sz - ba_sz) / (bb_sz + ba_sz)
                if (bb_sz is not None and ba_sz is not None and (bb_sz + ba_sz) > 0)
                else 0.0
            )


                # ---- CHANGE GUARD: only write if TOB changed ----
            curr_tob = (float(bb), float(ba), float(bb_sz), float(ba_sz))

            if last_tob is not None and _tob_same(last_tob, curr_tob):
                continue  # no change -> skip writing a row

            last_tob = curr_tob

            out_rows.append(
                {
                    "ts": ts_msg,
                    "segment_id": segment_id,
                    "best_bid": curr_tob[0],
                    "best_ask": curr_tob[1],
                    "bid_sz": curr_tob[2],
                    "ask_sz": curr_tob[3],
                    "spread": spread,
                    "mid": mid,
                    "microprice": micro,
                    "imbalance_1": imb1,
                }


            )

    if not out_rows:
        raise RuntimeError(
            f"No reconstructed rows for run {run_dir.name}. "
            "Common causes: no snapshot captured or schema mismatch."
        )

    tob = pd.DataFrame(out_rows).set_index("ts").sort_index()
    tob = tob[~tob.index.duplicated(keep="last")]
    tob_s = tob.resample(cfg.sample).last().ffill()

    # IMPORTANT: enforce stable dtypes (segment_id int64)
    tob_s = _enforce_tob_dtypes(tob_s)

    if cfg.verbose:
        print(
            f"  [run done] msgs={total_msgs:,} updates={total_updates:,} applied={total_applied:,} "
            f"raw_rows={len(tob):,} resampled_rows={len(tob_s):,} segments={tob_s['segment_id'].nunique()}"
        )

    return tob_s


def write_combined_streaming(out_path: Path, dfs: Iterator[pd.DataFrame]) -> None:
    """
    Write combined parquet without holding everything in memory.
    Requires pyarrow.
    FIX:
      - enforce dtypes (segment_id int64)
      - strip schema metadata and cast every chunk to the exact same schema
    """
    if not _HAS_PYARROW:
        raise RuntimeError("pyarrow not installed but write_combined_streaming() was called.")

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None

    try:
        for df in dfs:
            df2 = df.reset_index()  # include 'ts' as a column

            # enforce stable dtypes BEFORE Arrow conversion
            if "segment_id" in df2.columns:
                df2["segment_id"] = df2["segment_id"].astype("int64")

            table = pa.Table.from_pandas(df2, preserve_index=False)
            table = table.replace_schema_metadata(None)  # drop pandas metadata

            if writer is None:
                schema = table.schema.remove_metadata()
                writer = pq.ParquetWriter(str(out_path), schema, compression="snappy")

            assert schema is not None
            table = table.cast(schema)  # enforce exact schema
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    slug = _product_slug(CFG.product)
    run_dirs = list_run_dirs(CFG.product)

    if not CFG.process_all_runs:
        run_dirs = [run_dirs[-1]]

    print("=" * 80)
    print(f"Product: {CFG.product} | sample={CFG.sample} | gap_seconds={CFG.gap_seconds}")
    print(f"Found run folders: {len(run_dirs)}")
    print("=" * 80)

    segment_offset = 0
    per_run_outputs: list[Path] = []

    combined_out = PROCESSED_DIR / f"tob_{slug}_ALL_{CFG.sample}.parquet"
    if CFG.write_combined and not _HAS_PYARROW:
        print("[note] pyarrow not installed -> will NOT write combined file (per-run outputs only).")
        print("       Install with: pip install pyarrow")

    def combined_iter():
        nonlocal segment_offset
        for rd in run_dirs:
            print(f"\n[run] {rd.name}")
            tob = reconstruct_one_run(rd, CFG, segment_offset=segment_offset)

            # next run gets a fresh offset above current max (keeps uniqueness)
            segment_offset = int(tob["segment_id"].max()) + 2

            run_tag = rd.name.replace("coinbase_l2_", "")
            out_path = RUNS_DIR / f"tob_{slug}_{run_tag}_{CFG.sample}.parquet"
            tob.to_parquet(out_path, index=True)
            per_run_outputs.append(out_path)
            print(f"  wrote: {out_path} (rows={len(tob):,})")

            yield tob

    if CFG.write_combined and _HAS_PYARROW:
        print(f"\n[combined] writing streaming parquet to: {combined_out}")
        write_combined_streaming(combined_out, combined_iter())
        print(f"[combined] done: {combined_out}")
    else:
        # just run through and write per-run
        list(combined_iter())

    print("\nDone.")
    print(f"Per-run outputs: {len(per_run_outputs)} written to {RUNS_DIR}")
    if CFG.write_combined and _HAS_PYARROW:
        print(f"Combined output: {combined_out}")


if __name__ == "__main__":
    main()
