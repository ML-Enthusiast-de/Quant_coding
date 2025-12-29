#!/usr/bin/env python
"""
FOR RESEARCH / EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE.
NOT PRODUCTION-GRADE. MAY BE WRONG/INCOMPLETE. USE AT YOUR OWN RISK.

Collect Coinbase L2 order book events (crypto) to local files in batches.

Goals:
- Fast callback path: parse -> append to in-memory buffer
- Robust batching: flush buffer -> writer thread -> part files + manifest
- Graceful shutdown: on timer end, Ctrl+C, VSCode "Stop", kernel interrupt, SIGTERM/SIGBREAK
  -> send WS close + flush remaining buffer + finish writing parts + write manifest

Output folder:
  L2_order_book_project/data/raw/coinbase_l2_<product>_<YYYYMMDD_HHMMSS>/
    part_00000.parquet (or .csv)
    part_00001.parquet
    ...
    manifest.json

Run:
  python L2_order_book_project/scripts/collect_coinbase_l2.py --product BTC-USD --minutes 30 --flush-rows 20000
"""

from __future__ import annotations

import argparse
import atexit
import json
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Optional parquet support
try:
    import pyarrow  # noqa: F401
    _HAS_PARQUET = True
except Exception:
    _HAS_PARQUET = False

try:
    import websocket  # pip install websocket-client
except Exception as e:
    raise ImportError(
        "Missing dependency: websocket-client. Install with: pip install websocket-client"
    ) from e


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "L2_order_book_project" / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# add near the top (imports)
import json
from collections import defaultdict

def _dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

def summarize_collected_data(raw_dir: Path) -> dict:
    """
    Summarize everything we've collected so far by reading manifest.json files.
    Fast (no parquet reads).
    """
    manifests = sorted(raw_dir.glob("coinbase_l2_*/*manifest.json"))
    per_product = defaultdict(lambda: {"runs": 0, "minutes": 0, "rows": 0, "parts": 0})

    total = {"runs": 0, "minutes": 0, "rows": 0, "parts": 0}
    missing_manifest_dirs = []

    # also detect run directories without manifests (e.g. force-kill)
    for run_dir in sorted(raw_dir.glob("coinbase_l2_*")):
        if run_dir.is_dir() and not (run_dir / "manifest.json").exists():
            # only count as "missing" if it has part files
            has_parts = any(run_dir.glob("part_*.parquet")) or any(run_dir.glob("part_*.csv"))
            if has_parts:
                missing_manifest_dirs.append(run_dir)

    for mpath in manifests:
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            continue

        product = m.get("product", "UNKNOWN")
        runs = 1
        minutes = int(m.get("minutes", 0) or 0)
        rows = int(m.get("rows_written", 0) or 0)
        parts = int(m.get("parts", 0) or 0)

        per_product[product]["runs"] += runs
        per_product[product]["minutes"] += minutes
        per_product[product]["rows"] += rows
        per_product[product]["parts"] += parts

        total["runs"] += runs
        total["minutes"] += minutes
        total["rows"] += rows
        total["parts"] += parts

    total_hours = total["minutes"] / 60.0
    total_size_gb = _dir_size_bytes(raw_dir) / (1024**3)

    return {
        "total": total,
        "total_hours": total_hours,
        "total_size_gb": total_size_gb,
        "per_product": dict(per_product),
        "missing_manifest_dirs": [str(p) for p in missing_manifest_dirs],
        "manifest_count": len(manifests),
    }

def print_collection_banner(raw_dir: Path) -> None:
    s = summarize_collected_data(raw_dir)
    total = s["total"]

    print("=" * 80)
    print("COINBASE L2 DATABASE SUMMARY (local)")
    print(f"Folder: {raw_dir}")
    print(
        f"Collected so far: {total['minutes']} min ({s['total_hours']:.2f} h) | "
        f"runs={total['runs']} | parts={total['parts']} | rows={total['rows']:,} | "
        f"disk={s['total_size_gb']:.3f} GB | manifests={s['manifest_count']}"
    )

    if s["per_product"]:
        for product, d in sorted(s["per_product"].items()):
            print(
                f"  - {product}: {d['minutes']} min ({d['minutes']/60:.2f} h), "
                f"runs={d['runs']}, parts={d['parts']}, rows={d['rows']:,}"
            )

    if s["missing_manifest_dirs"]:
        print("WARNING: Found run dirs with data but missing manifest.json (likely force-kill):")
        for p in s["missing_manifest_dirs"][:5]:
            print(f"  - {p}")
        if len(s["missing_manifest_dirs"]) > 5:
            print(f"  ... and {len(s['missing_manifest_dirs']) - 5} more")

    print("=" * 80)
    print()


@dataclass
class Config:
    product: str = "BTC-USD"
    minutes: int = 120
    out_format: str = "parquet"  # parquet|csv
    ws_url: str = "wss://advanced-trade-ws.coinbase.com"
    flush_rows: int = 20_000
    flush_seconds: int = 300
    progress_every: int = 2_000
    queue_max_parts: int = 50
    shutdown_timeout_s: int = 8  # how long we wait for clean close/writes


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_subscribe_message(product: str) -> dict[str, Any]:
    # Keep this isolated: if Coinbase changes channel naming, update here.
    return {"type": "subscribe", "channel": "level2", "product_ids": [product]}


def parse_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    mtype = msg.get("type") or msg.get("event") or msg.get("channel")

    # ignore acks/heartbeats
    if mtype in {"subscriptions", "subscribed", "heartbeat"}:
        return None

    product = msg.get("product_id") or msg.get("product") or None
    exchange_ts = msg.get("time") or msg.get("timestamp") or None

    return {
        "receive_ts": utc_now_iso(),
        "exchange_ts": exchange_ts,
        "product": product,
        "msg_type": mtype,
        "raw": json.dumps(msg, ensure_ascii=False),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", type=str, default="BTC-USD")
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--format", type=str, default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--flush-rows", type=int, default=20_000)
    ap.add_argument("--flush-seconds", type=int, default=300)
    ap.add_argument("--progress-every", type=int, default=2_000)
    ap.add_argument("--shutdown-timeout", type=int, default=8)
    args = ap.parse_args()

    cfg = Config(
        product=args.product,
        minutes=args.minutes,
        out_format=args.format,
        flush_rows=args.flush_rows,
        flush_seconds=args.flush_seconds,
        progress_every=args.progress_every,
        shutdown_timeout_s=args.shutdown_timeout,
    )

    if cfg.out_format == "parquet" and not _HAS_PARQUET:
        print("pyarrow not installed -> switching output to CSV")
        cfg.out_format = "csv"

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RAW_DIR / f"coinbase_l2_{cfg.product.replace('-', '').lower()}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {cfg.ws_url} for {cfg.minutes} min, product={cfg.product}")
    print(f"Writing batches to: {run_dir}")

    print_collection_banner(RAW_DIR)

    ap = argparse.ArgumentParser()

    # ----------------------------
    # Shutdown coordination
    # ----------------------------
    stop_event = threading.Event()         # request stop (signals / timer / keyboard)
    close_requested = threading.Event()    # we already called ws.close()
    ws_closed_event = threading.Event()    # on_close fired

    ws_app: Optional[websocket.WebSocketApp] = None

    def request_close(reason: str):
        """Best-effort graceful close: send WS close + stop loop."""
        if close_requested.is_set():
            return
        close_requested.set()
        stop_event.set()
        try:
            if ws_app is not None:
                # websocket-client sets keep_running=False and closes the socket
                ws_app.close()
        except Exception as e:
            print(f"[shutdown] ws.close() failed ({reason}): {e}")

    # Signal handlers (VSCode "Stop" / Ctrl+C / TERM)
    def _handle_signal(sig, _frame):
        request_close(f"signal {sig}")

    # Register signals (availability differs by OS)
    for _sig in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
        if hasattr(signal, _sig):
            try:
                signal.signal(getattr(signal, _sig), _handle_signal)
            except Exception:
                pass

    # atexit (covers normal interpreter shutdown paths)
    atexit.register(lambda: request_close("atexit"))

    # ----------------------------
    # Writer thread (non-blocking callback)
    # ----------------------------
    from queue import Queue, Empty  # local import for clarity

    q_out: "Queue[Optional[pd.DataFrame]]" = Queue(maxsize=cfg.queue_max_parts)

    writer_done = threading.Event()
    part_idx_lock = threading.Lock()
    part_idx = 0
    rows_written = 0

    def write_manifest(extra: dict[str, Any] | None = None):
        manifest = {
            "product": cfg.product,
            "ws_url": cfg.ws_url,
            "run_id": run_id,
            "minutes": cfg.minutes,
            "out_format": cfg.out_format,
            "flush_rows": cfg.flush_rows,
            "flush_seconds": cfg.flush_seconds,
            "parts": part_idx,
            "rows_written": rows_written,
            "ended_utc": utc_now_iso(),
            "ws_closed": bool(ws_closed_event.is_set()),
        }
        if extra:
            manifest.update(extra)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def writer_loop():
        nonlocal part_idx, rows_written
        try:
            while True:
                try:
                    df_chunk = q_out.get(timeout=0.5)
                except Empty:
                    # if shutdown requested and nothing left to write -> exit
                    if stop_event.is_set() and q_out.empty():
                        break
                    continue

                # Sentinel = finish
                if df_chunk is None:
                    q_out.task_done()
                    break

                with part_idx_lock:
                    out_path = run_dir / f"part_{part_idx:05d}.{cfg.out_format}"
                    part_idx += 1

                try:
                    if cfg.out_format == "parquet":
                        df_chunk.to_parquet(out_path, index=False)
                    else:
                        df_chunk.to_csv(out_path, index=False)

                    rows_written += len(df_chunk)
                except Exception as e:
                    print(f"[writer][error] failed writing {out_path}: {e}")

                q_out.task_done()
        finally:
            # always write a manifest on writer exit
            write_manifest()
            writer_done.set()
            print(f"[writer] done. parts={part_idx} rows_written={rows_written:,}")

    t_writer = threading.Thread(target=writer_loop, daemon=True)
    t_writer.start()

    # ----------------------------
    # Collection state
    # ----------------------------
    buffer_lock = threading.Lock()
    buffer: list[dict[str, Any]] = []
    last_flush = time.time()

    msg_count = 0
    row_count_queued = 0  # rows queued to writer (not necessarily written yet)

    start = time.time()
    end = start + cfg.minutes * 60

    def flush_buffer(reason: str):
        nonlocal buffer, last_flush, row_count_queued
        with buffer_lock:
            if not buffer:
                last_flush = time.time()
                return
            df_chunk = pd.DataFrame(buffer)
            buffer = []
            last_flush = time.time()

        # queue to writer (block if necessary to avoid dropping data)
        try:
            q_out.put(df_chunk, timeout=cfg.shutdown_timeout_s)
            row_count_queued += len(df_chunk)
            print(
                f"[flush:{reason}] queued_rows={len(df_chunk):,} "
                f"total_queued={row_count_queued:,} parts_so_far={part_idx}"
            )
        except Exception as e:
            # Last resort: write synchronously if queue is jammed during shutdown
            print(f"[flush:{reason}][warn] writer queue jammed: {e} -> writing synchronously")
            out_path = run_dir / f"part_sync_{int(time.time())}.{cfg.out_format}"
            try:
                if cfg.out_format == "parquet":
                    df_chunk.to_parquet(out_path, index=False)
                else:
                    df_chunk.to_csv(out_path, index=False)
                row_count_queued += len(df_chunk)
            except Exception as e2:
                print(f"[flush:{reason}][error] sync write failed: {e2}")

    # ----------------------------
    # WebSocket callbacks
    # ----------------------------
    def on_open(ws):
        sub = build_subscribe_message(cfg.product)
        ws.send(json.dumps(sub))
        print(f"[open] subscribed: {sub}")

    def on_message(ws, message: str):
        nonlocal msg_count, last_flush
        msg_count += 1

        if stop_event.is_set():
            request_close("stop_event")
            return

        try:
            msg = json.loads(message)
        except Exception:
            return

        row = parse_message(msg)
        if row is not None:
            with buffer_lock:
                buffer.append(row)

        now = time.time()

        if msg_count % cfg.progress_every == 0:
            elapsed = int(now - start)
            with buffer_lock:
                buf_n = len(buffer)
            print(f"[progress] msgs={msg_count:,} buffer={buf_n:,} elapsed={elapsed}s")

        # flush conditions
        with buffer_lock:
            buf_len = len(buffer)

        if buf_len >= cfg.flush_rows:
            flush_buffer("rows")
        elif (now - last_flush) >= cfg.flush_seconds:
            flush_buffer("time")

        # stop condition
        if now >= end:
            request_close("timer")

    def on_error(ws, error):
        print(f"[error] {error}")
        # if exchange errors out, still exit cleanly
        request_close("ws_error")

    def on_close(ws, status_code, msg):
        print(f"[close] status={status_code} msg={msg}")
        ws_closed_event.set()

    # Create WS app (and assign to outer ref for signal handlers)
    websocket.enableTrace(False)
    ws_app = websocket.WebSocketApp(
        cfg.ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    # ----------------------------
    # Run + graceful shutdown
    # ----------------------------
    try:
        ws_app.run_forever(ping_interval=20, ping_timeout=10)
    except KeyboardInterrupt:
        # Jupyter / terminal interrupts
        request_close("KeyboardInterrupt")
    finally:
        # flush whatever is left
        flush_buffer("final")

        # tell writer we’re done
        try:
            q_out.put(None, timeout=cfg.shutdown_timeout_s)  # sentinel
        except Exception:
            pass

        # wait for writer to finish (bounded)
        t0 = time.time()
        while not writer_done.is_set() and (time.time() - t0) < cfg.shutdown_timeout_s:
            time.sleep(0.1)

        # best-effort manifest update with final counters
        write_manifest(
            extra={
                "msgs_received": msg_count,
                "rows_queued_total": row_count_queued,
                "shutdown_reason": "requested" if close_requested.is_set() else "unknown",
            }
        )

        print(
            f"Done. msgs={msg_count:,} rows_queued_total={row_count_queued:,} "
            f"ws_closed={ws_closed_event.is_set()} output_dir={run_dir}"
        )
        print("Note: a force-kill (hard terminate) can’t guarantee a clean WS close frame.")


if __name__ == "__main__":
    main()
