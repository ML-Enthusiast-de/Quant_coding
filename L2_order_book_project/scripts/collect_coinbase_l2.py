#!/usr/bin/env python
"""
FOR RESEARCH PURPOSE ONLY. NOT FOR PRODUCTION USE.

Collect Coinbase L2 order book events (crypto) to a local file.

- Connects to Coinbase Advanced Trade WebSocket
- Subscribes to level2 updates for one or more products (default: BTC-USD)
- Logs raw events with timestamps for later book reconstruction + feature engineering

Output:
  L2_order_book_project/data/raw/coinbase_l2_<product>_<YYYYMMDD_HHMMSS>.parquet
  (fallback to CSV if pyarrow is not installed)

Run:
  python L2_order_book_project/scripts/collect_coinbase_l2.py --product BTC-USD --minutes 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
DATA_DIR = PROJECT_ROOT / "L2_order_book_project" / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    product: str = "BTC-USD"
    minutes: int = 10
    out_format: str = "parquet"  # parquet|csv
    ws_url: str = "wss://advanced-trade-ws.coinbase.com"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_subscribe_message(product: str) -> dict[str, Any]:
    """
    Coinbase Advanced Trade WS subscription message.
    Channel naming can change; keep this isolated for easy updates.
    """
    return {
        "type": "subscribe",
        "channel": "level2",     # if this changes, update here
        "product_ids": [product],
    }


def parse_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """
    Normalize incoming messages into a flat event row.
    We store both:
      - receive_ts: our local timestamp (UTC)
      - exchange_ts: if provided by exchange
    """
    mtype = msg.get("type") or msg.get("event") or msg.get("channel")

    # Heartbeats / subscriptions / acks etc.
    if mtype in {"subscriptions", "subscribed", "heartbeat"}:
        return None

    # Store raw payload (as JSON) + a few keys for indexing
    product = msg.get("product_id") or msg.get("product") or None

    exchange_ts = msg.get("time") or msg.get("timestamp") or None

    return {
        "receive_ts": utc_now_iso(),
        "exchange_ts": exchange_ts,
        "product": product,
        "msg_type": mtype,
        "raw": json.dumps(msg, ensure_ascii=False),
    }

import threading

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", type=str, default="BTC-USD")
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--format", type=str, default="parquet", choices=["parquet", "csv"])
    args = ap.parse_args()

    cfg = Config(product=args.product, minutes=args.minutes, out_format=args.format)
    if cfg.out_format == "parquet" and not _HAS_PARQUET:
        print("pyarrow not installed -> switching output to CSV", flush=True)
        cfg.out_format = "csv"

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"coinbase_l2_{cfg.product.replace('-', '').lower()}_{run_id}.{cfg.out_format}"

    rows: list[dict[str, Any]] = []
    msg_count = 0
    start = time.time()
    end = start + cfg.minutes * 60
    last_print = start

    def on_open(ws):
        sub = build_subscribe_message(cfg.product)
        ws.send(json.dumps(sub))
        print(f"[open] subscribed: {sub}", flush=True)

    def on_message(ws, message: str):
        nonlocal msg_count, last_print
        msg_count += 1

        try:
            msg = json.loads(message)
        except Exception:
            return

        row = parse_message(msg)
        if row is not None:
            rows.append(row)

        now = time.time()
        if now - last_print >= 5:
            print(f"[progress] msgs={msg_count:,} rows={len(rows):,} elapsed={int(now-start)}s", flush=True)
            last_print = now

    def on_error(ws, error):
        print(f"[error] {error}", flush=True)

    def on_close(ws, status_code, msg):
        print(f"[close] status={status_code} msg={msg}", flush=True)

    print(f"Connecting to {cfg.ws_url} for {cfg.minutes} min, product={cfg.product}", flush=True)
    ws = websocket.WebSocketApp(
        cfg.ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    # HARD STOP after cfg.minutes regardless of whether messages arrive
    stop_timer = threading.Timer(cfg.minutes * 60, lambda: ws.close())
    stop_timer.daemon = True
    stop_timer.start()

    try:
        ws.run_forever(ping_interval=20, ping_timeout=10)
    finally:
        stop_timer.cancel()

        if not rows:
            print("No rows collected. Nothing saved.", flush=True)
            return

        df = pd.DataFrame(rows)
        print(f"Collected rows: {len(df):,}", flush=True)
        print("Sample types:", df["msg_type"].value_counts().head(10).to_dict(), flush=True)

        if cfg.out_format == "parquet":
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)

        print(f"Saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
