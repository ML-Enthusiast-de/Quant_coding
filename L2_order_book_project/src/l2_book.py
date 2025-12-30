from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class L2Book:
    bids: Dict[float, float]
    asks: Dict[float, float]

    def __init__(self) -> None:
        self.bids = {}
        self.asks = {}

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()

    def apply_update(self, side: str, price_level: str, new_quantity: str) -> None:
        """
        Coinbase Advanced Trade L2:
          - side: "bid" or "ask"
          - price_level: str price
          - new_quantity: str size at that price (absolute). "0" => delete level.
        """
        p = float(price_level)
        q = float(new_quantity)

        book = self.bids if side.lower() == "bid" else self.asks
        if q <= 0.0:
            book.pop(p, None)
        else:
            book[p] = q

    def best_bid_ask(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        bb = max(self.bids) if self.bids else None
        ba = min(self.asks) if self.asks else None
        bb_sz = self.bids.get(bb) if bb is not None else None
        ba_sz = self.asks.get(ba) if ba is not None else None
        return bb, ba, bb_sz, ba_sz
