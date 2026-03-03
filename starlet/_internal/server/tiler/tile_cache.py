"""LRU cache for MVT tile bytes, backed by :class:`collections.OrderedDict`."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional


class TileCache:
    """Bounded LRU cache mapping (z, x, y) keys to tile byte payloads."""

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self.store: OrderedDict = OrderedDict()

    def get(self, key: Any) -> Optional[bytes]:
        if key not in self.store:
            return None
        value = self.store.pop(key)
        self.store[key] = value
        return value

    def put(self, key: Any, value: bytes) -> None:
        if key in self.store:
            self.store.pop(key)
        self.store[key] = value

        if len(self.store) > self.capacity:
            self.store.popitem(last=False)
