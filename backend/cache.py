"""Small bounded, thread-safe TTL cache for reusable backend calculations."""

from collections import OrderedDict
from copy import deepcopy
import threading
import time


MISSING = object()


class TTLCache:
    def __init__(self, max_entries=256):
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0, "misses": 0, "expired": 0, "evictions": 0,
            "invalidations": 0, "provider_calls": 0, "provider_seconds_total": 0.0,
        }

    def get(self, key):
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return MISSING
            expires_at, value = entry
            if expires_at <= now:
                del self._entries[key]
                self._stats["expired"] += 1
                self._stats["misses"] += 1
                return MISSING
            self._entries.move_to_end(key)
            self._stats["hits"] += 1
            return deepcopy(value)

    def set(self, key, value, ttl_seconds):
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._entries[key] = (time.monotonic() + ttl_seconds, deepcopy(value))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._stats["evictions"] += 1

    def invalidate(self, key):
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                self._stats["invalidations"] += 1

    def clear(self):
        with self._lock:
            self._stats["invalidations"] += len(self._entries)
            self._entries.clear()

    def record_provider_call(self, elapsed_seconds):
        with self._lock:
            self._stats["provider_calls"] += 1
            self._stats["provider_seconds_total"] += max(0.0, float(elapsed_seconds))

    def status(self):
        with self._lock:
            return {**self._stats, "entries": len(self._entries), "max_entries": self.max_entries}


cache = TTLCache(max_entries=512)
