#!/usr/bin/env python3
"""
PowerTrader AI — IPC bridge

Abstracts signal and JSON state reads/writes behind a single interface so
the rest of the codebase can switch between file-based IPC (default, zero
extra dependencies) and Redis pub/sub (opt-in, for multi-machine or
sub-100 ms latency requirements) via a single env variable.

    # file mode (default)
    ipc = IPCBridge()

    # Redis mode (falls back to file if Redis is unreachable)
    REDIS_URL=redis://localhost:6379 python pt_trader.py

Usage pattern (read side — e.g. pt_trader.py):
    from pt_ipc import ipc
    level = ipc.read_signal("BTC", "long", base_dir="/app/BTC")

Usage pattern (write side — e.g. pt_thinker.py):
    from pt_ipc import ipc
    ipc.write_signal("BTC", "long", value, base_dir=".")
    ipc.write_signal("BTC", "short", value, base_dir=".")
"""

from __future__ import annotations

import json
import os
from typing import Optional

# Redis key prefix — avoids collisions with other apps on the same instance
_NS = "pt"


class IPCBridge:
    """
    Thin read/write wrapper.  When Redis is configured and reachable the
    bridge publishes to a channel AND sets a last-value key so late-joining
    readers always get the most recent state without needing to replay the
    stream.  All operations fall back silently to file IPC on any error.
    """

    def __init__(self) -> None:
        self.backend = "file"
        self._r = None

        redis_url = os.environ.get("REDIS_URL", "").strip()
        if redis_url:
            try:
                import redis  # type: ignore[import]

                client = redis.from_url(redis_url, decode_responses=True, socket_timeout=2)
                client.ping()
                self._r = client
                self.backend = "redis"
                print(f"[ipc] Redis backend active: {redis_url}")
            except Exception as exc:
                print(f"[ipc] Redis unavailable ({exc}), using file backend.")

    # ------------------------------------------------------------------
    # Signal helpers (int 0-7 written to {direction}_dca_signal.txt)
    # ------------------------------------------------------------------

    def write_signal(self, coin: str, direction: str, value: int, base_dir: str = ".") -> None:
        """Write a DCA signal value.  Always writes the file; also publishes
        to Redis when enabled so subscribers receive it immediately."""
        val_str = str(int(value))
        path = os.path.join(base_dir, f"{direction}_dca_signal.txt")

        # Always write file (keeps file-IPC readers happy regardless of backend)
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(val_str)
            os.replace(tmp, path)
        except Exception:
            pass

        if self.backend == "redis" and self._r is not None:
            try:
                key = f"{_NS}:signal:{coin.upper()}:{direction}"
                self._r.set(key, val_str)
                self._r.publish(key, val_str)
            except Exception:
                pass

    def read_signal(self, coin: str, direction: str, base_dir: str = ".") -> int:
        """Read a DCA signal value.  Prefers Redis (lower latency) then falls
        back to the signal file."""
        if self.backend == "redis" and self._r is not None:
            try:
                key = f"{_NS}:signal:{coin.upper()}:{direction}"
                val = self._r.get(key)
                if val is not None:
                    return int(val)
            except Exception:
                pass

        # File fallback
        path = os.path.join(base_dir, f"{direction}_dca_signal.txt")
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return int(f.read().strip())
        except Exception:
            pass
        return 0

    # ------------------------------------------------------------------
    # JSON state helpers (trader_status, pnl_ledger, etc.)
    # ------------------------------------------------------------------

    def write_json(self, key: str, data: dict, path: str) -> None:
        """Atomically write a JSON state file; also cache in Redis."""
        serialised = json.dumps(data, indent=2)

        # Atomic file write (always)
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(serialised)
            os.replace(tmp, path)
        except Exception:
            pass

        if self.backend == "redis" and self._r is not None:
            try:
                rkey = f"{_NS}:json:{key}"
                self._r.set(rkey, serialised)
                self._r.publish(rkey, serialised)
            except Exception:
                pass

    def read_json(self, key: str, path: str) -> Optional[dict]:
        """Read a JSON state value.  Prefers Redis then falls back to file."""
        if self.backend == "redis" and self._r is not None:
            try:
                rkey = f"{_NS}:json:{key}"
                val = self._r.get(rkey)
                if val is not None:
                    return json.loads(val)
            except Exception:
                pass

        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Subscribe (Redis-only convenience; no-op in file mode)
    # ------------------------------------------------------------------

    def subscribe_signal(self, coin: str, direction: str):
        """Return a Redis PubSub object subscribed to a signal channel, or
        None when using file backend.  Caller is responsible for cleanup."""
        if self.backend != "redis" or self._r is None:
            return None
        try:
            key = f"{_NS}:signal:{coin.upper()}:{direction}"
            ps = self._r.pubsub(ignore_subscribe_messages=True)
            ps.subscribe(key)
            return ps
        except Exception:
            return None


# Module-level singleton — import this everywhere instead of constructing
# a new instance per process.
ipc: IPCBridge = IPCBridge()
