"""
STRATIX Protocol Connection Pool

Manages SSE and HTTP connections for protocol adapters with configurable
limits per protocol type and per endpoint.
"""

from __future__ import annotations

import time
import logging
import threading
from typing import Any
from dataclasses import field, dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConnectionSlot:
    """A single connection slot in the pool."""

    endpoint: str
    protocol: str
    created_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class ProtocolConnectionPool:
    """
    Thread-safe connection pool for protocol adapters.

    Manages connection slots per (protocol, endpoint) pair with configurable
    limits. Does not manage actual transport connections — those are handled
    by the protocol-specific adapter. This pool tracks *slots* so adapters
    can enforce concurrency limits.
    """

    def __init__(
        self,
        max_per_endpoint: int = 5,
        max_total: int = 50,
        idle_timeout_s: float = 300.0,
    ) -> None:
        self._max_per_endpoint = max_per_endpoint
        self._max_total = max_total
        self._idle_timeout_s = idle_timeout_s
        self._lock = threading.Lock()
        self._slots: dict[str, list[ConnectionSlot]] = {}  # key = protocol:endpoint

    def _key(self, protocol: str, endpoint: str) -> str:
        return f"{protocol}:{endpoint}"

    @property
    def total_active(self) -> int:
        with self._lock:
            return sum(sum(1 for s in slots if s.active) for slots in self._slots.values())

    def acquire(self, protocol: str, endpoint: str) -> ConnectionSlot | None:
        """
        Acquire a connection slot.

        Returns None if pool limits are exceeded.
        """
        key = self._key(protocol, endpoint)
        with self._lock:
            # Evict idle connections first
            self._evict_idle_locked()

            total = sum(sum(1 for s in slots if s.active) for slots in self._slots.values())
            if total >= self._max_total:
                return None

            slots = self._slots.setdefault(key, [])
            active_count = sum(1 for s in slots if s.active)
            if active_count >= self._max_per_endpoint:
                return None

            slot = ConnectionSlot(endpoint=endpoint, protocol=protocol)
            slots.append(slot)
            return slot

    def release(self, slot: ConnectionSlot) -> None:
        """Mark a connection slot as inactive."""
        with self._lock:
            slot.active = False

    def _evict_idle_locked(self) -> None:
        """Remove slots that have been idle beyond the timeout. Caller holds lock."""
        now = time.monotonic()
        for key in list(self._slots.keys()):
            self._slots[key] = [
                s
                for s in self._slots[key]
                if s.active or (now - s.last_used_at) < self._idle_timeout_s
            ]
            if not self._slots[key]:
                del self._slots[key]

    def stats(self) -> dict[str, Any]:
        """Return pool statistics."""
        with self._lock:
            active = 0
            inactive = 0
            per_endpoint: dict[str, int] = {}
            for key, slots in self._slots.items():
                a = sum(1 for s in slots if s.active)
                active += a
                inactive += len(slots) - a
                per_endpoint[key] = a
            return {
                "active": active,
                "inactive": inactive,
                "per_endpoint": per_endpoint,
                "max_per_endpoint": self._max_per_endpoint,
                "max_total": self._max_total,
            }

    def close_all(self) -> None:
        """Mark all slots as inactive."""
        with self._lock:
            for slots in self._slots.values():
                for s in slots:
                    s.active = False
            self._slots.clear()
