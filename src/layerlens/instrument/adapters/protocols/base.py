"""
STRATIX Base Protocol Adapter

Abstract base class for protocol-level adapters (A2A, AG-UI, MCP Extensions).
Extends BaseAdapter with protocol-specific lifecycle: connection pooling,
health probes, protocol version negotiation, and async emission support.
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from typing import Any

from layerlens.instrument.adapters._base.adapter import (
    BaseAdapter,
    AdapterHealth,
)

logger = logging.getLogger(__name__)


class BaseProtocolAdapter(BaseAdapter):
    """
    Abstract base class for protocol-level adapters.

    Adds to BaseAdapter:
    - Protocol version negotiation
    - Async event emission via ``emit_event_async``
    - Protocol health probing
    - Connection pool awareness
    """

    # Subclasses MUST set
    PROTOCOL: str = ""
    PROTOCOL_VERSION: str = ""

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: Any | None = None,
        event_sinks: list[Any] | None = None,
        max_connections: int = 10,
        retry_max_attempts: int = 3,
        retry_backoff_base: float = 1.0,
        *,
        org_id: str | None = None,
    ) -> None:
        super().__init__(
            stratix=stratix,
            capture_config=capture_config,
            event_sinks=event_sinks,
            org_id=org_id,
        )
        self._max_connections = max_connections
        self._retry_max_attempts = retry_max_attempts
        self._retry_backoff_base = retry_backoff_base
        self._protocol_version_negotiated: str | None = None
        self._connection_pool: dict[str, Any] = {}
        self._pool_active_count = 0

    # --- Protocol-specific abstractions ---

    @abstractmethod
    def probe_health(self, endpoint: str | None = None) -> dict[str, Any]:
        """
        Probe the health of a protocol endpoint.

        Args:
            endpoint: Optional endpoint URL. If None, probe default endpoint.

        Returns:
            Dict with keys: reachable (bool), latency_ms (float), protocol_version (str|None)
        """
        ...

    def negotiate_version(self, server_versions: list[str]) -> str | None:
        """
        Negotiate protocol version with a remote endpoint.

        Args:
            server_versions: Versions the server supports.

        Returns:
            The negotiated version, or None if no compatible version found.
        """
        if self.PROTOCOL_VERSION in server_versions:
            self._protocol_version_negotiated = self.PROTOCOL_VERSION
            return self.PROTOCOL_VERSION
        # Fallback: pick the highest version we recognise
        for v in sorted(server_versions, reverse=True):
            if v.startswith(self.PROTOCOL_VERSION.split(".")[0]):
                self._protocol_version_negotiated = v
                return v
        return None

    # --- Async emission ---

    async def emit_event_async(
        self,
        payload: Any,
        privacy_level: Any | None = None,
    ) -> None:
        """
        Async wrapper around ``emit_event``.

        Protocol streams are high-throughput and often run inside an
        ``asyncio`` event loop. This wrapper avoids blocking the loop.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.emit_event, payload, privacy_level)

    # --- Connection pool helpers ---

    def _acquire_connection(self, endpoint: str) -> Any:
        """
        Acquire a connection slot for *endpoint*.

        Returns the connection object (or None if pool exhausted).
        Tracks connections per endpoint so repeated calls reuse the same slot.
        """
        if not hasattr(self, "_connections"):
            self._connections: dict[str, str] = {}

        # Reuse existing connection for this endpoint
        if endpoint in self._connections:
            return self._connections[endpoint]

        if self._pool_active_count >= self._max_connections:
            logger.warning(
                "%s connection pool exhausted (%d/%d)",
                self.PROTOCOL,
                self._pool_active_count,
                self._max_connections,
            )
            return None
        self._pool_active_count += 1
        self._connections[endpoint] = endpoint
        return self._connections[endpoint]

    def _release_connection(self, endpoint: str) -> None:
        """Release a connection slot for *endpoint*."""
        if hasattr(self, "_connections") and endpoint in self._connections:
            del self._connections[endpoint]
        self._pool_active_count = max(0, self._pool_active_count - 1)

    # --- Retry with backoff ---

    async def _retry_async(self, coro_factory: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Retry an async callable with exponential backoff.

        Args:
            coro_factory: Callable that returns a coroutine.

        Returns:
            The result of the coroutine.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(self._retry_max_attempts):
            try:
                return await coro_factory(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                delay = self._retry_backoff_base * (2**attempt)
                logger.debug(
                    "%s retry %d/%d after %.1fs: %s",
                    self.PROTOCOL,
                    attempt + 1,
                    self._retry_max_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    # --- Default health_check implementation ---

    def health_check(self) -> AdapterHealth:
        probe = self.probe_health()
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            framework_version=probe.get("protocol_version"),
            adapter_version=self.VERSION,
            message=f"reachable={probe.get('reachable', False)}",
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )
