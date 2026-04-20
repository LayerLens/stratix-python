"""Base class for protocol adapters (MCP, A2A, AG-UI, commerce protocols).

Provides shared lifecycle behavior: protocol version negotiation, async event
emission (wraps the sync collector emit via an executor), connection pooling,
retry with exponential backoff, and a health probe hook. Individual protocol
adapters subclass :class:`BaseProtocolAdapter` and monkey-patch SDK entry
points just like the provider adapters.
"""

from __future__ import annotations

import abc
import uuid
import asyncio
import logging
from typing import Any, Dict, List, Callable, Optional, Awaitable
from dataclasses import dataclass

from .._base import AdapterInfo, BaseAdapter
from ..._context import _current_span_id, _current_collector

log = logging.getLogger(__name__)


@dataclass
class ProtocolHealth:
    reachable: bool
    latency_ms: float
    protocol_version: Optional[str] = None
    error: Optional[str] = None


class BaseProtocolAdapter(BaseAdapter, abc.ABC):
    """Shared behavior for protocol-level instrumentation."""

    #: Subclasses MUST override.
    PROTOCOL: str = ""
    PROTOCOL_VERSION: str = ""

    def __init__(
        self,
        *,
        max_connections: int = 10,
        retry_max_attempts: int = 3,
        retry_backoff_base: float = 1.0,
    ) -> None:
        self._client: Any = None
        self._originals: Dict[str, Any] = {}
        self._max_connections = max_connections
        self._retry_max_attempts = retry_max_attempts
        self._retry_backoff_base = retry_backoff_base
        self._negotiated_version: Optional[str] = None
        self._connection_semaphore = asyncio.Semaphore(max_connections)

    # --- BaseAdapter contract ---

    @abc.abstractmethod
    def connect(self, target: Any = None, **kwargs: Any) -> Any: ...

    def disconnect(self) -> None:
        if self._client is None:
            return
        for attr, orig in self._originals.items():
            try:
                parts = attr.split(".")
                obj = self._client
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], orig)
            except Exception:
                log.warning("Could not restore %s on %s adapter", attr, self.PROTOCOL)
        self._client = None
        self._originals.clear()

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name=self.PROTOCOL or self.__class__.__name__.lower(),
            adapter_type="protocol",
            version=self.PROTOCOL_VERSION or "0.1.0",
            connected=self._client is not None,
            metadata={"negotiated_version": self._negotiated_version} if self._negotiated_version else {},
        )

    # --- Version negotiation ---

    def negotiate_version(self, server_versions: List[str]) -> Optional[str]:
        """Pick a mutually-supported protocol version, preferring our own."""
        if self.PROTOCOL_VERSION in server_versions:
            self._negotiated_version = self.PROTOCOL_VERSION
            return self.PROTOCOL_VERSION
        major = self.PROTOCOL_VERSION.split(".")[0] if self.PROTOCOL_VERSION else ""
        for v in sorted(server_versions, reverse=True):
            if major and v.startswith(major):
                self._negotiated_version = v
                return v
        return None

    # --- Health probing (subclasses implement) ---

    def probe_health(self, endpoint: Optional[str] = None) -> ProtocolHealth:  # noqa: ARG002
        """Default: treat "connected" as healthy. Subclasses override for real probes."""
        return ProtocolHealth(reachable=self._client is not None, latency_ms=0.0)

    # --- Event emission ---

    def emit(self, event_name: str, payload: Dict[str, Any], *, parent_span_id: Optional[str] = None) -> None:
        collector = _current_collector.get()
        if collector is None:
            return
        collector.emit(
            event_name,
            {"protocol": self.PROTOCOL, **payload},
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent_span_id or _current_span_id.get(),
        )

    async def emit_async(
        self,
        event_name: str,
        payload: Dict[str, Any],
        *,
        parent_span_id: Optional[str] = None,
    ) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.emit, event_name, payload, parent_span_id)

    # --- Retry helper ---

    async def retry_async(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        max_attempts: Optional[int] = None,
        base_delay: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        attempts = max_attempts or self._retry_max_attempts
        delay = base_delay if base_delay is not None else self._retry_backoff_base
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt == attempts - 1:
                    break
                await asyncio.sleep(delay * (2**attempt))
        assert last_exc is not None
        raise last_exc

    # --- Connection pool ---

    async def acquire_connection(self) -> None:
        await self._connection_semaphore.acquire()

    def release_connection(self) -> None:
        self._connection_semaphore.release()
