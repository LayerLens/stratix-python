"""Base class for framework adapters.

Subclasses MUST set ``name`` and implement ``connect()``.
Subclasses SHOULD call ``super().disconnect()`` after unhooking.
"""
from __future__ import annotations

import uuid
import logging
import threading
from typing import Any, Dict, Generator, Optional
from contextlib import contextmanager

from .._base import AdapterInfo, BaseAdapter
from ..._collector import TraceCollector
from ..._capture_config import CaptureConfig
from ..._context import _current_collector, _current_span_id, _push_span, _pop_span

log = logging.getLogger(__name__)


class FrameworkAdapter(BaseAdapter):
    """Base for framework adapters with collector lifecycle management."""

    name: str  # Subclass must set: "crewai", "llamaindex", etc.

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        self._client = client
        self._config = capture_config or CaptureConfig.standard()
        self._lock = threading.Lock()
        self._connected = False
        self._collector: Optional[TraceCollector] = None
        self._root_span_id: Optional[str] = None
        self._using_shared_collector = False
        # Optional run_id → span_id mapping for callback-style frameworks
        self._span_ids: Dict[str, str] = {}
        # Subclasses populate during connect() for adapter_info() metadata
        self._metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Collector lifecycle
    # ------------------------------------------------------------------

    def _ensure_collector(self) -> TraceCollector:
        """Return the shared collector from ContextVars, or create a private one."""
        shared = _current_collector.get()
        if shared is not None:
            self._using_shared_collector = True
            if self._root_span_id is None:
                self._root_span_id = _current_span_id.get()
            return shared

        if self._collector is None:
            self._using_shared_collector = False
            self._collector = TraceCollector(self._client, self._config)
            self._root_span_id = uuid.uuid4().hex[:16]
        return self._collector

    @staticmethod
    def _new_span_id() -> str:
        return uuid.uuid4().hex[:16]

    # ------------------------------------------------------------------
    # Callback scope — bridges framework callbacks to ContextVars
    # ------------------------------------------------------------------

    @contextmanager
    def _callback_scope(
        self,
        span_name: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Push collector + new span into ContextVars; yields the span_id."""
        collector = self._ensure_collector()
        span_id = self._new_span_id()

        # Only set the collector ContextVar if no shared one exists already
        needs_collector_push = _current_collector.get() is None
        col_token = None
        if needs_collector_push:
            col_token = _current_collector.set(collector)

        snapshot = _push_span(span_id, span_name)
        try:
            yield span_id
        finally:
            _pop_span(snapshot)
            if col_token is not None:
                _current_collector.reset(col_token)

    def _traced_call(
        self,
        original: Any,
        *args: Any,
        _span_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Call *original* inside a _callback_scope so providers see this collector."""
        with self._callback_scope(_span_name):
            return original(*args, **kwargs)

    async def _async_traced_call(
        self,
        original: Any,
        *args: Any,
        _span_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Async version of _traced_call."""
        with self._callback_scope(_span_name):
            return await original(*args, **kwargs)

    # ------------------------------------------------------------------
    # Event emission (thread-safe)
    # ------------------------------------------------------------------

    def _emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        span_name: Optional[str] = None,
    ) -> None:
        """Thread-safe event emission through the collector."""
        with self._lock:
            collector = self._ensure_collector()
            sid = span_id or self._new_span_id()
            parent = parent_span_id or self._root_span_id
            collector.emit(
                event_type, payload,
                span_id=sid, parent_span_id=parent, span_name=span_name,
            )

    # ------------------------------------------------------------------
    # Run ID → span ID mapping (opt-in for callback-style frameworks)
    # ------------------------------------------------------------------

    def _span_id_for(self, run_id: Any, parent_run_id: Any = None) -> tuple[str, Optional[str]]:
        """Map a framework run_id to a (span_id, parent_span_id) pair."""
        rid = str(run_id)
        if rid not in self._span_ids:
            self._span_ids[rid] = self._new_span_id()
        span_id = self._span_ids[rid]
        parent_span_id = self._span_ids.get(str(parent_run_id)) if parent_run_id else None
        return span_id, parent_span_id

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def _flush_collector(self) -> None:
        """Flush private collector (no-op for shared collectors)."""
        with self._lock:
            collector = self._collector
            is_shared = self._using_shared_collector
            self._collector = None
            self._root_span_id = None
            self._using_shared_collector = False
            self._span_ids.clear()
        if collector is not None and not is_shared:
            collector.flush()

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def connect(self, target: Any = None, **kwargs: Any) -> Any:
        """Mark as connected. Subclasses override for framework registration."""
        self._connected = True
        return target

    def disconnect(self) -> None:
        """Flush remaining events and mark as disconnected."""
        self._flush_collector()
        self._connected = False
        self._metadata.clear()

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name=self.name,
            adapter_type="framework",
            connected=self._connected,
            metadata=self._metadata,
        )
