"""Unified base class for all framework adapters.

Framework adapters hook into a framework's callback / event / tracing
system and emit LayerLens events.  They share a common lifecycle:

  1. Lazy-init a :class:`TraceCollector` on first event.
  2. Emit events through a thread-safe helper.
  3. Flush the collector when a logical trace ends (root span completes,
     agent run finishes, disconnect, etc.).

Subclasses MUST set ``name`` and implement ``connect()``.
Subclasses SHOULD call ``super().disconnect()`` after unhooking.
"""
from __future__ import annotations

import uuid
import threading
from typing import Any, Dict, Optional

from .._base import AdapterInfo, BaseAdapter
from ..._collector import TraceCollector
from ..._capture_config import CaptureConfig


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
        # Optional run_id → span_id mapping for callback-style frameworks
        self._span_ids: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Collector lifecycle
    # ------------------------------------------------------------------

    def _ensure_collector(self) -> TraceCollector:
        """Lazily create a collector and root span ID."""
        if self._collector is None:
            self._collector = TraceCollector(self._client, self._config)
            self._root_span_id = uuid.uuid4().hex[:16]
        return self._collector

    @staticmethod
    def _new_span_id() -> str:
        return uuid.uuid4().hex[:16]

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
        """Map a framework run_id to a span_id, creating one if needed.

        Returns ``(span_id, parent_span_id)``.  Useful for frameworks
        (LangChain, CrewAI, OpenAI Agents) that assign their own run
        identifiers to each step.
        """
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
        """Flush the current collector and reset state."""
        with self._lock:
            collector = self._collector
            self._collector = None
            self._root_span_id = None
            self._span_ids.clear()
        if collector is not None:
            collector.flush()

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def connect(self, target: Any = None, **kwargs: Any) -> Any:
        """Mark the adapter as connected.

        Callback-style adapters (LangChain, LangGraph) are passed directly
        to the framework, so ``connect()`` just flips the flag.  Adapters
        that need registration (CrewAI, LlamaIndex, etc.) should override.
        """
        self._connected = True
        return target

    def disconnect(self) -> None:
        """Flush remaining events and mark as disconnected.

        Subclasses should unhook from the framework first, then call
        ``super().disconnect()``.
        """
        self._flush_collector()
        self._connected = False

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name=self.name,
            adapter_type="framework",
            connected=self._connected,
        )
