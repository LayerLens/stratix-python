"""Base class for framework adapters.

Subclasses MUST set ``name`` and implement ``connect()``.
Subclasses SHOULD call ``super().disconnect()`` after unhooking.
"""

from __future__ import annotations

import time
import uuid
import logging
import threading
from typing import Any, Dict, Optional

from .._base import AdapterInfo, BaseAdapter
from ..._context import (
    RunState,
    _pop_span,
    _push_span,
    _current_run,
    _current_span_id,
    _current_collector,
)
from ..._collector import TraceCollector
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)


class FrameworkAdapter(BaseAdapter):
    """Base for framework adapters with collector lifecycle management.

    Every adapter call that produces events MUST be inside a
    ``_begin_run`` / ``_end_run`` pair. ``_begin_run`` pushes the
    collector and root span into ContextVars so provider adapters
    can see it automatically.
    """

    name: str  # Subclass must set: "langchain", "pydantic-ai", etc.
    package: str = ""  # pip extra name, e.g. "semantic-kernel"

    def _check_dependency(self, available: bool) -> None:
        """Raise ImportError with a helpful install message if the dependency is missing."""
        if not available:
            pkg = self.package or self.name
            raise ImportError(
                "The '%s' package is required for %s instrumentation. "
                "Install it with: pip install layerlens[%s]" % (pkg, self.name, pkg)
            )

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        self._client = client
        self._config = capture_config or CaptureConfig.standard()
        self._lock = threading.Lock()
        self._connected = False
        # Subclasses populate during connect() for adapter_info() metadata
        self._metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Per-run state (ContextVar-based isolation for concurrent runs)
    # ------------------------------------------------------------------

    def _begin_run(self) -> RunState:
        """Start a new run with its own collector, root span, and timers.

        Pushes the collector and root span into ContextVars so that:
        - Subsequent ``_emit`` calls route to this run's collector
        - Provider adapters see the collector via ``_current_collector``
        - ContextVars are automatically isolated per ``asyncio.Task``

        If called inside an existing ``trace_context()``, reuses the
        shared collector instead of creating a new one.
        """
        existing = _current_collector.get()
        if existing is not None:
            collector = existing
            col_token = None
        else:
            collector = TraceCollector(self._client, self._config)
            col_token = _current_collector.set(collector)

        root_span_id = uuid.uuid4().hex[:16]
        span_snapshot = _push_span(root_span_id, f"{self.name}:root")

        run = RunState(
            collector=collector,
            root_span_id=root_span_id,
            _token=None,
            _col_token=col_token,
            _span_snapshot=span_snapshot,
        )
        run._token = _current_run.set(run)
        return run

    def _end_run(self) -> None:
        """Pop ContextVars and flush the collector."""
        run = _current_run.get()
        if run is None:
            return

        # Restore ContextVars — use try/except for each because
        # frameworks like PydanticAI can copy contexts between hook
        # callbacks, making tokens invalid in the current Context.
        if run._span_snapshot is not None:
            try:
                _pop_span(run._span_snapshot)
            except ValueError:
                pass
        if run._col_token is not None:
            try:
                _current_collector.reset(run._col_token)
            except ValueError:
                _current_collector.set(None)
        if run._token is not None:
            try:
                _current_run.reset(run._token)
            except ValueError:
                _current_run.set(None)
        else:
            _current_run.set(None)

        # Only flush if we own the collector (not shared from trace_context)
        if run._col_token is not None:
            run.collector.flush()

    def _get_run(self) -> Optional[RunState]:
        """Return the current RunState, or None if not inside a ``_begin_run`` scope."""
        return _current_run.get()

    @staticmethod
    def _new_span_id() -> str:
        return uuid.uuid4().hex[:16]

    # ------------------------------------------------------------------
    # Shared helpers — payload, timing, tokens, content gating
    # ------------------------------------------------------------------

    def _payload(self, **extra: Any) -> Dict[str, Any]:
        """Start a payload dict with ``framework: self.name``."""
        p: Dict[str, Any] = {"framework": self.name}
        if extra:
            p.update(extra)
        return p

    def _get_root_span(self) -> str:
        """Return the root span ID for the current run.

        Returns a new random span ID if no run is active — callers should
        only call this inside a ``_begin_run`` scope.
        """
        run = _current_run.get()
        if run is not None:
            return run.root_span_id
        log.debug("layerlens: _get_root_span called outside _begin_run scope")
        return self._new_span_id()

    def _start_timer(self, key: str) -> None:
        """Record a start timestamp (nanoseconds) under *key*."""
        run = _current_run.get()
        if run is not None:
            run.timers[key] = time.time_ns()

    def _stop_timer(self, key: str) -> Optional[float]:
        """Pop the start time for *key* and return elapsed ``latency_ms``, or ``None``."""
        run = _current_run.get()
        if run is not None:
            start_ns = run.timers.pop(key, 0)
        else:
            start_ns = 0
        if not start_ns:
            return None
        return (time.time_ns() - start_ns) / 1_000_000

    @staticmethod
    def _normalize_tokens(usage: Any) -> Dict[str, Any]:
        """Extract token counts from any usage object or dict.

        Handles field-name variants across providers:
        ``prompt_tokens`` / ``input_tokens`` -> ``tokens_prompt``
        ``completion_tokens`` / ``output_tokens`` -> ``tokens_completion``

        Returns a dict with ``tokens_prompt``, ``tokens_completion``,
        ``tokens_total`` -- only keys that have non-zero values.
        Returns empty dict when all values are zero.
        """
        tokens: Dict[str, Any] = {}
        if usage is None:
            return tokens

        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            if prompt is None:
                prompt = usage.get("input_tokens")
            completion = usage.get("completion_tokens")
            if completion is None:
                completion = usage.get("output_tokens")
            total = usage.get("total_tokens")
        else:
            prompt = getattr(usage, "prompt_tokens", None)
            if prompt is None:
                prompt = getattr(usage, "input_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            if completion is None:
                completion = getattr(usage, "output_tokens", None)
            total = getattr(usage, "total_tokens", None)

        if prompt is not None:
            tokens["tokens_prompt"] = int(prompt)
        if completion is not None:
            tokens["tokens_completion"] = int(completion)
        if prompt is not None and completion is not None:
            tokens["tokens_total"] = int(prompt) + int(completion)
        elif total is not None:
            tokens["tokens_total"] = int(total)

        # Strip all-zero results so callers can use ``if tokens:``
        if tokens and not any(tokens.values()):
            return {}
        return tokens

    def _set_if_capturing(self, payload: Dict[str, Any], key: str, value: Any) -> None:
        """Set ``payload[key] = value`` only if ``capture_content`` is enabled."""
        if self._config.capture_content and value is not None:
            payload[key] = value

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        span_name: Optional[str] = None,
        run_id: Any = None,
        parent_run_id: Any = None,
    ) -> None:
        """Emit an event into the active collector.

        Single path: reads ``_current_collector``.  If there's also a
        RunState, uses it for run_id mapping and root_span_id fallback.
        No-op when no collector is active.
        """
        collector = _current_collector.get()
        if collector is None:
            return

        run = _current_run.get()

        if run_id is not None and run is not None:
            span_id, parent_span_id = self._span_id_for(run_id, parent_run_id)

        sid = span_id or self._new_span_id()
        if parent_span_id is None:
            parent_span_id = run.root_span_id if run is not None else _current_span_id.get()

        collector.emit(
            event_type,
            payload,
            span_id=sid,
            parent_span_id=parent_span_id,
            span_name=span_name,
        )

    # ------------------------------------------------------------------
    # Run ID -> span ID mapping (for callback-style frameworks)
    # ------------------------------------------------------------------

    def _span_id_for(self, run_id: Any, parent_run_id: Any = None) -> tuple[str, Optional[str]]:
        """Map a framework run_id to a (span_id, parent_span_id) pair.

        Span IDs are stored per-run in ``run.data["span_ids"]``.
        """
        run = _current_run.get()
        if run is None:
            return self._new_span_id(), None
        span_ids = run.data.setdefault("span_ids", {})
        rid = str(run_id)
        if rid not in span_ids:
            span_ids[rid] = self._new_span_id()
        span_id = span_ids[rid]
        parent_span_id = span_ids.get(str(parent_run_id)) if parent_run_id else None
        return span_id, parent_span_id

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def connect(self, target: Any = None, **kwargs: Any) -> Any:
        """Check dependencies, run framework-specific setup, and mark as connected."""
        self._on_connect(target, **kwargs)
        self._connected = True
        return target

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        """Override to set up framework-specific resources (subscribe, wrap, etc.)."""
        pass

    def disconnect(self) -> None:
        """Clean up framework resources and mark as disconnected."""
        self._on_disconnect()
        self._connected = False
        self._metadata.clear()

    def _on_disconnect(self) -> None:
        """Override to clean up framework-specific resources (unsubscribe, restore, etc.)."""
        pass

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name=self.name,
            adapter_type="framework",
            connected=self._connected,
            metadata=self._metadata,
        )
