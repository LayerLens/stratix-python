"""Base class for framework adapters.

Subclasses MUST set ``name`` and implement ``connect()``.
Subclasses SHOULD call ``super().disconnect()`` after unhooking.
"""
from __future__ import annotations

import time
import uuid
import logging
import threading
from typing import Any, Dict, Generator, Optional
from contextlib import contextmanager

from .._base import AdapterInfo, BaseAdapter
from ..._collector import TraceCollector
from ..._capture_config import CaptureConfig
from ..._context import _current_collector, _current_span_id, _push_span, _pop_span, _current_run, RunState

log = logging.getLogger(__name__)

_UNSET: Any = object()  # sentinel: distinguish "not passed" from explicit None


class FrameworkAdapter(BaseAdapter):
    """Base for framework adapters with collector lifecycle management."""

    name: str  # Subclass must set: "crewai", "llamaindex", etc.
    package: str = ""  # pip extra name, e.g. "crewai" → pip install layerlens[crewai]

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
        self._collector: Optional[TraceCollector] = None
        self._root_span_id: Optional[str] = None
        self._using_shared_collector = False
        # Optional run_id → span_id mapping for callback-style frameworks
        self._span_ids: Dict[str, str] = {}
        # Root run tracking for auto-flush on outermost callback completion
        self._root_run_id: Optional[str] = None
        # Timing: key → start_ns for _start_timer / _stop_timer
        self._timers: Dict[str, int] = {}
        # Subclasses populate during connect() for adapter_info() metadata
        self._metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Per-run state (ContextVar-based isolation for concurrent runs)
    # ------------------------------------------------------------------

    def _begin_run(self) -> RunState:
        """Start a new run with its own collector, root span, and timers.

        Stores the RunState in a ContextVar so all subsequent calls to
        ``_ensure_collector``, ``_start_timer``, ``_stop_timer``, and
        ``_get_root_span`` use per-run state instead of instance state.

        ContextVars are automatically isolated per ``asyncio.Task``, so
        concurrent runs on the same adapter get independent state.
        """
        run = RunState(
            collector=TraceCollector(self._client, self._config),
            root_span_id=uuid.uuid4().hex[:16],
        )
        run._token = _current_run.set(run)
        return run

    def _end_run(self) -> None:
        """Flush the current run's collector and restore the previous ContextVar state."""
        run = _current_run.get()
        if run is None:
            return
        if run._token is not None:
            try:
                _current_run.reset(run._token)
            except ValueError:
                # Token created in a different Context (e.g. framework copies
                # contexts between hook callbacks). Fall back to plain set.
                _current_run.set(None)
        else:
            _current_run.set(None)
        run.collector.flush()

    def _get_run(self) -> Optional[RunState]:
        """Return the current RunState, or None if not inside a ``_begin_run`` scope."""
        return _current_run.get()

    # ------------------------------------------------------------------
    # Collector lifecycle
    # ------------------------------------------------------------------

    def _ensure_collector(self) -> TraceCollector:
        """Return the collector for the current context.

        Checks (in order): active RunState, shared collector from ContextVars,
        then creates a private instance-level collector as fallback.
        """
        run = _current_run.get()
        if run is not None:
            return run.collector

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
    # Shared helpers — payload, timing, tokens, content gating
    # ------------------------------------------------------------------

    def _payload(self, **extra: Any) -> Dict[str, Any]:
        """Start a payload dict with ``framework: self.name``.

        Usage::

            payload = self._payload(agent_name="foo", status="ok")
        """
        p: Dict[str, Any] = {"framework": self.name}
        if extra:
            p.update(extra)
        return p

    def _get_root_span(self) -> str:
        """Return the root span ID for the current run.

        Checks RunState first, then falls back to instance-level ``_root_span_id``.
        If neither is set, generates a new one.
        """
        run = _current_run.get()
        if run is not None:
            return run.root_span_id

        with self._lock:
            sid = self._root_span_id
        if sid is not None:
            return sid
        sid = self._new_span_id()
        with self._lock:
            self._root_span_id = sid
        return sid

    def _start_timer(self, key: str) -> None:
        """Record a start timestamp (nanoseconds) under *key*."""
        run = _current_run.get()
        if run is not None:
            run.timers[key] = time.time_ns()
            return
        with self._lock:
            self._timers[key] = time.time_ns()

    def _stop_timer(self, key: str) -> Optional[float]:
        """Pop the start time for *key* and return elapsed ``latency_ms``, or ``None``."""
        run = _current_run.get()
        if run is not None:
            start_ns = run.timers.pop(key, 0)
        else:
            with self._lock:
                start_ns = self._timers.pop(key, 0)
        if not start_ns:
            return None
        return (time.time_ns() - start_ns) / 1_000_000

    @staticmethod
    def _normalize_tokens(usage: Any) -> Dict[str, Any]:
        """Extract token counts from any usage object or dict.

        Handles field-name variants across providers:
        ``prompt_tokens`` / ``input_tokens`` → ``tokens_prompt``
        ``completion_tokens`` / ``output_tokens`` → ``tokens_completion``

        Returns a dict with ``tokens_prompt``, ``tokens_completion``,
        ``tokens_total`` — only keys that have non-zero values.
        """
        tokens: Dict[str, Any] = {}
        if usage is None:
            return tokens

        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
            completion = usage.get("completion_tokens") or usage.get("output_tokens")
            total = usage.get("total_tokens")
        else:
            prompt = (
                getattr(usage, "prompt_tokens", None)
                or getattr(usage, "input_tokens", None)
            )
            completion = (
                getattr(usage, "completion_tokens", None)
                or getattr(usage, "output_tokens", None)
            )
            total = getattr(usage, "total_tokens", None)

        if prompt is not None:
            tokens["tokens_prompt"] = int(prompt)
        if completion is not None:
            tokens["tokens_completion"] = int(completion)
        if prompt is not None and completion is not None:
            tokens["tokens_total"] = int(prompt) + int(completion)
        elif total is not None:
            tokens["tokens_total"] = int(total)
        return tokens

    def _set_if_capturing(self, payload: Dict[str, Any], key: str, value: Any) -> None:
        """Set ``payload[key] = value`` only if ``capture_content`` is enabled."""
        if self._config.capture_content and value is not None:
            payload[key] = value

    # ------------------------------------------------------------------
    # Callback scope — bridges framework callbacks to ContextVars
    # ------------------------------------------------------------------

    def _push_context(self, span_id: str, span_name: Optional[str] = None) -> Any:
        """Push collector + span into ContextVars. Returns an opaque token for ``_pop_context``."""
        with self._lock:
            collector = self._ensure_collector()
        needs_collector_push = _current_collector.get() is None
        col_token = _current_collector.set(collector) if needs_collector_push else None
        snapshot = _push_span(span_id, span_name)
        return (snapshot, col_token)

    def _pop_context(self, token: Any) -> None:
        """Restore ContextVars from a token returned by ``_push_context``."""
        if token is None:
            return
        snapshot, col_token = token
        _pop_span(snapshot)
        if col_token is not None:
            _current_collector.reset(col_token)

    @contextmanager
    def _callback_scope(
        self,
        span_name: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Push collector + new span into ContextVars; yields the span_id."""
        span_id = self._new_span_id()
        token = self._push_context(span_id, span_name)
        try:
            yield span_id
        finally:
            self._pop_context(token)

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
        parent_span_id: Any = _UNSET,
        span_name: Optional[str] = None,
        run_id: Any = None,
        parent_run_id: Any = None,
    ) -> None:
        """Thread-safe event emission through the collector.

        When *run_id* is provided, it is translated to a span_id via
        ``_span_id_for`` and the first run_id seen is tracked as the root
        (for flush-on-completion in callback-style frameworks).

        When *parent_span_id* is omitted, falls back to ``_root_span_id``.
        Pass ``parent_span_id=None`` explicitly to emit with no parent
        (for adapters that manage their own span hierarchy).
        """
        # RunState path: per-run isolation, no lock needed
        run = _current_run.get()
        if run is not None:
            if run_id is not None:
                span_id, parent_span_id = self._span_id_for(run_id, parent_run_id)
            sid = span_id or self._new_span_id()
            parent = run.root_span_id if parent_span_id is _UNSET else parent_span_id
            run.collector.emit(
                event_type, payload,
                span_id=sid, parent_span_id=parent, span_name=span_name,
            )
            return

        # Legacy path: instance-level state with lock
        if run_id is not None:
            span_id, parent_span_id = self._span_id_for(run_id, parent_run_id)
            if self._root_run_id is None:
                self._root_run_id = str(run_id)
        with self._lock:
            collector = self._ensure_collector()
            sid = span_id or self._new_span_id()
            parent = self._root_span_id if parent_span_id is _UNSET else parent_span_id
            collector.emit(
                event_type, payload,
                span_id=sid, parent_span_id=parent, span_name=span_name,
            )

    # ------------------------------------------------------------------
    # Run ID → span ID mapping (opt-in for callback-style frameworks)
    # ------------------------------------------------------------------

    def _span_id_for(self, run_id: Any, parent_run_id: Any = None) -> tuple[str, Optional[str]]:
        """Map a framework run_id to a (span_id, parent_span_id) pair.

        When a RunState is active, span_ids are stored per-run in
        ``run.data["span_ids"]`` for concurrent-run isolation.
        Falls back to instance-level ``_span_ids`` otherwise.
        """
        run = _current_run.get()
        span_ids = run.data.setdefault("span_ids", {}) if run is not None else self._span_ids
        rid = str(run_id)
        if rid not in span_ids:
            span_ids[rid] = self._new_span_id()
        span_id = span_ids[rid]
        parent_span_id = span_ids.get(str(parent_run_id)) if parent_run_id else None
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
        """Check dependencies, run framework-specific setup, and mark as connected."""
        self._on_connect(target, **kwargs)
        self._connected = True
        return target

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        """Override to set up framework-specific resources (subscribe, wrap, etc.)."""
        pass

    def disconnect(self) -> None:
        """Clean up framework resources, flush events, and mark as disconnected."""
        self._on_disconnect()
        self._flush_collector()
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
