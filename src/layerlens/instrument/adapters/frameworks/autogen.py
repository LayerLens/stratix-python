from __future__ import annotations

import logging
import itertools
from typing import Any, Dict, Optional

from ._utils import truncate, safe_serialize
from ..._context import RunState, _current_run, _current_collector
from ..._collector import TraceCollector
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    from autogen_core import (
        EVENT_LOGGER_NAME as _EVENT_LOGGER_NAME,
    )  # pyright: ignore[reportMissingImports]

    _HAS_AUTOGEN = True
except ImportError:
    _HAS_AUTOGEN = False
    _EVENT_LOGGER_NAME = "autogen_core.events"


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _get_field(event: Any, name: str, default: Any = None) -> Any:
    kw = getattr(event, "kwargs", None)
    if isinstance(kw, dict) and name in kw:
        return kw[name]
    val = getattr(event, name, default)
    return val if val is not default else default


def _extract_model(event: Any) -> Optional[str]:
    response = _get_field(event, "response")
    if isinstance(response, dict):
        model = response.get("model")
        if model:
            return str(model)
    model = _get_field(event, "model")
    return str(model) if model else None


def _enum_name(value: Any) -> str:
    s = str(value)
    if "." in s:
        return s.rsplit(".", 1)[-1]
    if hasattr(value, "name"):
        return value.name
    return s


class AutoGenAdapter(FrameworkAdapter):
    """AutoGen adapter using the structured event logging API (autogen-core >= 0.4).

    Concurrency model (LAY-3576 / A6 fix)
    -------------------------------------
    AutoGen has no per-run callback: it logs ``LLMCallEvent`` / ``MessageEvent``
    through the **module-global** ``EVENT_LOGGER_NAME`` logger, and those events
    carry no run/topic/session id (``LLMCallEvent`` has only ``agent_id``;
    ``MessageEvent`` only sender/receiver). The old adapter funnelled every run
    into one lazily-created ``self._collector`` shared across all conversations —
    two concurrent ``team.run()`` calls merged into a single trace.

    But each AgentChat team owns its own ``SingleThreadedAgentRuntime`` whose
    message loop drains on its own asyncio task / thread, and the logging handler
    runs **inline on the emitting task/thread** (no context-copy). So a
    ``RunState`` opened lazily on the first event and bound to ``_current_run``
    is isolated per concurrent run by the task/thread ContextVar copy (verified
    live for two interleaved runtimes — distinct runs even for the sender-less
    ``LLMCallEvent``). The adapter keeps one ``RunState`` per run in
    ``self._runs`` and flushes each as its own trace on ``disconnect()``. This
    mirrors ``openai_agents.py`` (keyed map + per-callback ``_current_collector``).

    Usage::

        adapter = AutoGenAdapter(client)
        adapter.connect()
        result = await team.run(task="hello")
        adapter.disconnect()
    """

    name = "autogen"
    package = "autogen"

    _EVENT_DISPATCH = {
        "LLMCallEvent": "_on_llm_call",
        "LLMStreamEndEvent": "_on_llm_call",
        "ToolCallEvent": "_on_tool_call",
        "MessageEvent": "_on_message",
        "MessageDroppedEvent": "_on_message_dropped",
        "MessageHandlerExceptionEvent": "_on_handler_exception",
        "AgentConstructionExceptionEvent": "_on_construction_exception",
    }

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._handler: Optional[_LayerLensHandler] = None
        # token -> RunState (one per concurrent conversation/run).
        self._runs: Dict[str, RunState] = {}
        self._run_seq = itertools.count()
        # Fallback slots for callers that drive ``_fire`` outside a run (the
        # cost-pricing invariant test instantiates via ``__new__``).
        self._fallback_collector: Optional[TraceCollector] = None
        self._fallback_root_span_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Per-run state access (resolved through the active _current_run)
    # ------------------------------------------------------------------

    @property
    def _collector(self) -> Optional[TraceCollector]:
        run = _current_run.get()
        if run is not None:
            return run.collector
        return self._fallback_collector

    @_collector.setter
    def _collector(self, value: Optional[TraceCollector]) -> None:
        self._fallback_collector = value

    @property
    def _root_span_id(self) -> Optional[str]:
        run = _current_run.get()
        if run is not None:
            return run.root_span_id
        return self._fallback_root_span_id

    @_root_span_id.setter
    def _root_span_id(self, value: Optional[str]) -> None:
        self._fallback_root_span_id = value

    @property
    def _conversations(self) -> Dict[str, Dict[str, Any]]:
        run = _current_run.get()
        if run is None:
            return {}
        return run.data.setdefault("conversations", {})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_AUTOGEN)
        self._handler = _LayerLensHandler(self)
        logger = logging.getLogger(_EVENT_LOGGER_NAME)
        logger.addHandler(self._handler)
        if logger.level == logging.NOTSET or logger.level > logging.DEBUG:
            logger.setLevel(logging.DEBUG)

    def _on_disconnect(self) -> None:
        if self._handler is not None:
            logger = logging.getLogger(_EVENT_LOGGER_NAME)
            logger.removeHandler(self._handler)
            self._handler = None
        # Flush every open run as its own trace (autogen gives no per-run end
        # signal, so completion is at disconnect).
        with self._lock:
            runs = list(self._runs.values())
            self._runs.clear()
        # Unbind the run bound to THIS context first, so a stale RunState never
        # leaks into the next adapter on the same thread.
        active = _current_run.get()
        for run in runs:
            self._flush_run(run)
        if active is not None and active in runs:
            self._unbind(active)
        self._fallback_collector = None
        self._fallback_root_span_id = None

    @staticmethod
    def _unbind(run: RunState) -> None:
        """Reset the ContextVars this run set when it opened."""
        if run._col_token is not None:
            try:
                _current_collector.reset(run._col_token)
            except ValueError:
                _current_collector.set(None)
            run._col_token = None
        if run._token is not None:
            try:
                _current_run.reset(run._token)
            except ValueError:
                _current_run.set(None)
            run._token = None

    # ------------------------------------------------------------------
    # Collector + run-state management
    # ------------------------------------------------------------------

    def _ensure_run(self) -> RunState:
        """Lazily open a run on the current context and bind it.

        Concurrent conversations call this on their own task/thread, so the
        ContextVar copy keeps each run isolated. Subsequent events on the same
        context re-resolve the same run via ``_current_run``.

        Collector resolution
        --------------------
        If a caller has already bound a :class:`TraceCollector` on
        ``_current_collector`` (the canonical pattern used by ``instrument()``
        and ``samples/adapters/_shared.capture_events``), the run reuses **that**
        collector instead of minting a private one. Otherwise the adapter falls
        back to a self-owned collector flushed at ``disconnect()``. Without this
        the events landed in a private collector that the caller never read, so
        the bound trace captured zero events even though the logging hook fired.
        """
        run = _current_run.get()
        if run is not None:
            return run
        bound = _current_collector.get()
        owns_collector = bound is None
        collector = bound if bound is not None else TraceCollector(self._client, self._config)
        root_span_id = self._new_span_id()
        run = RunState(collector=collector, root_span_id=root_span_id, data={"conversations": {}})
        token = f"autogen-run-{next(self._run_seq)}"
        run.data["token"] = token
        # Only this adapter flushes collectors it owns; a caller-bound collector
        # is flushed by its owner (``capture_events`` / ``instrument()``).
        run.data["owns_collector"] = owns_collector
        with self._lock:
            self._runs[token] = run
        # Re-set the collector ContextVar only when we created it, so the run's
        # _col_token unbinds exactly what it bound (never the caller's binding).
        if owns_collector:
            run._col_token = _current_collector.set(collector)
        run._token = _current_run.set(run)
        return run

    def _flush_run(self, run: RunState) -> None:
        """Emit per-run conversation summaries, then flush owned collectors.

        A caller-bound collector (``owns_collector`` False) is flushed by its
        owner (``capture_events`` / ``instrument()``), so we only emit the
        summaries onto it and leave the flush to the owner.
        """
        collector = run.collector
        conversations = run.data.get("conversations", {})
        for conv_id, state in list(conversations.items()):
            collector.emit(
                "conversation.ended",
                self._payload(
                    conversation_id=conv_id,
                    participants=sorted(state["participants"]),
                    message_count=state["message_count"],
                    turn_count=state["turn_count"],
                    reason="trace_end",
                ),
                span_id=self._new_span_id(),
                parent_span_id=run.root_span_id,
            )
        if not run.data.get("owns_collector", True):
            return
        try:
            collector.flush()
        except Exception:
            log.warning("layerlens: error flushing AutoGen run", exc_info=True)

    def _fire(
        self,
        event_type: str,
        payload: Dict[str, Any],
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        span_name: Optional[str] = None,
    ) -> None:
        c = self._collector
        if c is None:
            return
        if event_type == "cost.record" and payload.get("cost_usd") is None:
            self._price_cost_record(payload)
        c.emit(
            event_type,
            payload,
            span_id=span_id or self._new_span_id(),
            parent_span_id=parent_span_id or self._root_span_id,
            span_name=span_name,
        )

    # ------------------------------------------------------------------
    # Event dispatch (called by handler)
    # ------------------------------------------------------------------

    def _dispatch(self, event: Any) -> None:
        event_class = type(event).__name__
        handler_name = self._EVENT_DISPATCH.get(event_class)
        if handler_name is None:
            return
        # Open/resolve this run on the current task/thread context BEFORE the
        # handler runs, so concurrent conversations stay isolated.
        self._ensure_run()
        try:
            getattr(self, handler_name)(event)
        except Exception:
            log.warning("layerlens: error in AutoGen event handler", exc_info=True)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_llm_call(self, event: Any) -> None:
        model = _extract_model(event)
        prompt_tokens = _get_field(event, "prompt_tokens", 0) or 0
        completion_tokens = _get_field(event, "completion_tokens", 0) or 0
        agent_id = _get_field(event, "agent_id")

        span_id = self._new_span_id()
        payload = self._payload()
        if model:
            payload["model"] = model
        if prompt_tokens:
            payload["tokens_prompt"] = prompt_tokens
        if completion_tokens:
            payload["tokens_completion"] = completion_tokens
        if prompt_tokens or completion_tokens:
            payload["tokens_total"] = prompt_tokens + completion_tokens
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)

        self._set_if_capturing(payload, "messages", safe_serialize(_get_field(event, "messages")))
        self._set_if_capturing(payload, "output_message", safe_serialize(_get_field(event, "response")))

        self._fire("model.invoke", payload, span_id=span_id)

        if prompt_tokens or completion_tokens:
            cost_payload = self._payload(
                tokens_prompt=prompt_tokens,
                tokens_completion=completion_tokens,
                tokens_total=prompt_tokens + completion_tokens,
            )
            if model:
                cost_payload["model"] = model
            self._fire("cost.record", cost_payload, span_id=span_id)

    def _on_tool_call(self, event: Any) -> None:
        tool_name = _get_field(event, "tool_name", "unknown")
        payload = self._payload(tool_name=tool_name)
        self._set_if_capturing(payload, "input", safe_serialize(_get_field(event, "arguments")))
        self._set_if_capturing(payload, "output", safe_serialize(_get_field(event, "result")))
        self._fire("tool.call", payload)

    def _on_message(self, event: Any) -> None:
        sender = _get_field(event, "sender")
        receiver = _get_field(event, "receiver")
        kind = _get_field(event, "kind")
        stage = _get_field(event, "delivery_stage")

        # Conversation tracking: group messages by topic/session ID so downstream
        # analysis can reason about multi-agent turn-taking.
        topic_id = _get_field(event, "topic_id") or _get_field(event, "session_id")
        conv_id = str(topic_id) if topic_id is not None else f"{sender}->{receiver}"
        conversations = self._conversations
        state = conversations.setdefault(
            conv_id,
            {
                "participants": set(),
                "turn_count": 0,
                "message_count": 0,
                "last_sender": None,
            },
        )
        if sender is not None:
            state["participants"].add(str(sender))
        if receiver is not None:
            state["participants"].add(str(receiver))
        state["message_count"] += 1
        last = state["last_sender"]
        if sender is not None and last is not None and str(sender) != last:
            state["turn_count"] += 1
        if sender is not None:
            state["last_sender"] = str(sender)

        payload = self._payload()
        payload["conversation_id"] = conv_id
        payload["turn_index"] = state["turn_count"]
        payload["message_index"] = state["message_count"]
        if sender is not None:
            payload["sender"] = str(sender)
        if receiver is not None:
            payload["receiver"] = str(receiver)
        if kind is not None:
            payload["message_kind"] = _enum_name(kind)
        if stage is not None:
            payload["delivery_stage"] = _enum_name(stage)
        self._set_if_capturing(
            payload,
            "content",
            truncate(str(_get_field(event, "payload", "")), 2000),
        )

        kind_str = _enum_name(kind) if kind is not None else ""
        if "RESPOND" in kind_str:
            self._fire("agent.output", payload)
        else:
            self._fire("agent.input", payload)

    def _on_message_dropped(self, event: Any) -> None:
        sender = _get_field(event, "sender")
        receiver = _get_field(event, "receiver")
        kind = _get_field(event, "kind")

        payload = self._payload(dropped=True)
        payload["error_type"] = "message_dropped"
        payload["status"] = "error"
        if sender is not None:
            payload["sender"] = str(sender)
        if receiver is not None:
            payload["receiver"] = str(receiver)
        if kind is not None:
            payload["message_kind"] = _enum_name(kind)
        self._fire("agent.error", payload)

    def _on_handler_exception(self, event: Any) -> None:
        agent_id = _get_field(event, "handling_agent")
        exc = _get_field(event, "exception")
        payload = self._payload(
            error=str(exc) if exc else "unknown error",
            error_type=(type(exc).__name__ if isinstance(exc, BaseException) else "Exception"),
        )
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        self._fire("agent.error", payload)

    def _on_construction_exception(self, event: Any) -> None:
        agent_id = _get_field(event, "agent_id")
        exc = _get_field(event, "exception")
        payload = self._payload(
            error=str(exc) if exc else "construction failed",
            error_type=(type(exc).__name__ if isinstance(exc, BaseException) else "Exception"),
        )
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        self._fire("agent.error", payload)


class _LayerLensHandler(logging.Handler):
    """Thin logging handler that delegates to the adapter."""

    def __init__(self, adapter: AutoGenAdapter) -> None:
        super().__init__()
        self._adapter = adapter

    def emit(self, record: logging.LogRecord) -> None:
        event = record.msg
        if event is not None:
            self._adapter._dispatch(event)
