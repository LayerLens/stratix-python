from __future__ import annotations

import re
import logging
import threading
from typing import Any, Dict, Optional

from ._utils import truncate, safe_serialize
from ..._context import RunState, _current_collector
from ..._identity import honest_agent_type
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


# AgentChat teams name each participant's runtime agent ``<agent-name>_<team-uuid>``
# and stringify an AgentId as ``<type>/<key>``. Strip that trailing per-team UUID
# so the graph node is the real agent name (``writer``), not ``writer_<uuid>``.
_TEAM_UUID_SUFFIX = re.compile(r"_[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _autogen_agent_name(agent_id: Any) -> Optional[str]:
    """Clean, honest graph-node name from an autogen AgentId (or its string).

    ``AgentId`` stringifies as ``<type>/<key>``; a team runtime sets the type to
    ``<agent-name>_<team-uuid>``. Take the type, strip the team-uuid suffix, drop
    the runtime plumbing that is not a real agent (the group-chat manager and the
    ``group_topic``/``output_topic`` routing topics), then apply the shared
    honesty guard (rejects model/class/generic names). Returns None when there is
    no honest agent — so plumbing never becomes a graph node.
    """
    if agent_id is None:
        return None
    typ = str(agent_id).split("/", 1)[0]
    typ = _TEAM_UUID_SUFFIX.sub("", typ)
    low = typ.lower()
    if "groupchatmanager" in low or low.endswith("_topic"):
        return None
    return honest_agent_type(typ)


class AutoGenAdapter(FrameworkAdapter):
    """AutoGen adapter using the structured event logging API (autogen-core >= 0.4).

    Concurrency model (LAY-3576 / A6 fix; run-grouping fix)
    -------------------------------------------------------
    AutoGen has no per-run callback: it logs ``LLMCallEvent`` / ``MessageEvent``
    through the **module-global** ``EVENT_LOGGER_NAME`` logger, and those events
    carry no run/topic/session id (``LLMCallEvent`` has only ``agent_id``;
    ``MessageEvent`` only sender/receiver). The old adapter funnelled every run
    into one shared ``self._collector`` — two concurrent ``team.run()`` calls
    merged into one trace.

    The grouping key is the **thread**. Each AgentChat team owns its own
    ``SingleThreadedAgentRuntime`` whose message loop drains on a single thread,
    and the logging handler runs inline on the emitting thread. A single
    ``team.run()`` fans out across many asyncio *tasks* on that one thread, so a
    per-``ContextVar`` run (copied per task) fragments one run into many partial
    traces — the bug this fixes. Keying the ``RunState`` by ``thread ident``
    keeps every task of one run together (one coherent trace) while two runs on
    separate threads stay isolated (the interleaved-run guard). The adapter keeps
    one ``RunState`` per thread in ``self._runs_by_thread`` and flushes each as
    its own trace on ``disconnect()``. A caller-bound ``_current_collector``
    (``instrument()`` / ``capture_events``) is still honoured and reused.

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
        # thread ident -> RunState. One run per thread: a single team.run's
        # asyncio tasks share the thread (one coherent trace); concurrent runs
        # use separate threads (isolated traces).
        self._runs_by_thread: Dict[int, RunState] = {}
        # Fallback slots for callers that drive ``_fire`` outside a run (the
        # cost-pricing invariant test instantiates via ``__new__``).
        self._fallback_collector: Optional[TraceCollector] = None
        self._fallback_root_span_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Per-run state access (resolved by the current thread's run)
    # ------------------------------------------------------------------

    def _run_for_thread(self) -> Optional[RunState]:
        # getattr guard: the cost-pricing invariant test drives _fire on an
        # adapter built via ``__new__`` (no ``__init__``), so ``_runs_by_thread``
        # may be absent. Fall through to the fallback slots instead of crashing,
        # mirroring how the old ``_current_run.get()`` path returned None there.
        runs = getattr(self, "_runs_by_thread", None)
        if not runs:
            return None
        return runs.get(threading.get_ident())

    @property
    def _collector(self) -> Optional[TraceCollector]:
        run = self._run_for_thread()
        if run is not None:
            return run.collector
        return self._fallback_collector

    @_collector.setter
    def _collector(self, value: Optional[TraceCollector]) -> None:
        self._fallback_collector = value

    @property
    def _root_span_id(self) -> Optional[str]:
        run = self._run_for_thread()
        if run is not None:
            return run.root_span_id
        return self._fallback_root_span_id

    @_root_span_id.setter
    def _root_span_id(self, value: Optional[str]) -> None:
        self._fallback_root_span_id = value

    @property
    def _conversations(self) -> Dict[str, Dict[str, Any]]:
        run = self._run_for_thread()
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
            runs = list(self._runs_by_thread.values())
            self._runs_by_thread.clear()
        for run in runs:
            self._flush_run(run)
        self._fallback_collector = None
        self._fallback_root_span_id = None

    # ------------------------------------------------------------------
    # Collector + run-state management
    # ------------------------------------------------------------------

    def _ensure_run(self) -> RunState:
        """Resolve (or lazily open) the run for the CURRENT THREAD.

        A single ``team.run()`` drains on one thread across many asyncio tasks,
        so keying by thread ident keeps all of its events in one run (one
        coherent trace). Concurrent ``team.run()`` calls execute on separate
        threads and therefore get separate runs (the interleaved-run guard).

        Collector resolution
        --------------------
        If a caller has already bound a :class:`TraceCollector` on
        ``_current_collector`` (the canonical pattern used by ``instrument()``
        and ``samples/adapters/_shared.capture_events``), the run reuses **that**
        collector instead of minting a private one. Otherwise the adapter falls
        back to a self-owned collector flushed at ``disconnect()``.
        """
        tid = threading.get_ident()
        run = self._runs_by_thread.get(tid)
        if run is not None:
            return run
        bound = _current_collector.get()
        owns_collector = bound is None
        collector = bound if bound is not None else TraceCollector(self._client, self._config)
        run = RunState(collector=collector, root_span_id=self._new_span_id(), data={"conversations": {}})
        # Only this adapter flushes collectors it owns; a caller-bound collector
        # is flushed by its owner (``capture_events`` / ``instrument()``).
        run.data["owns_collector"] = owns_collector
        with self._lock:
            self._runs_by_thread[tid] = run
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
        name = _autogen_agent_name(agent_id)
        if name:
            # Honest graph-node identity: the calling agent's real name (team
            # runtime uuid stripped, plumbing dropped) so the graph engine
            # attributes this model call to a clean agent node.
            payload["agent_id"] = name
            payload["agent_name"] = name

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

        # autogen logs each message at both the SEND and DELIVER stages; the
        # ateam autogen contract translates only SEND so a single message is
        # not double-counted (a group chat logs ~2x MessageEvents otherwise).
        stage_str = _enum_name(stage) if stage is not None else ""
        if stage_str and "SEND" not in stage_str.upper():
            return

        kind_str = _enum_name(kind) if kind is not None else ""
        is_respond = "RESPOND" in kind_str
        # Honest graph-node identity: the acting agent's real name — the
        # responder on RESPOND, else the receiver processing the input. Runtime
        # plumbing (group-chat manager, group_topic/output_topic broadcast
        # targets) resolves to None. Such a message is pub/sub plumbing, not an
        # agent turn, so it is skipped rather than emitted as an unattributed
        # event — this is what keeps a RoundRobinGroupChat trace to its real
        # per-agent turns instead of the runtime's dozens of broadcast deliveries.
        acting_name = _autogen_agent_name(sender if is_respond else receiver)
        if acting_name is None:
            return

        # Conversation tracking (real-agent turns only), grouped by topic/session
        # so downstream analysis can reason about multi-agent turn-taking.
        topic_id = _get_field(event, "topic_id") or _get_field(event, "session_id")
        conv_id = str(topic_id) if topic_id is not None else f"{sender}->{receiver}"
        conversations = self._conversations
        state = conversations.setdefault(
            conv_id,
            {
                "participants": set(),
                "turn_count": 0,
                "message_count": 0,
                "last_actor": None,
            },
        )
        # Participants are recorded by honest name (plumbing endpoints dropped).
        for who in (sender, receiver):
            honest = _autogen_agent_name(who)
            if honest:
                state["participants"].add(honest)
        state["message_count"] += 1
        last = state["last_actor"]
        if last is not None and acting_name != last:
            state["turn_count"] += 1
        state["last_actor"] = acting_name

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

        payload["agent_name"] = acting_name
        if is_respond:
            self._fire("agent.output", payload)
        else:
            self._fire("agent.input", payload)

        # Topology edge: a message between two DIFFERENT honest agents is a
        # handoff the graph engine renders. Emitted once per (conversation,
        # from->to); self-loops and generic containers are skipped. This restores
        # the multi-agent topology the ateam autogen adapter emitted (which the
        # SDK rewrite had dropped), so the trace no longer renders blank.
        from_t = _autogen_agent_name(sender)
        to_t = _autogen_agent_name(receiver)
        if from_t and to_t and from_t != to_t:
            seen = state.setdefault("handoffs", set())
            if (from_t, to_t) not in seen:
                seen.add((from_t, to_t))
                self._fire(
                    "agent.handoff",
                    self._payload(from_agent=from_t, to_agent=to_t, conversation_id=conv_id),
                )

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
            status="error",  # uniform agent.error shape across all 3 paths (S20e)
        )
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
            at = _autogen_agent_name(agent_id)
            if at:
                payload["agent_name"] = at
        self._fire("agent.error", payload)

    def _on_construction_exception(self, event: Any) -> None:
        agent_id = _get_field(event, "agent_id")
        exc = _get_field(event, "exception")
        payload = self._payload(
            error=str(exc) if exc else "construction failed",
            error_type=(type(exc).__name__ if isinstance(exc, BaseException) else "Exception"),
            status="error",  # uniform agent.error shape across all 3 paths (S20e)
        )
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
            at = _autogen_agent_name(agent_id)
            if at:
                payload["agent_name"] = at
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
