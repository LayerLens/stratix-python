from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Tuple, Optional

from ._utils import safe_serialize
from ._handoff import scrub_context
from ..._context import RunState
from ..._collector import TraceCollector
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig
from ....attestation._hash import compute_hash

log = logging.getLogger(__name__)

# CrewAI's built-in delegation tools (case-insensitive substring match).
# https://docs.crewai.com/concepts/agents#agent-delegation
_DELEGATION_TOOL_PATTERNS: Tuple[str, ...] = (
    "delegate work to coworker",
    "ask question to coworker",
)

# crewai event ``type`` strings that open a new run root (a per-kickoff /
# per-flow trace). These are the only events that create a RunState; every
# other event resolves its run by walking the event lineage up to one of these.
_ROOT_EVENT_TYPES: Tuple[str, ...] = ("crew_kickoff_started", "flow_started")


def _is_delegation_tool(tool_name: Optional[str]) -> bool:
    if not tool_name:
        return False
    low = tool_name.lower()
    return any(pat in low for pat in _DELEGATION_TOOL_PATTERNS)


try:
    from crewai.events import (
        BaseEventListener as _BaseEventListener,
    )  # pyright: ignore[reportMissingImports]
except (ImportError, TypeError):
    _BaseEventListener = None


class CrewAIAdapter(FrameworkAdapter):
    """CrewAI adapter using the typed event bus API (crewai >= 1.0).

    CrewAI's event bus dispatches every handler through a fresh
    ``contextvars.copy_context()`` on a thread-pool worker, so a ContextVar
    set in one handler is invisible to the next — the standard
    ``_begin_run``/ContextVar ``RunState`` isolation used by the other
    framework adapters is impossible here.

    Instead the bus stamps event *lineage* (``event_id`` /
    ``parent_event_id`` / ``started_event_id``) synchronously in the emitting
    thread, *before* the thread-pool dispatch (``crewai/events/event_bus.py``
    ``_prepare_event``). The lineage therefore lives ON THE EVENT and is immune
    to ``copy_context``. This adapter keeps a LOCKED per-run map
    (``self._runs`` keyed by the root event_id) and resolves each event to its
    owning run by walking that lineage — so two concurrent crew kickoffs
    through ONE shared adapter never corrupt each other's traces (LAY-3576).

    Usage::

        adapter = CrewAIAdapter(client)
        adapter.connect()
        crew.kickoff()
        adapter.disconnect()
    """

    name = "crewai"
    # CrewAI 0.30+ is Pydantic v2-only (LAY-3447 catalog manifest AC).
    requires_pydantic: str = "2"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._registered_handlers: list = []
        # run_key (root event_id) -> RunState. All run state lives here, so two
        # concurrent kickoffs through one adapter are fully isolated.
        self._runs: Dict[str, RunState] = {}
        # event_id -> resolved run_key memo (so we walk lineage at most once
        # per event and can evict an entire run's memo on completion).
        self._event_root: Dict[str, str] = {}

    @staticmethod
    def _llm_timer_key(event: Any) -> str:
        """Stable timer key for an LLM call.

        Uses ``call_id`` when present (older crewai versions), otherwise
        falls back to ``agent_id``/``task_id`` (newer versions dropped
        ``call_id``). We deliberately keep a single key when none of these
        are present — LLM calls within a crew are serial, so the matching
        start/complete event pair shares the key.
        """
        call_id = getattr(event, "call_id", None)
        if call_id:
            return f"llm:{call_id}"
        agent_id = getattr(event, "agent_id", None)
        task_id = getattr(event, "task_id", None)
        if agent_id or task_id:
            return f"llm:{agent_id}:{task_id}"
        return "llm:current"

    _EVENT_MAP = [
        ("CrewKickoffStartedEvent", "_on_crew_started"),
        ("CrewKickoffCompletedEvent", "_on_crew_completed"),
        ("CrewKickoffFailedEvent", "_on_crew_failed"),
        ("TaskStartedEvent", "_on_task_started"),
        ("TaskCompletedEvent", "_on_task_completed"),
        ("TaskFailedEvent", "_on_task_failed"),
        ("AgentExecutionStartedEvent", "_on_agent_execution_started"),
        ("AgentExecutionCompletedEvent", "_on_agent_execution_completed"),
        ("AgentExecutionErrorEvent", "_on_agent_execution_error"),
        ("LLMCallStartedEvent", "_on_llm_started"),
        ("LLMCallCompletedEvent", "_on_llm_completed"),
        ("LLMCallFailedEvent", "_on_llm_failed"),
        ("ToolUsageStartedEvent", "_on_tool_started"),
        ("ToolUsageFinishedEvent", "_on_tool_finished"),
        ("ToolUsageErrorEvent", "_on_tool_error"),
        ("FlowStartedEvent", "_on_flow_started"),
        ("FlowFinishedEvent", "_on_flow_finished"),
    ]

    # Optional events — the class names below are not present in every crewai
    # release the adapter supports (crewai>=0.30.0). MCP tool events and the
    # delegation events were each introduced in later versions, so we subscribe
    # to them best-effort at connect time and skip any the installed version
    # does not expose (e.g. crewai 0.193.2 ships no MCP* events).
    _OPTIONAL_EVENT_MAP = [
        ("MCPToolExecutionCompletedEvent", "_on_mcp_tool_completed"),
        ("MCPToolExecutionFailedEvent", "_on_mcp_tool_failed"),
        ("AgentDelegationStartedEvent", "_on_delegation_started"),
        ("AgentDelegationCompletedEvent", "_on_delegation_completed"),
        ("DelegationEvent", "_on_delegation_started"),
    ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_BaseEventListener is not None)
        self._subscribe()

    def _on_disconnect(self) -> None:
        self._unsubscribe()
        self._registered_handlers.clear()
        self._drain_all_runs()

    def _subscribe(self) -> None:
        import crewai.events as ev  # pyright: ignore[reportMissingImports]

        # crewai >=1.x inspects the handler's param count and passes a
        # third `state` positional when there are 3 params, which would
        # silently clobber a default-arg closure. Bind via a factory so
        # the visible signature is exactly (source, event).
        def _make_handler(target):
            def _handler(source: Any, event: Any) -> None:
                try:
                    target(source, event)
                except Exception:
                    log.warning("layerlens: error in CrewAI event handler", exc_info=True)

            return _handler

        for event_name, method_name in self._EVENT_MAP:
            event_cls = getattr(ev, event_name)
            method = getattr(self, method_name)
            handler = _make_handler(method)
            ev.crewai_event_bus.on(event_cls)(handler)
            self._registered_handlers.append((event_cls, handler))

        # Optional events are not shipped by every supported crewai version —
        # subscribe best-effort and skip any class the installed version lacks.
        for event_name, method_name in self._OPTIONAL_EVENT_MAP:
            event_cls = getattr(ev, event_name, None)
            if event_cls is None:
                continue
            method = getattr(self, method_name)
            handler = _make_handler(method)
            ev.crewai_event_bus.on(event_cls)(handler)
            self._registered_handlers.append((event_cls, handler))

    def _unsubscribe(self) -> None:
        try:
            from crewai.events import (
                crewai_event_bus,
            )  # pyright: ignore[reportMissingImports]
        except ImportError:
            return
        for event_cls, handler in self._registered_handlers:
            try:
                off = getattr(crewai_event_bus, "off", None)
                if off is not None:
                    off(event_cls, handler)
                else:
                    # crewai < 1.14 has no off() API — remove our handler from
                    # the bus's internal registries directly so disconnect
                    # leaves no trace on every supported crewai version.
                    self._remove_handler_directly(crewai_event_bus, event_cls, handler)
            except Exception:
                log.debug(
                    "layerlens: could not unregister %s handler",
                    event_cls.__name__,
                    exc_info=True,
                )

    @staticmethod
    def _remove_handler_directly(bus: Any, event_cls: Any, handler: Any) -> None:
        for attr in ("_sync_handlers", "_async_handlers", "_handlers"):
            registry = getattr(bus, attr, None)
            if not isinstance(registry, dict) or event_cls not in registry:
                continue
            handlers = registry[event_cls]
            if handler not in handlers:
                continue
            if isinstance(handlers, frozenset):
                registry[event_cls] = handlers - {handler}
            elif isinstance(handlers, set):
                handlers.discard(handler)
            elif isinstance(handlers, list):
                registry[event_cls] = [h for h in handlers if h is not handler]

    # ------------------------------------------------------------------
    # Per-run state management (lineage-keyed isolation, LAY-3576)
    # ------------------------------------------------------------------

    def _create_run(self, event: Any) -> RunState:
        """Construct and register a RunState for a root (kickoff/flow) event.

        The run is keyed by the root event's ``event_id`` (its run_key);
        ``self._event_root`` memoizes the event_id -> run_key mapping. Must be
        called under ``self._lock``.
        """
        run_key = self._run_key(event)
        run = RunState(
            collector=TraceCollector(self._client, self._config),
            root_span_id=self._new_span_id(),
        )
        # Per-run bookkeeping previously held in instance scalars.
        data = run.data
        data["task_span_ids"] = {}
        data["current_task_span_id"] = None
        data["agent_span_ids"] = {}
        data["current_agent_span_id"] = None
        data["current_agent_role"] = None
        data["tool_span_ids"] = {}
        data["llm_in_flight_model"] = None
        data["delegation_seq"] = 0
        data["delegation_chain"] = []
        self._runs[run_key] = run
        self._event_root[run_key] = run_key
        return run

    @staticmethod
    def _run_key(event: Any) -> Optional[str]:
        """The event's own lineage id (``event_id``), or None when absent."""
        return getattr(event, "event_id", None)

    @staticmethod
    def _is_root_event(event: Any) -> bool:
        return getattr(event, "type", None) in _ROOT_EVENT_TYPES

    def _resolve_root(self, event: Any) -> Optional[str]:
        """Walk the event lineage to its owning run_key, memoizing as we go.

        crewai stamps lineage synchronously before thread-pool dispatch, so
        every real event carries it: children link via ``parent_event_id``;
        ending events (``*_completed`` / ``*_failed`` / ``*_finished``) carry
        ``parent_event_id`` and/or ``started_event_id``. Walk
        ``parent_event_id`` then fall back to ``started_event_id`` until we
        reach a known run_key. Returns the resolved run_key, or None if the
        event has no usable lineage / its root isn't (or is no longer) active.

        Must be called under ``self._lock``.
        """
        eid = getattr(event, "event_id", None)
        if eid is not None and eid in self._event_root:
            return self._event_root[eid]

        # Walk the lineage chain, collecting the visited ids so we can memoize
        # the whole path onto the resolved root in one pass.
        path: List[str] = []
        seen = set()
        cur_id = eid
        next_id = getattr(event, "parent_event_id", None) or getattr(event, "started_event_id", None)
        resolved: Optional[str] = None
        while next_id is not None and next_id not in seen:
            seen.add(next_id)
            if next_id in self._event_root:
                resolved = self._event_root[next_id]
                break
            if next_id in self._runs:
                resolved = next_id
                break
            # Continue walking only if the parent itself was memoized as a
            # lineage node; absent the parent event object we cannot climb
            # further, so the chain stops here.
            path.append(next_id)
            next_id = None

        if resolved is None:
            return None
        if cur_id is not None:
            self._event_root[cur_id] = resolved
        for node in path:
            self._event_root[node] = resolved
        return resolved

    def _run_for(self, event: Any) -> Optional[RunState]:
        """Resolve the RunState that owns *event* (or None to no-op).

        * Root events (crew kickoff / flow start) create and return a new run.
        * Other events resolve their root by walking the stamped lineage.
        * FALLBACK: events with no usable lineage (the hand-built unit-test
          doubles carry no event_id/parent and source=None) attach to the
          single active run when exactly one exists — safe, because one run
          cannot interleave. With zero runs there's nothing to attach to;
          with >1 run and no lineage we refuse to guess (that would
          reintroduce contamination) — this never happens with real crewai,
          whose every event carries lineage.
        """
        with self._lock:
            if self._is_root_event(event):
                return self._create_run(event)

            run_key = self._resolve_root(event)
            if run_key is not None:
                return self._runs.get(run_key)

            # No usable lineage — fall back to the single active run if unique.
            if len(self._runs) == 1:
                return next(iter(self._runs.values()))
            if not self._runs:
                return None
            log.debug(
                "layerlens: crewai event %r has no usable lineage and %d runs are active; cannot attribute it to a run",
                getattr(event, "type", type(event).__name__),
                len(self._runs),
            )
            return None

    def _evict_run(self, run: RunState) -> Optional[TraceCollector]:
        """Remove *run* and its lineage memo, returning its collector to flush.

        Called on crew/flow completion/failure. Must NOT be called while
        holding ``self._lock`` for the flush (flush is done by the caller
        after releasing the lock).
        """
        with self._lock:
            run_key = None
            for key, candidate in self._runs.items():
                if candidate is run:
                    run_key = key
                    break
            if run_key is None:
                return None
            self._runs.pop(run_key, None)
            # Drop every memo entry pointing at this run so the maps don't grow
            # unbounded across many kickoffs.
            stale = [eid for eid, root in self._event_root.items() if root == run_key]
            for eid in stale:
                self._event_root.pop(eid, None)
        return run.collector

    def _drain_all_runs(self) -> None:
        """Flush + clear every still-open run (disconnect path).

        Mirrors openai_agents ``_flush_pending_runs``: any run whose crew/flow
        never emitted a completion event is uploaded partially rather than
        lost.
        """
        with self._lock:
            collectors = [run.collector for run in self._runs.values()]
            self._runs.clear()
            self._event_root.clear()
        for collector in collectors:
            try:
                collector.flush()
            except Exception:
                log.warning("layerlens: error flushing pending CrewAI run", exc_info=True)

    # ------------------------------------------------------------------
    # Emission + per-run helpers
    # ------------------------------------------------------------------

    def _fire(
        self,
        run: RunState,
        event_type: str,
        payload: Dict[str, Any],
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        span_name: Optional[str] = None,
    ) -> None:
        """Emit directly to the run's collector."""
        run.collector.emit(
            event_type,
            payload,
            span_id=span_id or self._new_span_id(),
            parent_span_id=parent_span_id,
            span_name=span_name,
        )

    @staticmethod
    def _leaf_parent(run: RunState) -> Optional[str]:
        data = run.data
        return data.get("current_agent_span_id") or data.get("current_task_span_id") or run.root_span_id

    @staticmethod
    def _tick(run: RunState, key: str) -> None:
        run.timers[key] = time.time_ns()

    @staticmethod
    def _tock(run: RunState, key: str) -> Optional[float]:
        start = run.timers.pop(key, 0)
        if not start:
            return None
        return (time.time_ns() - start) / 1_000_000

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_name(obj: Any) -> str:
        return getattr(obj, "name", None) or type(obj).__name__

    @staticmethod
    def _get_task_name(event: Any) -> str:
        name = getattr(event, "task_name", None)
        if name:
            return str(name)
        task = getattr(event, "task", None)
        if task:
            return str(getattr(task, "description", None) or getattr(task, "name", ""))[:200]
        return ""

    @staticmethod
    def _tool_key(event: Any) -> str:
        tool_name = getattr(event, "tool_name", None) or ""
        agent_key = getattr(event, "agent_key", None) or ""
        return f"{tool_name}:{agent_key}"

    # ------------------------------------------------------------------
    # Crew lifecycle
    # ------------------------------------------------------------------

    def _on_crew_started(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        span_id = run.root_span_id
        self._tick(run, "crew")
        crew_name = getattr(event, "crew_name", None) or self._get_name(source)
        payload = self._payload(crew_name=crew_name)
        self._set_if_capturing(payload, "input", safe_serialize(getattr(event, "inputs", None)))
        self._fire(
            run,
            "agent.input",
            payload,
            span_id=span_id,
            parent_span_id=None,
            span_name=crew_name,
        )

    def _on_crew_completed(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        latency_ms = self._tock(run, "crew")
        crew_name = getattr(event, "crew_name", None) or self._get_name(source)
        span_id = run.root_span_id
        payload = self._payload(crew_name=crew_name)
        if latency_ms is not None:
            payload["duration_ns"] = int(latency_ms * 1_000_000)
        self._set_if_capturing(payload, "output", safe_serialize(getattr(event, "output", None)))
        total_tokens = getattr(event, "total_tokens", None)
        if total_tokens is not None:
            payload["tokens_total"] = total_tokens
        self._fire(
            run,
            "agent.output",
            payload,
            span_id=span_id,
            parent_span_id=None,
            span_name=crew_name,
        )
        if total_tokens:
            self._fire(
                run,
                "cost.record",
                self._payload(tokens_total=total_tokens),
                span_id=span_id,
                parent_span_id=None,
            )
        collector = self._evict_run(run)
        if collector is not None:
            collector.flush()

    def _on_crew_failed(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        error = str(getattr(event, "error", "unknown error"))
        crew_name = getattr(event, "crew_name", None) or self._get_name(source)
        span_id = run.root_span_id
        self._fire(
            run,
            "agent.error",
            self._payload(crew_name=crew_name, error=error, error_type="crew_error", status="error"),
            span_id=span_id,
            parent_span_id=None,
            span_name=crew_name,
        )
        collector = self._evict_run(run)
        if collector is not None:
            collector.flush()

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def _on_task_started(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        task_name = self._get_task_name(event)
        span_id = self._new_span_id()
        with self._lock:
            run.data["task_span_ids"][task_name] = span_id
            run.data["current_task_span_id"] = span_id
            parent = run.root_span_id
        agent_role = getattr(event, "agent_role", None)
        payload = self._payload(task_name=task_name)
        if agent_role:
            payload["agent_role"] = agent_role
        if self._config.capture_content:
            context = getattr(event, "context", None)
            if context:
                payload["context"] = str(context)[:500]
        self._fire(
            run,
            "agent.input",
            payload,
            span_id=span_id,
            parent_span_id=parent,
            span_name=f"task:{task_name[:60]}",
        )

    def _on_task_completed(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        task_name = self._get_task_name(event)
        with self._lock:
            span_id = run.data["task_span_ids"].pop(
                task_name, run.data.get("current_task_span_id") or self._new_span_id()
            )
            parent = run.root_span_id
        payload = self._payload(task_name=task_name)
        self._set_if_capturing(payload, "output", safe_serialize(getattr(event, "output", None)))
        self._fire(
            run,
            "agent.output",
            payload,
            span_id=span_id,
            parent_span_id=parent,
            span_name=f"task:{task_name[:60]}",
        )

    def _on_task_failed(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        task_name = self._get_task_name(event)
        with self._lock:
            span_id = run.data["task_span_ids"].pop(
                task_name, run.data.get("current_task_span_id") or self._new_span_id()
            )
            parent = run.root_span_id
        self._fire(
            run,
            "agent.error",
            self._payload(
                task_name=task_name,
                error=str(getattr(event, "error", "unknown error")),
                error_type="task_error",
                status="error",
            ),
            span_id=span_id,
            parent_span_id=parent,
        )

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    def _on_agent_execution_started(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        agent = getattr(event, "agent", None)
        agent_role = (
            getattr(event, "agent_role", None) or (getattr(agent, "role", None) if agent else None) or "unknown"
        )
        span_id = self._new_span_id()
        with self._lock:
            run.data["agent_span_ids"][agent_role] = span_id
            run.data["current_agent_span_id"] = span_id
            run.data["current_agent_role"] = agent_role
            parent = run.data.get("current_task_span_id") or run.root_span_id
        payload = self._payload(agent_role=agent_role)
        # Capture manager-agent context so hierarchical crews are visible.
        allow_delegation = getattr(agent, "allow_delegation", None) if agent else None
        if allow_delegation is not None:
            payload["allow_delegation"] = bool(allow_delegation)
        is_manager = getattr(agent, "is_manager", None) if agent else None
        if is_manager is not None:
            payload["is_manager"] = bool(is_manager)
        tools = getattr(event, "tools", None)
        if tools:
            payload["tools"] = [getattr(t, "name", str(t)) for t in tools]
        if self._config.capture_content:
            task_prompt = getattr(event, "task_prompt", None)
            if task_prompt:
                payload["task_prompt"] = str(task_prompt)[:500]
        self._fire(
            run,
            "agent.input",
            payload,
            span_id=span_id,
            parent_span_id=parent,
            span_name=f"agent:{agent_role[:60]}",
        )

    def _on_agent_execution_completed(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        agent = getattr(event, "agent", None)
        agent_role = (
            getattr(event, "agent_role", None) or (getattr(agent, "role", None) if agent else None) or "unknown"
        )
        with self._lock:
            span_id = run.data["agent_span_ids"].pop(
                agent_role, run.data.get("current_agent_span_id") or self._new_span_id()
            )
            parent = run.data.get("current_task_span_id") or run.root_span_id
            if run.data.get("current_agent_span_id") == span_id:
                run.data["current_agent_span_id"] = None
        payload = self._payload(agent_role=agent_role, status="ok")
        self._set_if_capturing(payload, "output", safe_serialize(getattr(event, "output", None)))
        self._fire(
            run,
            "agent.output",
            payload,
            span_id=span_id,
            parent_span_id=parent,
            span_name=f"agent:{agent_role[:60]}",
        )

    def _on_agent_execution_error(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        agent = getattr(event, "agent", None)
        agent_role = (
            getattr(event, "agent_role", None) or (getattr(agent, "role", None) if agent else None) or "unknown"
        )
        error = str(getattr(event, "error", "unknown error"))
        with self._lock:
            span_id = run.data["agent_span_ids"].pop(
                agent_role, run.data.get("current_agent_span_id") or self._new_span_id()
            )
            parent = run.data.get("current_task_span_id") or run.root_span_id
            if run.data.get("current_agent_span_id") == span_id:
                run.data["current_agent_span_id"] = None
        self._fire(
            run,
            "agent.error",
            self._payload(agent_role=agent_role, error=error, error_type="agent_error", status="error"),
            span_id=span_id,
            parent_span_id=parent,
            span_name=f"agent:{agent_role[:60]}",
        )

    # ------------------------------------------------------------------
    # Delegation / handoff (hierarchical crews)
    # ------------------------------------------------------------------

    def _next_delegation_seq(self, run: RunState, from_agent: str, to_agent: str) -> int:
        """Bump the run's delegation counter and record the (from, to) pair.

        Returns the new sequence number. Bookkeeping protected by ``self._lock``.
        """
        with self._lock:
            run.data["delegation_seq"] += 1
            run.data["delegation_chain"].append((from_agent, to_agent))
            return run.data["delegation_seq"]

    @staticmethod
    def _extract_delegation_args(tool_args: Any) -> Dict[str, Any]:
        """Pull ``task`` / ``context`` / ``coworker`` out of whatever crewai passed in.

        ``tool_args`` may be a dict, a JSON-encoded string, or ``None``.
        """
        if tool_args is None:
            return {}
        if isinstance(tool_args, dict):
            return tool_args
        if isinstance(tool_args, str):
            import json

            try:
                parsed = json.loads(tool_args)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, TypeError):
                pass
        return {}

    def _emit_delegation_from_tool(self, run: RunState, event: Any, tool_name: str, tool_span_id: str) -> None:
        """Emit ``agent.handoff`` for a built-in coworker-delegation tool call.

        Bridges the gap between crewai versions: newer versions fire
        ``AgentDelegationStartedEvent`` which we handle below; older
        versions only emit the tool call, so we synthesize the handoff
        from the tool args.
        """
        tool_args = self._extract_delegation_args(getattr(event, "tool_args", None))
        to_agent = str(tool_args.get("coworker") or "unknown")
        from_agent = run.data.get("current_agent_role") or "unknown"
        seq = self._next_delegation_seq(run, from_agent, to_agent)

        summary = scrub_context(
            {
                "task": tool_args.get("task"),
                "context": tool_args.get("context"),
            }
        )
        payload = self._payload(
            from_agent=from_agent,
            to_agent=to_agent,
            reason="delegation",
            delegation_seq=seq,
            tool_name=tool_name,
        )
        if summary:
            try:
                payload["handoff_context_hash"] = compute_hash(summary)
            except TypeError:
                payload["handoff_context_hash"] = compute_hash({"_repr": repr(summary)})
            if self._config.capture_content:
                payload["context"] = summary
        self._fire(run, "agent.handoff", payload, parent_span_id=tool_span_id)

    def _on_delegation_started(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        from_role = (
            getattr(event, "from_agent", None)
            or getattr(event, "manager_role", None)
            or getattr(event, "source_agent", None)
            or "manager"
        )
        to_role = (
            getattr(event, "to_agent", None)
            or getattr(event, "delegate_role", None)
            or getattr(event, "target_agent", None)
            or "worker"
        )
        seq = self._next_delegation_seq(run, str(from_role), str(to_role))
        task_name = self._get_task_name(event) or getattr(event, "description", "") or ""
        payload = self._payload(
            from_agent=str(from_role),
            to_agent=str(to_role),
            phase="start",
            reason="delegation",
            delegation_seq=seq,
        )
        if task_name:
            payload["task"] = str(task_name)[:200]
        self._set_if_capturing(payload, "context", safe_serialize(getattr(event, "context", None)))
        self._fire(run, "agent.handoff", payload, parent_span_id=self._leaf_parent(run))

    def _on_delegation_completed(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        from_role = getattr(event, "from_agent", None) or getattr(event, "manager_role", None) or "manager"
        to_role = getattr(event, "to_agent", None) or getattr(event, "delegate_role", None) or "worker"
        payload = self._payload(from_agent=str(from_role), to_agent=str(to_role), phase="complete")
        self._set_if_capturing(payload, "result", safe_serialize(getattr(event, "result", None)))
        self._fire(run, "agent.handoff", payload, parent_span_id=self._leaf_parent(run))

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _on_llm_started(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        key = self._llm_timer_key(event)
        self._tick(run, key)
        # Remember the model for the paired completed/failed event, which in
        # newer crewai drops ``call_id`` and may also drop ``model`` on failure.
        with self._lock:
            run.data["llm_in_flight_model"] = getattr(event, "model", None)

    def _on_llm_completed(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        model = getattr(event, "model", None) or run.data.get("llm_in_flight_model")
        response = getattr(event, "response", None)
        usage = (
            getattr(response, "usage", None)
            if response and not isinstance(response, dict)
            else (response.get("usage") if isinstance(response, dict) else None)
        )
        tokens = self._normalize_tokens(usage)
        payload = self._payload()
        if model:
            payload["model"] = model
        key = self._llm_timer_key(event)
        latency_ms = self._tock(run, key)
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        payload.update(tokens)
        parent = self._leaf_parent(run)
        span_id = self._new_span_id()
        self._fire(run, "model.invoke", payload, span_id=span_id, parent_span_id=parent)
        if tokens:
            self._fire(
                run,
                "cost.record",
                self._payload(model=model, **tokens),
                span_id=span_id,
                parent_span_id=parent,
            )
        with self._lock:
            run.data["llm_in_flight_model"] = None

    def _on_llm_failed(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        error = str(getattr(event, "error", "unknown error"))
        model = getattr(event, "model", None) or run.data.get("llm_in_flight_model")
        payload = self._payload(error=error, error_type="llm_error", status="error")
        if model:
            payload["model"] = model
        self._fire(run, "agent.error", payload, parent_span_id=self._leaf_parent(run))

    # ------------------------------------------------------------------
    # Tool usage
    # ------------------------------------------------------------------

    def _on_tool_started(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        tool_name = getattr(event, "tool_name", None) or "unknown"
        span_id = self._new_span_id()
        key = self._tool_key(event)
        with self._lock:
            run.data["tool_span_ids"][key] = span_id
        payload = self._payload(tool_name=tool_name)
        self._set_if_capturing(payload, "input", safe_serialize(getattr(event, "tool_args", None)))
        self._fire(run, "tool.call", payload, span_id=span_id, parent_span_id=self._leaf_parent(run))

        # Detect delegation invoked via the built-in coworker tools — older
        # crewai versions don't fire typed delegation events, so without this
        # the handoff is invisible in the trace.
        if _is_delegation_tool(tool_name):
            self._emit_delegation_from_tool(run, event, tool_name, span_id)

    def _on_tool_finished(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        tool_name = getattr(event, "tool_name", None) or "unknown"
        key = self._tool_key(event)
        with self._lock:
            span_id = run.data["tool_span_ids"].pop(key, None)
        if span_id is None:
            span_id = self._new_span_id()
        payload = self._payload(tool_name=tool_name)
        self._set_if_capturing(payload, "output", safe_serialize(getattr(event, "output", None)))
        started_at = getattr(event, "started_at", None)
        finished_at = getattr(event, "finished_at", None)
        if started_at is not None and finished_at is not None:
            try:
                payload["latency_ms"] = (finished_at - started_at).total_seconds() * 1000
            except Exception:
                pass
        if getattr(event, "from_cache", None):
            payload["from_cache"] = True
        self._fire(run, "tool.result", payload, span_id=span_id, parent_span_id=self._leaf_parent(run))

    def _on_tool_error(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        tool_name = getattr(event, "tool_name", None) or "unknown"
        error = str(getattr(event, "error", "unknown error"))
        key = self._tool_key(event)
        with self._lock:
            run.data["tool_span_ids"].pop(key, None)
        self._fire(
            run,
            "agent.error",
            self._payload(tool_name=tool_name, error=error, error_type="tool_error", status="error"),
            parent_span_id=self._leaf_parent(run),
        )

    # ------------------------------------------------------------------
    # Flow events
    # ------------------------------------------------------------------

    def _on_flow_started(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        span_id = run.root_span_id
        self._tick(run, "crew")
        flow_name = getattr(event, "flow_name", None) or self._get_name(source)
        payload = self._payload(flow_name=flow_name)
        self._set_if_capturing(payload, "input", safe_serialize(getattr(event, "inputs", None)))
        self._fire(
            run,
            "agent.input",
            payload,
            span_id=span_id,
            parent_span_id=None,
            span_name=f"flow:{flow_name}",
        )

    def _on_flow_finished(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        latency_ms = self._tock(run, "crew")
        flow_name = getattr(event, "flow_name", None) or self._get_name(source)
        span_id = run.root_span_id
        payload = self._payload(flow_name=flow_name)
        if latency_ms is not None:
            payload["duration_ns"] = int(latency_ms * 1_000_000)
        self._set_if_capturing(payload, "output", safe_serialize(getattr(event, "result", None)))
        self._fire(
            run,
            "agent.output",
            payload,
            span_id=span_id,
            parent_span_id=None,
            span_name=f"flow:{flow_name}",
        )
        collector = self._evict_run(run)
        if collector is not None:
            collector.flush()

    # ------------------------------------------------------------------
    # MCP tool events
    # ------------------------------------------------------------------

    def _on_mcp_tool_completed(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        tool_name = getattr(event, "tool_name", None) or "unknown"
        server_name = getattr(event, "server_name", None)
        latency_ms = getattr(event, "execution_duration_ms", None)
        payload = self._payload(tool_name=tool_name)
        self._set_if_capturing(payload, "output", safe_serialize(getattr(event, "result", None)))
        if server_name:
            payload["mcp_server"] = server_name
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        self._fire(run, "tool.call", payload, parent_span_id=self._leaf_parent(run))

    def _on_mcp_tool_failed(self, source: Any, event: Any) -> None:
        run = self._run_for(event)
        if run is None:
            return
        tool_name = getattr(event, "tool_name", None) or "unknown"
        error = str(getattr(event, "error", "unknown error"))
        server_name = getattr(event, "server_name", None)
        payload = self._payload(tool_name=tool_name, error=error)
        payload["error_type"] = "mcp_tool_error"
        payload["status"] = "error"
        if server_name:
            payload["mcp_server"] = server_name
        self._fire(run, "agent.error", payload, parent_span_id=self._leaf_parent(run))
