from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional

from ._utils import safe_serialize
from ..._collector import TraceCollector
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    from crewai.events import BaseEventListener as _BaseEventListener  # pyright: ignore[reportMissingImports]
except (ImportError, TypeError):
    _BaseEventListener = None


class CrewAIAdapter(FrameworkAdapter):
    """CrewAI adapter using the typed event bus API (crewai >= 1.0).

    CrewAI's event bus dispatches handlers across threads, so this
    adapter manages its own collector and span state on the instance
    rather than using ContextVar-based RunState.

    Usage::

        adapter = CrewAIAdapter(client)
        adapter.connect()
        crew.kickoff()
        adapter.disconnect()
    """

    name = "crewai"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._registered_handlers: list = []
        self._collector: Optional[TraceCollector] = None
        self._crew_span_id: Optional[str] = None
        self._task_span_ids: Dict[str, str] = {}
        self._current_task_span_id: Optional[str] = None
        self._agent_span_ids: Dict[str, str] = {}
        self._current_agent_span_id: Optional[str] = None
        self._tool_span_ids: Dict[str, str] = {}
        self._timers: Dict[str, int] = {}

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
        ("MCPToolExecutionCompletedEvent", "_on_mcp_tool_completed"),
        ("MCPToolExecutionFailedEvent", "_on_mcp_tool_failed"),
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
        self._end_trace()

    def _subscribe(self) -> None:
        import crewai.events as ev  # pyright: ignore[reportMissingImports]

        for event_name, method_name in self._EVENT_MAP:
            event_cls = getattr(ev, event_name)
            method = getattr(self, method_name)

            def _handler(source: Any, event: Any, _m: Any = method) -> None:
                try:
                    _m(source, event)
                except Exception:
                    log.warning("layerlens: error in CrewAI event handler", exc_info=True)

            ev.crewai_event_bus.on(event_cls)(_handler)
            self._registered_handlers.append((event_cls, _handler))

    def _unsubscribe(self) -> None:
        try:
            from crewai.events import crewai_event_bus  # pyright: ignore[reportMissingImports]
        except ImportError:
            return
        for event_cls, handler in self._registered_handlers:
            try:
                crewai_event_bus.off(event_cls, handler)
            except Exception:
                log.debug("layerlens: could not unregister %s handler", event_cls.__name__, exc_info=True)

    # ------------------------------------------------------------------
    # Collector + state management
    # ------------------------------------------------------------------

    def _fire(
        self,
        event_type: str,
        payload: Dict[str, Any],
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        span_name: Optional[str] = None,
    ) -> None:
        """Emit directly to the instance collector."""
        c = self._collector
        if c is None:
            return
        c.emit(
            event_type,
            payload,
            span_id=span_id or self._new_span_id(),
            parent_span_id=parent_span_id,
            span_name=span_name,
        )

    def _leaf_parent(self) -> Optional[str]:
        return self._current_agent_span_id or self._current_task_span_id or self._crew_span_id

    def _tick(self, key: str) -> None:
        self._timers[key] = time.time_ns()

    def _tock(self, key: str) -> Optional[float]:
        start = self._timers.pop(key, 0)
        if not start:
            return None
        return (time.time_ns() - start) / 1_000_000

    def _end_trace(self) -> None:
        with self._lock:
            collector = self._collector
            self._collector = None
            self._crew_span_id = None
            self._task_span_ids.clear()
            self._current_task_span_id = None
            self._agent_span_ids.clear()
            self._current_agent_span_id = None
            self._tool_span_ids.clear()
            self._timers.clear()
        if collector is not None:
            collector.flush()

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
        span_id = self._new_span_id()
        with self._lock:
            self._collector = TraceCollector(self._client, self._config)
            self._crew_span_id = span_id
        self._tick("crew")
        crew_name = getattr(event, "crew_name", None) or self._get_name(source)
        payload = self._payload(crew_name=crew_name)
        self._set_if_capturing(payload, "input", safe_serialize(getattr(event, "inputs", None)))
        self._fire("agent.input", payload, span_id=span_id, parent_span_id=None, span_name=crew_name)

    def _on_crew_completed(self, source: Any, event: Any) -> None:
        latency_ms = self._tock("crew")
        crew_name = getattr(event, "crew_name", None) or self._get_name(source)
        span_id = self._crew_span_id or self._new_span_id()
        payload = self._payload(crew_name=crew_name)
        if latency_ms is not None:
            payload["duration_ns"] = int(latency_ms * 1_000_000)
        self._set_if_capturing(payload, "output", safe_serialize(getattr(event, "output", None)))
        total_tokens = getattr(event, "total_tokens", None)
        if total_tokens is not None:
            payload["tokens_total"] = total_tokens
        self._fire("agent.output", payload, span_id=span_id, parent_span_id=None, span_name=crew_name)
        if total_tokens:
            self._fire("cost.record", self._payload(tokens_total=total_tokens), span_id=span_id, parent_span_id=None)
        self._end_trace()

    def _on_crew_failed(self, source: Any, event: Any) -> None:
        error = str(getattr(event, "error", "unknown error"))
        crew_name = getattr(event, "crew_name", None) or self._get_name(source)
        span_id = self._crew_span_id or self._new_span_id()
        self._fire(
            "agent.error",
            self._payload(crew_name=crew_name, error=error),
            span_id=span_id,
            parent_span_id=None,
            span_name=crew_name,
        )
        self._end_trace()

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def _on_task_started(self, source: Any, event: Any) -> None:
        task_name = self._get_task_name(event)
        span_id = self._new_span_id()
        with self._lock:
            self._task_span_ids[task_name] = span_id
            self._current_task_span_id = span_id
            parent = self._crew_span_id
        agent_role = getattr(event, "agent_role", None)
        payload = self._payload(task_name=task_name)
        if agent_role:
            payload["agent_role"] = agent_role
        if self._config.capture_content:
            context = getattr(event, "context", None)
            if context:
                payload["context"] = str(context)[:500]
        self._fire("agent.input", payload, span_id=span_id, parent_span_id=parent, span_name=f"task:{task_name[:60]}")

    def _on_task_completed(self, source: Any, event: Any) -> None:
        task_name = self._get_task_name(event)
        with self._lock:
            span_id = self._task_span_ids.pop(task_name, self._current_task_span_id or self._new_span_id())
            parent = self._crew_span_id
        payload = self._payload(task_name=task_name)
        self._set_if_capturing(payload, "output", safe_serialize(getattr(event, "output", None)))
        self._fire("agent.output", payload, span_id=span_id, parent_span_id=parent, span_name=f"task:{task_name[:60]}")

    def _on_task_failed(self, source: Any, event: Any) -> None:
        task_name = self._get_task_name(event)
        with self._lock:
            span_id = self._task_span_ids.pop(task_name, self._current_task_span_id or self._new_span_id())
            parent = self._crew_span_id
        self._fire(
            "agent.error",
            self._payload(task_name=task_name, error=str(getattr(event, "error", "unknown error"))),
            span_id=span_id,
            parent_span_id=parent,
        )

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    def _on_agent_execution_started(self, source: Any, event: Any) -> None:
        agent = getattr(event, "agent", None)
        agent_role = (
            getattr(event, "agent_role", None) or (getattr(agent, "role", None) if agent else None) or "unknown"
        )
        span_id = self._new_span_id()
        with self._lock:
            self._agent_span_ids[agent_role] = span_id
            self._current_agent_span_id = span_id
            parent = self._current_task_span_id or self._crew_span_id
        payload = self._payload(agent_role=agent_role)
        tools = getattr(event, "tools", None)
        if tools:
            payload["tools"] = [getattr(t, "name", str(t)) for t in tools]
        if self._config.capture_content:
            task_prompt = getattr(event, "task_prompt", None)
            if task_prompt:
                payload["task_prompt"] = str(task_prompt)[:500]
        self._fire("agent.input", payload, span_id=span_id, parent_span_id=parent, span_name=f"agent:{agent_role[:60]}")

    def _on_agent_execution_completed(self, source: Any, event: Any) -> None:
        agent = getattr(event, "agent", None)
        agent_role = (
            getattr(event, "agent_role", None) or (getattr(agent, "role", None) if agent else None) or "unknown"
        )
        with self._lock:
            span_id = self._agent_span_ids.pop(agent_role, self._current_agent_span_id or self._new_span_id())
            parent = self._current_task_span_id or self._crew_span_id
            if self._current_agent_span_id == span_id:
                self._current_agent_span_id = None
        payload = self._payload(agent_role=agent_role, status="ok")
        self._set_if_capturing(payload, "output", safe_serialize(getattr(event, "output", None)))
        self._fire(
            "agent.output", payload, span_id=span_id, parent_span_id=parent, span_name=f"agent:{agent_role[:60]}"
        )

    def _on_agent_execution_error(self, source: Any, event: Any) -> None:
        agent = getattr(event, "agent", None)
        agent_role = (
            getattr(event, "agent_role", None) or (getattr(agent, "role", None) if agent else None) or "unknown"
        )
        error = str(getattr(event, "error", "unknown error"))
        with self._lock:
            span_id = self._agent_span_ids.pop(agent_role, self._current_agent_span_id or self._new_span_id())
            parent = self._current_task_span_id or self._crew_span_id
            if self._current_agent_span_id == span_id:
                self._current_agent_span_id = None
        self._fire(
            "agent.error",
            self._payload(agent_role=agent_role, error=error),
            span_id=span_id,
            parent_span_id=parent,
            span_name=f"agent:{agent_role[:60]}",
        )

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _on_llm_started(self, source: Any, event: Any) -> None:
        call_id = getattr(event, "call_id", None)
        if call_id:
            self._tick(f"llm:{call_id}")

    def _on_llm_completed(self, source: Any, event: Any) -> None:
        model = getattr(event, "model", None)
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
        call_id = getattr(event, "call_id", None)
        if call_id:
            latency_ms = self._tock(f"llm:{call_id}")
            if latency_ms is not None:
                payload["latency_ms"] = latency_ms
        payload.update(tokens)
        parent = self._leaf_parent()
        span_id = self._new_span_id()
        self._fire("model.invoke", payload, span_id=span_id, parent_span_id=parent)
        if tokens:
            self._fire("cost.record", self._payload(model=model, **tokens), span_id=span_id, parent_span_id=parent)

    def _on_llm_failed(self, source: Any, event: Any) -> None:
        error = str(getattr(event, "error", "unknown error"))
        model = getattr(event, "model", None)
        payload = self._payload(error=error)
        if model:
            payload["model"] = model
        self._fire("agent.error", payload, parent_span_id=self._leaf_parent())

    # ------------------------------------------------------------------
    # Tool usage
    # ------------------------------------------------------------------

    def _on_tool_started(self, source: Any, event: Any) -> None:
        tool_name = getattr(event, "tool_name", None) or "unknown"
        span_id = self._new_span_id()
        key = self._tool_key(event)
        with self._lock:
            self._tool_span_ids[key] = span_id
        payload = self._payload(tool_name=tool_name)
        self._set_if_capturing(payload, "input", safe_serialize(getattr(event, "tool_args", None)))
        self._fire("tool.call", payload, span_id=span_id, parent_span_id=self._leaf_parent())

    def _on_tool_finished(self, source: Any, event: Any) -> None:
        tool_name = getattr(event, "tool_name", None) or "unknown"
        key = self._tool_key(event)
        with self._lock:
            span_id = self._tool_span_ids.pop(key, None)
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
        self._fire("tool.result", payload, span_id=span_id, parent_span_id=self._leaf_parent())

    def _on_tool_error(self, source: Any, event: Any) -> None:
        tool_name = getattr(event, "tool_name", None) or "unknown"
        error = str(getattr(event, "error", "unknown error"))
        key = self._tool_key(event)
        with self._lock:
            self._tool_span_ids.pop(key, None)
        self._fire("agent.error", self._payload(tool_name=tool_name, error=error), parent_span_id=self._leaf_parent())

    # ------------------------------------------------------------------
    # Flow events
    # ------------------------------------------------------------------

    def _on_flow_started(self, source: Any, event: Any) -> None:
        span_id = self._new_span_id()
        with self._lock:
            self._collector = TraceCollector(self._client, self._config)
            self._crew_span_id = span_id
        self._tick("crew")
        flow_name = getattr(event, "flow_name", None) or self._get_name(source)
        payload = self._payload(flow_name=flow_name)
        self._set_if_capturing(payload, "input", safe_serialize(getattr(event, "inputs", None)))
        self._fire("agent.input", payload, span_id=span_id, parent_span_id=None, span_name=f"flow:{flow_name}")

    def _on_flow_finished(self, source: Any, event: Any) -> None:
        latency_ms = self._tock("crew")
        flow_name = getattr(event, "flow_name", None) or self._get_name(source)
        span_id = self._crew_span_id or self._new_span_id()
        payload = self._payload(flow_name=flow_name)
        if latency_ms is not None:
            payload["duration_ns"] = int(latency_ms * 1_000_000)
        self._set_if_capturing(payload, "output", safe_serialize(getattr(event, "result", None)))
        self._fire("agent.output", payload, span_id=span_id, parent_span_id=None, span_name=f"flow:{flow_name}")
        self._end_trace()

    # ------------------------------------------------------------------
    # MCP tool events
    # ------------------------------------------------------------------

    def _on_mcp_tool_completed(self, source: Any, event: Any) -> None:
        tool_name = getattr(event, "tool_name", None) or "unknown"
        server_name = getattr(event, "server_name", None)
        latency_ms = getattr(event, "execution_duration_ms", None)
        payload = self._payload(tool_name=tool_name)
        self._set_if_capturing(payload, "output", safe_serialize(getattr(event, "result", None)))
        if server_name:
            payload["mcp_server"] = server_name
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        self._fire("tool.call", payload, parent_span_id=self._leaf_parent())

    def _on_mcp_tool_failed(self, source: Any, event: Any) -> None:
        tool_name = getattr(event, "tool_name", None) or "unknown"
        error = str(getattr(event, "error", "unknown error"))
        server_name = getattr(event, "server_name", None)
        payload = self._payload(tool_name=tool_name, error=error)
        if server_name:
            payload["mcp_server"] = server_name
        self._fire("agent.error", payload, parent_span_id=self._leaf_parent())
