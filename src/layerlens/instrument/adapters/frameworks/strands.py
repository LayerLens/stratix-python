from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional

from .._base import StateFilter, resilient_callback
from ._utils import safe_serialize
from ..._collector import TraceCollector
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

_HAS_STRANDS = False
try:
    from strands.hooks.events import (  # pyright: ignore[reportMissingImports]
        AfterToolCallEvent as _AfterToolCallEvent,
        AfterModelCallEvent as _AfterModelCallEvent,
        BeforeToolCallEvent as _BeforeToolCallEvent,
        AfterInvocationEvent as _AfterInvocationEvent,
        BeforeModelCallEvent as _BeforeModelCallEvent,
        AgentInitializedEvent as _AgentInitializedEvent,
        BeforeInvocationEvent as _BeforeInvocationEvent,
    )

    _HAS_STRANDS = True
except ImportError:
    pass


class StrandsAdapter(FrameworkAdapter):
    """AWS Strands Agents adapter using the native hook system.

    Implements ``HookProvider`` and registers for all lifecycle events:
    agent init, invocation start/end, model calls, and tool calls.

    Usage::

        adapter = StrandsAdapter(client)
        adapter.connect()

        # Pass the adapter as a hook provider at construction:
        agent = Agent(model=model, hooks=[adapter])
        result = agent("Hello!")

        # Or register on an existing agent:
        adapter.connect(target=agent)
        result = agent("Hello!")

        adapter.disconnect()
    """

    name = "strands"

    def __init__(
        self,
        client: Any,
        capture_config: Optional[CaptureConfig] = None,
        state_filter: Optional[StateFilter] = None,
    ) -> None:
        super().__init__(client, capture_config, state_filter=state_filter)
        self._collector: Optional[TraceCollector] = None
        self._run_span_id: Optional[str] = None
        self._current_agent_name: Optional[str] = None
        self._timers: Dict[str, int] = {}
        self._seen_agents: set = set()
        self._target: Optional[Any] = None
        self._registered_callbacks: list = []
        self._model_span_ids: list = []  # span_ids of emitted model.invoke events

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> Any:
        self._check_dependency(_HAS_STRANDS)
        self._metadata["framework_version"] = _get_version()
        if target is not None:
            self._target = target
            self._register_on_agent(target)
        return target

    def _on_disconnect(self) -> None:
        self._deregister_callbacks()
        self._end_trace()
        self._target = None
        self._seen_agents.clear()

    # ------------------------------------------------------------------
    # HookProvider protocol
    # ------------------------------------------------------------------

    def register_hooks(self, registry: Any) -> None:
        """Called by Strands when this adapter is passed as ``hooks=[adapter]``."""
        self._add_callbacks(registry)

    def _register_on_agent(self, agent: Any) -> None:
        """Register hooks on an existing agent's hook registry."""
        hooks = getattr(agent, "hooks", None)
        if hooks is not None and hasattr(hooks, "add_callback"):
            self._add_callbacks(hooks)

    def _add_callbacks(self, registry: Any) -> None:
        callbacks = [
            (_AgentInitializedEvent, self._on_agent_initialized),
            (_BeforeInvocationEvent, self._on_before_invocation),
            (_AfterInvocationEvent, self._on_after_invocation),
            (_BeforeModelCallEvent, self._on_before_model),
            (_AfterModelCallEvent, self._on_after_model),
            (_BeforeToolCallEvent, self._on_before_tool),
            (_AfterToolCallEvent, self._on_after_tool),
        ]
        for event_type, callback in callbacks:
            if event_type is not None:
                registry.add_callback(event_type, callback)
                self._registered_callbacks.append((event_type, callback))

    def _deregister_callbacks(self) -> None:
        agent = self._target
        if agent is None:
            self._registered_callbacks.clear()
            return
        hooks = getattr(agent, "hooks", None)
        if hooks is None or not hasattr(hooks, "_registered_callbacks"):
            self._registered_callbacks.clear()
            return
        for event_type, callback in self._registered_callbacks:
            cbs = hooks._registered_callbacks.get(event_type, [])
            try:
                cbs.remove(callback)
            except ValueError:
                pass
        self._registered_callbacks.clear()

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
            self._run_span_id = None
            self._current_agent_name = None
            self._timers.clear()
            self._model_span_ids.clear()
        if collector is not None:
            collector.flush()

    # ------------------------------------------------------------------
    # Hook handlers
    # ------------------------------------------------------------------

    @resilient_callback(callback_name="_on_agent_initialized")
    def _on_agent_initialized(self, event: Any) -> None:
        """Sync-only callback fired when an agent is constructed."""
        agent = event.agent
        name = _agent_name(agent)
        self._emit_agent_config(name, agent)

    @resilient_callback(callback_name="_on_before_invocation")
    def _on_before_invocation(self, event: Any) -> None:
        agent = event.agent
        name = _agent_name(agent)
        span_id = self._new_span_id()
        with self._lock:
            self._collector = TraceCollector(self._client, self._config)
            self._run_span_id = span_id
            self._current_agent_name = name
        self._tick("run")

        # Re-emit config if we haven't seen this agent yet
        self._emit_agent_config(name, agent)

        payload = self._payload(agent_name=name)
        model_id = _model_id(agent)
        if model_id:
            payload["model"] = model_id

        messages = getattr(event, "messages", None)
        self._set_if_capturing(payload, "input", safe_serialize(messages))
        self._filter_payload(payload, "input")
        self._fire("agent.input", payload, span_id=span_id, span_name=name)

    @resilient_callback(callback_name="_on_after_invocation")
    def _on_after_invocation(self, event: Any) -> None:
        agent = event.agent
        name = _agent_name(agent)
        latency_ms = self._tock("run")
        span_id = self._run_span_id or self._new_span_id()

        payload = self._payload(agent_name=name)
        if latency_ms is not None:
            payload["duration_ns"] = int(latency_ms * 1_000_000)

        result = getattr(event, "result", None)
        if result is not None:
            stop_reason = getattr(result, "stop_reason", None)
            if stop_reason:
                payload["stop_reason"] = str(stop_reason)

            message = getattr(result, "message", None)
            self._set_if_capturing(payload, "output", safe_serialize(message))

        # Emit per-cycle cost.record events matched to model spans.
        # accumulated_usage updates AFTER AfterModelCallEvent fires,
        # so we read per-cycle tokens here instead.
        self._emit_per_cycle_tokens(agent)

        self._filter_payload(payload, "output")
        self._fire("agent.output", payload, span_id=span_id, span_name=name)
        self._end_trace()

    @resilient_callback(callback_name="_on_before_model")
    def _on_before_model(self, event: Any) -> None:
        agent = event.agent
        name = _agent_name(agent)
        self._tick(f"model:{name}")

    @resilient_callback(callback_name="_on_after_model")
    def _on_after_model(self, event: Any) -> None:
        """Emit model.invoke with timing and error info.

        Per-call token usage is NOT available here — Strands updates
        accumulated_usage AFTER this hook fires.  Tokens are emitted
        per-cycle from _on_after_invocation using the cycle data.
        """
        agent = event.agent
        name = _agent_name(agent)
        latency_ms = self._tock(f"model:{name}")

        model_id = _model_id(agent)
        payload = self._payload()
        if model_id:
            payload["model"] = model_id

        if latency_ms is not None:
            payload["latency_ms"] = latency_ms

        exception = getattr(event, "exception", None)
        if exception is not None:
            payload["error"] = str(exception)
            payload["error_type"] = type(exception).__name__

        stop_response = getattr(event, "stop_response", None)
        if stop_response is not None:
            stop_reason = getattr(stop_response, "stop_reason", None)
            if stop_reason:
                payload["stop_reason"] = str(stop_reason)

        parent = self._run_span_id
        span_id = self._new_span_id()
        self._fire("model.invoke", payload, span_id=span_id, parent_span_id=parent)
        with self._lock:
            self._model_span_ids.append(span_id)

    @resilient_callback(callback_name="_on_before_tool")
    def _on_before_tool(self, event: Any) -> None:
        tool_use = event.tool_use
        tool_name = (
            tool_use.get("name", "unknown") if isinstance(tool_use, dict) else getattr(tool_use, "name", "unknown")
        )
        tool_id = (
            tool_use.get("toolUseId", tool_name)
            if isinstance(tool_use, dict)
            else getattr(tool_use, "toolUseId", tool_name)
        )
        self._tick(f"tool:{tool_id}")

    @resilient_callback(callback_name="_on_after_tool")
    def _on_after_tool(self, event: Any) -> None:
        tool_use = event.tool_use
        tool_name = (
            tool_use.get("name", "unknown") if isinstance(tool_use, dict) else getattr(tool_use, "name", "unknown")
        )
        tool_id = (
            tool_use.get("toolUseId", tool_name)
            if isinstance(tool_use, dict)
            else getattr(tool_use, "toolUseId", tool_name)
        )
        tool_input = tool_use.get("input", None) if isinstance(tool_use, dict) else getattr(tool_use, "input", None)
        latency_ms = self._tock(f"tool:{tool_id}")

        parent = self._run_span_id
        span_id = self._new_span_id()

        call_payload = self._payload(tool_name=tool_name)
        self._set_if_capturing(call_payload, "input", safe_serialize(tool_input))
        if latency_ms is not None:
            call_payload["latency_ms"] = latency_ms
        self._filter_payload(call_payload, "input")
        self._fire("tool.call", call_payload, span_id=span_id, parent_span_id=parent, span_name=f"tool:{tool_name}")

        result = getattr(event, "result", None)
        result_payload = self._payload(tool_name=tool_name)
        if result is not None:
            status = result.get("status", None) if isinstance(result, dict) else getattr(result, "status", None)
            if status:
                result_payload["status"] = str(status)
            content = result.get("content", None) if isinstance(result, dict) else getattr(result, "content", None)
            self._set_if_capturing(result_payload, "output", safe_serialize(content))

        exception = getattr(event, "exception", None)
        if exception is not None:
            result_payload["error"] = str(exception)
            result_payload["error_type"] = type(exception).__name__

        self._filter_payload(result_payload, "output")
        self._fire(
            "tool.result", result_payload, span_id=span_id, parent_span_id=parent, span_name=f"tool:{tool_name}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit_agent_config(self, name: str, agent: Any) -> None:
        with self._lock:
            if name in self._seen_agents:
                return
            self._seen_agents.add(name)

        payload = self._payload(agent_name=name, agent_type=type(agent).__name__)

        mid = _model_id(agent)
        if mid:
            payload["model"] = mid

        system_prompt = getattr(agent, "system_prompt", None)
        if system_prompt and self._config.capture_content:
            payload["system_prompt"] = str(system_prompt)[:500]

        tool_names = getattr(agent, "tool_names", None)
        if tool_names:
            payload["tools"] = list(tool_names)

        self._fire("environment.config", payload, parent_span_id=self._run_span_id, span_name=f"config:{name}")

    def _emit_per_cycle_tokens(self, agent: Any) -> None:
        """Emit cost.record per model call using per-cycle token data.

        Strands stores per-cycle usage at:
            agent.event_loop_metrics.agent_invocations[-1].cycles[i].usage
        Each cycle maps 1:1 with a model call, so we zip cycles with
        the stored ``_model_span_ids`` to attribute tokens correctly.
        """
        model_id = _model_id(agent)
        with self._lock:
            span_ids = list(self._model_span_ids)

        cycles = _get_cycles(agent)
        if not cycles and not span_ids:
            return

        # Zip cycles with model span_ids — if counts differ, emit what we can
        for i, cycle in enumerate(cycles):
            usage = getattr(cycle, "usage", None) if not isinstance(cycle, dict) else cycle.get("usage")
            if usage is None:
                continue
            if isinstance(usage, dict):
                input_t = usage.get("inputTokens", 0)
                output_t = usage.get("outputTokens", 0)
            else:
                input_t = getattr(usage, "inputTokens", 0) or 0
                output_t = getattr(usage, "outputTokens", 0) or 0

            if not input_t and not output_t:
                continue

            tokens: Dict[str, int] = {}
            if input_t:
                tokens["tokens_prompt"] = input_t
            if output_t:
                tokens["tokens_completion"] = output_t
            tokens["tokens_total"] = input_t + output_t

            # Per-cycle timing — Strands stores start/end on each cycle;
            # surface the duration so we can chart tokens/sec per cycle.
            cycle_latency_ms = _cycle_latency_ms(cycle)
            if cycle_latency_ms is not None:
                tokens["cycle_latency_ms"] = int(cycle_latency_ms)
            # Per-cycle stop reason (set when the cycle exits due to e.g.
            # tool_use, end_turn, max_tokens, etc.).
            stop_reason = _cycle_stop_reason(cycle)
            if stop_reason:
                tokens["stop_reason"] = stop_reason

            cost_payload = self._payload(**tokens)
            cost_payload["cycle_index"] = i
            if model_id:
                cost_payload["model"] = model_id

            parent = span_ids[i] if i < len(span_ids) else self._run_span_id
            self._fire("cost.record", cost_payload, parent_span_id=parent)


# -- Module-level helpers --------------------------------------------------


def _get_cycles(agent: Any) -> list:
    """Extract per-cycle data from the most recent invocation.

    Path: agent.event_loop_metrics.agent_invocations[-1].cycles
    """
    try:
        metrics = getattr(agent, "event_loop_metrics", None)
        if metrics is None:
            return []
        invocations = getattr(metrics, "agent_invocations", None)
        if not invocations:
            return []
        last = invocations[-1]
        cycles = getattr(last, "cycles", None)
        return list(cycles) if cycles else []
    except Exception:
        return []


def _agent_name(agent: Any) -> str:
    if agent is None:
        return "unknown"
    return getattr(agent, "name", None) or type(agent).__name__


def _model_id(agent: Any) -> Optional[str]:
    if agent is None:
        return None
    model = getattr(agent, "model", None)
    if model is None:
        return None
    config = getattr(model, "config", None)
    if isinstance(config, dict):
        mid = config.get("model_id")
        if mid:
            return str(mid)
    return str(model) if model else None


def _get_version() -> str:
    try:
        import strands as _mod  # pyright: ignore[reportMissingImports]

        return getattr(_mod, "__version__", "unknown")
    except Exception:
        return "unknown"


def _cycle_latency_ms(cycle: Any) -> Optional[float]:
    if isinstance(cycle, dict):
        start = cycle.get("start_time") or cycle.get("startTime")
        end = cycle.get("end_time") or cycle.get("endTime")
    else:
        start = getattr(cycle, "start_time", None) or getattr(cycle, "startTime", None)
        end = getattr(cycle, "end_time", None) or getattr(cycle, "endTime", None)
    if start is None or end is None:
        return None
    try:
        return (end - start) * 1000 if isinstance(start, (int, float)) else None
    except Exception:
        return None


def _cycle_stop_reason(cycle: Any) -> Optional[str]:
    if isinstance(cycle, dict):
        return cycle.get("stop_reason") or cycle.get("stopReason")
    return getattr(cycle, "stop_reason", None) or getattr(cycle, "stopReason", None)
