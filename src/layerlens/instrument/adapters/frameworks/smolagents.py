from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

from ._base_framework import FrameworkAdapter
from ._utils import safe_serialize
from ..._collector import TraceCollector
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

_HAS_SMOLAGENTS = False
try:
    from smolagents import (  # pyright: ignore[reportMissingImports]
        ActionStep as _ActionStep,
        PlanningStep as _PlanningStep,
        FinalAnswerStep as _FinalAnswerStep,
    )

    _HAS_SMOLAGENTS = True
except ImportError:
    _ActionStep = _PlanningStep = _FinalAnswerStep = None  # type: ignore[assignment,misc]


class SmolAgentsAdapter(FrameworkAdapter):
    """SmoLAgents (HuggingFace) adapter using step callbacks + run wrapper.

    SmoLAgents fires post-step callbacks via ``CallbackRegistry`` on the
    agent's ``step_callbacks``.  This adapter registers for ``ActionStep``,
    ``PlanningStep``, and ``FinalAnswerStep`` to capture per-step detail
    (tool calls, model invocations, planning), and wraps ``agent.run()``
    for the outer lifecycle boundary (collector creation / flush).

    Usage::

        adapter = SmolAgentsAdapter(client)
        agent = adapter.connect(target=agent)
        result = agent.run("Summarise this document.")
        adapter.disconnect()
    """

    name = "smolagents"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._collector: Optional[TraceCollector] = None
        self._run_span_id: Optional[str] = None
        self._current_step_span_id: Optional[str] = None
        self._step_count: int = 0
        self._timers: Dict[str, int] = {}
        self._original_run: Optional[Any] = None
        self._target_agent: Optional[Any] = None
        self._callbacks: List[Any] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> Any:
        self._check_dependency(_HAS_SMOLAGENTS)
        if target is None:
            raise ValueError("SmolAgentsAdapter.connect() requires a target agent.")
        self._target_agent = target
        self._metadata["framework_version"] = _get_version()
        self._wrap_run(target)
        self._register_callbacks(target)
        return target

    def _on_disconnect(self) -> None:
        self._unwrap_run()
        self._deregister_callbacks()
        self._end_trace()
        self._target_agent = None

    # ------------------------------------------------------------------
    # Run wrapper
    # ------------------------------------------------------------------

    def _wrap_run(self, agent: Any) -> None:
        if not hasattr(agent, "run"):
            return
        self._original_run = agent.run
        adapter = self

        def _traced_run(*args: Any, **kwargs: Any) -> Any:
            task = args[0] if args else kwargs.get("task")
            adapter._on_run_start(agent, task)
            error: Optional[Exception] = None
            result: Any = None
            try:
                result = adapter._original_run(*args, **kwargs)
            except Exception as exc:
                error = exc
                adapter._on_run_error(agent, exc)
                raise
            finally:
                adapter._on_run_end(agent, result, error)
            return result

        _traced_run._layerlens_original = self._original_run  # type: ignore[attr-defined]
        agent.run = _traced_run

    def _unwrap_run(self) -> None:
        if self._target_agent is not None and self._original_run is not None:
            try:
                self._target_agent.run = self._original_run
            except Exception:
                log.debug("layerlens: could not unwrap run()", exc_info=True)
        self._original_run = None

    # ------------------------------------------------------------------
    # Step callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self, agent: Any) -> None:
        registry = getattr(agent, "step_callbacks", None)
        if registry is None or not hasattr(registry, "register"):
            return
        for step_cls, method in [
            (_ActionStep, self._on_action_step),
            (_PlanningStep, self._on_planning_step),
            (_FinalAnswerStep, self._on_final_answer_step),
        ]:
            if step_cls is not None:
                registry.register(step_cls, method)
                self._callbacks.append((step_cls, method))

    def _deregister_callbacks(self) -> None:
        agent = self._target_agent
        if agent is None:
            return
        registry = getattr(agent, "step_callbacks", None)
        if registry is None:
            self._callbacks.clear()
            return
        for step_cls, method in self._callbacks:
            cbs = registry._callbacks.get(step_cls, [])
            try:
                cbs.remove(method)
            except ValueError:
                pass
        self._callbacks.clear()

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
            event_type, payload,
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
            self._current_step_span_id = None
            self._step_count = 0
            self._timers.clear()
        if collector is not None:
            collector.flush()

    # ------------------------------------------------------------------
    # Run lifecycle handlers
    # ------------------------------------------------------------------

    def _on_run_start(self, agent: Any, task: Any) -> None:
        span_id = self._new_span_id()
        with self._lock:
            self._collector = TraceCollector(self._client, self._config)
            self._run_span_id = span_id
        self._tick("run")

        agent_name = _agent_name(agent)
        payload = self._payload(agent_name=agent_name, agent_type=type(agent).__name__)

        model_id = _model_id(agent)
        if model_id:
            payload["model"] = model_id

        tools = getattr(agent, "tools", None)
        if tools:
            payload["tools"] = list(tools.keys()) if isinstance(tools, dict) else [getattr(t, "name", str(t)) for t in tools]

        managed = getattr(agent, "managed_agents", None)
        if managed:
            payload["managed_agents"] = list(managed.keys()) if isinstance(managed, dict) else [getattr(a, "name", str(a)) for a in managed]

        self._set_if_capturing(payload, "input", safe_serialize(task))
        self._fire("agent.input", payload, span_id=span_id, span_name=agent_name)

    def _on_run_end(self, agent: Any, result: Any, error: Optional[Exception]) -> None:
        latency_ms = self._tock("run")
        span_id = self._run_span_id or self._new_span_id()
        agent_name = _agent_name(agent)
        payload = self._payload(agent_name=agent_name)
        if latency_ms is not None:
            payload["duration_ns"] = int(latency_ms * 1_000_000)
        if error:
            payload["error"] = str(error)
        self._set_if_capturing(payload, "output", safe_serialize(result))
        self._fire("agent.output", payload, span_id=span_id, span_name=agent_name)
        self._end_trace()

    def _on_run_error(self, agent: Any, exc: Exception) -> None:
        agent_name = _agent_name(agent)
        self._fire(
            "agent.error",
            self._payload(agent_name=agent_name, error=str(exc), error_type=type(exc).__name__),
            parent_span_id=self._run_span_id,
        )

    # ------------------------------------------------------------------
    # Step handlers (registered as step_callbacks)
    # ------------------------------------------------------------------

    def _on_action_step(self, step: Any, agent: Any = None) -> None:
        try:
            self._handle_action_step(step, agent)
        except Exception:
            log.warning("layerlens: error in SmolAgents action step handler", exc_info=True)

    def _on_planning_step(self, step: Any, agent: Any = None) -> None:
        try:
            self._handle_planning_step(step, agent)
        except Exception:
            log.warning("layerlens: error in SmolAgents planning step handler", exc_info=True)

    def _on_final_answer_step(self, step: Any, agent: Any = None) -> None:
        pass  # run wrapper handles final output + flush

    # ------------------------------------------------------------------
    # ActionStep processing
    # ------------------------------------------------------------------

    def _handle_action_step(self, step: Any, agent: Any) -> None:
        self._step_count += 1
        step_span_id = self._new_span_id()
        with self._lock:
            self._current_step_span_id = step_span_id

        model_id = _model_id(agent) if agent else None

        # model.invoke — from token_usage on the step
        token_usage = getattr(step, "token_usage", None)
        if token_usage is not None:
            self._emit_model_invoke(step, model_id, step_span_id)

        # tool calls — from step.tool_calls
        tool_calls = getattr(step, "tool_calls", None)
        if tool_calls:
            self._emit_tool_calls(tool_calls, step, step_span_id)

        # step event
        step_payload = self._payload(step_number=self._step_count)
        if model_id:
            step_payload["model"] = model_id

        timing = getattr(step, "timing", None)
        if timing is not None:
            start = getattr(timing, "start_time", None)
            end = getattr(timing, "end_time", None)
            if start is not None and end is not None:
                step_payload["duration_ns"] = int((end - start) * 1_000_000_000)

        error = getattr(step, "error", None)
        if error is not None:
            step_payload["error"] = str(error)

        is_final = getattr(step, "is_final_answer", False)
        if is_final:
            step_payload["is_final_answer"] = True

        code_action = getattr(step, "code_action", None)
        if code_action and self._config.capture_content:
            step_payload["code_action"] = str(code_action)[:2000]

        self._set_if_capturing(step_payload, "observations", safe_serialize(getattr(step, "observations", None)))
        self._fire("agent.step", step_payload, span_id=step_span_id, parent_span_id=self._run_span_id, span_name=f"step:{self._step_count}")

    def _emit_model_invoke(self, step: Any, model_id: Optional[str], parent_span_id: str) -> None:
        token_usage = getattr(step, "token_usage", None)
        tokens = self._normalize_tokens(token_usage)
        payload = self._payload()
        if model_id:
            payload["model"] = model_id
        payload.update(tokens)
        span_id = self._new_span_id()
        self._fire("model.invoke", payload, span_id=span_id, parent_span_id=parent_span_id)
        if tokens:
            cost_payload = self._payload(**tokens)
            if model_id:
                cost_payload["model"] = model_id
            self._fire("cost.record", cost_payload, span_id=span_id, parent_span_id=parent_span_id)

    def _emit_tool_calls(self, tool_calls: List[Any], step: Any, parent_span_id: str) -> None:
        observations = getattr(step, "observations", None) or ""
        for tc in tool_calls:
            name = getattr(tc, "name", None) or "unknown"
            if name == "final_answer":
                continue
            span_id = self._new_span_id()
            call_payload = self._payload(tool_name=name)
            self._set_if_capturing(call_payload, "input", safe_serialize(getattr(tc, "arguments", None)))
            self._fire("tool.call", call_payload, span_id=span_id, parent_span_id=parent_span_id)
            result_payload = self._payload(tool_name=name)
            self._set_if_capturing(result_payload, "output", safe_serialize(observations))
            self._fire("tool.result", result_payload, span_id=span_id, parent_span_id=parent_span_id)

    # ------------------------------------------------------------------
    # PlanningStep processing
    # ------------------------------------------------------------------

    def _handle_planning_step(self, step: Any, agent: Any) -> None:
        span_id = self._new_span_id()
        model_id = _model_id(agent) if agent else None

        payload = self._payload()
        if model_id:
            payload["model"] = model_id

        timing = getattr(step, "timing", None)
        if timing is not None:
            start = getattr(timing, "start_time", None)
            end = getattr(timing, "end_time", None)
            if start is not None and end is not None:
                payload["duration_ns"] = int((end - start) * 1_000_000_000)

        self._set_if_capturing(payload, "plan", safe_serialize(getattr(step, "plan", None)))
        self._fire("agent.step", payload, span_id=span_id, parent_span_id=self._run_span_id, span_name="planning")

        # model.invoke for the planning LLM call
        token_usage = getattr(step, "token_usage", None)
        if token_usage is not None:
            self._emit_model_invoke(step, model_id, span_id)


# -- Module-level helpers --------------------------------------------------


def _agent_name(agent: Any) -> str:
    return getattr(agent, "name", None) or type(agent).__name__


def _model_id(agent: Any) -> Optional[str]:
    if agent is None:
        return None
    model = getattr(agent, "model", None)
    if model is None:
        return None
    return getattr(model, "model_id", None) or str(model)


def _get_version() -> str:
    try:
        import smolagents  # pyright: ignore[reportMissingImports]
        return getattr(smolagents, "__version__", "unknown")
    except Exception:
        return "unknown"
