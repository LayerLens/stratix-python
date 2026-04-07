from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional

from ._base_framework import FrameworkAdapter
from ._utils import safe_serialize
from ..._collector import TraceCollector
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

_HAS_GOOGLE_ADK = False
try:
    from google.adk.plugins import BasePlugin as _BasePlugin  # pyright: ignore[reportMissingImports]

    _HAS_GOOGLE_ADK = True
except ImportError:
    _BasePlugin = None  # type: ignore[assignment,misc]


class GoogleADKAdapter(FrameworkAdapter):
    """Google Agent Development Kit (ADK) adapter using the plugin system.

    Registers a ``BasePlugin`` subclass on the ADK ``Runner`` to capture
    the full agent lifecycle: run start/end, agent enter/exit, model
    calls (with tokens), tool calls, errors, and handoffs.

    Usage::

        adapter = GoogleADKAdapter(client)
        adapter.connect()

        # Pass the plugin to the Runner
        runner = Runner(
            app_name="my_app",
            agent=agent,
            session_service=session_service,
            plugins=[adapter.plugin],
        )

        # Or register on an existing runner:
        runner._plugin_manager.register_plugin(adapter.plugin)

        # Run your agent
        async for event in runner.run_async(...):
            ...

        adapter.disconnect()
    """

    name = "google_adk"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._collector: Optional[TraceCollector] = None
        self._run_span_id: Optional[str] = None
        self._agent_span_ids: Dict[str, str] = {}
        self._current_agent_name: Optional[str] = None
        self._timers: Dict[str, int] = {}
        self._seen_agents: set = set()
        self._plugin: Optional[Any] = None

    @property
    def plugin(self) -> Any:
        """The ADK plugin instance. Pass this to ``Runner(plugins=[...])``."""
        return self._plugin

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> Any:
        self._check_dependency(_HAS_GOOGLE_ADK)
        self._metadata["framework_version"] = _get_version()
        self._plugin = _make_plugin(self)
        return target

    def _on_disconnect(self) -> None:
        self._end_trace()
        self._plugin = None
        self._seen_agents.clear()

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

    def _leaf_parent(self) -> Optional[str]:
        if self._current_agent_name:
            return self._agent_span_ids.get(self._current_agent_name, self._run_span_id)
        return self._run_span_id

    def _end_trace(self) -> None:
        with self._lock:
            collector = self._collector
            self._collector = None
            self._run_span_id = None
            self._agent_span_ids.clear()
            self._current_agent_name = None
            self._timers.clear()
        if collector is not None:
            collector.flush()

    # ------------------------------------------------------------------
    # Run lifecycle handlers (called from plugin)
    # ------------------------------------------------------------------

    def _on_before_run(self, invocation_context: Any) -> None:
        span_id = self._new_span_id()
        with self._lock:
            self._collector = TraceCollector(self._client, self._config)
            self._run_span_id = span_id
        self._tick("run")

        agent = getattr(invocation_context, "agent", None)
        agent_name = _agent_name(agent)
        payload = self._payload(agent_name=agent_name)

        session = getattr(invocation_context, "session", None)
        if session is not None:
            sid = getattr(session, "id", None)
            if sid:
                payload["session_id"] = str(sid)

        invocation_id = getattr(invocation_context, "invocation_id", None)
        if invocation_id:
            payload["invocation_id"] = str(invocation_id)

        user_content = getattr(invocation_context, "user_content", None)
        self._set_if_capturing(payload, "input", safe_serialize(user_content))
        self._fire("agent.input", payload, span_id=span_id, span_name=agent_name)

    def _on_after_run(self, invocation_context: Any) -> None:
        latency_ms = self._tock("run")
        span_id = self._run_span_id or self._new_span_id()
        agent = getattr(invocation_context, "agent", None)
        agent_name = _agent_name(agent)
        payload = self._payload(agent_name=agent_name)
        if latency_ms is not None:
            payload["duration_ns"] = int(latency_ms * 1_000_000)
        self._fire("agent.output", payload, span_id=span_id, span_name=agent_name)
        self._end_trace()

    # ------------------------------------------------------------------
    # Agent lifecycle handlers
    # ------------------------------------------------------------------

    def _on_before_agent(self, agent: Any, callback_context: Any) -> None:
        name = _agent_name(agent)
        span_id = self._new_span_id()
        with self._lock:
            self._agent_span_ids[name] = span_id
            self._current_agent_name = name
        self._tick(f"agent:{name}")

        self._emit_agent_config(name, agent, callback_context)

        payload = self._payload(agent_name=name)
        user_content = getattr(callback_context, "user_content", None)
        self._set_if_capturing(payload, "input", safe_serialize(user_content))
        self._fire("agent.input", payload, span_id=span_id, parent_span_id=self._run_span_id, span_name=f"agent:{name}")

    def _on_after_agent(self, agent: Any, callback_context: Any) -> None:
        name = _agent_name(agent)
        latency_ms = self._tock(f"agent:{name}")
        with self._lock:
            span_id = self._agent_span_ids.pop(name, self._new_span_id())
            if self._current_agent_name == name:
                self._current_agent_name = None

        payload = self._payload(agent_name=name)
        if latency_ms is not None:
            payload["duration_ns"] = int(latency_ms * 1_000_000)
        self._fire("agent.output", payload, span_id=span_id, parent_span_id=self._run_span_id, span_name=f"agent:{name}")

    # ------------------------------------------------------------------
    # Model lifecycle handlers
    # ------------------------------------------------------------------

    def _on_before_model(self, callback_context: Any, llm_request: Any) -> None:
        agent_name = getattr(callback_context, "agent_name", None) or "unknown"
        self._tick(f"model:{agent_name}")

    def _on_after_model(self, callback_context: Any, llm_response: Any) -> None:
        agent_name = getattr(callback_context, "agent_name", None) or "unknown"
        latency_ms = self._tock(f"model:{agent_name}")

        payload = self._payload()

        # Model name — prefer request model, fall back to response model_version
        model = getattr(llm_response, "model_version", None)
        if model:
            payload["model"] = str(model)
            payload["provider"] = "google"

        # Tokens from usage_metadata
        usage = getattr(llm_response, "usage_metadata", None)
        tokens = {}
        if usage is not None:
            prompt = getattr(usage, "prompt_token_count", None) or 0
            completion = getattr(usage, "candidates_token_count", None) or 0
            if prompt:
                tokens["tokens_prompt"] = prompt
            if completion:
                tokens["tokens_completion"] = completion
            if prompt or completion:
                tokens["tokens_total"] = prompt + completion
        payload.update(tokens)

        if latency_ms is not None:
            payload["latency_ms"] = latency_ms

        parent = self._leaf_parent()
        span_id = self._new_span_id()
        self._fire("model.invoke", payload, span_id=span_id, parent_span_id=parent)
        if tokens:
            cost_payload = self._payload(**tokens)
            if model:
                cost_payload["model"] = str(model)
            self._fire("cost.record", cost_payload, span_id=span_id, parent_span_id=parent)

    def _on_model_error(self, callback_context: Any, llm_request: Any, error: Exception) -> None:
        agent_name = getattr(callback_context, "agent_name", None) or "unknown"
        self._tock(f"model:{agent_name}")  # clear timer
        model = getattr(llm_request, "model", None)
        payload = self._payload(error=str(error), error_type=type(error).__name__)
        if model:
            payload["model"] = str(model)
        self._fire("agent.error", payload, parent_span_id=self._leaf_parent())

    # ------------------------------------------------------------------
    # Tool lifecycle handlers
    # ------------------------------------------------------------------

    def _on_before_tool(self, tool: Any, tool_args: Any, tool_context: Any) -> None:
        tool_name = getattr(tool, "name", None) or "unknown"
        call_id = getattr(tool_context, "function_call_id", None) or tool_name
        self._tick(f"tool:{call_id}")

    def _on_after_tool(self, tool: Any, tool_args: Any, tool_context: Any, result: Any) -> None:
        tool_name = getattr(tool, "name", None) or "unknown"
        call_id = getattr(tool_context, "function_call_id", None) or tool_name
        latency_ms = self._tock(f"tool:{call_id}")

        span_id = self._new_span_id()
        parent = self._leaf_parent()

        call_payload = self._payload(tool_name=tool_name)
        self._set_if_capturing(call_payload, "input", safe_serialize(tool_args))
        if latency_ms is not None:
            call_payload["latency_ms"] = latency_ms
        self._fire("tool.call", call_payload, span_id=span_id, parent_span_id=parent, span_name=f"tool:{tool_name}")

        result_payload = self._payload(tool_name=tool_name)
        self._set_if_capturing(result_payload, "output", safe_serialize(result))
        self._fire("tool.result", result_payload, span_id=span_id, parent_span_id=parent, span_name=f"tool:{tool_name}")

    def _on_tool_error(self, tool: Any, tool_args: Any, tool_context: Any, error: Exception) -> None:
        tool_name = getattr(tool, "name", None) or "unknown"
        call_id = getattr(tool_context, "function_call_id", None) or tool_name
        self._tock(f"tool:{call_id}")  # clear timer
        self._fire(
            "agent.error",
            self._payload(tool_name=tool_name, error=str(error), error_type=type(error).__name__),
            parent_span_id=self._leaf_parent(),
        )

    # ------------------------------------------------------------------
    # Event callback
    # ------------------------------------------------------------------

    def _on_event(self, invocation_context: Any, event: Any) -> None:
        # Detect agent handoffs from event actions
        actions = getattr(event, "actions", None)
        if actions is None:
            return
        transfer_to = getattr(actions, "transfer_to_agent", None)
        if transfer_to:
            author = getattr(event, "author", None) or "unknown"
            self._fire(
                "agent.handoff",
                self._payload(from_agent=author, to_agent=str(transfer_to)),
                parent_span_id=self._run_span_id,
                span_name=f"handoff:{author}->{transfer_to}",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit_agent_config(self, name: str, agent: Any, callback_context: Any) -> None:
        with self._lock:
            if name in self._seen_agents:
                return
            self._seen_agents.add(name)

        payload = self._payload(agent_name=name, agent_type=type(agent).__name__)

        for attr in ("description", "instruction"):
            val = getattr(agent, attr, None)
            if val is not None:
                payload[attr] = str(val)[:500]

        model = getattr(agent, "model", None)
        if model is not None:
            payload["model"] = str(model)

        tools = getattr(agent, "tools", None)
        if tools:
            payload["tools"] = [getattr(t, "name", str(t)) for t in tools]

        sub_agents = getattr(agent, "sub_agents", None)
        if sub_agents:
            payload["sub_agents"] = [getattr(a, "name", str(a)) for a in sub_agents]

        session = getattr(callback_context, "session", None)
        if session is not None:
            sid = getattr(session, "id", None)
            if sid:
                payload["session_id"] = str(sid)

        self._fire("environment.config", payload, parent_span_id=self._run_span_id, span_name=f"config:{name}")


# -- Plugin factory --------------------------------------------------------


def _make_plugin(adapter: GoogleADKAdapter) -> Any:
    """Create a BasePlugin subclass that delegates all callbacks to the adapter."""
    if _BasePlugin is None:
        raise ImportError("google-adk is required for GoogleADKAdapter")

    class _LayerLensPlugin(_BasePlugin):
        def __init__(self) -> None:
            super().__init__(name="layerlens")

        async def before_run_callback(self, *, invocation_context: Any) -> None:
            try:
                adapter._on_before_run(invocation_context)
            except Exception:
                log.warning("layerlens: error in before_run_callback", exc_info=True)
            return None

        async def after_run_callback(self, *, invocation_context: Any) -> None:
            try:
                adapter._on_after_run(invocation_context)
            except Exception:
                log.warning("layerlens: error in after_run_callback", exc_info=True)

        async def before_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
            try:
                adapter._on_before_agent(agent, callback_context)
            except Exception:
                log.warning("layerlens: error in before_agent_callback", exc_info=True)
            return None

        async def after_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
            try:
                adapter._on_after_agent(agent, callback_context)
            except Exception:
                log.warning("layerlens: error in after_agent_callback", exc_info=True)
            return None

        async def before_model_callback(self, *, callback_context: Any, llm_request: Any) -> None:
            try:
                adapter._on_before_model(callback_context, llm_request)
            except Exception:
                log.warning("layerlens: error in before_model_callback", exc_info=True)
            return None

        async def after_model_callback(self, *, callback_context: Any, llm_response: Any) -> None:
            try:
                adapter._on_after_model(callback_context, llm_response)
            except Exception:
                log.warning("layerlens: error in after_model_callback", exc_info=True)
            return None

        async def on_model_error_callback(self, *, callback_context: Any, llm_request: Any, error: Exception) -> None:
            try:
                adapter._on_model_error(callback_context, llm_request, error)
            except Exception:
                log.warning("layerlens: error in on_model_error_callback", exc_info=True)
            return None

        async def before_tool_callback(self, *, tool: Any, tool_args: Any, tool_context: Any) -> None:
            try:
                adapter._on_before_tool(tool, tool_args, tool_context)
            except Exception:
                log.warning("layerlens: error in before_tool_callback", exc_info=True)
            return None

        async def after_tool_callback(self, *, tool: Any, tool_args: Any, tool_context: Any, result: Any) -> None:
            try:
                adapter._on_after_tool(tool, tool_args, tool_context, result)
            except Exception:
                log.warning("layerlens: error in after_tool_callback", exc_info=True)
            return None

        async def on_tool_error_callback(self, *, tool: Any, tool_args: Any, tool_context: Any, error: Exception) -> None:
            try:
                adapter._on_tool_error(tool, tool_args, tool_context, error)
            except Exception:
                log.warning("layerlens: error in on_tool_error_callback", exc_info=True)
            return None

        async def on_event_callback(self, *, invocation_context: Any, event: Any) -> None:
            try:
                adapter._on_event(invocation_context, event)
            except Exception:
                log.warning("layerlens: error in on_event_callback", exc_info=True)
            return None

    return _LayerLensPlugin()


# -- Module-level helpers --------------------------------------------------


def _agent_name(agent: Any) -> str:
    if agent is None:
        return "unknown"
    return getattr(agent, "name", None) or type(agent).__name__


def _get_version() -> str:
    try:
        import google.adk as _adk  # pyright: ignore[reportMissingImports]
        return getattr(_adk, "__version__", "unknown")
    except Exception:
        return "unknown"
