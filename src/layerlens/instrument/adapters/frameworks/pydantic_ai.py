from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ._base_framework import FrameworkAdapter
from ._utils import safe_serialize
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    from pydantic_ai import Agent as _AgentCheck  # pyright: ignore[reportMissingImports]  # noqa: F401

    _HAS_PYDANTIC_AI = True
    del _AgentCheck
except ImportError:
    _HAS_PYDANTIC_AI = False


class PydanticAIAdapter(FrameworkAdapter):
    """PydanticAI adapter using the native Hooks capability API.

    Injects a ``Hooks`` capability into the target agent to receive
    real-time lifecycle callbacks for run start/end, per-model-call,
    and per-tool-execution events — with precise per-step timing.

    Concurrent runs on the same agent are safe: each run gets its own
    RunState via ContextVar, so collectors, timers, and tool spans
    are fully isolated per ``asyncio.Task``.

    Usage::

        adapter = PydanticAIAdapter(client)
        adapter.connect(target=agent)   # injects hooks capability
        result = agent.run_sync("hello")
        adapter.disconnect()            # removes hooks capability
    """

    name = "pydantic-ai"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._target: Any = None
        self._hooks: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_PYDANTIC_AI)
        if target is None:
            raise ValueError("PydanticAIAdapter requires a target agent: adapter.connect(target=agent)")

        from pydantic_ai.capabilities.hooks import Hooks  # pyright: ignore[reportMissingImports]

        self._target = target
        self._hooks = Hooks()
        self._register_hooks(self._hooks)
        target._root_capability.capabilities.append(self._hooks)

    def _on_disconnect(self) -> None:
        if self._target is not None and self._hooks is not None:
            try:
                caps = self._target._root_capability.capabilities
                if self._hooks in caps:
                    caps.remove(self._hooks)
            except Exception:
                log.warning("Could not remove PydanticAI hooks capability")
        self._hooks = None
        self._target = None

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self, hooks: Any) -> None:
        hooks.on.before_run(self._on_before_run)
        hooks.on.after_run(self._on_after_run)
        hooks.on.run_error(self._on_run_error)
        hooks.on.after_model_request(self._on_after_model_request)
        hooks.on.model_request_error(self._on_model_request_error)
        hooks.on.before_tool_execute(self._on_before_tool_execute)
        hooks.on.after_tool_execute(self._on_after_tool_execute)
        hooks.on.tool_execute_error(self._on_tool_execute_error)

    # ------------------------------------------------------------------
    # Run lifecycle hooks
    # ------------------------------------------------------------------

    def _on_before_run(self, ctx: Any) -> None:
        run = self._begin_run()
        agent_name = self._get_agent_name(ctx)
        model_name = self._get_model_name(ctx)

        payload = self._payload(agent_name=agent_name)
        if model_name:
            payload["model"] = model_name
        self._set_if_capturing(payload, "input", safe_serialize(ctx.prompt))

        run.collector.emit(
            "agent.input", payload,
            span_id=run.root_span_id, parent_span_id=None,
            span_name=f"pydantic_ai:{agent_name}",
        )
        self._start_timer("run")

    def _on_after_run(self, ctx: Any, *, result: Any) -> Any:
        latency_ms = self._stop_timer("run")
        agent_name = self._get_agent_name(ctx)
        model_name = self._get_model_name(ctx)
        root_span = self._get_root_span()
        collector = self._ensure_collector()

        output = self._extract_output(result)
        usage = self._extract_usage(result)

        payload = self._payload(agent_name=agent_name, status="ok")
        if model_name:
            payload["model"] = model_name
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        self._set_if_capturing(payload, "output", output)
        payload.update(usage)
        collector.emit(
            "agent.output", payload,
            span_id=root_span, parent_span_id=None,
            span_name=f"pydantic_ai:{agent_name}",
        )

        if usage:
            cost_payload = self._payload()
            if model_name:
                cost_payload["model"] = model_name
            cost_payload.update(usage)
            collector.emit(
                "cost.record", cost_payload,
                span_id=self._new_span_id(), parent_span_id=root_span,
            )

        self._end_run()
        return result

    def _on_run_error(self, ctx: Any, *, error: BaseException) -> None:
        latency_ms = self._stop_timer("run")
        agent_name = self._get_agent_name(ctx)
        root_span = self._get_root_span()
        collector = self._ensure_collector()

        payload = self._payload(
            agent_name=agent_name,
            error=str(error),
            error_type=type(error).__name__,
        )
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        collector.emit(
            "agent.error", payload,
            span_id=root_span, parent_span_id=None,
            span_name=f"pydantic_ai:{agent_name}",
        )

        self._end_run()
        raise error

    # ------------------------------------------------------------------
    # Model request hooks
    # ------------------------------------------------------------------

    def _on_after_model_request(
        self, ctx: Any, *, request_context: Any, response: Any,
    ) -> Any:
        root_span = self._get_root_span()
        collector = self._ensure_collector()

        model_name = getattr(response, "model_name", None)
        usage = getattr(response, "usage", None)
        tokens = self._normalize_tokens(usage)

        payload = self._payload()
        if model_name:
            payload["model"] = model_name
        payload.update(tokens)

        model_span = self._new_span_id()
        collector.emit(
            "model.invoke", payload,
            span_id=model_span, parent_span_id=root_span,
        )

        parts = getattr(response, "parts", None) or []
        for part in parts:
            if type(part).__name__ == "ToolCallPart":
                tool_name = getattr(part, "tool_name", "unknown")
                tool_payload = self._payload(tool_name=tool_name)
                self._set_if_capturing(
                    tool_payload, "input",
                    safe_serialize(getattr(part, "args", None)),
                )
                collector.emit(
                    "tool.call", tool_payload,
                    span_id=self._new_span_id(), parent_span_id=root_span,
                )

        return response

    def _on_model_request_error(
        self, ctx: Any, *, request_context: Any, error: Exception,
    ) -> None:
        root_span = self._get_root_span()
        collector = self._ensure_collector()

        payload = self._payload(
            error=str(error),
            error_type=type(error).__name__,
        )
        collector.emit(
            "agent.error", payload,
            span_id=self._new_span_id(), parent_span_id=root_span,
        )
        raise error

    # ------------------------------------------------------------------
    # Tool execution hooks
    # ------------------------------------------------------------------

    def _on_before_tool_execute(
        self, ctx: Any, *, call: Any, tool_def: Any, args: Any,
    ) -> Any:
        tool_name = getattr(call, "tool_name", "unknown")
        span_id = self._new_span_id()
        run = self._get_run()
        if run is not None:
            run.data.setdefault("tool_spans", {})[tool_name] = span_id
        self._start_timer(f"tool:{tool_name}")
        return args

    def _on_after_tool_execute(
        self, ctx: Any, *, call: Any, tool_def: Any, args: Any, result: Any,
    ) -> Any:
        tool_name = getattr(call, "tool_name", "unknown")
        latency_ms = self._stop_timer(f"tool:{tool_name}")

        run = self._get_run()
        tool_spans = run.data.get("tool_spans", {}) if run is not None else {}
        span_id = tool_spans.pop(tool_name, self._new_span_id())

        root_span = self._get_root_span()
        collector = self._ensure_collector()

        payload = self._payload(tool_name=tool_name)
        self._set_if_capturing(payload, "output", safe_serialize(result))
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        collector.emit(
            "tool.result", payload,
            span_id=span_id, parent_span_id=root_span,
        )
        return result

    def _on_tool_execute_error(
        self, ctx: Any, *, call: Any, tool_def: Any, args: Any, error: Exception,
    ) -> None:
        tool_name = getattr(call, "tool_name", "unknown")
        self._stop_timer(f"tool:{tool_name}")

        run = self._get_run()
        if run is not None:
            run.data.get("tool_spans", {}).pop(tool_name, None)

        root_span = self._get_root_span()
        collector = self._ensure_collector()

        payload = self._payload(
            tool_name=tool_name,
            error=str(error),
            error_type=type(error).__name__,
        )
        collector.emit(
            "agent.error", payload,
            span_id=self._new_span_id(), parent_span_id=root_span,
        )
        raise error

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_agent_name(ctx: Any) -> str:
        agent = getattr(ctx, "agent", None)
        if agent is not None:
            name = getattr(agent, "name", None)
            if name:
                return str(name)
        return PydanticAIAdapter._get_model_name(ctx) or "pydantic_ai_agent"

    @staticmethod
    def _get_model_name(ctx: Any) -> Optional[str]:
        model = getattr(ctx, "model", None)
        if model is None:
            agent = getattr(ctx, "agent", None)
            model = getattr(agent, "model", None) if agent else None
        if model is None:
            return None
        if isinstance(model, str):
            return model
        name = getattr(model, "model_name", None)
        if name:
            return str(name)
        return str(model)

    @staticmethod
    def _extract_output(result: Any) -> Any:
        if result is None:
            return None
        output = getattr(result, "output", None)
        if output is not None:
            return safe_serialize(output)
        return None

    @staticmethod
    def _extract_usage(result: Any) -> Dict[str, Any]:
        tokens: Dict[str, Any] = {}
        usage = getattr(result, "usage", None)
        if usage is None:
            return tokens

        if callable(usage):
            try:
                usage = usage()
            except Exception:
                return tokens

        input_t = getattr(usage, "input_tokens", 0) or 0
        output_t = getattr(usage, "output_tokens", 0) or 0

        if input_t:
            tokens["tokens_prompt"] = input_t
        if output_t:
            tokens["tokens_completion"] = output_t
        if input_t or output_t:
            tokens["tokens_total"] = input_t + output_t

        requests = getattr(usage, "requests", 0) or 0
        if requests:
            tokens["model_requests"] = requests

        return tokens
