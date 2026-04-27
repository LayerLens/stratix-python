from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .._base import StateFilter, resilient_callback
from ._utils import safe_serialize
from ._base_framework import FrameworkAdapter
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
        adapter.connect(target=agent)  # injects hooks capability
        result = agent.run_sync("hello")
        adapter.disconnect()  # removes hooks capability
    """

    name = "pydantic-ai"

    def __init__(
        self,
        client: Any,
        capture_config: Optional[CaptureConfig] = None,
        state_filter: Optional[StateFilter] = None,
    ) -> None:
        super().__init__(client, capture_config, state_filter=state_filter)
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
        # Streaming hooks are optional — pydantic-ai >= 0.5 exposes on.stream_chunk
        # / on.after_stream. Older versions simply don't have them.
        for hook_name, method in (
            ("stream_chunk", self._on_stream_chunk),
            ("after_stream", self._on_after_stream),
        ):
            attr = getattr(hooks.on, hook_name, None)
            if callable(attr):
                attr(method)

    # ------------------------------------------------------------------
    # Run lifecycle hooks
    # ------------------------------------------------------------------

    @resilient_callback(callback_name="_on_before_run")
    def _on_before_run(self, ctx: Any) -> None:
        self._begin_run()
        root = self._get_root_span()
        agent_name = self._get_agent_name(ctx)
        model_name = self._get_model_name(ctx)

        payload = self._payload(agent_name=agent_name)
        if model_name:
            payload["model"] = model_name
        self._set_if_capturing(payload, "input", safe_serialize(ctx.prompt))

        # Surface the declared result/output type and dependency shape so
        # downstream telemetry can reason about what the agent is configured
        # to return, independent of any single response.
        agent = getattr(ctx, "agent", None) or getattr(ctx, "_agent", None)
        if agent is not None:
            result_type = (
                getattr(agent, "output_type", None)
                or getattr(agent, "result_type", None)
                or getattr(agent, "_output_type", None)
            )
            if result_type is not None:
                payload["result_type"] = _describe_type(result_type)
            deps_type = getattr(agent, "deps_type", None) or getattr(agent, "_deps_type", None)
            if deps_type is not None:
                payload["deps_type"] = _describe_type(deps_type)
        # Record the deps instance (not raw — key/type summary only) so
        # result-injection-driven runs can be differentiated.
        deps = getattr(ctx, "deps", None)
        if deps is not None and self._config.capture_content:
            payload["deps_summary"] = (
                safe_serialize(deps)[:500] if isinstance(safe_serialize(deps), str) else _summarize_deps(deps)
            )

        # Filter agent input + deps_summary before emit. ``deps`` in
        # PydanticAI is the canonical request-scoped object — DB
        # handles, request_id, user objects all live here. The filter
        # is the last line of defence between the deps shape and the
        # event sink.
        self._filter_payload(payload, "input", "deps_summary")
        self._emit(
            "agent.input",
            payload,
            span_id=root,
            parent_span_id=None,
            span_name=f"pydantic_ai:{agent_name}",
        )
        self._start_timer("run")

    @resilient_callback(callback_name="_on_after_run", passthrough_arg="result")
    def _on_after_run(self, ctx: Any, *, result: Any) -> Any:
        latency_ms = self._stop_timer("run")
        root = self._get_root_span()
        agent_name = self._get_agent_name(ctx)
        model_name = self._get_model_name(ctx)

        output = self._extract_output(result)
        usage = self._extract_usage(result)

        payload = self._payload(agent_name=agent_name, status="ok")
        if model_name:
            payload["model"] = model_name
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        self._set_if_capturing(payload, "output", output)
        payload.update(usage)
        self._filter_payload(payload, "output")
        self._emit(
            "agent.output",
            payload,
            span_id=root,
            parent_span_id=None,
            span_name=f"pydantic_ai:{agent_name}",
        )

        if usage:
            cost_payload = self._payload()
            if model_name:
                cost_payload["model"] = model_name
            cost_payload.update(usage)
            self._emit("cost.record", cost_payload)

        self._end_run()
        return result

    def _on_run_error(self, ctx: Any, *, error: BaseException) -> None:
        # Telemetry is best-effort; we MUST always re-raise the
        # framework's original error or PydanticAI loses its error
        # propagation contract. Keep telemetry inside a resilient
        # helper so adapter-side bugs can never swallow the framework
        # error.
        self._emit_run_error_telemetry(ctx, error=error)
        raise error

    @resilient_callback(callback_name="_on_run_error")
    def _emit_run_error_telemetry(self, ctx: Any, *, error: BaseException) -> None:
        latency_ms = self._stop_timer("run")
        root = self._get_root_span()
        agent_name = self._get_agent_name(ctx)

        payload = self._payload(
            agent_name=agent_name,
            error=str(error),
            error_type=type(error).__name__,
        )
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        self._emit(
            "agent.error",
            payload,
            span_id=root,
            parent_span_id=None,
            span_name=f"pydantic_ai:{agent_name}",
        )

        self._end_run()

    # ------------------------------------------------------------------
    # Model request hooks
    # ------------------------------------------------------------------

    @resilient_callback(callback_name="_on_after_model_request", passthrough_arg="response")
    def _on_after_model_request(
        self,
        ctx: Any,
        *,
        request_context: Any,
        response: Any,
    ) -> Any:
        model_name = getattr(response, "model_name", None)
        usage = getattr(response, "usage", None)
        tokens = self._normalize_tokens(usage)

        payload = self._payload()
        if model_name:
            payload["model"] = model_name
        payload.update(tokens)

        self._emit("model.invoke", payload)

        parts = getattr(response, "parts", None) or []
        for part in parts:
            if type(part).__name__ == "ToolCallPart":
                tool_name = getattr(part, "tool_name", "unknown")
                tool_payload = self._payload(tool_name=tool_name)
                self._set_if_capturing(
                    tool_payload,
                    "input",
                    safe_serialize(getattr(part, "args", None)),
                )
                self._filter_payload(tool_payload, "input")
                self._emit("tool.call", tool_payload)

        return response

    def _on_model_request_error(
        self,
        ctx: Any,
        *,
        request_context: Any,
        error: Exception,
    ) -> None:
        # Telemetry first (resilient), THEN re-raise the framework's
        # error so PydanticAI's own error propagation is preserved.
        self._emit_model_request_error_telemetry(ctx, request_context=request_context, error=error)
        raise error

    @resilient_callback(callback_name="_on_model_request_error")
    def _emit_model_request_error_telemetry(
        self,
        ctx: Any,
        *,
        request_context: Any,
        error: Exception,
    ) -> None:
        payload = self._payload(
            error=str(error),
            error_type=type(error).__name__,
        )
        self._emit("agent.error", payload)

    # ------------------------------------------------------------------
    # Tool execution hooks
    # ------------------------------------------------------------------

    @resilient_callback(callback_name="_on_before_tool_execute", passthrough_arg="args")
    def _on_before_tool_execute(
        self,
        ctx: Any,
        *,
        call: Any,
        tool_def: Any,
        args: Any,
    ) -> Any:
        tool_name = getattr(call, "tool_name", "unknown")
        call_id = getattr(call, "id", None) or tool_name
        span_id = self._new_span_id()
        run = self._get_run()
        if run is not None:
            run.data.setdefault("tool_spans", {})[call_id] = span_id
        self._start_timer(f"tool:{call_id}")
        return args

    @resilient_callback(callback_name="_on_after_tool_execute", passthrough_arg="result")
    def _on_after_tool_execute(
        self,
        ctx: Any,
        *,
        call: Any,
        tool_def: Any,
        args: Any,
        result: Any,
    ) -> Any:
        tool_name = getattr(call, "tool_name", "unknown")
        call_id = getattr(call, "id", None) or tool_name
        latency_ms = self._stop_timer(f"tool:{call_id}")

        run = self._get_run()
        tool_spans = run.data.get("tool_spans", {}) if run is not None else {}
        span_id = tool_spans.pop(call_id, self._new_span_id())

        payload = self._payload(tool_name=tool_name)
        self._set_if_capturing(payload, "output", safe_serialize(result))
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        self._filter_payload(payload, "output")
        self._emit("tool.result", payload, span_id=span_id)
        return result

    def _on_tool_execute_error(
        self,
        ctx: Any,
        *,
        call: Any,
        tool_def: Any,
        args: Any,
        error: Exception,
    ) -> None:
        # Telemetry first (resilient), THEN re-raise the framework's
        # error so PydanticAI can propagate the tool failure.
        self._emit_tool_execute_error_telemetry(ctx, call=call, tool_def=tool_def, args=args, error=error)
        raise error

    @resilient_callback(callback_name="_on_tool_execute_error")
    def _emit_tool_execute_error_telemetry(
        self,
        ctx: Any,
        *,
        call: Any,
        tool_def: Any,
        args: Any,
        error: Exception,
    ) -> None:
        tool_name = getattr(call, "tool_name", "unknown")
        call_id = getattr(call, "id", None) or tool_name
        self._stop_timer(f"tool:{call_id}")

        run = self._get_run()
        if run is not None:
            run.data.get("tool_spans", {}).pop(call_id, None)

        payload = self._payload(
            tool_name=tool_name,
            error=str(error),
            error_type=type(error).__name__,
        )
        self._emit("agent.error", payload)

    # ------------------------------------------------------------------
    # Streaming hooks (pydantic-ai >= 0.5)
    # ------------------------------------------------------------------

    @resilient_callback(callback_name="_on_stream_chunk")
    def _on_stream_chunk(self, ctx: Any, *, chunk: Any, **_kwargs: Any) -> None:
        """Accumulate streaming chunks on the RunState; aggregated at stream end."""
        run = self._get_run()
        if run is None:
            return
        buf = run.data.setdefault("stream_buffer", [])
        buf.append(chunk)

    @resilient_callback(callback_name="_on_after_stream")
    def _on_after_stream(self, ctx: Any, *, response: Any = None, **_kwargs: Any) -> None:
        run = self._get_run()
        if run is None:
            return
        chunks = run.data.pop("stream_buffer", [])
        payload = self._payload(streaming=True, streamed_chunks=len(chunks))
        model_name = self._get_model_name(ctx)
        if model_name:
            payload["model"] = model_name
        if response is not None:
            usage = getattr(response, "usage", None)
            payload.update(self._normalize_tokens(usage))
            if self._config.capture_content:
                output = self._extract_output(response)
                if output is not None:
                    payload["output_message"] = output
        self._filter_payload(payload, "output_message")
        self._emit("model.invoke", payload)

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


def _describe_type(t: Any) -> str:
    """Render a type hint as a readable string for telemetry."""
    if t is None:
        return "None"
    name = getattr(t, "__name__", None)
    if name:
        mod = getattr(t, "__module__", "")
        return f"{mod}.{name}" if mod and mod != "builtins" else name
    return str(t)[:200]


def _summarize_deps(deps: Any) -> Dict[str, Any]:
    """Dependencies are often request-scoped (request_id, user, db handle).
    Capture shape only — key names + value types — so we never log raw data.
    """
    out: Dict[str, Any] = {"type": type(deps).__name__}
    try:
        if hasattr(deps, "__dict__"):
            out["fields"] = {k: type(v).__name__ for k, v in vars(deps).items() if not k.startswith("_")}
        elif isinstance(deps, dict):
            out["fields"] = {k: type(v).__name__ for k, v in deps.items()}
    except Exception:
        pass
    return out
