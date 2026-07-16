"""PydanticAI adapter — wraps the agent's model and run methods.

pydantic-ai has no stable hook/callback API, so the adapter instruments
two seams of its public surface (LAY-3567 B2):

- ``agent.model`` is swapped for a :class:`pydantic_ai.models.wrapper.WrapperModel`
  subclass that emits ``model.invoke`` for every model request, ``tool.call``
  for each ``ToolCallPart`` in a response, and ``tool.result`` for each
  ``ToolReturnPart`` echoed back on the following request.
- ``run`` / ``run_sync`` / ``run_stream`` are wrapped on the agent *instance*
  (reentrancy-guarded — ``run_sync`` delegates to ``run`` internally) to emit
  ``agent.input`` / ``agent.output`` / ``agent.error`` and one ``cost.record``
  per run.

Concurrent runs on the same agent are safe: per-run state (collector,
timers, tool spans) lives in ContextVars, isolated per ``asyncio.Task``.

Usage::

    adapter = PydanticAIAdapter(client)
    adapter.connect(target=agent)  # wraps agent.model + run methods
    result = agent.run_sync("hello")
    adapter.disconnect()  # restores the agent
"""

from __future__ import annotations

import time
import logging
import functools
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager
from contextvars import ContextVar

from ._utils import safe_serialize
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    from pydantic_ai import Agent as _AgentCheck  # pyright: ignore[reportMissingImports]  # noqa: F401
    from pydantic_ai.models.wrapper import WrapperModel  # pyright: ignore[reportMissingImports]

    _HAS_PYDANTIC_AI = True
    del _AgentCheck
except ImportError:
    _HAS_PYDANTIC_AI = False
    WrapperModel = object  # type: ignore[assignment,misc]

#: Agent entry points wrapped on the instance during connect().
_RUN_METHODS = ("run", "run_sync", "run_stream")

#: Sentinel distinguishing "no pre-resolved output supplied" (read it off the
#: result) from a genuine ``None`` output — the streaming path resolves the
#: output itself via ``await get_output()`` and hands it in.
_UNRESOLVED = object()


class PydanticAIAdapter(FrameworkAdapter):
    """PydanticAI adapter — see module docstring for the instrumentation model."""

    name = "pydantic-ai"
    package = "pydantic-ai"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._target: Any = None
        self._original_model: Any = None
        # run_sync() calls run() internally — only the outermost wrapped
        # method may own the run lifecycle.
        self._in_run: ContextVar[bool] = ContextVar(f"layerlens_pydantic_ai_in_run_{id(self)}", default=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_PYDANTIC_AI)
        if target is None:
            raise ValueError("PydanticAIAdapter requires a target agent: adapter.connect(target=agent)")

        self._target = target

        model = getattr(target, "model", None)
        if model is not None:
            try:
                target.model = _InstrumentedModel(model, self)
                self._original_model = model
            except Exception:
                self._original_model = None
                log.warning(
                    "layerlens: could not wrap the pydantic-ai model; model-level events disabled",
                    exc_info=True,
                )
        else:
            log.warning(
                "layerlens: agent has no model bound at connect(); model.invoke/tool events disabled "
                "(pass the model to Agent(...) rather than per-run for full instrumentation)"
            )

        for method_name in _RUN_METHODS:
            original = getattr(target, method_name, None)
            if original is None:
                continue
            setattr(target, method_name, self._wrap_run_method(method_name, original))

    def _on_disconnect(self) -> None:
        target = self._target
        if target is not None:
            for method_name in _RUN_METHODS:
                target.__dict__.pop(method_name, None)
            if self._original_model is not None:
                try:
                    target.model = self._original_model
                except Exception:
                    log.warning("layerlens: could not restore the pydantic-ai model")
        self._original_model = None
        self._target = None

    # ------------------------------------------------------------------
    # Run wrapping
    # ------------------------------------------------------------------

    def _wrap_run_method(self, method_name: str, original: Any) -> Any:
        if method_name == "run":

            @functools.wraps(original)
            async def wrapped_run(*args: Any, **kwargs: Any) -> Any:
                if self._in_run.get():
                    return await original(*args, **kwargs)
                token = self._in_run.set(True)
                self._start_run_events(args, kwargs)
                try:
                    result = await original(*args, **kwargs)
                except BaseException as exc:
                    self._finish_run_error(exc)
                    raise
                else:
                    self._finish_run_ok(result)
                    return result
                finally:
                    self._in_run.reset(token)

            return wrapped_run

        if method_name == "run_sync":

            @functools.wraps(original)
            def wrapped_run_sync(*args: Any, **kwargs: Any) -> Any:
                if self._in_run.get():
                    return original(*args, **kwargs)
                token = self._in_run.set(True)
                self._start_run_events(args, kwargs)
                try:
                    result = original(*args, **kwargs)
                except BaseException as exc:
                    self._finish_run_error(exc)
                    raise
                else:
                    self._finish_run_ok(result)
                    return result
                finally:
                    self._in_run.reset(token)

            return wrapped_run_sync

        # run_stream returns an async context manager
        @functools.wraps(original)
        def wrapped_run_stream(*args: Any, **kwargs: Any) -> Any:
            if self._in_run.get():
                return original(*args, **kwargs)
            return self._instrumented_run_stream(original, args, kwargs)

        return wrapped_run_stream

    @asynccontextmanager
    async def _instrumented_run_stream(self, original: Any, args: Any, kwargs: Any) -> Any:
        token = self._in_run.set(True)
        self._start_run_events(args, kwargs)
        stream_result: Any = None
        try:
            async with original(*args, **kwargs) as stream_result:
                yield stream_result
        except BaseException as exc:
            self._finish_run_error(exc)
            raise
        else:
            # StreamedRunResult exposes its result ONLY via ``await get_output()``
            # (no ``.output`` attribute) — resolve it here, in async context,
            # AFTER the consumer's ``async with`` body has run (stream consumed or
            # abandoned). Guard it: a caller that abandoned the stream (or a
            # transport that cannot replay it) must not turn a successful run into
            # a crash — fall back to the honest "no output" instead.
            resolved: Any = _UNRESOLVED
            try:
                resolved = await stream_result.get_output()
            except BaseException:
                log.debug(
                    "layerlens: could not resolve streamed pydantic-ai output; emitting agent.output without content",
                    exc_info=True,
                )
            self._finish_run_ok(stream_result, streaming=True, resolved_output=resolved)
        finally:
            self._in_run.reset(token)

    # ------------------------------------------------------------------
    # Run lifecycle events
    # ------------------------------------------------------------------

    def _start_run_events(self, args: Any, kwargs: Dict[str, Any]) -> None:
        self._begin_run()
        root = self._get_root_span()
        agent = self._target
        agent_name = self._agent_display_name(agent)
        model_name = self._model_display_name(agent)
        prompt = args[0] if args else kwargs.get("user_prompt")

        payload = self._payload()
        if agent_name:
            payload["agent_name"] = agent_name
        if model_name:
            payload["model"] = model_name
        self._set_if_capturing(payload, "input", safe_serialize(prompt))

        # Surface the declared result/output type and dependency shape so
        # downstream telemetry can reason about what the agent is configured
        # to return, independent of any single response.
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
        # result-injection-driven runs can be differentiated. Deps are
        # request-scoped secrets (tokens, db handles); _summarize_deps captures
        # names + value TYPES only. NEVER serialize deps to a string — for a
        # dataclass/arbitrary object safe_serialize() falls back to str(deps),
        # whose repr embeds the raw values verbatim (a privacy leak).
        deps = kwargs.get("deps")
        if deps is not None and self._config.capture_content:
            payload["deps_summary"] = _summarize_deps(deps)

        self._emit(
            "agent.input",
            payload,
            span_id=root,
            parent_span_id=None,
            span_name=f"pydantic_ai:{self._span_label(agent)}",
        )
        self._start_timer("run")

    def _finish_run_ok(self, result: Any, streaming: bool = False, resolved_output: Any = _UNRESOLVED) -> None:
        latency_ms = self._stop_timer("run")
        root = self._get_root_span()
        agent_name = self._agent_display_name(self._target)
        model_name = self._model_display_name(self._target)

        # Streaming resolves its output out-of-band (``await get_output()`` — the
        # StreamedRunResult has no ``.output`` attr for ``_extract_output`` to read);
        # honor the pre-resolved value when supplied, else read it off the result.
        if resolved_output is _UNRESOLVED:
            output = self._extract_output(result)
        else:
            output = safe_serialize(resolved_output) if resolved_output is not None else None
        usage = self._extract_usage(result)

        payload = self._payload(status="ok")
        if agent_name:
            payload["agent_name"] = agent_name
        if streaming:
            payload["streaming"] = True
        if model_name:
            payload["model"] = model_name
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        self._set_if_capturing(payload, "output", output)
        payload.update(usage)
        self._emit(
            "agent.output",
            payload,
            span_id=root,
            parent_span_id=None,
            span_name=f"pydantic_ai:{self._span_label(self._target)}",
        )

        if usage:
            cost_payload = self._payload()
            if model_name:
                cost_payload["model"] = model_name
            cost_payload.update(usage)
            self._emit("cost.record", cost_payload)

        self._end_run()

    def _finish_run_error(self, error: BaseException) -> None:
        latency_ms = self._stop_timer("run")
        root = self._get_root_span()
        agent_name = self._agent_display_name(self._target)

        payload = self._payload(
            error=str(error),
            error_type=type(error).__name__,
        )
        if agent_name:
            payload["agent_name"] = agent_name
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        self._emit(
            "agent.error",
            payload,
            span_id=root,
            parent_span_id=None,
            span_name=f"pydantic_ai:{self._span_label(self._target)}",
        )

        self._end_run()

    # ------------------------------------------------------------------
    # Model-level events (called by _InstrumentedModel)
    # ------------------------------------------------------------------

    def _emit_model_response(self, response: Any, latency_ms: float, fallback_model_name: Optional[str]) -> None:
        model_name = getattr(response, "model_name", None) or fallback_model_name
        payload = self._payload()
        # Attribute the model call to the DECLARED agent (never the model — that
        # would be model-as-agent) so the graph engine places this node-bearing
        # event on the honest node; stays absent for an unnamed agent.
        agent_name = self._agent_display_name(self._target)
        if agent_name:
            payload["agent_name"] = agent_name
        if model_name:
            payload["model"] = str(model_name)
        prid = getattr(response, "provider_response_id", None)
        if prid:
            payload["response_id"] = str(prid)
        payload["latency_ms"] = latency_ms
        payload.update(self._normalize_tokens(getattr(response, "usage", None)))
        self._emit("model.invoke", payload)

        run = self._get_run()
        for part in getattr(response, "parts", None) or []:
            if type(part).__name__ != "ToolCallPart":
                continue
            tool_name = getattr(part, "tool_name", "unknown")
            tool_payload = self._payload(tool_name=tool_name)
            self._set_if_capturing(tool_payload, "input", safe_serialize(getattr(part, "args", None)))
            self._emit("tool.call", tool_payload)
            call_id = getattr(part, "tool_call_id", None) or tool_name
            if run is not None:
                run.data.setdefault("tool_started", {})[call_id] = time.time_ns()

    def _emit_tool_results(self, messages: Any) -> None:
        """ToolReturnParts for just-executed tools arrive on the *next* model
        request's trailing message — turn them into ``tool.result`` events."""
        if not messages:
            return
        parts = getattr(messages[-1], "parts", None) or []
        run = self._get_run()
        started = run.data.get("tool_started", {}) if run is not None else {}
        for part in parts:
            if type(part).__name__ != "ToolReturnPart":
                continue
            tool_name = getattr(part, "tool_name", "unknown")
            call_id = getattr(part, "tool_call_id", None) or tool_name
            start_ns = started.pop(call_id, None)
            payload = self._payload(tool_name=tool_name)
            self._set_if_capturing(payload, "output", safe_serialize(getattr(part, "content", None)))
            payload["latency_ms"] = (time.time_ns() - start_ns) / 1_000_000 if start_ns else 0.0
            self._emit("tool.result", payload)

    def _emit_model_error(self, error: BaseException, latency_ms: float) -> None:
        payload = self._payload(
            error=str(error),
            error_type=type(error).__name__,
            latency_ms=latency_ms,
        )
        self._emit("agent.error", payload)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_display_name(agent: Any) -> Optional[str]:
        """The agent's DECLARED name, or None. Never the model — an unnamed
        pydantic Agent has no honest agent identity, and returning the model name
        here would surface model-as-agent in the Agent column (fabrication). The
        model is captured separately (payload.model); the span label falls back to
        it for display, but ``agent_name`` stays absent when there is no name."""
        name = getattr(agent, "name", None)
        return str(name) if name else None

    @staticmethod
    def _span_label(agent: Any) -> str:
        """A cosmetic span label (NOT an agent identity): the declared name, else
        the model, else a generic marker."""
        return PydanticAIAdapter._agent_display_name(agent) or PydanticAIAdapter._model_display_name(agent) or "agent"

    @staticmethod
    def _model_display_name(agent: Any) -> Optional[str]:
        model = getattr(agent, "model", None)
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


class _InstrumentedModel(WrapperModel):  # type: ignore[misc,valid-type]
    """WrapperModel that reports model requests/tool traffic to the adapter."""

    def __init__(self, wrapped: Any, adapter: PydanticAIAdapter) -> None:
        super().__init__(wrapped)
        self._adapter = adapter

    async def request(self, messages: Any, model_settings: Any, model_request_parameters: Any) -> Any:
        adapter = self._adapter
        adapter._emit_tool_results(messages)
        start_ns = time.time_ns()
        try:
            response = await super().request(messages, model_settings, model_request_parameters)
        except Exception as exc:
            adapter._emit_model_error(exc, (time.time_ns() - start_ns) / 1_000_000)
            raise
        adapter._emit_model_response(
            response,
            (time.time_ns() - start_ns) / 1_000_000,
            fallback_model_name=getattr(self.wrapped, "model_name", None),
        )
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: Any,
        model_settings: Any,
        model_request_parameters: Any,
        run_context: Any = None,
    ) -> Any:
        adapter = self._adapter
        adapter._emit_tool_results(messages)
        start_ns = time.time_ns()
        streamed: Any = None
        try:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed:
                yield streamed
        except Exception as exc:
            adapter._emit_model_error(exc, (time.time_ns() - start_ns) / 1_000_000)
            raise
        # Best-effort: by the time the consumer exits the stream context the
        # StreamedResponse knows its usage/model; emit one streaming invoke.
        latency_ms = (time.time_ns() - start_ns) / 1_000_000
        payload = adapter._payload(streaming=True)
        agent_name = adapter._agent_display_name(adapter._target)
        if agent_name:
            payload["agent_name"] = agent_name
        model_name = getattr(streamed, "model_name", None) or getattr(self.wrapped, "model_name", None)
        if model_name:
            payload["model"] = str(model_name)
        payload["latency_ms"] = latency_ms
        usage = getattr(streamed, "usage", None)
        if callable(usage):
            try:
                usage = usage()
            except Exception:
                usage = None
        payload.update(adapter._normalize_tokens(usage))
        adapter._emit("model.invoke", payload)


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
