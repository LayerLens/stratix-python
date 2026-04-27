from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .._base import StateFilter, resilient_callback
from ._utils import safe_serialize
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    import agno  # pyright: ignore[reportMissingImports]  # noqa: F401

    _HAS_AGNO = True
except ImportError:
    _HAS_AGNO = False


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _model_id(agent: Any) -> Optional[str]:
    model = getattr(agent, "model", None)
    if model is None:
        return None
    return getattr(model, "id", None) or str(model)


def _extract_tokens(result: Any) -> Dict[str, int]:
    metrics = getattr(result, "metrics", None)
    if metrics is None:
        return {}

    inp = getattr(metrics, "input_tokens", None)
    out = getattr(metrics, "output_tokens", None)
    reasoning = getattr(metrics, "reasoning_tokens", None) or getattr(metrics, "thinking_tokens", None)
    cached = getattr(metrics, "cached_tokens", None) or getattr(metrics, "cache_read_tokens", None)
    audio = getattr(metrics, "audio_tokens", None)
    time_ms = getattr(metrics, "duration_ms", None) or getattr(metrics, "time", None)

    if inp is not None or out is not None:
        tokens: Dict[str, int] = {}
        if inp:
            tokens["tokens_prompt"] = int(inp)
        if out:
            tokens["tokens_completion"] = int(out)
        if inp or out:
            tokens["tokens_total"] = (int(inp) if inp else 0) + (int(out) if out else 0)
        if reasoning:
            tokens["reasoning_tokens"] = int(reasoning)
        if cached:
            tokens["cached_tokens"] = int(cached)
        if audio:
            tokens["audio_tokens"] = int(audio)
        if time_ms:
            try:
                tokens["duration_ms"] = int(float(time_ms))
            except (TypeError, ValueError):
                pass
        return tokens

    details = getattr(metrics, "details", None)
    if not isinstance(details, dict):
        return {}
    total_in = total_out = total_reason = total_cached = 0
    per_model: Dict[str, Dict[str, int]] = {}
    for model_name, model_metrics_list in details.items():
        if not isinstance(model_metrics_list, list):
            continue
        model_in = model_out = 0
        for mm in model_metrics_list:
            model_in += getattr(mm, "input_tokens", 0) or 0
            model_out += getattr(mm, "output_tokens", 0) or 0
            total_reason += getattr(mm, "reasoning_tokens", 0) or 0
            total_cached += getattr(mm, "cached_tokens", 0) or 0
        total_in += model_in
        total_out += model_out
        if model_in or model_out:
            per_model[str(model_name)] = {
                "tokens_prompt": model_in,
                "tokens_completion": model_out,
                "tokens_total": model_in + model_out,
            }
    if not total_in and not total_out:
        return {}
    tokens = {}
    if total_in:
        tokens["tokens_prompt"] = total_in
    if total_out:
        tokens["tokens_completion"] = total_out
    tokens["tokens_total"] = total_in + total_out
    if total_reason:
        tokens["reasoning_tokens"] = total_reason
    if total_cached:
        tokens["cached_tokens"] = total_cached
    # Multi-model aggregation: surface per-model breakdown so we can see which
    # model contributed how many tokens in a hybrid run.
    if len(per_model) > 1:
        tokens["per_model"] = per_model  # type: ignore[assignment]
    return tokens


def _extract_tools(result: Any) -> List[Dict[str, Any]]:
    tools = getattr(result, "tools", None)
    if not tools:
        return []
    out = []
    for te in tools:
        entry: Dict[str, Any] = {
            "tool_name": getattr(te, "tool_name", None) or getattr(te, "name", "unknown"),
            "tool_args": getattr(te, "tool_args", None) or getattr(te, "arguments", None),
            "result": getattr(te, "result", None),
        }
        te_metrics = getattr(te, "metrics", None)
        if te_metrics is not None:
            duration = getattr(te_metrics, "execution_time", None) or getattr(te_metrics, "duration", None)
            if duration is not None:
                entry["latency_ms"] = float(duration) * 1000
        out.append(entry)
    return out


class AgnoAdapter(FrameworkAdapter):
    """Agno adapter wrapping ``Agent.run()`` / ``Agent.arun()``.

    Uses ``_begin_run`` / ``_end_run`` for ContextVar-based collector
    lifecycle. All telemetry is extracted post-hoc from ``RunOutput``.

    Usage::

        adapter = AgnoAdapter(client)
        agent = adapter.connect(target=agent)
        result = agent.run("hello")
        adapter.disconnect()
    """

    name = "agno"

    def __init__(
        self,
        client: Any,
        capture_config: Optional[CaptureConfig] = None,
        state_filter: Optional[StateFilter] = None,
    ) -> None:
        super().__init__(client, capture_config, state_filter=state_filter)
        self._originals: Dict[int, Dict[str, Any]] = {}
        self._wrapped_agents: List[Any] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_AGNO)
        if target is not None:
            self._instrument_agent(target)

    def _on_disconnect(self) -> None:
        for agent in self._wrapped_agents:
            self._unwrap_agent(agent)
        self._wrapped_agents.clear()
        self._originals.clear()

    # ------------------------------------------------------------------
    # Instrumentation
    # ------------------------------------------------------------------

    def _instrument_agent(self, agent: Any) -> None:
        agent_id = id(agent)
        if agent_id in self._originals:
            return
        originals: Dict[str, Any] = {}
        if hasattr(agent, "run"):
            originals["run"] = agent.run
            agent.run = self._wrap_sync(agent, agent.run)
        if hasattr(agent, "arun"):
            originals["arun"] = agent.arun
            agent.arun = self._wrap_async(agent, agent.arun)
        self._originals[agent_id] = originals
        self._wrapped_agents.append(agent)

    def _unwrap_agent(self, agent: Any) -> None:
        originals = self._originals.get(id(agent))
        if not originals:
            return
        for method_name, original in originals.items():
            try:
                setattr(agent, method_name, original)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Wrappers
    # ------------------------------------------------------------------

    def _wrap_sync(self, agent: Any, original: Any) -> Any:
        adapter = self

        def _traced_run(*args: Any, **kwargs: Any) -> Any:
            if not adapter._connected:
                return original(*args, **kwargs)
            input_data = kwargs.get("message") or (args[0] if args else None)
            adapter._begin_run()
            adapter._start_timer("run")
            adapter._on_run_start(agent, input_data)
            error: Optional[Exception] = None
            result = None
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                adapter._on_run_end(agent, result, error)
                adapter._end_run()
            return result

        _traced_run._layerlens_original = original  # type: ignore[attr-defined]
        return _traced_run

    def _wrap_async(self, agent: Any, original: Any) -> Any:
        adapter = self

        async def _traced_arun(*args: Any, **kwargs: Any) -> Any:
            if not adapter._connected:
                return await original(*args, **kwargs)
            input_data = kwargs.get("message") or (args[0] if args else None)
            adapter._begin_run()
            adapter._start_timer("run")
            adapter._on_run_start(agent, input_data)
            error: Optional[Exception] = None
            result = None
            try:
                result = await original(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                adapter._on_run_end(agent, result, error)
                adapter._end_run()
            return result

        _traced_arun._layerlens_original = original  # type: ignore[attr-defined]
        return _traced_arun

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    @resilient_callback(callback_name="_on_run_start")
    def _on_run_start(self, agent: Any, input_data: Any) -> None:
        root = self._get_root_span()
        name = _agent_name(agent)
        model = _model_id(agent)
        payload = self._payload(agent_name=name)
        if model:
            payload["model"] = model
        self._set_if_capturing(payload, "input", safe_serialize(input_data))
        self._filter_payload(payload, "input")
        self._emit("agent.input", payload, span_id=root, parent_span_id=None, span_name=f"agno:{name}")

    @resilient_callback(callback_name="_on_run_end")
    def _on_run_end(self, agent: Any, result: Any, error: Optional[Exception]) -> None:
        self._emit_output(agent, result, error)
        if result is not None:
            self._emit_model(agent, result)
            self._emit_tools(result)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _emit_output(self, agent: Any, result: Any, error: Optional[Exception]) -> None:
        root = self._get_root_span()
        name = _agent_name(agent)
        model = _model_id(agent)
        latency_ms = self._stop_timer("run")

        output = getattr(result, "content", None) if result is not None else None
        payload = self._payload(agent_name=name)
        if model:
            payload["model"] = model
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if error:
            payload["error"] = str(error)
            payload["error_type"] = type(error).__name__
        self._set_if_capturing(payload, "output", safe_serialize(output))
        self._filter_payload(payload, "output")
        self._emit("agent.output", payload, span_id=root, parent_span_id=None, span_name=f"agno:{name}")

    def _emit_model(self, agent: Any, result: Any) -> None:
        model = _model_id(agent)
        if not model:
            return
        root = self._get_root_span()
        tokens = _extract_tokens(result)

        span_id = self._new_span_id()
        payload = self._payload(model=model)
        payload.update(tokens)
        self._emit("model.invoke", payload, span_id=span_id, parent_span_id=root, span_name="model.invoke")

        if tokens:
            cost_payload = self._payload(model=model)
            cost_payload.update(tokens)
            self._emit("cost.record", cost_payload, span_id=span_id, parent_span_id=root)

    def _emit_tools(self, result: Any) -> None:
        root = self._get_root_span()
        for tool in _extract_tools(result):
            span_id = self._new_span_id()

            call_payload = self._payload(tool_name=tool["tool_name"])
            self._set_if_capturing(call_payload, "input", safe_serialize(tool.get("tool_args")))
            self._filter_payload(call_payload, "input")
            self._emit("tool.call", call_payload, span_id=span_id, parent_span_id=root)

            result_payload = self._payload(tool_name=tool["tool_name"])
            self._set_if_capturing(result_payload, "output", safe_serialize(tool.get("result")))
            if tool.get("latency_ms") is not None:
                result_payload["latency_ms"] = tool["latency_ms"]
            self._filter_payload(result_payload, "output")
            self._emit("tool.result", result_payload, span_id=span_id, parent_span_id=root)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _agent_name(agent: Any) -> str:
    return getattr(agent, "name", None) or "agno_agent"
