from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from ._base_framework import FrameworkAdapter
from ._utils import safe_serialize
from ..._collector import TraceCollector
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

_HAS_OPENAI_AGENTS = False
try:
    from agents.tracing import TracingProcessor  # pyright: ignore[reportMissingImports]

    _HAS_OPENAI_AGENTS = True
except (ImportError, Exception):
    TracingProcessor = None  # type: ignore[assignment,misc]

# Real TracingProcessor when installed, plain object otherwise.
_Base: Any = TracingProcessor if _HAS_OPENAI_AGENTS else object


class OpenAIAgentsAdapter(_Base, FrameworkAdapter):
    """OpenAI Agents SDK adapter using the TracingProcessor API.

    The adapter *is* the trace processor — it registers itself globally
    to receive all span lifecycle events, then maps agent, generation,
    function, handoff, and guardrail spans to flat layerlens events.

    Unlike other adapters that use a single collector, this adapter manages
    per-trace collectors because the SDK can run multiple concurrent traces
    through the same global processor.

    Usage::

        adapter = OpenAIAgentsAdapter(client)
        adapter.connect()
        result = await Runner.run(agent, "hello")
        adapter.disconnect()
    """

    name = "openai-agents"
    package = "openai-agents"

    _SPAN_HANDLERS = {
        "agent": "_handle_agent_span",
        "generation": "_handle_generation_span",
        "function": "_handle_function_span",
        "handoff": "_handle_handoff_span",
        "guardrail": "_handle_guardrail_span",
        "response": "_handle_response_span",
    }

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        FrameworkAdapter.__init__(self, client, capture_config)
        self._collectors: Dict[str, TraceCollector] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_OPENAI_AGENTS)
        from agents import add_trace_processor  # pyright: ignore[reportMissingImports]

        add_trace_processor(self)  # type: ignore[arg-type]

    def _on_disconnect(self) -> None:
        from agents import set_trace_processors  # pyright: ignore[reportMissingImports]

        set_trace_processors([])
        with self._lock:
            self._collectors.clear()

    # ------------------------------------------------------------------
    # TracingProcessor interface
    # ------------------------------------------------------------------

    def on_trace_start(self, trace: Any) -> None:
        try:
            self._get_collector(trace.trace_id)
        except Exception:
            log.warning("layerlens: error in on_trace_start", exc_info=True)

    def on_trace_end(self, trace: Any) -> None:
        try:
            with self._lock:
                collector = self._collectors.pop(trace.trace_id, None)
            if collector is not None:
                collector.flush()
        except Exception:
            log.warning("layerlens: error in on_trace_end", exc_info=True)

    def on_span_start(self, span: Any) -> None:
        pass

    def on_span_end(self, span: Any) -> None:
        try:
            span_type = getattr(span.span_data, "type", None) or ""
            handler_name = self._SPAN_HANDLERS.get(span_type)
            if handler_name is not None:
                getattr(self, handler_name)(span)
        except Exception:
            log.warning("layerlens: error handling OpenAI Agents span", exc_info=True)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Per-trace collector
    # ------------------------------------------------------------------

    def _get_collector(self, trace_id: str) -> TraceCollector:
        with self._lock:
            if trace_id not in self._collectors:
                self._collectors[trace_id] = TraceCollector(self._client, self._config)
            return self._collectors[trace_id]

    # ------------------------------------------------------------------
    # Span handlers
    # ------------------------------------------------------------------

    def _handle_agent_span(self, span: Any) -> None:
        data = span.span_data
        collector = self._get_collector(span.trace_id)
        agent_name = getattr(data, "name", "unknown")
        span_id = span.span_id or self._new_span_id()
        parent_id = span.parent_id

        input_payload = self._payload(agent_name=agent_name)
        for key in ("tools", "handoffs", "output_type"):
            val = getattr(data, key, None)
            if val:
                input_payload[key] = val

        collector.emit(
            "agent.input", input_payload,
            span_id=span_id, parent_span_id=parent_id,
            span_name=f"agent:{agent_name}",
        )

        event_type = "agent.error" if span.error else "agent.output"
        out_payload = self._payload(
            agent_name=agent_name,
            status="error" if span.error else "ok",
        )
        duration_ms = _compute_duration_ms(span)
        if duration_ms is not None:
            out_payload["duration_ms"] = duration_ms
        if span.error:
            out_payload["error"] = safe_serialize(span.error)

        collector.emit(
            event_type, out_payload,
            span_id=span_id, parent_span_id=parent_id,
            span_name=f"agent:{agent_name}",
        )

    def _handle_generation_span(self, span: Any) -> None:
        data = span.span_data
        collector = self._get_collector(span.trace_id)
        model = getattr(data, "model", None) or "unknown"
        span_id = span.span_id or self._new_span_id()
        parent_id = span.parent_id

        payload = self._payload(model=model)
        tokens = self._normalize_tokens(getattr(data, "usage", None))
        payload.update(tokens)

        duration_ms = _compute_duration_ms(span)
        if duration_ms is not None:
            payload["latency_ms"] = duration_ms

        model_config = getattr(data, "model_config", None)
        if model_config:
            payload["model_config"] = safe_serialize(model_config)

        self._set_if_capturing(payload, "messages", safe_serialize(getattr(data, "input", None)))
        self._set_if_capturing(payload, "output_message", safe_serialize(getattr(data, "output", None)))

        if span.error:
            payload["error"] = safe_serialize(span.error)
            collector.emit("agent.error", payload, span_id=span_id, parent_span_id=parent_id)
        else:
            collector.emit("model.invoke", payload, span_id=span_id, parent_span_id=parent_id)

        if tokens:
            cost_payload = self._payload(model=model)
            cost_payload.update(tokens)
            collector.emit("cost.record", cost_payload, span_id=span_id, parent_span_id=parent_id)

    def _handle_function_span(self, span: Any) -> None:
        data = span.span_data
        collector = self._get_collector(span.trace_id)
        tool_name = getattr(data, "name", "unknown")
        span_id = span.span_id or self._new_span_id()
        parent_id = span.parent_id

        payload = self._payload(tool_name=tool_name)
        self._set_if_capturing(payload, "input", safe_serialize(getattr(data, "input", None)))
        self._set_if_capturing(payload, "output", safe_serialize(getattr(data, "output", None)))

        duration_ms = _compute_duration_ms(span)
        if duration_ms is not None:
            payload["latency_ms"] = duration_ms

        mcp_data = getattr(data, "mcp_data", None)
        if mcp_data:
            payload["mcp_data"] = safe_serialize(mcp_data)

        if span.error:
            payload["error"] = safe_serialize(span.error)
            collector.emit("agent.error", payload, span_id=span_id, parent_span_id=parent_id)
        else:
            collector.emit("tool.call", payload, span_id=span_id, parent_span_id=parent_id)

    def _handle_handoff_span(self, span: Any) -> None:
        data = span.span_data
        self._get_collector(span.trace_id).emit(
            "agent.handoff",
            self._payload(
                from_agent=getattr(data, "from_agent", None) or "unknown",
                to_agent=getattr(data, "to_agent", None) or "unknown",
            ),
            span_id=span.span_id or self._new_span_id(),
            parent_span_id=span.parent_id,
        )

    def _handle_guardrail_span(self, span: Any) -> None:
        data = span.span_data
        self._get_collector(span.trace_id).emit(
            "evaluation.result",
            self._payload(
                guardrail_name=getattr(data, "name", "unknown"),
                triggered=getattr(data, "triggered", False),
            ),
            span_id=span.span_id or self._new_span_id(),
            parent_span_id=span.parent_id,
        )

    def _handle_response_span(self, span: Any) -> None:
        data = span.span_data
        response = getattr(data, "response", None)
        if response is None:
            return

        collector = self._get_collector(span.trace_id)
        span_id = span.span_id or self._new_span_id()
        parent_id = span.parent_id
        payload = self._payload()

        model = getattr(response, "model", None)
        if model:
            payload["model"] = model

        usage = getattr(response, "usage", None)
        tokens = self._normalize_tokens(usage)
        # OpenAI-specific detailed token breakdowns
        if usage is not None:
            input_details = getattr(usage, "input_tokens_details", None)
            if input_details:
                cached = getattr(input_details, "cached_tokens", 0) or 0
                if cached:
                    tokens["cached_tokens"] = cached
            output_details = getattr(usage, "output_tokens_details", None)
            if output_details:
                reasoning = getattr(output_details, "reasoning_tokens", 0) or 0
                if reasoning:
                    tokens["reasoning_tokens"] = reasoning
        payload.update(tokens)

        duration_ms = _compute_duration_ms(span)
        if duration_ms is not None:
            payload["latency_ms"] = duration_ms

        if span.error:
            payload["error"] = safe_serialize(span.error)
            collector.emit("agent.error", payload, span_id=span_id, parent_span_id=parent_id)
        else:
            collector.emit("model.invoke", payload, span_id=span_id, parent_span_id=parent_id)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _compute_duration_ms(span: Any) -> Optional[float]:
    started = getattr(span, "started_at", None)
    ended = getattr(span, "ended_at", None)
    if started is None or ended is None:
        return None
    try:
        if isinstance(started, str):
            started = datetime.fromisoformat(started)
        if isinstance(ended, str):
            ended = datetime.fromisoformat(ended)
        return (ended - started).total_seconds() * 1000
    except Exception:
        return None
