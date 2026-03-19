"""STRATIX native output formatter.

Produces STRATIX canonical event dicts matching the structure
from stratix/ingestion/normalizer.py normalize_otel_span output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..span_model import SimulatedSpan, SimulatedTrace, SpanType
from .base import BaseOutputFormatter


def _ns_to_iso(ns: int) -> str:
    """Convert nanosecond timestamp to ISO 8601 string."""
    dt = datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)
    return dt.isoformat()


def _infer_event_type(span: SimulatedSpan) -> str:
    """Map span type to STRATIX event type."""
    event_type_map = {
        SpanType.LLM: "model.invoke",
        SpanType.TOOL: "tool.call",
        SpanType.AGENT: "agent.input",
        SpanType.EVALUATION: "evaluation.result",
    }
    return event_type_map.get(span.span_type, "unknown")


def _build_payload(span: SimulatedSpan) -> dict[str, Any]:
    """Build STRATIX event payload from span."""
    payload: dict[str, Any] = {}

    if span.span_type == SpanType.LLM:
        payload["provider"] = span.provider
        payload["model"] = span.model
        payload["operation"] = span.operation
        if span.token_usage:
            payload["prompt_tokens"] = span.token_usage.prompt_tokens
            payload["completion_tokens"] = span.token_usage.completion_tokens
            payload["total_tokens"] = span.token_usage.total_tokens
        if span.temperature is not None:
            payload["temperature"] = span.temperature
        if span.finish_reasons:
            payload["finish_reasons"] = span.finish_reasons
        if span.input_messages:
            payload["input_messages"] = span.input_messages
        if span.output_message:
            payload["output_message"] = span.output_message
        payload["latency_ms"] = span.duration_ms

    elif span.span_type == SpanType.TOOL:
        payload["tool_name"] = span.tool_name
        payload["tool_call_id"] = span.tool_call_id
        if span.tool_input:
            payload["input"] = span.tool_input
        if span.tool_output:
            payload["output"] = span.tool_output
        payload["latency_ms"] = span.duration_ms

    elif span.span_type == SpanType.AGENT:
        payload["agent_name"] = span.agent_name
        if span.agent_description:
            payload["agent_description"] = span.agent_description

    elif span.span_type == SpanType.EVALUATION:
        payload["dimension"] = span.eval_dimension
        payload["score"] = span.eval_score
        payload["label"] = span.eval_label
        if span.eval_grader_id:
            payload["grader_id"] = span.eval_grader_id

    if span.error_type:
        payload["error"] = {
            "type": span.error_type,
            "http_status_code": span.http_status_code,
            "message": span.status_message,
        }

    return payload


class STRATIXNativeFormatter(BaseOutputFormatter):
    """STRATIX canonical event dict output format.

    Matches the structure from normalizer.normalize_otel_span().
    """

    def format_trace(self, trace: SimulatedTrace) -> dict[str, Any]:
        """Format a SimulatedTrace as STRATIX native event dicts."""
        events = []
        for span in trace.spans:
            event = {
                "event_type": _infer_event_type(span),
                "identity": {
                    "trace_id": trace.trace_id,
                    "span_id": span.span_id,
                },
                "timestamp": _ns_to_iso(span.start_time_unix_nano),
                "payload": _build_payload(span),
            }
            if span.parent_span_id:
                event["identity"]["parent_span_id"] = span.parent_span_id
            events.append(event)

        return {
            "trace_id": trace.trace_id,
            "scenario": trace.scenario,
            "topic": trace.topic,
            "events": events,
        }
