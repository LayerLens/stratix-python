"""Langfuse JSON output formatter.

Produces trace + observation structure matching Langfuse's data model,
compatible with agentforce-synthetic-data/scenario_*/langfuse/ structure.
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


def _span_to_observation(span: SimulatedSpan, trace_id: str) -> dict[str, Any]:
    """Convert a SimulatedSpan to a Langfuse observation."""
    obs_type_map = {
        SpanType.LLM: "GENERATION",
        SpanType.TOOL: "SPAN",
        SpanType.AGENT: "SPAN",
        SpanType.EVALUATION: "EVENT",
    }

    observation: dict[str, Any] = {
        "id": span.span_id,
        "traceId": trace_id,
        "type": obs_type_map.get(span.span_type, "SPAN"),
        "name": span.name,
        "startTime": _ns_to_iso(span.start_time_unix_nano),
        "endTime": _ns_to_iso(span.end_time_unix_nano),
        "metadata": {},
    }

    if span.parent_span_id:
        observation["parentObservationId"] = span.parent_span_id

    if span.span_type == SpanType.LLM:
        observation["model"] = span.model
        observation["modelParameters"] = {}
        if span.temperature is not None:
            observation["modelParameters"]["temperature"] = span.temperature
        if span.max_tokens is not None:
            observation["modelParameters"]["max_tokens"] = span.max_tokens

        if span.token_usage:
            observation["usage"] = {
                "input": span.token_usage.prompt_tokens,
                "output": span.token_usage.completion_tokens,
                "total": span.token_usage.total_tokens,
            }

        if span.input_messages:
            observation["input"] = span.input_messages
        if span.output_message:
            observation["output"] = span.output_message

        if span.finish_reasons:
            observation["completionStartTime"] = _ns_to_iso(span.start_time_unix_nano)

    elif span.span_type == SpanType.TOOL:
        if span.tool_input:
            observation["input"] = span.tool_input
        if span.tool_output:
            observation["output"] = span.tool_output

    elif span.span_type == SpanType.EVALUATION:
        observation["metadata"]["score"] = span.eval_score
        observation["metadata"]["dimension"] = span.eval_dimension
        observation["metadata"]["label"] = span.eval_label

    return observation


class LangfuseJSONFormatter(BaseOutputFormatter):
    """Langfuse trace + observations JSON output format."""

    def format_trace(self, trace: SimulatedTrace) -> dict[str, Any]:
        """Format a SimulatedTrace as Langfuse-compatible JSON."""
        observations = []
        for span in trace.spans:
            obs = _span_to_observation(span, trace.trace_id)
            observations.append(obs)

        root = trace.root_span
        return {
            "id": trace.trace_id,
            "name": root.name if root else "trace",
            "timestamp": _ns_to_iso(
                trace.spans[0].start_time_unix_nano if trace.spans else 0
            ),
            "metadata": {
                "scenario": trace.scenario,
                "topic": trace.topic,
                "source_format": trace.source_format,
                "seed": trace.seed,
            },
            "observations": observations,
            "tags": [trace.scenario or "unknown", trace.source_format or "unknown"],
        }
