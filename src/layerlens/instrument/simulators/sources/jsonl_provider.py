"""JSONL source formatter.

Produces one event per line in STRATIX's native JSONL schema.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan, SpanType
from .base import BaseSourceFormatter, ProviderProfile


class JSONLSourceFormatter(BaseSourceFormatter):
    """Source 12: JSONL (one event per line, STRATIX schema)."""

    def get_default_profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider_name="openai",
            default_model="gpt-4o",
            service_name="jsonl-import",
            service_version="1.0.0",
        )

    def enrich_span(
        self,
        span: SimulatedSpan,
        profile: ProviderProfile,
        include_content: bool = False,
    ) -> SimulatedSpan:
        self._add_common_genai_attributes(span, profile)
        self._add_tool_attributes(span)
        self._add_agent_attributes(span)
        self._add_evaluation_attributes(span)
        self._add_error_attributes(span)
        self._add_streaming_attributes(span)
        self._add_jsonl_attributes(span)
        return span

    def _add_jsonl_attributes(self, span: SimulatedSpan) -> None:
        attrs = span.attributes
        attrs["stratix.import.format"] = "jsonl"
        # Map span type to STRATIX event type
        event_type_map = {
            SpanType.AGENT: "agent.input",
            SpanType.LLM: "model.invoke",
            SpanType.TOOL: "tool.call",
            SpanType.EVALUATION: "evaluation.result",
        }
        attrs["stratix.event_type"] = event_type_map.get(span.span_type, "unknown")

    def get_resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "jsonl-import",
            "service.version": "1.0.0",
            "stratix.import.format": "jsonl",
        }

    def get_scope(self) -> tuple[str, str]:
        return ("stratix.jsonl", "0.1.0")
