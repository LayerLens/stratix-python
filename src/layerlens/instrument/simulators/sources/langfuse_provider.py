"""Langfuse source formatter.

Produces trace + observation structure matching Langfuse's data model.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan, SpanType
from .base import BaseSourceFormatter, ProviderProfile


class LangfuseSourceFormatter(BaseSourceFormatter):
    """Source 11: Langfuse (trace + observation structure)."""

    def get_default_profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider_name="openai",
            default_model="gpt-4o",
            service_name="langfuse-service",
            service_version="1.0.0",
            extra={
                "langfuse.project_id": "proj_abc123",
                "langfuse.host": "https://cloud.langfuse.com",
            },
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
        self._add_langfuse_attributes(span, profile)
        return span

    def _add_langfuse_attributes(
        self, span: SimulatedSpan, profile: ProviderProfile
    ) -> None:
        attrs = span.attributes
        extra = profile.extra
        attrs["langfuse.project_id"] = extra.get("langfuse.project_id", "proj_abc123")

        if span.span_type == SpanType.LLM:
            attrs["langfuse.observation_type"] = "generation"
            attrs["langfuse.model"] = span.model or profile.default_model
            if span.token_usage:
                attrs["langfuse.usage.input"] = span.token_usage.prompt_tokens
                attrs["langfuse.usage.output"] = span.token_usage.completion_tokens
                attrs["langfuse.usage.total"] = span.token_usage.total_tokens
        elif span.span_type == SpanType.TOOL:
            attrs["langfuse.observation_type"] = "span"
        elif span.span_type == SpanType.AGENT:
            attrs["langfuse.observation_type"] = "span"
            attrs["langfuse.trace.name"] = span.agent_name
        elif span.span_type == SpanType.EVALUATION:
            attrs["langfuse.observation_type"] = "event"
            if span.eval_score is not None:
                attrs["langfuse.score.value"] = span.eval_score
                attrs["langfuse.score.name"] = span.eval_dimension

    def get_resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "langfuse-service",
            "service.version": "1.0.0",
            "langfuse.project_id": "proj_abc123",
        }

    def get_scope(self) -> tuple[str, str]:
        return ("stratix.langfuse", "0.1.0")
