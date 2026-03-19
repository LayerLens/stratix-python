"""Anthropic source formatter.

Adds Anthropic-specific attributes: cache tokens, stop_reason.
Matches stratix/sdk/python/adapters/llm_providers/anthropic_adapter.py.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan, SpanType
from .base import BaseSourceFormatter, ProviderProfile


class AnthropicSourceFormatter(BaseSourceFormatter):
    """Source 5: Anthropic (cache tokens, stop_reason)."""

    def get_default_profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider_name="anthropic",
            default_model="claude-sonnet-4-20250514",
            models=[
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-haiku-4-20250414",
                "claude-3-5-sonnet-20241022",
            ],
            service_name="anthropic-service",
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
        self._add_anthropic_attributes(span)
        return span

    def _add_anthropic_attributes(self, span: SimulatedSpan) -> None:
        if span.span_type != SpanType.LLM:
            return
        attrs = span.attributes
        # Anthropic cache token attributes
        if span.token_usage and span.token_usage.cached_tokens:
            attrs["gen_ai.usage.cache_read_input_tokens"] = span.token_usage.cached_tokens
        attrs["gen_ai.usage.cache_creation_input_tokens"] = 0
        # Anthropic uses "end_turn" instead of "stop"
        if span.finish_reasons == ["stop"]:
            attrs["gen_ai.response.finish_reasons"] = ["end_turn"]
            span.finish_reasons = list(["end_turn"])  # New list, not mutating original

    def get_resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "anthropic-service",
            "service.version": "1.0.0",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.29.0",
        }

    def get_scope(self) -> tuple[str, str]:
        return ("stratix.anthropic", "0.1.0")
