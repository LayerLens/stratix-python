"""OpenAI source formatter.

Adds OpenAI-specific attributes: system_fingerprint, seed, service_tier.
Matches the output of stratix/sdk/python/adapters/llm_providers/openai_adapter.py.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan, SpanType
from .base import BaseSourceFormatter, ProviderProfile


class OpenAISourceFormatter(BaseSourceFormatter):
    """Source 4: OpenAI (system_fingerprint, seed, service_tier)."""

    def get_default_profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider_name="openai",
            default_model="gpt-4o",
            models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o3-mini"],
            service_name="openai-service",
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
        self._add_openai_attributes(span, profile)
        return span

    def _add_openai_attributes(
        self, span: SimulatedSpan, profile: ProviderProfile
    ) -> None:
        if span.span_type != SpanType.LLM:
            return
        attrs = span.attributes
        attrs["gen_ai.openai.response.system_fingerprint"] = "fp_" + span.span_id[:10]
        attrs["gen_ai.openai.response.service_tier"] = "default"
        # Only set seed attribute when an actual seed value exists (OTel attrs cannot be None)
        if span.attributes.get("gen_ai.request.seed") is not None:
            attrs["gen_ai.openai.request.seed"] = span.attributes["gen_ai.request.seed"]

    def get_resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "openai-service",
            "service.version": "1.0.0",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.29.0",
        }

    def get_scope(self) -> tuple[str, str]:
        return ("stratix.openai", "0.1.0")
