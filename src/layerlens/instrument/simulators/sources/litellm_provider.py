"""LiteLLM source formatter.

Adds LiteLLM-specific attributes: model prefix routing, callback attrs.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan, SpanType
from .base import BaseSourceFormatter, ProviderProfile


class LiteLLMSourceFormatter(BaseSourceFormatter):
    """Source 10: LiteLLM (model prefix routing, callback attrs)."""

    def get_default_profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider_name="litellm",
            default_model="gpt-4o",
            models=["gpt-4o", "claude-sonnet-4-20250514", "gemini-1.5-pro"],
            service_name="litellm-proxy",
            service_version="1.0.0",
            extra={
                "litellm.proxy_base_url": "http://localhost:4000",
                "litellm.model_group": "default",
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
        self._add_litellm_attributes(span, profile)
        return span

    def _add_litellm_attributes(
        self, span: SimulatedSpan, profile: ProviderProfile
    ) -> None:
        if span.span_type != SpanType.LLM:
            return
        attrs = span.attributes
        extra = profile.extra
        attrs["gen_ai.system"] = "litellm"
        attrs["litellm.proxy_base_url"] = extra.get(
            "litellm.proxy_base_url", "http://localhost:4000"
        )
        attrs["litellm.model_group"] = extra.get("litellm.model_group", "default")
        # LiteLLM uses model prefix routing
        model = span.model or profile.default_model
        if "/" not in model:
            attrs["litellm.routed_model"] = f"openai/{model}"
        else:
            attrs["litellm.routed_model"] = model
        attrs["litellm.cache_hit"] = False

    def get_resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "litellm-proxy",
            "service.version": "1.0.0",
            "server.address": "localhost:4000",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.29.0",
        }

    def get_scope(self) -> tuple[str, str]:
        return ("stratix.litellm", "0.1.0")
