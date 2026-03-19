"""Generic OTel source formatter.

Produces spans with gen_ai.* semantic conventions only,
without any provider-specific extensions.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan
from .base import BaseSourceFormatter, ProviderProfile


class GenericOTelFormatter(BaseSourceFormatter):
    """Source 1: Generic OpenTelemetry gen_ai.* spans."""

    def get_default_profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider_name="openai",
            default_model="gpt-4o",
            service_name="generic-otel-service",
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
        return span

    def get_resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "generic-otel-service",
            "service.version": "1.0.0",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.29.0",
        }

    def get_scope(self) -> tuple[str, str]:
        return ("opentelemetry.instrumentation.genai", "0.1.0")
