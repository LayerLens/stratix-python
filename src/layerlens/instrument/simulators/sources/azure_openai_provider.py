"""Azure OpenAI source formatter.

Adds Azure-specific attributes: deployment, endpoint, api_version.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan, SpanType
from .base import BaseSourceFormatter, ProviderProfile


class AzureOpenAISourceFormatter(BaseSourceFormatter):
    """Source 6: Azure OpenAI (deployment, endpoint, api_version)."""

    def get_default_profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider_name="azure_openai",
            default_model="gpt-4o",
            models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            service_name="azure-openai-service",
            service_version="1.0.0",
            extra={
                "azure.deployment": "gpt-4o-deployment",
                "azure.endpoint": "https://myresource.openai.azure.com",
                "azure.api_version": "2024-10-21",
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
        self._add_azure_attributes(span, profile)
        return span

    def _add_azure_attributes(
        self, span: SimulatedSpan, profile: ProviderProfile
    ) -> None:
        if span.span_type != SpanType.LLM:
            return
        attrs = span.attributes
        extra = profile.extra
        attrs["az.namespace"] = "Microsoft.CognitiveServices"
        attrs["gen_ai.azure.deployment"] = extra.get("azure.deployment", "gpt-4o-deployment")
        attrs["server.address"] = extra.get("azure.endpoint", "https://myresource.openai.azure.com")
        attrs["gen_ai.azure.api_version"] = extra.get("azure.api_version", "2024-10-21")
        # Azure OpenAI also has system_fingerprint
        attrs["gen_ai.openai.response.system_fingerprint"] = "fp_" + span.span_id[:10]

    def get_resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "azure-openai-service",
            "service.version": "1.0.0",
            "cloud.provider": "azure",
            "cloud.platform": "azure_openai",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.29.0",
        }

    def get_scope(self) -> tuple[str, str]:
        return ("stratix.azure_openai", "0.1.0")
