"""AgentForce OTLP source formatter.

Produces spans with both sf.* (Salesforce) and gen_ai.* attributes,
matching the AgentForce OTel integration pattern.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan, SpanType
from .base import BaseSourceFormatter, ProviderProfile


class AgentForceOTLPFormatter(BaseSourceFormatter):
    """Source 2: AgentForce OTLP (sf.* + gen_ai.*)."""

    def get_default_profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider_name="openai",
            default_model="gpt-4o",
            service_name="agentforce-service",
            service_version="1.0.0",
            extra={
                "sf.org_id": "00D5f000000XXXX",
                "sf.agent_id": "0XxAF000000YYYY",
                "sf.bot_version": "1.0",
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
        self._add_salesforce_attributes(span, profile)
        return span

    def _add_salesforce_attributes(
        self, span: SimulatedSpan, profile: ProviderProfile
    ) -> None:
        attrs = span.attributes
        extra = profile.extra

        attrs["sf.org_id"] = extra.get("sf.org_id", "00D5f000000XXXX")
        attrs["sf.agent_id"] = extra.get("sf.agent_id", "0XxAF000000YYYY")
        attrs["sf.bot_version"] = extra.get("sf.bot_version", "1.0")

        if span.span_type == SpanType.AGENT:
            attrs["sf.agent.name"] = span.agent_name or "AgentForce_Agent"
            attrs["sf.agent.type"] = "copilot"
            attrs["sf.conversation.id"] = "conv_sf_001"
        elif span.span_type == SpanType.LLM:
            attrs["sf.llm.api_type"] = "chat_completion"
            attrs["sf.llm.trust_layer.enabled"] = True
        elif span.span_type == SpanType.TOOL:
            attrs["sf.action.type"] = "flow"
            attrs["sf.action.api_name"] = span.tool_name or "Unknown_Action"

    def get_resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "agentforce-service",
            "service.version": "1.0.0",
            "sf.org_id": "00D5f000000XXXX",
            "sf.instance_url": "https://myorg.my.salesforce.com",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.29.0",
        }

    def get_scope(self) -> tuple[str, str]:
        return ("stratix.agentforce.otel", "0.1.0")
