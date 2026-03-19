"""AgentForce SOQL DMO source formatter.

Produces records matching the 5 Salesforce Data Model Objects (DMOs)
used in AgentForce SOQL-based trace ingestion.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan, SpanType
from .base import BaseSourceFormatter, ProviderProfile


class AgentForceSOQLFormatter(BaseSourceFormatter):
    """Source 3: AgentForce SOQL DMO records."""

    def get_default_profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider_name="openai",
            default_model="gpt-4o",
            service_name="agentforce-soql",
            service_version="1.0.0",
            extra={
                "sf.org_id": "00D5f000000XXXX",
                "sf.bot_definition_id": "0XxAF000000YYYY",
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
        self._add_soql_attributes(span, profile)
        return span

    def _add_soql_attributes(
        self, span: SimulatedSpan, profile: ProviderProfile
    ) -> None:
        attrs = span.attributes
        extra = profile.extra
        attrs["sf.org_id"] = extra.get("sf.org_id", "00D5f000000XXXX")
        attrs["sf.dmo.source"] = "soql"

        if span.span_type == SpanType.AGENT:
            attrs["sf.dmo.type"] = "BotSession"
            attrs["sf.bot_definition_id"] = extra.get("sf.bot_definition_id", "0XxAF000000YYYY")
            attrs["sf.session.status"] = "Completed"
        elif span.span_type == SpanType.LLM:
            attrs["sf.dmo.type"] = "GenAiInteraction"
            attrs["sf.interaction.type"] = "chat_completion"
        elif span.span_type == SpanType.TOOL:
            attrs["sf.dmo.type"] = "BotSessionAction"
            attrs["sf.action.invocation_status"] = (
                "Error" if span.error_type else "Success"
            )

    def get_resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "agentforce-soql",
            "service.version": "1.0.0",
            "sf.org_id": "00D5f000000XXXX",
            "sf.dmo.source": "soql",
        }

    def get_scope(self) -> tuple[str, str]:
        return ("stratix.agentforce.soql", "0.1.0")
