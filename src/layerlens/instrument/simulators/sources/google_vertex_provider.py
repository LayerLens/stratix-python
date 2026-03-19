"""Google Vertex AI source formatter.

Adds Vertex-specific attributes: enum finish_reason, safety_ratings.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan, SpanType
from .base import BaseSourceFormatter, ProviderProfile


class GoogleVertexSourceFormatter(BaseSourceFormatter):
    """Source 8: Google Vertex AI (enum finish_reason, safety_ratings)."""

    def get_default_profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider_name="google_vertex",
            default_model="gemini-1.5-pro",
            models=[
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-2.0-flash",
                "gemini-2.5-pro",
            ],
            service_name="vertex-ai-service",
            service_version="1.0.0",
            extra={
                "gcp.project_id": "my-project-123",
                "gcp.region": "us-central1",
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
        self._add_vertex_attributes(span, profile)
        return span

    def _add_vertex_attributes(
        self, span: SimulatedSpan, profile: ProviderProfile
    ) -> None:
        if span.span_type != SpanType.LLM:
            return
        attrs = span.attributes
        extra = profile.extra
        attrs["cloud.provider"] = "gcp"
        attrs["cloud.region"] = extra.get("gcp.region", "us-central1")
        attrs["gcp.project_id"] = extra.get("gcp.project_id", "my-project-123")
        attrs["gen_ai.system"] = "vertex_ai"
        # Vertex uses STOP enum instead of "stop" string
        if span.finish_reasons == ["stop"]:
            attrs["gen_ai.response.finish_reasons"] = ["STOP"]
        # Safety ratings (Vertex-specific) — serialized to JSON string since
        # OTel attributes cannot contain nested dicts/objects
        import json
        attrs["gen_ai.google.safety_ratings"] = json.dumps([
            {"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "NEGLIGIBLE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "NEGLIGIBLE"},
        ])

    def get_resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "vertex-ai-service",
            "service.version": "1.0.0",
            "cloud.provider": "gcp",
            "cloud.platform": "gcp_vertex_ai",
            "cloud.region": "us-central1",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.29.0",
        }

    def get_scope(self) -> tuple[str, str]:
        return ("stratix.google_vertex", "0.1.0")
