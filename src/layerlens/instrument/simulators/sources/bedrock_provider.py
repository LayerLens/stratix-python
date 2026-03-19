"""AWS Bedrock source formatter.

Supports 6 model families: Anthropic, Meta, Cohere, Amazon, AI21, Mistral.
Adds Bedrock-specific attributes: guardrail_id, knowledge_base_id, agent_id.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan, SpanType
from .base import BaseSourceFormatter, ProviderProfile


BEDROCK_FAMILIES = {
    "anthropic": {
        "models": [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
        ],
        "default": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    },
    "meta": {
        "models": ["meta.llama3-1-70b-instruct-v1:0", "meta.llama3-1-8b-instruct-v1:0"],
        "default": "meta.llama3-1-70b-instruct-v1:0",
    },
    "cohere": {
        "models": ["cohere.command-r-plus-v1:0", "cohere.command-r-v1:0"],
        "default": "cohere.command-r-plus-v1:0",
    },
    "amazon": {
        "models": ["amazon.titan-text-premier-v2:0", "amazon.titan-text-express-v1"],
        "default": "amazon.titan-text-premier-v2:0",
    },
    "ai21": {
        "models": ["ai21.jamba-1-5-large-v1:0", "ai21.jamba-1-5-mini-v1:0"],
        "default": "ai21.jamba-1-5-large-v1:0",
    },
    "mistral": {
        "models": ["mistral.mistral-large-2407-v1:0", "mistral.mixtral-8x7b-instruct-v0:1"],
        "default": "mistral.mistral-large-2407-v1:0",
    },
}


class BedrockSourceFormatter(BaseSourceFormatter):
    """Source 7: AWS Bedrock (6 model families)."""

    def __init__(self, family: str = "anthropic"):
        self._family = family

    def get_default_profile(self) -> ProviderProfile:
        family_info = BEDROCK_FAMILIES.get(self._family, BEDROCK_FAMILIES["anthropic"])
        return ProviderProfile(
            provider_name="bedrock",
            default_model=family_info["default"],
            models=family_info["models"],
            service_name="bedrock-service",
            service_version="1.0.0",
            extra={
                "aws.region": "us-east-1",
                "aws.bedrock.family": self._family,
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
        self._add_bedrock_attributes(span, profile)
        return span

    def _add_bedrock_attributes(
        self, span: SimulatedSpan, profile: ProviderProfile
    ) -> None:
        if span.span_type != SpanType.LLM:
            return
        attrs = span.attributes
        extra = profile.extra
        attrs["cloud.provider"] = "aws"
        attrs["cloud.region"] = extra.get("aws.region", "us-east-1")
        attrs["aws.bedrock.family"] = extra.get("aws.bedrock.family", "anthropic")
        attrs["gen_ai.system"] = "aws.bedrock"
        # Bedrock-specific optional attributes — only set when actual values exist
        # (OTel attributes cannot be None)

    def get_resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "bedrock-service",
            "service.version": "1.0.0",
            "cloud.provider": "aws",
            "cloud.platform": "aws_bedrock",
            "cloud.region": "us-east-1",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.29.0",
        }

    def get_scope(self) -> tuple[str, str]:
        return ("stratix.bedrock", "0.1.0")
