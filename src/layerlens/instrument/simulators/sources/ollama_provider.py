"""Ollama source formatter.

Adds Ollama-specific attributes: prompt_eval_count, eval_count, $0 cost.
"""

from __future__ import annotations

from typing import Any

from ..span_model import SimulatedSpan, SpanType
from .base import BaseSourceFormatter, ProviderProfile


class OllamaSourceFormatter(BaseSourceFormatter):
    """Source 9: Ollama (local inference, $0 cost)."""

    def get_default_profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider_name="ollama",
            default_model="llama3.1:70b",
            models=[
                "llama3.1:70b",
                "llama3.1:8b",
                "mistral:7b",
                "codellama:34b",
                "phi3:14b",
            ],
            service_name="ollama-service",
            service_version="0.5.0",
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
        self._add_ollama_attributes(span)
        return span

    def _add_ollama_attributes(self, span: SimulatedSpan) -> None:
        if span.span_type != SpanType.LLM:
            return
        attrs = span.attributes
        attrs["gen_ai.system"] = "ollama"
        # Ollama-specific eval metrics
        if span.token_usage:
            attrs["gen_ai.ollama.prompt_eval_count"] = span.token_usage.prompt_tokens
            attrs["gen_ai.ollama.eval_count"] = span.token_usage.completion_tokens
        # Ollama is local — $0 cost
        attrs["gen_ai.usage.cost"] = 0.0
        attrs["server.address"] = "localhost:11434"

    def get_resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "ollama-service",
            "service.version": "0.5.0",
            "server.address": "localhost:11434",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "1.29.0",
        }

    def get_scope(self) -> tuple[str, str]:
        return ("stratix.ollama", "0.1.0")
