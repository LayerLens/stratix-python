"""Base source formatter ABC.

Source formatters enrich provider-neutral SimulatedSpans with
source-specific attributes (gen_ai.*, sf.*, aws.bedrock.*, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..span_model import SimulatedSpan, SpanType


@dataclass
class ProviderProfile:
    """Provider-specific default values."""

    provider_name: str
    default_model: str
    default_operation: str = "chat"
    models: list[str] = field(default_factory=list)
    service_name: str = ""
    service_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class BaseSourceFormatter(ABC):
    """Abstract base for source formatters.

    Each of the 12 sources implements this to add source-specific
    attributes to SimulatedSpans.
    """

    @abstractmethod
    def get_default_profile(self) -> ProviderProfile:
        """Return the default provider profile for this source."""

    @abstractmethod
    def enrich_span(
        self,
        span: SimulatedSpan,
        profile: ProviderProfile,
        include_content: bool = False,
    ) -> SimulatedSpan:
        """Enrich a span with source-specific attributes."""

    @abstractmethod
    def get_resource_attributes(self) -> dict[str, Any]:
        """Return resource-level attributes for this source."""

    @abstractmethod
    def get_scope(self) -> tuple[str, str]:
        """Return (scope_name, scope_version) for this source."""

    def _add_common_genai_attributes(
        self,
        span: SimulatedSpan,
        profile: ProviderProfile,
    ) -> None:
        """Add common gen_ai.* attributes to any LLM span."""
        if span.span_type != SpanType.LLM:
            return

        attrs = span.attributes
        attrs["gen_ai.system"] = profile.provider_name
        attrs["gen_ai.operation.name"] = span.operation
        attrs["gen_ai.request.model"] = span.model or profile.default_model
        attrs["gen_ai.response.model"] = span.model or profile.default_model

        if span.token_usage:
            attrs["gen_ai.usage.input_tokens"] = span.token_usage.prompt_tokens
            attrs["gen_ai.usage.output_tokens"] = span.token_usage.completion_tokens

        if span.temperature is not None:
            attrs["gen_ai.request.temperature"] = span.temperature
        if span.max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = span.max_tokens
        if span.top_p is not None:
            attrs["gen_ai.request.top_p"] = span.top_p
        if span.finish_reasons:
            attrs["gen_ai.response.finish_reasons"] = span.finish_reasons
        if span.response_id:
            attrs["gen_ai.response.id"] = span.response_id

    def _add_tool_attributes(self, span: SimulatedSpan) -> None:
        """Add gen_ai.tool.* attributes to tool spans."""
        if span.span_type != SpanType.TOOL:
            return
        attrs = span.attributes
        if span.tool_name:
            attrs["gen_ai.tool.name"] = span.tool_name
        if span.tool_description:
            attrs["gen_ai.tool.description"] = span.tool_description
        if span.tool_call_id:
            attrs["gen_ai.tool.call.id"] = span.tool_call_id

    def _add_agent_attributes(self, span: SimulatedSpan) -> None:
        """Add gen_ai.agent.* attributes to agent spans."""
        if span.span_type != SpanType.AGENT:
            return
        attrs = span.attributes
        if span.agent_name:
            attrs["gen_ai.agent.name"] = span.agent_name
        if span.agent_description:
            attrs["gen_ai.agent.description"] = span.agent_description

    def _add_evaluation_attributes(self, span: SimulatedSpan) -> None:
        """Add evaluation attributes."""
        if span.span_type != SpanType.EVALUATION:
            return
        attrs = span.attributes
        if span.eval_score is not None:
            attrs["gen_ai.evaluation.score.value"] = span.eval_score
        if span.eval_dimension:
            attrs["gen_ai.evaluation.name"] = span.eval_dimension
        if span.eval_label:
            attrs["gen_ai.evaluation.score.label"] = span.eval_label
        if span.eval_grader_id:
            attrs["stratix.evaluation.grader_id"] = span.eval_grader_id

    def _add_error_attributes(self, span: SimulatedSpan) -> None:
        """Add error-related attributes."""
        if span.error_type:
            span.attributes["error.type"] = span.error_type
        if span.http_status_code:
            span.attributes["http.response.status_code"] = span.http_status_code

    def _add_streaming_attributes(self, span: SimulatedSpan) -> None:
        """Add streaming-related attributes."""
        if not span.is_streaming:
            return
        span.attributes["gen_ai.is_streaming"] = True
        if span.ttft_ms is not None:
            span.attributes["gen_ai.server.time_to_first_token"] = span.ttft_ms / 1000.0
        if span.tpot_ms is not None:
            span.attributes["gen_ai.server.time_per_output_token"] = span.tpot_ms / 1000.0
