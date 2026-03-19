"""Simulated span and trace models.

Provider-neutral Pydantic models representing generated trace data.
SimulatedTrace is the internal representation that flows through the
3-layer architecture: Scenario → Source → Output.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SpanType(str, Enum):
    """Span type classification."""

    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    EVALUATION = "evaluation"


class SpanKind(int, Enum):
    """OTel SpanKind values (proto numeric)."""

    INTERNAL = 1
    SERVER = 2
    CLIENT = 3
    PRODUCER = 4
    CONSUMER = 5


class SpanStatus(str, Enum):
    """Span status."""

    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


SPAN_TYPE_TO_KIND: dict[SpanType, SpanKind] = {
    SpanType.AGENT: SpanKind.SERVER,
    SpanType.LLM: SpanKind.CLIENT,
    SpanType.TOOL: SpanKind.INTERNAL,
    SpanType.EVALUATION: SpanKind.INTERNAL,
}


class TokenUsage(BaseModel):
    """Token usage for LLM spans."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens


class SimulatedSpan(BaseModel):
    """A single simulated span within a trace.

    Provider-neutral representation enriched by source formatters.
    """

    span_id: str
    parent_span_id: str | None = None
    span_type: SpanType
    name: str
    start_time_unix_nano: int
    end_time_unix_nano: int
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.OK
    status_message: str = ""

    # LLM-specific
    provider: str | None = None
    model: str | None = None
    operation: str = "chat"
    token_usage: TokenUsage | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    finish_reasons: list[str] = Field(default_factory=list)
    response_id: str | None = None

    # Tool-specific
    tool_name: str | None = None
    tool_description: str | None = None
    tool_call_id: str | None = None

    # Agent-specific
    agent_name: str | None = None
    agent_description: str | None = None

    # Evaluation-specific
    eval_dimension: str | None = None
    eval_score: float | None = None
    eval_label: str | None = None
    eval_grader_id: str | None = None

    # Content (optional, gated by include_content)
    input_messages: list[dict[str, Any]] = Field(default_factory=list)
    output_message: dict[str, Any] | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: dict[str, Any] | None = None

    # Source-enriched attributes (set by source formatters)
    attributes: dict[str, Any] = Field(default_factory=dict)

    # Streaming
    is_streaming: bool = False
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    chunk_count: int | None = None

    # Error injection
    error_type: str | None = None
    http_status_code: int | None = None

    # Span events (OTel events attached to span)
    events: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_time_unix_nano - self.start_time_unix_nano) / 1_000_000

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


class SimulatedTrace(BaseModel):
    """A complete simulated trace containing multiple spans.

    The internal representation that flows through the 3-layer architecture.
    """

    trace_id: str
    spans: list[SimulatedSpan] = Field(default_factory=list)

    # Metadata
    source_format: str | None = None
    scenario: str | None = None
    topic: str | None = None
    seed: int | None = None

    # Resource attributes (set by source formatters)
    resource_attributes: dict[str, Any] = Field(default_factory=dict)

    # Scope info (set by source formatters)
    scope_name: str = "stratix.simulator"
    scope_version: str = "0.1.0"

    # Conversation
    session_id: str | None = None
    turn_number: int | None = None

    @property
    def root_span(self) -> SimulatedSpan | None:
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return self.spans[0] if self.spans else None

    @property
    def span_count(self) -> int:
        return len(self.spans)

    @property
    def duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        start = min(s.start_time_unix_nano for s in self.spans)
        end = max(s.end_time_unix_nano for s in self.spans)
        return (end - start) / 1_000_000

    @property
    def llm_spans(self) -> list[SimulatedSpan]:
        return [s for s in self.spans if s.span_type == SpanType.LLM]

    @property
    def tool_spans(self) -> list[SimulatedSpan]:
        return [s for s in self.spans if s.span_type == SpanType.TOOL]

    @property
    def total_tokens(self) -> int:
        return sum(
            s.token_usage.total_tokens
            for s in self.spans
            if s.token_usage is not None
        )

    def add_span(self, span: SimulatedSpan) -> None:
        self.spans.append(span)

    def get_span(self, span_id: str) -> SimulatedSpan | None:
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_children(self, span_id: str) -> list[SimulatedSpan]:
        return [s for s in self.spans if s.parent_span_id == span_id]
