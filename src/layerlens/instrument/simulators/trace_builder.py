"""Fluent TraceBuilder API for constructing simulated traces.

Provides a chainable API for building SimulatedTrace objects with
agent, LLM, tool, and evaluation spans.
"""

from __future__ import annotations

from typing import Any

from .clock import DeterministicClock
from .identifiers import IDGenerator
from .span_model import (
    SPAN_TYPE_TO_KIND,
    SimulatedSpan,
    SimulatedTrace,
    SpanStatus,
    SpanType,
    TokenUsage,
)


class TraceBuilder:
    """Fluent API for building simulated traces.

    Usage:
        trace = (
            TraceBuilder(seed=42)
            .with_scenario("customer_service")
            .with_source("openai")
            .add_agent_span("Case_Resolution_Agent")
            .add_llm_span(provider="openai", model="gpt-4o",
                          prompt_tokens=250, completion_tokens=180)
            .add_tool_span(name="Get_Order_Details", latency_ms=350.0)
            .add_llm_span(provider="openai", model="gpt-4o",
                          prompt_tokens=400, completion_tokens=220)
            .add_evaluation_span(dimension="factual_accuracy", score=0.92)
            .with_error(error_type="rate_limit", span_index=-1)
            .with_streaming(ttft_ms=120.0, tpot_ms=35.0)
            .build()
        )
    """

    def __init__(self, seed: int | None = None):
        self._clock = DeterministicClock(seed=seed)
        self._ids = IDGenerator(seed=seed)
        self._trace_id = self._ids.trace_id()
        self._spans: list[SimulatedSpan] = []
        self._scenario: str | None = None
        self._topic: str | None = None
        self._source: str | None = None
        self._session_id: str | None = None
        self._turn_number: int | None = None
        self._seed = seed
        self._current_parent_id: str | None = None
        self._agent_span_id: str | None = None

    def with_scenario(self, scenario: str, topic: str | None = None) -> TraceBuilder:
        self._scenario = scenario
        self._topic = topic
        return self

    def with_source(self, source: str) -> TraceBuilder:
        self._source = source
        return self

    def with_session(self, session_id: str | None = None, turn: int = 1) -> TraceBuilder:
        self._session_id = session_id or self._ids.session_id()
        self._turn_number = turn
        return self

    def add_agent_span(
        self,
        name: str,
        description: str | None = None,
        duration_ms: float | None = None,
    ) -> TraceBuilder:
        """Add a root agent span. Subsequent spans become children."""
        span_id = self._ids.span_id()
        start_ns = self._clock.now_ns()
        dur = duration_ms or self._clock.agent_span_duration_ms()

        span = SimulatedSpan(
            span_id=span_id,
            parent_span_id=None,
            span_type=SpanType.AGENT,
            name=f"agent {name}",
            start_time_unix_nano=start_ns,
            end_time_unix_nano=start_ns + int(dur * 1_000_000),
            kind=SPAN_TYPE_TO_KIND[SpanType.AGENT],
            agent_name=name,
            agent_description=description,
        )
        self._spans.append(span)
        self._agent_span_id = span_id
        self._current_parent_id = span_id
        # Advance past a small gap
        self._clock.advance_ms(5.0)
        return self

    def add_llm_span(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        prompt_tokens: int = 200,
        completion_tokens: int = 150,
        cached_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        finish_reasons: list[str] | None = None,
        duration_ms: float | None = None,
        operation: str = "chat",
        input_messages: list[dict[str, Any]] | None = None,
        output_message: dict[str, Any] | None = None,
    ) -> TraceBuilder:
        """Add an LLM call span as child of current agent span."""
        span_id = self._ids.span_id()
        start_ns = self._clock.now_ns()
        dur = duration_ms or self._clock.llm_span_duration_ms()
        end_ns = start_ns + int(dur * 1_000_000)

        span = SimulatedSpan(
            span_id=span_id,
            parent_span_id=self._current_parent_id,
            span_type=SpanType.LLM,
            name=f"{operation} {model}",
            start_time_unix_nano=start_ns,
            end_time_unix_nano=end_ns,
            kind=SPAN_TYPE_TO_KIND[SpanType.LLM],
            provider=provider,
            model=model,
            operation=operation,
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            finish_reasons=finish_reasons or ["stop"],
            response_id=self._provider_response_id(provider),
            input_messages=input_messages or [],
            output_message=output_message,
        )
        self._spans.append(span)
        self._clock.advance_ms(dur)
        self._clock.advance_random_ms(1.0, 10.0)
        return self

    def _provider_response_id(self, provider: str) -> str:
        """Generate a provider-appropriate response ID."""
        provider_lower = provider.lower()
        if "anthropic" in provider_lower:
            return self._ids.response_id_anthropic()
        elif "vertex" in provider_lower or "google" in provider_lower:
            return self._ids.response_id_vertex()
        elif "bedrock" in provider_lower:
            return self._ids.response_id_bedrock()
        else:
            return self._ids.response_id_openai()

    def add_tool_span(
        self,
        name: str,
        description: str | None = None,
        latency_ms: float | None = None,
        tool_input: dict[str, Any] | None = None,
        tool_output: dict[str, Any] | None = None,
    ) -> TraceBuilder:
        """Add a tool call span as child of current agent span."""
        span_id = self._ids.span_id()
        start_ns = self._clock.now_ns()
        dur = latency_ms or self._clock.tool_span_duration_ms()
        end_ns = start_ns + int(dur * 1_000_000)

        span = SimulatedSpan(
            span_id=span_id,
            parent_span_id=self._current_parent_id,
            span_type=SpanType.TOOL,
            name=f"tool {name}",
            start_time_unix_nano=start_ns,
            end_time_unix_nano=end_ns,
            kind=SPAN_TYPE_TO_KIND[SpanType.TOOL],
            tool_name=name,
            tool_description=description,
            tool_call_id=self._ids.tool_call_id(),
            tool_input=tool_input,
            tool_output=tool_output,
        )
        self._spans.append(span)
        self._clock.advance_ms(dur)
        self._clock.advance_random_ms(1.0, 10.0)
        return self

    def add_evaluation_span(
        self,
        dimension: str,
        score: float,
        label: str | None = None,
        grader_id: str | None = None,
        duration_ms: float | None = None,
    ) -> TraceBuilder:
        """Add an evaluation result span as child of current agent span."""
        span_id = self._ids.span_id()
        start_ns = self._clock.now_ns()
        dur = duration_ms or self._clock.eval_span_duration_ms()
        end_ns = start_ns + int(dur * 1_000_000)

        if label is None:
            label = "pass" if score >= 0.7 else "fail"

        span = SimulatedSpan(
            span_id=span_id,
            parent_span_id=self._current_parent_id,
            span_type=SpanType.EVALUATION,
            name=f"evaluation {dimension}",
            start_time_unix_nano=start_ns,
            end_time_unix_nano=end_ns,
            kind=SPAN_TYPE_TO_KIND[SpanType.EVALUATION],
            eval_dimension=dimension,
            eval_score=score,
            eval_label=label,
            eval_grader_id=grader_id,
        )
        self._spans.append(span)
        self._clock.advance_ms(dur)
        self._clock.advance_random_ms(1.0, 10.0)
        return self

    def with_error(
        self,
        error_type: str,
        span_index: int = -1,
        http_status_code: int | None = None,
        message: str | None = None,
    ) -> TraceBuilder:
        """Inject an error into a specific span."""
        if not self._spans:
            return self
        span = self._spans[span_index]
        span.error_type = error_type
        span.status = SpanStatus.ERROR
        span.status_message = message or f"Simulated {error_type} error"

        status_map = {
            "rate_limit": 429,
            "timeout": 504,
            "auth_failure": 401,
            "content_filter": 200,
            "server_error": 500,
        }
        span.http_status_code = http_status_code or status_map.get(error_type, 500)
        return self

    def with_streaming(
        self,
        ttft_ms: float = 120.0,
        tpot_ms: float = 35.0,
        chunk_count: int | None = None,
        span_index: int = -1,
    ) -> TraceBuilder:
        """Mark an LLM span as streaming with timing parameters."""
        # Find the target LLM span
        llm_spans = [
            (i, s) for i, s in enumerate(self._spans) if s.span_type == SpanType.LLM
        ]
        if not llm_spans:
            return self

        if span_index == -1:
            _, span = llm_spans[-1]
        else:
            _, span = llm_spans[span_index % len(llm_spans)]

        span.is_streaming = True
        span.ttft_ms = ttft_ms
        span.tpot_ms = tpot_ms
        if chunk_count is None and span.token_usage:
            span.chunk_count = max(1, span.token_usage.completion_tokens // 5)
        else:
            span.chunk_count = chunk_count or 10
        return self

    def with_content(
        self,
        span_index: int,
        input_messages: list[dict[str, Any]] | None = None,
        output_message: dict[str, Any] | None = None,
    ) -> TraceBuilder:
        """Add content to a specific span."""
        if 0 <= span_index < len(self._spans) or (
            span_index < 0 and abs(span_index) <= len(self._spans)
        ):
            span = self._spans[span_index]
            if input_messages:
                span.input_messages = input_messages
            if output_message:
                span.output_message = output_message
        return self

    def build(self) -> SimulatedTrace:
        """Build and return the SimulatedTrace."""
        # Fixup agent span end time to encompass all children
        if self._agent_span_id:
            agent_span = next(
                (s for s in self._spans if s.span_id == self._agent_span_id), None
            )
            if agent_span and len(self._spans) > 1:
                max_end = max(s.end_time_unix_nano for s in self._spans)
                agent_span.end_time_unix_nano = max_end + int(
                    self._clock.inter_span_gap_ms() * 1_000_000
                )

        return SimulatedTrace(
            trace_id=self._trace_id,
            spans=self._spans,
            source_format=self._source,
            scenario=self._scenario,
            topic=self._topic,
            seed=self._seed,
            session_id=self._session_id,
            turn_number=self._turn_number,
        )
