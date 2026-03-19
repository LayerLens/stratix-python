"""Tests for SimulatedSpan and SimulatedTrace models."""

import pytest

from layerlens.instrument.simulators.span_model import (
    SPAN_TYPE_TO_KIND,
    SimulatedSpan,
    SimulatedTrace,
    SpanKind,
    SpanStatus,
    SpanType,
    TokenUsage,
)


class TestTokenUsage:
    def test_auto_total(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150

    def test_explicit_total(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=200)
        assert usage.total_tokens == 200

    def test_zero_tokens(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_optional_fields(self):
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=20,
            reasoning_tokens=30,
        )
        assert usage.cached_tokens == 20
        assert usage.reasoning_tokens == 30


class TestSpanType:
    def test_all_types(self):
        assert len(list(SpanType)) == 4
        assert SpanType.AGENT.value == "agent"
        assert SpanType.LLM.value == "llm"
        assert SpanType.TOOL.value == "tool"
        assert SpanType.EVALUATION.value == "evaluation"

    def test_span_kind_mapping(self):
        assert SPAN_TYPE_TO_KIND[SpanType.AGENT] == SpanKind.SERVER
        assert SPAN_TYPE_TO_KIND[SpanType.LLM] == SpanKind.CLIENT
        assert SPAN_TYPE_TO_KIND[SpanType.TOOL] == SpanKind.INTERNAL
        assert SPAN_TYPE_TO_KIND[SpanType.EVALUATION] == SpanKind.INTERNAL


class TestSimulatedSpan:
    def test_basic_creation(self):
        span = SimulatedSpan(
            span_id="abc123",
            span_type=SpanType.LLM,
            name="chat gpt-4o",
            start_time_unix_nano=1_700_000_000_000_000_000,
            end_time_unix_nano=1_700_000_001_000_000_000,
        )
        assert span.span_id == "abc123"
        assert span.span_type == SpanType.LLM
        assert span.status == SpanStatus.OK
        assert span.parent_span_id is None

    def test_duration_ms(self):
        span = SimulatedSpan(
            span_id="abc",
            span_type=SpanType.LLM,
            name="test",
            start_time_unix_nano=1_000_000_000,
            end_time_unix_nano=1_500_000_000,
        )
        assert span.duration_ms == 500.0

    def test_duration_s(self):
        span = SimulatedSpan(
            span_id="abc",
            span_type=SpanType.LLM,
            name="test",
            start_time_unix_nano=0,
            end_time_unix_nano=2_000_000_000,
        )
        assert span.duration_s == 2.0

    def test_llm_fields(self):
        span = SimulatedSpan(
            span_id="abc",
            span_type=SpanType.LLM,
            name="chat gpt-4o",
            start_time_unix_nano=0,
            end_time_unix_nano=1_000_000_000,
            provider="openai",
            model="gpt-4o",
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            finish_reasons=["stop"],
        )
        assert span.provider == "openai"
        assert span.model == "gpt-4o"
        assert span.token_usage.total_tokens == 150

    def test_tool_fields(self):
        span = SimulatedSpan(
            span_id="abc",
            span_type=SpanType.TOOL,
            name="tool Get_Order",
            start_time_unix_nano=0,
            end_time_unix_nano=500_000_000,
            tool_name="Get_Order",
            tool_call_id="call_abc123",
        )
        assert span.tool_name == "Get_Order"
        assert span.tool_call_id == "call_abc123"

    def test_attributes_dict(self):
        span = SimulatedSpan(
            span_id="abc",
            span_type=SpanType.LLM,
            name="test",
            start_time_unix_nano=0,
            end_time_unix_nano=1_000_000_000,
            attributes={"gen_ai.system": "openai"},
        )
        assert span.attributes["gen_ai.system"] == "openai"


class TestSimulatedTrace:
    def _make_span(self, span_id: str, span_type: SpanType, parent_id: str | None = None, **kwargs):
        return SimulatedSpan(
            span_id=span_id,
            parent_span_id=parent_id,
            span_type=span_type,
            name=f"test_{span_id}",
            start_time_unix_nano=kwargs.get("start", 1_000_000_000_000),
            end_time_unix_nano=kwargs.get("end", 2_000_000_000_000),
            token_usage=kwargs.get("token_usage"),
        )

    def test_empty_trace(self):
        trace = SimulatedTrace(trace_id="abc123")
        assert trace.span_count == 0
        assert trace.duration_ms == 0.0
        assert trace.root_span is None

    def test_root_span(self):
        agent_span = self._make_span("s1", SpanType.AGENT)
        llm_span = self._make_span("s2", SpanType.LLM, parent_id="s1")
        trace = SimulatedTrace(trace_id="t1", spans=[agent_span, llm_span])
        assert trace.root_span is agent_span

    def test_span_count(self):
        spans = [
            self._make_span("s1", SpanType.AGENT),
            self._make_span("s2", SpanType.LLM, parent_id="s1"),
            self._make_span("s3", SpanType.TOOL, parent_id="s1"),
        ]
        trace = SimulatedTrace(trace_id="t1", spans=spans)
        assert trace.span_count == 3

    def test_llm_spans_filter(self):
        spans = [
            self._make_span("s1", SpanType.AGENT),
            self._make_span("s2", SpanType.LLM, parent_id="s1"),
            self._make_span("s3", SpanType.TOOL, parent_id="s1"),
            self._make_span("s4", SpanType.LLM, parent_id="s1"),
        ]
        trace = SimulatedTrace(trace_id="t1", spans=spans)
        assert len(trace.llm_spans) == 2
        assert len(trace.tool_spans) == 1

    def test_total_tokens(self):
        spans = [
            self._make_span("s1", SpanType.AGENT),
            self._make_span(
                "s2",
                SpanType.LLM,
                parent_id="s1",
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            ),
            self._make_span(
                "s3",
                SpanType.LLM,
                parent_id="s1",
                token_usage=TokenUsage(prompt_tokens=200, completion_tokens=100),
            ),
        ]
        trace = SimulatedTrace(trace_id="t1", spans=spans)
        assert trace.total_tokens == 450

    def test_add_span(self):
        trace = SimulatedTrace(trace_id="t1")
        trace.add_span(self._make_span("s1", SpanType.AGENT))
        assert trace.span_count == 1

    def test_get_span(self):
        span = self._make_span("s1", SpanType.AGENT)
        trace = SimulatedTrace(trace_id="t1", spans=[span])
        assert trace.get_span("s1") is span
        assert trace.get_span("nonexistent") is None

    def test_get_children(self):
        spans = [
            self._make_span("s1", SpanType.AGENT),
            self._make_span("s2", SpanType.LLM, parent_id="s1"),
            self._make_span("s3", SpanType.TOOL, parent_id="s1"),
            self._make_span("s4", SpanType.LLM, parent_id="s2"),
        ]
        trace = SimulatedTrace(trace_id="t1", spans=spans)
        children = trace.get_children("s1")
        assert len(children) == 2

    def test_duration_ms(self):
        spans = [
            SimulatedSpan(
                span_id="s1",
                span_type=SpanType.AGENT,
                name="test",
                start_time_unix_nano=1_000_000_000,
                end_time_unix_nano=5_000_000_000,
            ),
            SimulatedSpan(
                span_id="s2",
                span_type=SpanType.LLM,
                name="test",
                start_time_unix_nano=1_500_000_000,
                end_time_unix_nano=3_000_000_000,
            ),
        ]
        trace = SimulatedTrace(trace_id="t1", spans=spans)
        assert trace.duration_ms == 4000.0

    def test_serialization(self):
        span = self._make_span("s1", SpanType.AGENT)
        trace = SimulatedTrace(
            trace_id="t1",
            spans=[span],
            scenario="customer_service",
            topic="Shipping_Delay",
        )
        data = trace.model_dump(mode="json")
        assert data["trace_id"] == "t1"
        assert data["scenario"] == "customer_service"
        restored = SimulatedTrace(**data)
        assert restored.trace_id == "t1"
        assert restored.span_count == 1
