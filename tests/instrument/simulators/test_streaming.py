"""Tests for streaming behavior."""

import pytest

from layerlens.instrument.simulators.config import StreamingConfig
from layerlens.instrument.simulators.span_model import (
    SimulatedSpan,
    SimulatedTrace,
    SpanKind,
    SpanType,
    TokenUsage,
)
from layerlens.instrument.simulators.streaming import StreamingBehavior


def _make_llm_span(
    span_id: str = "llm001",
    completion_tokens: int = 180,
) -> SimulatedSpan:
    return SimulatedSpan(
        span_id=span_id,
        span_type=SpanType.LLM,
        name="chat gpt-4o",
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_001_000_000_000,
        kind=SpanKind.CLIENT,
        provider="openai",
        model="gpt-4o",
        token_usage=TokenUsage(prompt_tokens=250, completion_tokens=completion_tokens),
        finish_reasons=["stop"],
    )


def _make_trace(num_llm: int = 2) -> SimulatedTrace:
    spans = [
        SimulatedSpan(
            span_id="agent001",
            span_type=SpanType.AGENT,
            name="agent Test",
            start_time_unix_nano=1_700_000_000_000_000_000,
            end_time_unix_nano=1_700_000_010_000_000_000,
            kind=SpanKind.SERVER,
        ),
    ]
    for i in range(num_llm):
        spans.append(_make_llm_span(span_id=f"llm{i:03d}"))
    return SimulatedTrace(trace_id="trace_test", spans=spans)


class TestStreamingBehavior:
    def test_disabled_noop(self):
        config = StreamingConfig(enabled=False)
        behavior = StreamingBehavior(config, seed=42)
        trace = _make_trace()
        result = behavior.apply(trace)
        for span in result.spans:
            assert span.is_streaming is False
            assert span.ttft_ms is None
            assert span.tpot_ms is None

    def test_applies_to_llm_spans_only(self):
        config = StreamingConfig(enabled=True)
        behavior = StreamingBehavior(config, seed=42)
        trace = _make_trace()
        behavior.apply(trace)
        agent = trace.spans[0]
        assert agent.is_streaming is False
        for span in trace.spans[1:]:
            assert span.is_streaming is True

    def test_ttft_in_range(self):
        config = StreamingConfig(
            enabled=True,
            ttft_ms_min=100.0,
            ttft_ms_max=500.0,
        )
        behavior = StreamingBehavior(config, seed=42)
        span = _make_llm_span()
        trace = SimulatedTrace(trace_id="t1", spans=[span])
        behavior.apply(trace)
        assert span.ttft_ms is not None
        assert 100.0 <= span.ttft_ms <= 500.0

    def test_tpot_in_range(self):
        config = StreamingConfig(
            enabled=True,
            tpot_ms_min=10.0,
            tpot_ms_max=80.0,
        )
        behavior = StreamingBehavior(config, seed=42)
        span = _make_llm_span()
        trace = SimulatedTrace(trace_id="t1", spans=[span])
        behavior.apply(trace)
        assert span.tpot_ms is not None
        assert 10.0 <= span.tpot_ms <= 80.0

    def test_chunk_count_positive(self):
        config = StreamingConfig(enabled=True)
        behavior = StreamingBehavior(config, seed=42)
        span = _make_llm_span(completion_tokens=200)
        trace = SimulatedTrace(trace_id="t1", spans=[span])
        behavior.apply(trace)
        assert span.chunk_count is not None
        assert span.chunk_count >= 1

    def test_chunk_count_zero_completion_tokens(self):
        config = StreamingConfig(enabled=True)
        behavior = StreamingBehavior(config, seed=42)
        span = _make_llm_span(completion_tokens=0)
        trace = SimulatedTrace(trace_id="t1", spans=[span])
        behavior.apply(trace)
        assert span.chunk_count is not None
        assert span.chunk_count >= 1

    def test_streaming_attributes_added(self):
        config = StreamingConfig(enabled=True)
        behavior = StreamingBehavior(config, seed=42)
        span = _make_llm_span()
        trace = SimulatedTrace(trace_id="t1", spans=[span])
        behavior.apply(trace)
        assert span.attributes["gen_ai.is_streaming"] is True
        assert "gen_ai.server.time_to_first_token" in span.attributes
        assert "gen_ai.server.time_per_output_token" in span.attributes

    def test_chunk_events_generated(self):
        config = StreamingConfig(enabled=True)
        behavior = StreamingBehavior(config, seed=42)
        span = _make_llm_span(completion_tokens=200)
        trace = SimulatedTrace(trace_id="t1", spans=[span])
        behavior.apply(trace)
        assert len(span.events) > 0
        for event in span.events:
            assert event["name"] == "gen_ai.content.chunk"
            assert "timeUnixNano" in event

    def test_chunk_events_ordered(self):
        config = StreamingConfig(enabled=True)
        behavior = StreamingBehavior(config, seed=42)
        span = _make_llm_span(completion_tokens=500)
        trace = SimulatedTrace(trace_id="t1", spans=[span])
        behavior.apply(trace)
        times = [int(e["timeUnixNano"]) for e in span.events]
        assert times == sorted(times)

    def test_chunk_events_within_span_duration(self):
        config = StreamingConfig(enabled=True)
        behavior = StreamingBehavior(config, seed=42)
        span = _make_llm_span(completion_tokens=200)
        trace = SimulatedTrace(trace_id="t1", spans=[span])
        behavior.apply(trace)
        for event in span.events:
            t = int(event["timeUnixNano"])
            assert t >= span.start_time_unix_nano
            assert t <= span.end_time_unix_nano

    def test_chunk_events_capped_at_50(self):
        config = StreamingConfig(
            enabled=True,
            chunks_min=100,
            chunks_max=200,
        )
        behavior = StreamingBehavior(config, seed=42)
        # Very long span to allow many chunks
        span = SimulatedSpan(
            span_id="llm001",
            span_type=SpanType.LLM,
            name="chat gpt-4o",
            start_time_unix_nano=1_700_000_000_000_000_000,
            end_time_unix_nano=1_700_000_100_000_000_000,  # 100 seconds
            kind=SpanKind.CLIENT,
            provider="openai",
            model="gpt-4o",
            token_usage=TokenUsage(prompt_tokens=250, completion_tokens=1000),
        )
        trace = SimulatedTrace(trace_id="t1", spans=[span])
        behavior.apply(trace)
        assert len(span.events) <= 50

    def test_deterministic(self):
        config = StreamingConfig(enabled=True)

        def run():
            behavior = StreamingBehavior(config, seed=42)
            trace = _make_trace(num_llm=5)
            behavior.apply(trace)
            return [
                (s.ttft_ms, s.tpot_ms, s.chunk_count)
                for s in trace.spans
                if s.span_type == SpanType.LLM
            ]

        assert run() == run()

    def test_chunk_event_index_attribute(self):
        config = StreamingConfig(enabled=True)
        behavior = StreamingBehavior(config, seed=42)
        span = _make_llm_span(completion_tokens=200)
        trace = SimulatedTrace(trace_id="t1", spans=[span])
        behavior.apply(trace)
        for i, event in enumerate(span.events):
            attrs = {a["key"]: a["value"] for a in event["attributes"]}
            assert "gen_ai.chunk.index" in attrs
            assert attrs["gen_ai.chunk.index"]["intValue"] == str(i)
