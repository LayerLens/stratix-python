"""Tests for testing utilities."""

import pytest

from layerlens.instrument.simulators import SimulatorConfig, TraceBuilder, TraceSimulator
from layerlens.instrument.simulators.outputs.otlp_json import OTLPJSONFormatter
from layerlens.instrument.simulators.sources import get_source_formatter
from layerlens.instrument.simulators.span_model import SimulatedTrace, SpanType
from layerlens.instrument.simulators.testing.assertions import (
    assert_deterministic,
    assert_genai_attributes,
    assert_round_trip,
    assert_span_tree,
    assert_token_counts,
    assert_valid_otlp_trace,
)
from layerlens.instrument.simulators.testing.round_trip import (
    RoundTripResult,
    validate_all_sources,
    validate_round_trip,
)


def _build_sample_trace() -> SimulatedTrace:
    return (
        TraceBuilder(seed=42)
        .with_scenario("customer_service")
        .add_agent_span("TestAgent")
        .add_llm_span(
            provider="openai", model="gpt-4o",
            prompt_tokens=250, completion_tokens=180,
        )
        .add_tool_span(name="TestTool", latency_ms=100.0)
        .add_llm_span(
            provider="openai", model="gpt-4o",
            prompt_tokens=400, completion_tokens=220,
        )
        .add_evaluation_span(dimension="accuracy", score=0.9)
        .build()
    )


class TestAssertValidOtlpTrace:
    def test_valid_trace_passes(self):
        trace = _build_sample_trace()
        formatter = get_source_formatter("openai")
        for span in trace.spans:
            formatter.enrich_span(span, formatter.get_default_profile())
        output = OTLPJSONFormatter()
        otlp = output.format_trace(trace)
        assert_valid_otlp_trace(otlp)

    def test_invalid_raises(self):
        with pytest.raises(AssertionError):
            assert_valid_otlp_trace({"invalid": "data"})


class TestAssertSpanTree:
    def test_valid_tree_passes(self):
        trace = _build_sample_trace()
        formatter = get_source_formatter("generic_otel")
        for span in trace.spans:
            formatter.enrich_span(span, formatter.get_default_profile())
        output = OTLPJSONFormatter()
        otlp = output.format_trace(trace)
        assert_span_tree(otlp)


class TestAssertTokenCounts:
    def test_valid_tokens_pass(self):
        trace = _build_sample_trace()
        assert_token_counts(trace)


class TestAssertDeterministic:
    def test_deterministic_generation(self):
        def gen():
            return (
                TraceBuilder(seed=42)
                .add_agent_span("Agent")
                .add_llm_span(
                    provider="openai", model="gpt-4o",
                    prompt_tokens=100, completion_tokens=50,
                )
                .build()
            )
        assert_deterministic(gen)


class TestAssertRoundTrip:
    def test_round_trip_passes(self):
        trace = _build_sample_trace()
        source = get_source_formatter("openai")
        output = OTLPJSONFormatter()
        assert_round_trip(trace, source, output)


class TestRoundTripValidation:
    def test_validate_single_source(self):
        result = validate_round_trip("openai", count=1, seed=42)
        assert isinstance(result, RoundTripResult)
        assert result.passed
        assert result.traces_generated >= 1

    def test_validate_all_sources(self):
        results = validate_all_sources(count=1, seed=42)
        assert len(results) == 12
        for r in results:
            assert r.passed, f"{r.source} failed: {r.errors}"
