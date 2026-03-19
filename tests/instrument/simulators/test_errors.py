"""Tests for error injection."""

import pytest

from layerlens.instrument.simulators.config import ErrorConfig
from layerlens.instrument.simulators.errors import inject_errors
from layerlens.instrument.simulators.errors.auth_failure import AuthFailureInjector
from layerlens.instrument.simulators.errors.content_filter import ContentFilterInjector
from layerlens.instrument.simulators.errors.rate_limit import RateLimitInjector
from layerlens.instrument.simulators.errors.server_error import ServerErrorInjector
from layerlens.instrument.simulators.errors.timeout import TimeoutInjector
from layerlens.instrument.simulators.span_model import (
    SimulatedSpan,
    SimulatedTrace,
    SpanKind,
    SpanStatus,
    SpanType,
    TokenUsage,
)


def _make_llm_span(span_id: str = "llm001") -> SimulatedSpan:
    return SimulatedSpan(
        span_id=span_id,
        span_type=SpanType.LLM,
        name="chat gpt-4o",
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_001_000_000_000,
        kind=SpanKind.CLIENT,
        provider="openai",
        model="gpt-4o",
        token_usage=TokenUsage(prompt_tokens=250, completion_tokens=180),
        finish_reasons=["stop"],
    )


def _make_trace(num_llm_spans: int = 3) -> SimulatedTrace:
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
    for i in range(num_llm_spans):
        spans.append(_make_llm_span(span_id=f"llm{i:03d}"))
    return SimulatedTrace(trace_id="trace_test", spans=spans)


class TestRateLimitInjector:
    def test_inject(self):
        span = _make_llm_span()
        injector = RateLimitInjector()
        injector.inject(span)
        assert span.status == SpanStatus.ERROR
        assert span.error_type == "rate_limit"
        assert span.http_status_code == 429
        assert span.attributes["http.response.status_code"] == 429
        assert span.attributes["retry-after"] == "30"
        assert span.token_usage.completion_tokens == 0
        assert span.finish_reasons == []


class TestTimeoutInjector:
    def test_inject(self):
        span = _make_llm_span()
        original_end = span.end_time_unix_nano
        injector = TimeoutInjector()
        injector.inject(span)
        assert span.status == SpanStatus.ERROR
        assert span.error_type == "timeout"
        assert span.http_status_code == 504
        assert span.token_usage.completion_tokens == 0
        # Verify end time was truncated to 30s deadline
        deadline_ns = span.start_time_unix_nano + 30_000_000_000
        assert span.end_time_unix_nano <= deadline_ns


class TestAuthFailureInjector:
    def test_inject(self):
        span = _make_llm_span()
        injector = AuthFailureInjector()
        injector.inject(span)
        assert span.status == SpanStatus.ERROR
        assert span.error_type == "auth_failure"
        assert span.http_status_code == 401
        assert span.token_usage.completion_tokens == 0


class TestContentFilterInjector:
    def test_inject(self):
        span = _make_llm_span()
        injector = ContentFilterInjector()
        injector.inject(span)
        assert span.error_type == "content_filter"
        assert span.status == SpanStatus.OK  # Content filter is not an error status
        assert span.finish_reasons == ["content_filter"]
        assert span.token_usage.completion_tokens <= 10
        assert span.http_status_code == 200


class TestServerErrorInjector:
    def test_inject(self):
        span = _make_llm_span()
        injector = ServerErrorInjector(seed=42)
        injector.inject(span)
        assert span.status == SpanStatus.ERROR
        assert span.error_type == "server_error"
        assert span.http_status_code in [500, 502, 503]
        assert span.token_usage.completion_tokens == 0


class TestInjectErrors:
    def test_disabled_config(self):
        trace = _make_trace()
        config = ErrorConfig(enabled=False)
        result = inject_errors(trace, config, seed=42)
        errors = [s for s in result.spans if s.error_type]
        assert len(errors) == 0

    def test_high_probability_injects(self):
        trace = _make_trace(num_llm_spans=10)
        config = ErrorConfig(
            enabled=True,
            rate_limit_probability=1.0,  # 100% = always inject
        )
        result = inject_errors(trace, config, seed=42)
        error_spans = [s for s in result.spans if s.error_type]
        assert len(error_spans) == 10  # All LLM spans get errors

    def test_zero_probability_no_errors(self):
        trace = _make_trace(num_llm_spans=10)
        config = ErrorConfig(
            enabled=True,
            rate_limit_probability=0.0,
            timeout_probability=0.0,
            auth_failure_probability=0.0,
            content_filter_probability=0.0,
            server_error_probability=0.0,
        )
        result = inject_errors(trace, config, seed=42)
        error_spans = [s for s in result.spans if s.error_type]
        assert len(error_spans) == 0

    def test_only_llm_spans_affected(self):
        trace = _make_trace()
        config = ErrorConfig(enabled=True, rate_limit_probability=1.0)
        result = inject_errors(trace, config, seed=42)
        agent_span = result.spans[0]
        assert agent_span.error_type is None  # Agent spans unaffected

    def test_deterministic(self):
        def run():
            trace = _make_trace(num_llm_spans=10)
            config = ErrorConfig(
                enabled=True,
                rate_limit_probability=0.3,
                timeout_probability=0.1,
            )
            return inject_errors(trace, config, seed=42)

        r1 = run()
        r2 = run()
        errors1 = [(s.span_id, s.error_type) for s in r1.spans if s.error_type]
        errors2 = [(s.span_id, s.error_type) for s in r2.spans if s.error_type]
        assert errors1 == errors2

    def test_one_error_per_span(self):
        """Each span should get at most one error type."""
        trace = _make_trace(num_llm_spans=20)
        config = ErrorConfig(
            enabled=True,
            rate_limit_probability=0.5,
            timeout_probability=0.5,
            server_error_probability=0.5,
        )
        inject_errors(trace, config, seed=42)
        for span in trace.spans:
            if span.error_type:
                # Only one error type
                assert span.error_type in [
                    "rate_limit", "timeout", "auth_failure",
                    "content_filter", "server_error",
                ]
