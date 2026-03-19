"""Tests for OTel GenAI Metrics Exporter."""

import pytest

from layerlens.instrument.exporters._otel_metrics import (
    OTelMetricsExporter,
    OPERATION_DURATION_BOUNDARIES,
    STREAMING_BOUNDARIES,
    TOKEN_USAGE_BOUNDARIES,
)


class TestOTelMetricsExporterInit:
    """Tests for OTelMetricsExporter initialization."""

    def test_default_init(self):
        """Test creating exporter with defaults."""
        exporter = OTelMetricsExporter()
        assert exporter._endpoint is None
        assert exporter._export_interval_ms == 60000
        assert exporter._initialized is False

    def test_custom_init(self):
        """Test creating exporter with custom settings."""
        exporter = OTelMetricsExporter(
            endpoint="localhost:4318",
            export_interval_ms=30000,
        )
        assert exporter._endpoint == "localhost:4318"
        assert exporter._export_interval_ms == 30000

    def test_not_initialized_before_use(self):
        """Test that histograms are None before initialization."""
        exporter = OTelMetricsExporter()
        assert exporter._meter is None
        assert exporter._token_usage_histogram is None
        assert exporter._operation_duration_histogram is None
        assert exporter._ttft_histogram is None
        assert exporter._tpot_histogram is None


class TestOTelMetricsRecording:
    """Tests for metrics recording methods (without OTel SDK)."""

    def test_record_token_usage_no_crash_without_sdk(self):
        """Test that recording tokens doesn't crash when OTel SDK is missing."""
        exporter = OTelMetricsExporter()
        # Should not raise even without OTel SDK
        exporter.record_token_usage(
            input_tokens=100,
            output_tokens=50,
            operation="chat",
            model="gpt-4o",
        )

    def test_record_operation_duration_no_crash_without_sdk(self):
        """Test that recording duration doesn't crash when OTel SDK is missing."""
        exporter = OTelMetricsExporter()
        exporter.record_operation_duration(
            duration_seconds=1.5,
            operation="chat",
            model="gpt-4o",
        )

    def test_record_ttft_no_crash_without_sdk(self):
        """Test that recording TTFT doesn't crash when OTel SDK is missing."""
        exporter = OTelMetricsExporter()
        exporter.record_time_to_first_token(
            ttft_seconds=0.35,
            operation="chat",
            model="gpt-4o",
        )

    def test_record_tpot_no_crash_without_sdk(self):
        """Test that recording TPoT doesn't crash when OTel SDK is missing."""
        exporter = OTelMetricsExporter()
        exporter.record_time_per_output_token(
            tpot_seconds=0.02,
            operation="chat",
            model="gpt-4o",
        )

    def test_record_token_usage_none_values(self):
        """Test recording with None token values."""
        exporter = OTelMetricsExporter()
        exporter.record_token_usage(
            input_tokens=None,
            output_tokens=None,
        )

    def test_record_token_usage_input_only(self):
        """Test recording input tokens only."""
        exporter = OTelMetricsExporter()
        exporter.record_token_usage(
            input_tokens=100,
            output_tokens=None,
        )

    def test_record_token_usage_output_only(self):
        """Test recording output tokens only."""
        exporter = OTelMetricsExporter()
        exporter.record_token_usage(
            input_tokens=None,
            output_tokens=50,
        )


class TestOTelMetricsWithMockSDK:
    """Tests for metrics recording with mock OTel SDK components."""

    def test_record_token_usage_calls_histogram(self):
        """Test that recording tokens calls the histogram with correct attributes."""
        exporter = OTelMetricsExporter()
        recorded = []

        class MockHistogram:
            def record(self, value, attributes=None):
                recorded.append({"value": value, "attributes": attributes})

        exporter._initialized = True
        exporter._token_usage_histogram = MockHistogram()

        exporter.record_token_usage(
            input_tokens=200,
            output_tokens=100,
            operation="chat",
            model="gpt-4o",
            provider="openai",
        )

        assert len(recorded) == 2
        # Input tokens
        assert recorded[0]["value"] == 200
        assert recorded[0]["attributes"]["gen_ai.operation.name"] == "chat"
        assert recorded[0]["attributes"]["gen_ai.request.model"] == "gpt-4o"
        assert recorded[0]["attributes"]["gen_ai.provider.name"] == "openai"
        assert recorded[0]["attributes"]["gen_ai.token.type"] == "input"
        # Output tokens
        assert recorded[1]["value"] == 100
        assert recorded[1]["attributes"]["gen_ai.token.type"] == "output"
        assert recorded[1]["attributes"]["gen_ai.provider.name"] == "openai"

    def test_record_operation_duration_calls_histogram(self):
        """Test that recording duration calls the histogram with correct attributes."""
        exporter = OTelMetricsExporter()
        recorded = []

        class MockHistogram:
            def record(self, value, attributes=None):
                recorded.append({"value": value, "attributes": attributes})

        exporter._initialized = True
        exporter._operation_duration_histogram = MockHistogram()

        exporter.record_operation_duration(
            duration_seconds=2.5,
            operation="embedding",
            model="text-embedding-3-small",
            provider="openai",
        )

        assert len(recorded) == 1
        assert recorded[0]["value"] == 2.5
        assert recorded[0]["attributes"]["gen_ai.operation.name"] == "embedding"
        assert recorded[0]["attributes"]["gen_ai.request.model"] == "text-embedding-3-small"
        assert recorded[0]["attributes"]["gen_ai.provider.name"] == "openai"

    def test_record_ttft_calls_histogram(self):
        """Test that recording TTFT calls the histogram with correct attributes."""
        exporter = OTelMetricsExporter()
        recorded = []

        class MockHistogram:
            def record(self, value, attributes=None):
                recorded.append({"value": value, "attributes": attributes})

        exporter._initialized = True
        exporter._ttft_histogram = MockHistogram()

        exporter.record_time_to_first_token(
            ttft_seconds=0.45,
            operation="chat",
            model="gpt-4o",
            provider="openai",
        )

        assert len(recorded) == 1
        assert recorded[0]["value"] == 0.45
        assert recorded[0]["attributes"]["gen_ai.operation.name"] == "chat"
        assert recorded[0]["attributes"]["gen_ai.provider.name"] == "openai"

    def test_record_tpot_calls_histogram(self):
        """Test that recording TPoT calls the histogram with correct attributes."""
        exporter = OTelMetricsExporter()
        recorded = []

        class MockHistogram:
            def record(self, value, attributes=None):
                recorded.append({"value": value, "attributes": attributes})

        exporter._initialized = True
        exporter._tpot_histogram = MockHistogram()

        exporter.record_time_per_output_token(
            tpot_seconds=0.015,
            operation="chat",
            model="claude-3-opus",
            provider="anthropic",
        )

        assert len(recorded) == 1
        assert recorded[0]["value"] == 0.015
        assert recorded[0]["attributes"]["gen_ai.request.model"] == "claude-3-opus"
        assert recorded[0]["attributes"]["gen_ai.provider.name"] == "anthropic"

    def test_idempotent_initialization(self):
        """Test that _initialize() is idempotent."""
        exporter = OTelMetricsExporter()
        exporter._initialized = True  # Pretend already initialized

        # Should not re-initialize
        exporter._initialize()
        assert exporter._initialized is True


class TestHistogramBucketBoundaries:
    """Tests for histogram bucket boundary constants."""

    def test_token_usage_boundaries_sorted(self):
        """Token usage boundaries are strictly increasing."""
        for i in range(1, len(TOKEN_USAGE_BOUNDARIES)):
            assert TOKEN_USAGE_BOUNDARIES[i] > TOKEN_USAGE_BOUNDARIES[i - 1]

    def test_token_usage_boundaries_powers_of_four(self):
        """Token usage boundaries are powers of 4 (per OTel GenAI spec)."""
        for i, b in enumerate(TOKEN_USAGE_BOUNDARIES):
            assert b == 4**i, f"{b} is not 4^{i}"

    def test_token_usage_boundaries_range(self):
        """Token usage boundaries cover 1 to 67108864."""
        assert TOKEN_USAGE_BOUNDARIES[0] == 1
        assert TOKEN_USAGE_BOUNDARIES[-1] == 67108864

    def test_operation_duration_boundaries_sorted(self):
        """Operation duration boundaries are strictly increasing."""
        for i in range(1, len(OPERATION_DURATION_BOUNDARIES)):
            assert OPERATION_DURATION_BOUNDARIES[i] > OPERATION_DURATION_BOUNDARIES[i - 1]

    def test_operation_duration_boundaries_range(self):
        """Operation duration boundaries cover 0.01s to 81.92s."""
        assert OPERATION_DURATION_BOUNDARIES[0] == pytest.approx(0.01)
        assert OPERATION_DURATION_BOUNDARIES[-1] == pytest.approx(81.92)

    def test_streaming_boundaries_sorted(self):
        """Streaming boundaries are strictly increasing."""
        for i in range(1, len(STREAMING_BOUNDARIES)):
            assert STREAMING_BOUNDARIES[i] > STREAMING_BOUNDARIES[i - 1]

    def test_streaming_boundaries_range(self):
        """Streaming boundaries cover 0.001s to 10.0s."""
        assert STREAMING_BOUNDARIES[0] == pytest.approx(0.001)
        assert STREAMING_BOUNDARIES[-1] == pytest.approx(10.0)

    def test_streaming_boundaries_sub_second_focus(self):
        """Streaming boundaries have finer granularity at sub-second range."""
        sub_second = [b for b in STREAMING_BOUNDARIES if b < 1.0]
        assert len(sub_second) >= 8

    def test_build_histogram_views_returns_list(self):
        """_build_histogram_views returns a list (possibly empty without OTel SDK)."""
        views = OTelMetricsExporter._build_histogram_views()
        assert isinstance(views, list)


class TestInputValidation:
    """Tests for input validation on metric recording."""

    def test_negative_tokens_ignored(self):
        """Negative token values should not be recorded."""
        exporter = OTelMetricsExporter()
        recorded = []

        class MockHistogram:
            def record(self, value, attributes=None):
                recorded.append(value)

        exporter._initialized = True
        exporter._token_usage_histogram = MockHistogram()

        exporter.record_token_usage(input_tokens=-5, output_tokens=-10)
        assert len(recorded) == 0

    def test_zero_tokens_recorded(self):
        """Zero token values should be recorded (valid edge case)."""
        exporter = OTelMetricsExporter()
        recorded = []

        class MockHistogram:
            def record(self, value, attributes=None):
                recorded.append(value)

        exporter._initialized = True
        exporter._token_usage_histogram = MockHistogram()

        exporter.record_token_usage(input_tokens=0, output_tokens=0)
        assert len(recorded) == 2

    def test_negative_duration_ignored(self):
        """Negative duration values should not be recorded."""
        exporter = OTelMetricsExporter()
        recorded = []

        class MockHistogram:
            def record(self, value, attributes=None):
                recorded.append(value)

        exporter._initialized = True
        exporter._operation_duration_histogram = MockHistogram()

        exporter.record_operation_duration(duration_seconds=-1.0)
        assert len(recorded) == 0

    def test_negative_ttft_ignored(self):
        """Negative TTFT values should not be recorded."""
        exporter = OTelMetricsExporter()
        recorded = []

        class MockHistogram:
            def record(self, value, attributes=None):
                recorded.append(value)

        exporter._initialized = True
        exporter._ttft_histogram = MockHistogram()

        exporter.record_time_to_first_token(ttft_seconds=-0.5)
        assert len(recorded) == 0

    def test_negative_tpot_ignored(self):
        """Negative TPoT values should not be recorded."""
        exporter = OTelMetricsExporter()
        recorded = []

        class MockHistogram:
            def record(self, value, attributes=None):
                recorded.append(value)

        exporter._initialized = True
        exporter._tpot_histogram = MockHistogram()

        exporter.record_time_per_output_token(tpot_seconds=-0.01)
        assert len(recorded) == 0
