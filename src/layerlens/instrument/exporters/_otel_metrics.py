"""
STRATIX OpenTelemetry Metrics Exporter

Exports OTel GenAI Semantic Convention metrics:
- gen_ai.client.token.usage: Histogram of token counts
- gen_ai.client.operation.duration: Histogram of operation durations
- gen_ai.server.time_to_first_token: Histogram of TTFT (P3)
- gen_ai.server.time_per_output_token: Histogram of inter-token latency (P3)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Histogram bucket boundaries per OTel GenAI Semantic Conventions
# Token usage: powers of 4 from 1 to 67108864 (per official spec)
TOKEN_USAGE_BOUNDARIES = [
    1, 4, 16, 64, 256, 1024, 4096, 16384,
    65536, 262144, 1048576, 4194304, 16777216, 67108864,
]

# Operation duration: exponential doubling from 0.01s to 81.92s (per official spec)
OPERATION_DURATION_BOUNDARIES = [
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64,
    1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
]

# TTFT and TPoT: same duration-style boundaries, finer at sub-second range
# Per OTel GenAI spec, streaming metrics share duration-scale boundaries
STREAMING_BOUNDARIES = [
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
    0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
]


class OTelMetricsExporter:
    """
    OpenTelemetry metrics exporter for STRATIX GenAI metrics.

    Records histogram metrics following OTel GenAI Semantic Conventions:
    - gen_ai.client.token.usage (tokens)
    - gen_ai.client.operation.duration (seconds)
    - gen_ai.server.time_to_first_token (seconds, P3)
    - gen_ai.server.time_per_output_token (seconds, P3)
    """

    def __init__(
        self,
        endpoint: str | None = None,
        export_interval_ms: int = 60000,
    ):
        self._endpoint = endpoint
        self._export_interval_ms = export_interval_ms
        self._meter = None
        self._token_usage_histogram = None
        self._operation_duration_histogram = None
        self._ttft_histogram = None
        self._tpot_histogram = None
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize OpenTelemetry metrics SDK."""
        if self._initialized:
            return

        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider

            # Configure explicit bucket boundaries per OTel GenAI semconv
            views = self._build_histogram_views()
            self._meter_provider = MeterProvider(views=views) if views else MeterProvider()
            # Get meter from local provider (avoid polluting global state)
            self._meter = self._meter_provider.get_meter("stratix.genai")

            # Token usage histogram
            self._token_usage_histogram = self._meter.create_histogram(
                name="gen_ai.client.token.usage",
                unit="tokens",
                description="Number of tokens used per GenAI operation",
            )

            # Operation duration histogram
            self._operation_duration_histogram = self._meter.create_histogram(
                name="gen_ai.client.operation.duration",
                unit="s",
                description="Duration of GenAI operations in seconds",
            )

            # Streaming metrics (P3)
            self._ttft_histogram = self._meter.create_histogram(
                name="gen_ai.server.time_to_first_token",
                unit="s",
                description="Time to first token for streaming operations",
            )

            self._tpot_histogram = self._meter.create_histogram(
                name="gen_ai.server.time_per_output_token",
                unit="s",
                description="Average time per output token for streaming operations",
            )

            self._initialized = True
            logger.info("OTel metrics exporter initialized")

        except ImportError as e:
            logger.warning("OpenTelemetry metrics SDK not available: %s", e)
        except Exception as e:
            logger.error("Failed to initialize OTel metrics: %s", e)

    @staticmethod
    def _build_histogram_views() -> list:
        """Build OTel SDK Views with explicit bucket boundaries."""
        try:
            from opentelemetry.sdk.metrics.view import View
            from opentelemetry.sdk.metrics.aggregation import (
                ExplicitBucketHistogramAggregation,
            )

            return [
                View(
                    instrument_name="gen_ai.client.token.usage",
                    aggregation=ExplicitBucketHistogramAggregation(
                        boundaries=TOKEN_USAGE_BOUNDARIES,
                    ),
                ),
                View(
                    instrument_name="gen_ai.client.operation.duration",
                    aggregation=ExplicitBucketHistogramAggregation(
                        boundaries=OPERATION_DURATION_BOUNDARIES,
                    ),
                ),
                View(
                    instrument_name="gen_ai.server.time_to_first_token",
                    aggregation=ExplicitBucketHistogramAggregation(
                        boundaries=STREAMING_BOUNDARIES,
                    ),
                ),
                View(
                    instrument_name="gen_ai.server.time_per_output_token",
                    aggregation=ExplicitBucketHistogramAggregation(
                        boundaries=STREAMING_BOUNDARIES,
                    ),
                ),
            ]
        except ImportError:
            return []

    def record_token_usage(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        operation: str = "chat",
        model: str = "",
        provider: str = "",
    ) -> None:
        """Record token usage histogram values."""
        self._initialize()
        if not self._token_usage_histogram:
            return

        attributes = {
            "gen_ai.operation.name": operation,
            "gen_ai.request.model": model,
            "gen_ai.provider.name": provider,
        }

        if input_tokens is not None and input_tokens >= 0:
            self._token_usage_histogram.record(
                input_tokens,
                {**attributes, "gen_ai.token.type": "input"},
            )

        if output_tokens is not None and output_tokens >= 0:
            self._token_usage_histogram.record(
                output_tokens,
                {**attributes, "gen_ai.token.type": "output"},
            )

    def record_operation_duration(
        self,
        duration_seconds: float,
        operation: str = "chat",
        model: str = "",
        provider: str = "",
    ) -> None:
        """Record operation duration histogram value."""
        if duration_seconds < 0:
            return
        self._initialize()
        if not self._operation_duration_histogram:
            return

        self._operation_duration_histogram.record(
            duration_seconds,
            {
                "gen_ai.operation.name": operation,
                "gen_ai.request.model": model,
                "gen_ai.provider.name": provider,
            },
        )

    def record_time_to_first_token(
        self,
        ttft_seconds: float,
        operation: str = "chat",
        model: str = "",
        provider: str = "",
    ) -> None:
        """Record time-to-first-token histogram value (P3)."""
        if ttft_seconds < 0:
            return
        self._initialize()
        if not self._ttft_histogram:
            return

        self._ttft_histogram.record(
            ttft_seconds,
            {
                "gen_ai.operation.name": operation,
                "gen_ai.request.model": model,
                "gen_ai.provider.name": provider,
            },
        )

    def record_time_per_output_token(
        self,
        tpot_seconds: float,
        operation: str = "chat",
        model: str = "",
        provider: str = "",
    ) -> None:
        """Record time-per-output-token histogram value (P3)."""
        if tpot_seconds < 0:
            return
        self._initialize()
        if not self._tpot_histogram:
            return

        self._tpot_histogram.record(
            tpot_seconds,
            {
                "gen_ai.operation.name": operation,
                "gen_ai.request.model": model,
                "gen_ai.provider.name": provider,
            },
        )
