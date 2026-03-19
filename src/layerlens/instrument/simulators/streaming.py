"""Streaming behavior for simulated LLM spans.

Adds streaming-related attributes and chunk events to LLM spans,
including time-to-first-token (TTFT) and time-per-output-token (TPoT).
"""

from __future__ import annotations

from .clock import DeterministicClock
from .config import StreamingConfig
from .span_model import SimulatedSpan, SimulatedTrace, SpanType


class StreamingBehavior:
    """Applies streaming behavior to LLM spans in a trace."""

    def __init__(self, config: StreamingConfig, seed: int | None = None):
        self._config = config
        self._clock = DeterministicClock(seed=seed)

    def apply(self, trace: SimulatedTrace) -> SimulatedTrace:
        """Apply streaming behavior to all LLM spans in the trace."""
        if not self._config.enabled:
            return trace

        for span in trace.spans:
            if span.span_type != SpanType.LLM:
                continue
            self._apply_to_span(span)

        return trace

    def _apply_to_span(self, span: SimulatedSpan) -> None:
        """Apply streaming to a single LLM span."""
        span.is_streaming = True

        # Generate TTFT and TPoT
        span.ttft_ms = self._clock.ttft_ms(
            self._config.ttft_ms_min, self._config.ttft_ms_max
        )
        span.tpot_ms = self._clock.tpot_ms(
            self._config.tpot_ms_min, self._config.tpot_ms_max
        )

        # Chunk count based on completion tokens
        if span.token_usage and span.token_usage.completion_tokens > 0:
            tokens = span.token_usage.completion_tokens
            chunk_size = self._clock.randint(
                self._config.chunks_min,
                self._config.chunks_max,
            )
            span.chunk_count = max(1, tokens // max(1, tokens // chunk_size))
        else:
            span.chunk_count = self._clock.randint(
                self._config.chunks_min,
                self._config.chunks_max,
            )

        # Add streaming attributes
        span.attributes["gen_ai.is_streaming"] = True
        span.attributes["gen_ai.server.time_to_first_token"] = span.ttft_ms / 1000.0
        span.attributes["gen_ai.server.time_per_output_token"] = span.tpot_ms / 1000.0

        # Generate chunk events
        if span.chunk_count and span.chunk_count > 0:
            self._add_chunk_events(span)

    def _add_chunk_events(self, span: SimulatedSpan) -> None:
        """Add chunk span events representing streaming chunks."""
        if not span.chunk_count or span.chunk_count <= 0:
            return

        ttft_ns = int((span.ttft_ms or 100.0) * 1_000_000)
        tpot_ns = int((span.tpot_ms or 30.0) * 1_000_000)

        for i in range(min(span.chunk_count, 50)):  # Cap at 50 events
            if i == 0:
                offset_ns = ttft_ns
            else:
                offset_ns = ttft_ns + (i * tpot_ns)

            event_time = span.start_time_unix_nano + offset_ns
            if event_time > span.end_time_unix_nano:
                break

            span.events.append({
                "name": "gen_ai.content.chunk",
                "timeUnixNano": str(event_time),
                "attributes": [
                    {"key": "gen_ai.chunk.index", "value": {"intValue": str(i)}},
                ],
            })
