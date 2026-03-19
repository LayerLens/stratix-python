"""Timeout (deadline exceeded) error injector."""

from ..span_model import SimulatedSpan
from .base import BaseErrorInjector


class TimeoutInjector(BaseErrorInjector):
    @property
    def error_type(self) -> str:
        return "timeout"

    def inject(self, span: SimulatedSpan) -> SimulatedSpan:
        self._set_error_status(
            span,
            message="Request timed out: deadline exceeded",
            http_status=504,
        )
        span.attributes["error.type"] = "timeout"
        # Truncate end time to simulate deadline
        deadline_ns = span.start_time_unix_nano + 30_000_000_000  # 30s deadline
        if span.end_time_unix_nano > deadline_ns:
            span.end_time_unix_nano = deadline_ns
        # Clear completion tokens
        if span.token_usage:
            span.token_usage.completion_tokens = 0
            span.token_usage.total_tokens = span.token_usage.prompt_tokens
        span.finish_reasons = []
        return span
