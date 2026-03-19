"""Rate limit (429) error injector."""

from ..span_model import SimulatedSpan
from .base import BaseErrorInjector


class RateLimitInjector(BaseErrorInjector):
    @property
    def error_type(self) -> str:
        return "rate_limit"

    def inject(self, span: SimulatedSpan) -> SimulatedSpan:
        self._set_error_status(
            span,
            message="Rate limit exceeded. Retry after 30 seconds.",
            http_status=429,
        )
        span.attributes["http.response.status_code"] = 429
        span.attributes["retry-after"] = "30"
        span.attributes["error.type"] = "rate_limit"
        # Clear completion tokens (no response generated)
        if span.token_usage:
            span.token_usage.completion_tokens = 0
            span.token_usage.total_tokens = span.token_usage.prompt_tokens
        span.finish_reasons = []
        return span
