"""Authentication failure (401/403) error injector."""

from ..span_model import SimulatedSpan
from .base import BaseErrorInjector


class AuthFailureInjector(BaseErrorInjector):
    @property
    def error_type(self) -> str:
        return "auth_failure"

    def inject(self, span: SimulatedSpan) -> SimulatedSpan:
        self._set_error_status(
            span,
            message="Authentication failed: invalid or expired API key",
            http_status=401,
        )
        span.attributes["error.type"] = "auth_failure"
        span.attributes["http.response.status_code"] = 401
        if span.token_usage:
            span.token_usage.completion_tokens = 0
            span.token_usage.total_tokens = span.token_usage.prompt_tokens
        span.finish_reasons = []
        return span
