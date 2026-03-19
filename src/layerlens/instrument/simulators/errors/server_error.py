"""Server error (500/502/503) injector."""

import random

from ..span_model import SimulatedSpan
from .base import BaseErrorInjector


class ServerErrorInjector(BaseErrorInjector):
    _STATUS_CODES = [500, 502, 503]

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    @property
    def error_type(self) -> str:
        return "server_error"

    def inject(self, span: SimulatedSpan) -> SimulatedSpan:
        status_code = self._rng.choice(self._STATUS_CODES)
        messages = {
            500: "Internal server error",
            502: "Bad gateway",
            503: "Service temporarily unavailable",
        }
        self._set_error_status(
            span,
            message=messages.get(status_code, "Server error"),
            http_status=status_code,
        )
        span.attributes["error.type"] = "server_error"
        span.attributes["http.response.status_code"] = status_code
        if span.token_usage:
            span.token_usage.completion_tokens = 0
            span.token_usage.total_tokens = span.token_usage.prompt_tokens
        span.finish_reasons = []
        return span
