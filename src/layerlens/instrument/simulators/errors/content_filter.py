"""Content filter error injector."""

from ..span_model import SimulatedSpan, SpanStatus
from .base import BaseErrorInjector


class ContentFilterInjector(BaseErrorInjector):
    @property
    def error_type(self) -> str:
        return "content_filter"

    def inject(self, span: SimulatedSpan) -> SimulatedSpan:
        span.status = SpanStatus.OK  # Content filter isn't necessarily an error
        span.status_message = "Response filtered by content policy"
        span.error_type = self.error_type
        span.http_status_code = 200
        span.finish_reasons = ["content_filter"]
        span.attributes["gen_ai.response.finish_reasons"] = ["content_filter"]
        span.attributes["error.type"] = "content_filter"
        # Partial tokens — some output before filter triggered
        if span.token_usage and span.token_usage.completion_tokens > 10:
            span.token_usage.completion_tokens = min(10, span.token_usage.completion_tokens)
            span.token_usage.total_tokens = (
                span.token_usage.prompt_tokens + span.token_usage.completion_tokens
            )
        return span
