"""Base error injector ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..span_model import SimulatedSpan, SpanStatus


class BaseErrorInjector(ABC):
    """Abstract base for error injectors.

    Errors are injected per-span with independent probability.
    """

    @property
    @abstractmethod
    def error_type(self) -> str:
        """Name of this error type."""

    @abstractmethod
    def inject(self, span: SimulatedSpan) -> SimulatedSpan:
        """Inject error into span, modifying it in place."""

    def _set_error_status(
        self,
        span: SimulatedSpan,
        message: str,
        http_status: int | None = None,
    ) -> None:
        """Helper to set error status on a span."""
        span.status = SpanStatus.ERROR
        span.status_message = message
        span.error_type = self.error_type
        if http_status is not None:
            span.http_status_code = http_status
