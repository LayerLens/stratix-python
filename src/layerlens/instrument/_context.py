from __future__ import annotations

from typing import Any, Optional, NamedTuple
from contextvars import ContextVar

from ._collector import TraceCollector

_current_collector: ContextVar[Optional[TraceCollector]] = ContextVar("_current_collector", default=None)
_current_span_id: ContextVar[Optional[str]] = ContextVar("_current_span_id", default=None)
_parent_span_id: ContextVar[Optional[str]] = ContextVar("_parent_span_id", default=None)
_current_span_name: ContextVar[Optional[str]] = ContextVar("_current_span_name", default=None)


class _SpanTokens(NamedTuple):
    span_id: Any
    parent_span_id: Any
    span_name: Any


def _push_span(span_id: str, name: Optional[str] = None) -> _SpanTokens:
    """Push a new span onto the context stack. The current span becomes the parent."""
    old_span_id = _current_span_id.get()
    return _SpanTokens(
        span_id=_current_span_id.set(span_id),
        parent_span_id=_parent_span_id.set(old_span_id),
        span_name=_current_span_name.set(name),
    )


def _pop_span(tokens: _SpanTokens) -> None:
    """Restore the previous span context."""
    _current_span_name.reset(tokens.span_name)
    _parent_span_id.reset(tokens.parent_span_id)
    _current_span_id.reset(tokens.span_id)
