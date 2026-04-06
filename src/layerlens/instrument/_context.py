from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, NamedTuple
from contextvars import ContextVar

from ._collector import TraceCollector

_current_collector: ContextVar[Optional[TraceCollector]] = ContextVar("_current_collector", default=None)
_current_span_id: ContextVar[Optional[str]] = ContextVar("_current_span_id", default=None)
_parent_span_id: ContextVar[Optional[str]] = ContextVar("_parent_span_id", default=None)
_current_span_name: ContextVar[Optional[str]] = ContextVar("_current_span_name", default=None)


@dataclass
class RunState:
    """Per-run state isolated via ContextVar.

    Each concurrent run (agent invocation, crew kickoff, etc.) gets its own
    RunState stored in ``_current_run``. This isolates the collector, root span,
    timers, and any adapter-specific data so concurrent runs on the same adapter
    instance don't clobber each other.
    """

    collector: TraceCollector
    root_span_id: str
    timers: Dict[str, int] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    _token: Any = field(default=None, repr=False)
    _col_token: Any = field(default=None, repr=False)
    _span_snapshot: Any = field(default=None, repr=False)


_current_run: ContextVar[Optional[RunState]] = ContextVar("_current_run", default=None)


class _SpanSnapshot(NamedTuple):
    span_id: Any
    parent_span_id: Any
    span_name: Any


def _push_span(span_id: str, name: Optional[str] = None) -> _SpanSnapshot:
    """Push a new span onto the context stack. The current span becomes the parent."""
    old_span_id = _current_span_id.get()
    return _SpanSnapshot(
        span_id=_current_span_id.set(span_id),
        parent_span_id=_parent_span_id.set(old_span_id),
        span_name=_current_span_name.set(name),
    )


def _pop_span(snapshot: _SpanSnapshot) -> None:
    """Restore the previous span context."""
    _current_span_name.reset(snapshot.span_name)
    _parent_span_id.reset(snapshot.parent_span_id)
    _current_span_id.reset(snapshot.span_id)
