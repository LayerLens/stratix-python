from __future__ import annotations

from typing import Any, Dict, Optional

from ._context import _current_collector, _current_span_id, _parent_span_id, _current_span_name


def emit(
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit an event into the current trace.

    Reads the active TraceCollector, span_id, parent_span_id, and span_name
    from context. No-op if called outside a @trace block.

    Args:
        event_type: Event type string (e.g. "tool.call", "model.invoke").
        payload: Event payload dict. Defaults to empty dict.
    """
    collector = _current_collector.get()
    if collector is None:
        return

    span_id = _current_span_id.get()
    if span_id is None:
        return

    collector.emit(
        event_type,
        payload or {},
        span_id=span_id,
        parent_span_id=_parent_span_id.get(),
        span_name=_current_span_name.get(),
    )
