from __future__ import annotations

import uuid
from typing import Generator
from contextlib import contextmanager

from ._context import _push_span, _pop_span


@contextmanager
def span(name: str) -> Generator[str, None, None]:
    """Create a child span for grouping events.

    Pushes a new span_id onto the context stack. Any events emitted
    inside the block will have this span_id, with the outer span as
    parent_span_id.

    Yields the span_id string.
    """
    new_span_id = uuid.uuid4().hex[:16]
    snapshot = _push_span(new_span_id, name)
    try:
        yield new_span_id
    finally:
        _pop_span(snapshot)
