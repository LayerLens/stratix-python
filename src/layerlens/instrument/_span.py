from __future__ import annotations

from typing import Any, Dict, Optional, Generator
from contextlib import contextmanager

from ._types import SpanData
from ._context import _current_span, _current_recorder


@contextmanager
def span(
    name: str,
    *,
    kind: str = "internal",
    input: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Generator[SpanData, None, None]:
    recorder = _current_recorder.get()
    parent = _current_span.get()

    if recorder is None or parent is None:
        yield SpanData(name=name, kind=kind, input=input, metadata=metadata or {})
        return

    s = SpanData(
        name=name,
        kind=kind,
        parent_id=parent.span_id,
        input=input,
        metadata=metadata or {},
    )
    parent.children.append(s)

    token = _current_span.set(s)
    try:
        yield s
    except Exception as exc:
        s.finish(error=str(exc))
        raise
    else:
        s.finish()
    finally:
        _current_span.reset(token)
