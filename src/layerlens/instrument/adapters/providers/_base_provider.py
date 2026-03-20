from __future__ import annotations

from typing import Any, Dict, Tuple, Callable, Optional

from ..._types import SpanData
from ..._context import _current_span, _current_recorder


def create_llm_span(
    name: str,
    kwargs: Dict[str, Any],
    capture_params: frozenset[str],
) -> Tuple[Optional[SpanData], Any]:
    recorder = _current_recorder.get()
    parent = _current_span.get()

    if recorder is None or parent is None:
        return None, None

    meta = {k: kwargs[k] for k in capture_params if k in kwargs}

    s = SpanData(
        name=name,
        kind="llm",
        parent_id=parent.span_id,
        input=_extract_messages(kwargs),
        metadata=meta,
    )
    parent.children.append(s)
    token = _current_span.set(s)
    return s, token


def finish_llm_span(
    span: SpanData,
    token: Any,
    response: Any,
    extract_output: Callable[[Any], Any],
    extract_meta: Callable[[Any], Dict[str, Any]],
) -> None:
    try:
        span.output = extract_output(response)
        span.metadata.update(extract_meta(response))
        span.finish()
    finally:
        _current_span.reset(token)


def fail_llm_span(span: SpanData, token: Any, error: Exception) -> None:
    try:
        span.finish(error=str(error))
    finally:
        _current_span.reset(token)


def _extract_messages(kwargs: Dict[str, Any]) -> Any:
    messages = kwargs.get("messages")
    if messages is not None:
        return [_serialize_message(m) for m in messages]
    for key in ("prompt", "contents", "input"):
        val = kwargs.get(key)
        if val is not None:
            return val
    return None


def _serialize_message(msg: Any) -> Any:
    if isinstance(msg, dict):
        return msg
    try:
        return {"role": msg.role, "content": msg.content}
    except AttributeError:
        return str(msg)
