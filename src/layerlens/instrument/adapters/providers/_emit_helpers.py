from __future__ import annotations

import uuid
from typing import Any, Dict, Callable

from ..._context import _current_span_id, _current_collector


def emit_llm_events(
    name: str,
    kwargs: Dict[str, Any],
    response: Any,
    extract_output: Callable[[Any], Any],
    extract_meta: Callable[[Any], Dict[str, Any]],
    capture_params: frozenset[str],
    latency_ms: float,
) -> None:
    """Emit model.invoke + cost.record events for an LLM call.

    Builds the full payload -- the collector handles CaptureConfig gating
    (L3 suppresses model.invoke entirely, capture_content strips messages).
    """
    collector = _current_collector.get()
    if collector is None:
        return

    parent_span_id = _current_span_id.get()
    span_id = uuid.uuid4().hex[:16]
    response_meta = extract_meta(response)

    # Resolve model name: prefer response_model (actual model used), fall back to kwargs
    model_name = response_meta.get("response_model") or kwargs.get("model")

    collector.emit(
        "model.invoke",
        {
            "name": name,
            "model": model_name,
            "latency_ms": latency_ms,
            "parameters": {k: kwargs[k] for k in capture_params if k in kwargs},
            "messages": _extract_messages(kwargs),
            "output_message": extract_output(response),
            **response_meta,
        },
        span_id=span_id,
        parent_span_id=parent_span_id,
    )

    usage = response_meta.get("usage", {})
    if usage:
        collector.emit(
            "cost.record",
            {
                "provider": name.split(".")[0],
                "model": response_meta.get("response_model", kwargs.get("model")),
                **usage,
            },
            span_id=span_id,
            parent_span_id=parent_span_id,
        )


def emit_llm_error(
    name: str,
    error: Exception,
    latency_ms: float,
) -> None:
    """Emit agent.error event for a failed LLM call."""
    collector = _current_collector.get()
    parent_span_id = _current_span_id.get()
    if collector is None:
        return

    span_id = uuid.uuid4().hex[:16]
    collector.emit(
        "agent.error",
        {"name": name, "error": str(error), "latency_ms": latency_ms},
        span_id=span_id,
        parent_span_id=parent_span_id,
    )


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
