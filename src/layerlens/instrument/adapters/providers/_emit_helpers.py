from __future__ import annotations

import uuid
from typing import Any, Dict, Callable, Optional

from .._base import AdapterInfo  # noqa: F401  (re-exported for typing)
from .pricing import PRICING, calculate_cost
from ..._events import (
    TOOL_CALL,
    AGENT_ERROR,
    COST_RECORD,
    MODEL_INVOKE,
)
from ..._context import _current_span_id, _current_collector
from .token_usage import NormalizedTokenUsage


def emit_llm_events(
    name: str,
    kwargs: Dict[str, Any],
    response: Any,
    extract_output: Callable[[Any], Any],
    extract_meta: Callable[[Any], Dict[str, Any]],
    capture_params: frozenset[str],
    latency_ms: float,
    *,
    pricing_table: Optional[dict[str, dict[str, float]]] = None,
    extract_tool_calls: Optional[Callable[[Any], list[dict[str, Any]]]] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit ``model.invoke`` + optional ``tool.call`` + ``cost.record`` events.

    Builds the full payload; the collector handles CaptureConfig gating
    (L3 suppresses model.invoke entirely; capture_content strips messages).
    """
    collector = _current_collector.get()
    if collector is None:
        return

    parent_span_id = _current_span_id.get()
    span_id = uuid.uuid4().hex[:16]
    response_meta = extract_meta(response)

    model_name = response_meta.get("response_model") or kwargs.get("model")

    parameters: Dict[str, Any] = {k: kwargs[k] for k in capture_params if k in kwargs}
    if extra_params:
        parameters.update(extra_params)

    collector.emit(
        MODEL_INVOKE,
        {
            "name": name,
            "model": model_name,
            "latency_ms": latency_ms,
            "parameters": parameters,
            "messages": _extract_messages(kwargs),
            "output_message": extract_output(response),
            **response_meta,
        },
        span_id=span_id,
        parent_span_id=parent_span_id,
    )

    if extract_tool_calls is not None:
        try:
            tool_calls = extract_tool_calls(response) or []
        except Exception:
            tool_calls = []
        for tc in tool_calls:
            collector.emit(
                TOOL_CALL,
                {
                    "provider": name.split(".")[0],
                    "model": model_name,
                    **tc,
                },
                span_id=uuid.uuid4().hex[:16],
                parent_span_id=span_id,
            )

    usage = response_meta.get("usage")
    if usage:
        _emit_cost(
            collector,
            provider=name.split(".")[0],
            model=model_name,
            usage=usage,
            pricing_table=pricing_table,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )


def emit_llm_error(
    name: str,
    error: Exception,
    latency_ms: float,
) -> None:
    """Emit agent.error for a failed LLM call."""
    collector = _current_collector.get()
    parent_span_id = _current_span_id.get()
    if collector is None:
        return
    span_id = uuid.uuid4().hex[:16]
    collector.emit(
        AGENT_ERROR,
        {"name": name, "error": str(error), "latency_ms": latency_ms},
        span_id=span_id,
        parent_span_id=parent_span_id,
    )


def emit_tool_call(
    *,
    provider: str,
    model: Optional[str],
    tool_name: str,
    arguments: Any,
    result: Any = None,
    parent_span_id: Optional[str] = None,
) -> None:
    """Explicit tool.call emission for adapters that observe tool dispatch directly."""
    collector = _current_collector.get()
    if collector is None:
        return
    collector.emit(
        TOOL_CALL,
        {
            "provider": provider,
            "model": model,
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
        },
        span_id=uuid.uuid4().hex[:16],
        parent_span_id=parent_span_id or _current_span_id.get(),
    )


def _emit_cost(
    collector: Any,
    *,
    provider: str,
    model: Optional[str],
    usage: Any,
    pricing_table: Optional[dict[str, dict[str, float]]],
    span_id: str,
    parent_span_id: Optional[str],
) -> None:
    """Emit cost.record. Accepts either a dict usage or NormalizedTokenUsage."""
    if isinstance(usage, NormalizedTokenUsage):
        normalized = usage
        usage_payload = usage.as_event_dict()
    elif isinstance(usage, dict):
        normalized = NormalizedTokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            cached_tokens=_opt_int(usage.get("cached_tokens") or usage.get("cache_read_input_tokens")),
            cache_creation_tokens=_opt_int(usage.get("cache_creation_input_tokens")),
            reasoning_tokens=_opt_int(usage.get("reasoning_tokens")),
            thinking_tokens=_opt_int(usage.get("thinking_tokens")),
        )
        usage_payload = dict(usage)
    else:
        return

    cost_usd = calculate_cost(model or "", normalized, pricing_table or PRICING) if model else None

    collector.emit(
        COST_RECORD,
        {
            "provider": provider,
            "model": model,
            "cost_usd": cost_usd,
            **usage_payload,
        },
        span_id=span_id,
        parent_span_id=parent_span_id,
    )


def _opt_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


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
