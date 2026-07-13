from __future__ import annotations

import uuid
from typing import Any, Dict, Callable, Optional

from ..._w3c import gen_ai_attributes
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
from ..._secret_scrub import safe_error


def _derive_operation(name: str) -> str:
    """Derive the OTel gen_ai.operation.name from our event name string."""
    low = name.lower()
    if "embedding" in low:
        return "embeddings"
    if "responses" in low:
        return "responses"
    if "completion" in low and "chat" not in low:
        return "text_completion"
    return "chat"


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
    ttft_ms: Optional[float] = None,
    streaming_duration_ms: Optional[float] = None,
    provider: Optional[str] = None,
    framework: Optional[str] = None,
) -> None:
    """Emit ``model.invoke`` + optional ``tool.call`` + ``cost.record`` events.

    Builds the full payload; the collector handles CaptureConfig gating
    (L3 suppresses model.invoke entirely; capture_content strips messages).

    ``provider`` overrides the default ``name.split(".")[0]`` derivation so
    routing layers (LiteLLM) can attribute the call to the underlying provider
    that actually served the request (LAY-3455).

    ``framework`` is the integration name (openai/anthropic/litellm/
    azure_openai/…). It is stamped on every emitted event so the framework
    column reflects the integration rather than the routed/underlying provider
    (litellm shows 'litellm' not the routed provider; azure_openai shows
    'azure_openai' not 'openai') — S19/F12. ``cost.record.provider`` stays the
    honest underlying provider, unchanged.
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

    resolved = provider or name.split(".")[0]
    # Integration-name stamp (S19/F12); omitted (honest blank) if not supplied.
    fw = {"framework": framework} if framework else {}
    otel_attrs = gen_ai_attributes(
        provider=resolved,
        operation=_derive_operation(name),
        parameters=parameters,
        response_meta=response_meta,
        usage=response_meta.get("usage"),
    )

    streaming_timing: Dict[str, float] = {}
    if ttft_ms is not None:
        streaming_timing["ttft_ms"] = ttft_ms
    if streaming_duration_ms is not None:
        streaming_timing["streaming_duration_ms"] = streaming_duration_ms

    collector.emit(
        MODEL_INVOKE,
        {
            "name": name,
            "model": model_name,
            "latency_ms": latency_ms,
            "parameters": parameters,
            "messages": _extract_messages(kwargs),
            "output_message": extract_output(response),
            "otel_gen_ai": otel_attrs,
            **fw,
            **streaming_timing,
            **response_meta,
            # Flat token counts beside the nested `usage` block, so the atlas
            # extractor (which reads top-level prompt/completion/total_tokens,
            # never usage.*) fills the tokens column (S11/F2). Spread last so the
            # normalized values win; empty {} when no usage was declared.
            **_flat_token_fields(response_meta.get("usage")),
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
                    "provider": resolved,
                    "model": model_name,
                    **fw,
                    **tc,
                },
                span_id=uuid.uuid4().hex[:16],
                parent_span_id=span_id,
            )

    usage = response_meta.get("usage")
    if usage:
        _emit_cost(
            collector,
            provider=resolved,
            model=model_name,
            usage=usage,
            pricing_table=pricing_table,
            span_id=span_id,
            parent_span_id=parent_span_id,
            service_tier=response_meta.get("service_tier"),
            framework=framework,
        )


def emit_llm_error(
    name: str,
    error: Exception,
    latency_ms: float,
    *,
    partial_meta: Optional[Dict[str, Any]] = None,
    partial_chunks: Optional[int] = None,
) -> None:
    """Emit agent.error for a failed LLM call.

    When the failure happened mid-stream, callers pass ``partial_meta`` with
    whatever was accumulated before the exception (token counts, response_id,
    stop_reason, etc.) along with ``partial_chunks`` — the number of chunks
    or events received pre-error. This satisfies the LAY-3329 / LAY-3332
    "partial event with error metadata" acceptance criterion.
    """
    collector = _current_collector.get()
    parent_span_id = _current_span_id.get()
    if collector is None:
        return
    span_id = uuid.uuid4().hex[:16]
    payload: Dict[str, Any] = {
        "name": name,
        # Provider auth/validation exceptions routinely echo the API key /
        # bearer token; scrub before it enters the (always-uploaded) payload.
        "error": safe_error(error),
        "error_type": type(error).__name__,
        "latency_ms": latency_ms,
    }
    if partial_chunks is not None:
        payload["partial_chunks"] = partial_chunks
    if partial_meta:
        payload["partial_meta"] = partial_meta
    collector.emit(
        AGENT_ERROR,
        payload,
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
    service_tier: Optional[str] = None,
    framework: Optional[str] = None,
) -> None:
    """Emit cost.record. Accepts either a dict usage or NormalizedTokenUsage.

    ``cost_usd`` is computed here for back-compat, but the authoritative price is
    (re)computed at the collector chokepoint (A1) from this payload, which is why
    ``service_tier`` is threaded into the payload — so the central formula can
    apply tier pricing uniformly.
    """
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

    cost_usd = (
        calculate_cost(model or "", normalized, pricing_table or PRICING, service_tier=service_tier) if model else None
    )

    cost_payload: Dict[str, Any] = {
        "provider": provider,
        "model": model,
        "cost_usd": cost_usd,
        **usage_payload,
    }
    # Integration name (S19/F12) — distinct from `provider`, the honest
    # underlying provider that priced the call (unchanged).
    if framework:
        cost_payload["framework"] = framework
    if service_tier is not None:
        cost_payload["service_tier"] = service_tier

    collector.emit(
        COST_RECORD,
        cost_payload,
        span_id=span_id,
        parent_span_id=parent_span_id,
    )


def _flat_token_fields(usage: Any) -> Dict[str, int]:
    """Flatten declared token counts to top-level model.invoke keys beside the
    nested ``usage`` block (S11/F2).

    The atlas extractor reads flat ``prompt_tokens``/``completion_tokens``/
    ``total_tokens`` (and the ``tokens_*`` spellings), never ``usage.*`` — so the
    tokens column stayed blank for every provider lane. This never fabricates:
    it returns ``{}`` when the producer declared no usable counts, and mirrors
    ``_emit_cost``'s normalization (prompt<-prompt_tokens|input_tokens, etc.).
    ``total_tokens`` is the honest sum when the provider omits it (Anthropic).
    """
    if isinstance(usage, NormalizedTokenUsage):
        normalized = usage
    elif isinstance(usage, dict):
        normalized = NormalizedTokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
        )
    else:
        return {}
    if not (normalized.prompt_tokens or normalized.completion_tokens or normalized.total_tokens):
        return {}
    return {
        "prompt_tokens": normalized.prompt_tokens,
        "completion_tokens": normalized.completion_tokens,
        "total_tokens": normalized.total_tokens,
    }


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
