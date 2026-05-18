"""W3C Trace Context propagation (https://www.w3.org/TR/trace-context/).

Lets user code stitch our traces into a wider distributed-tracing system
without taking a hard dependency on OpenTelemetry. Two entry points:

- ``inject_headers(headers)`` — copy the active layerlens trace/span into
  ``traceparent`` (and ``tracestate`` if any), so an HTTP call our code
  makes carries our trace id forward to the receiver.
- ``extract_headers(headers)`` — parse ``traceparent``/``tracestate`` from
  an inbound request so server-side code can adopt the caller's trace
  before opening its own ``trace_context()``.

If OpenTelemetry is installed in the user's environment we delegate to
its propagator so OTel-aware peers see the same context; otherwise we
build the headers by hand from the current ``TraceCollector`` + active
span ContextVars.
"""

from __future__ import annotations

import os
import uuid
import logging
from typing import Any, Dict, Optional

from ._context import _current_span_id, _current_collector

log = logging.getLogger(__name__)

TRACEPARENT_HEADER = "traceparent"
TRACESTATE_HEADER = "tracestate"

_W3C_VERSION = "00"
_DEFAULT_FLAGS = "01"  # sampled=true


def _expand_trace_id(short: str) -> str:
    """Pad our 16-hex-char trace id up to W3C's 32-hex requirement.

    The left half is a deterministic per-process prefix derived from the
    PID; the right half is the layerlens trace id (left-padded with zeros
    to 16 chars). Result is always exactly 32 hex chars.
    """
    short = short.lower().lstrip("0x")
    if len(short) >= 32:
        return short[:32]
    prefix = (f"{os.getpid():016x}")[-16:]  # exactly 16
    padded_short = short.rjust(16, "0")[-16:]  # exactly 16
    return prefix + padded_short


def _shorten_trace_id(full: str) -> str:
    """Inverse of ``_expand_trace_id``: keep the layerlens half."""
    full = full.lower().lstrip("0x")
    return full[-16:] if len(full) >= 16 else full


def _build_traceparent(trace_id: str, span_id: str, flags: str = _DEFAULT_FLAGS) -> str:
    full_trace = _expand_trace_id(trace_id)
    span = span_id.lower()[:16].rjust(16, "0")
    return f"{_W3C_VERSION}-{full_trace}-{span}-{flags}"


def _parse_traceparent(value: str) -> Optional[Dict[str, str]]:
    parts = value.strip().split("-")
    if len(parts) < 4:
        return None
    version, trace_id, parent_span_id, flags = parts[:4]
    if len(trace_id) != 32 or len(parent_span_id) != 16:
        return None
    return {
        "version": version,
        "trace_id": trace_id,
        "parent_span_id": parent_span_id,
        "trace_flags": flags,
    }


def inject_headers(headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Add ``traceparent`` (and ``tracestate``) to *headers* based on the
    active layerlens trace context.

    The dict is mutated in place AND returned. If OpenTelemetry is
    installed, its propagator is used so existing OTel users keep working;
    otherwise we generate the header from the active TraceCollector +
    current span. No-op (returns headers unchanged) when no trace is active.
    """
    if headers is None:
        headers = {}

    # Prefer OTel if the user has it installed.
    try:
        from opentelemetry.propagate import inject as otel_inject

        otel_inject(headers)
        if TRACEPARENT_HEADER in headers:
            return headers
    except ImportError:
        pass
    except Exception:  # pragma: no cover — defensive against OTel version drift
        log.debug(
            "layerlens._w3c: OpenTelemetry propagator raised; falling back",
            exc_info=True,
        )

    collector = _current_collector.get()
    span_id = _current_span_id.get()
    if collector is None or not span_id:
        return headers

    headers[TRACEPARENT_HEADER] = _build_traceparent(collector.trace_id, span_id)
    return headers


def extract_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Parse ``traceparent`` / ``tracestate`` from *headers*.

    Returns a dict with ``trace_id``, ``parent_span_id``, ``trace_flags``,
    and optionally ``tracestate``. ``trace_id`` is the layerlens-style
    short id (16 hex chars) — use ``raw_trace_id`` for the full 32-char
    W3C form if you need to re-emit it. Returns ``{}`` when no header is
    present or the value is malformed.
    """
    raw = headers.get(TRACEPARENT_HEADER) or headers.get(TRACEPARENT_HEADER.title())
    if not raw:
        return {}
    parsed = _parse_traceparent(raw)
    if parsed is None:
        return {}

    result: Dict[str, str] = {
        "trace_id": _shorten_trace_id(parsed["trace_id"]),
        "raw_trace_id": parsed["trace_id"],
        "parent_span_id": parsed["parent_span_id"],
        "trace_flags": parsed["trace_flags"],
    }
    state = headers.get(TRACESTATE_HEADER) or headers.get(TRACESTATE_HEADER.title())
    if state:
        result["tracestate"] = state
    return result


def new_traceparent(trace_id: Optional[str] = None, span_id: Optional[str] = None) -> str:
    """Build a fresh ``traceparent`` header value (e.g. for outbound HTTP).

    Uses *trace_id* and *span_id* if supplied; otherwise reads the active
    layerlens context. Generates random ids when no context is available
    — handy for one-off requests.
    """
    if trace_id is None:
        collector = _current_collector.get()
        trace_id = collector.trace_id if collector is not None else uuid.uuid4().hex[:16]
    if span_id is None:
        span_id = _current_span_id.get() or uuid.uuid4().hex[:16]
    return _build_traceparent(trace_id, span_id)


# ----------------------------------------------------------------------
# OTel GenAI semantic conventions
#
# https://opentelemetry.io/docs/specs/semconv/gen-ai/
# ----------------------------------------------------------------------


# Map our capture_params -> gen_ai.request.* attribute names. Keys without
# a mapping are dropped (the raw value is still available in the original
# ``parameters`` dict on the event payload).
_GEN_AI_REQUEST_ATTR: Dict[str, str] = {
    "model": "gen_ai.request.model",
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "top_k": "gen_ai.request.top_k",
    "max_tokens": "gen_ai.request.max_tokens",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
    "stop": "gen_ai.request.stop_sequences",
    "seed": "gen_ai.request.seed",
}

# Provider name -> gen_ai.system value. Matches the OTel registry.
_GEN_AI_SYSTEM: Dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "azure_openai": "az.ai.openai",
    "google_vertex": "gcp.vertex_ai",
    "bedrock": "aws.bedrock",
    "ollama": "ollama",
    "litellm": "litellm",
}


def gen_ai_attributes(
    *,
    provider: str,
    operation: str,
    parameters: Dict[str, Any],
    response_meta: Dict[str, Any],
    usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a dict of OTel GenAI semantic-convention attributes.

    Caller passes the provider name (``"openai"``, ``"anthropic"``, ...),
    the operation (``"chat"``, ``"embeddings"``, ``"text_completion"``),
    the request parameters dict, the extracted response metadata, and
    optionally the usage dict. Returned dict can be merged into the
    ``model.invoke`` event payload under ``otel_gen_ai`` (or any key).

    Missing values are dropped — the returned dict only contains keys
    whose values are non-``None`` and non-empty.
    """
    attrs: Dict[str, Any] = {
        "gen_ai.system": _GEN_AI_SYSTEM.get(provider, provider),
        "gen_ai.operation.name": operation,
    }
    for key, value in parameters.items():
        attr = _GEN_AI_REQUEST_ATTR.get(key)
        if attr is not None and value is not None:
            attrs[attr] = value

    response_model = response_meta.get("response_model")
    if response_model:
        attrs["gen_ai.response.model"] = response_model
    response_id = response_meta.get("response_id")
    if response_id:
        attrs["gen_ai.response.id"] = response_id
    finish_reason = response_meta.get("finish_reason")
    if finish_reason:
        attrs["gen_ai.response.finish_reasons"] = [finish_reason]

    if usage:
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion = usage.get("completion_tokens") or usage.get("output_tokens")
        if prompt is not None:
            attrs["gen_ai.usage.input_tokens"] = int(prompt)
        if completion is not None:
            attrs["gen_ai.usage.output_tokens"] = int(completion)

    return attrs
