"""Assertion helpers for testing simulator output.

Provides structured validation of OTLP traces, GenAI semantic attributes,
span tree integrity, token counts, determinism, and round-trip fidelity.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from ..span_model import SimulatedTrace, SpanType


# --------------------------------------------------------------------------- #
# OTLP structure validation
# --------------------------------------------------------------------------- #

_HEX_RE = re.compile(r"^[0-9a-f]+$")
_VALID_SPAN_KINDS = {1, 2, 3, 4, 5}


def assert_valid_otlp_trace(otlp_dict: dict[str, Any]) -> None:
    """Assert *otlp_dict* is a valid OTLP ``resourceSpans`` structure.

    Checks:
    * ``resourceSpans`` key exists and is a non-empty list.
    * Each resourceSpan has ``resource`` and ``scopeSpans``.
    * Each scopeSpan has ``scope`` and ``spans``.
    * Every span has valid ``traceId`` (32 hex), ``spanId`` (16 hex).
    * Span ``kind`` is in 1-5.
    * ``startTimeUnixNano < endTimeUnixNano``.
    * Attributes are properly encoded as ``[{key, value}]`` dicts.
    """
    assert "resourceSpans" in otlp_dict, "Missing 'resourceSpans' key"
    resource_spans = otlp_dict["resourceSpans"]
    assert isinstance(resource_spans, list) and len(resource_spans) > 0, (
        "resourceSpans must be a non-empty list"
    )

    for rs_idx, rs in enumerate(resource_spans):
        _prefix = f"resourceSpans[{rs_idx}]"
        assert "resource" in rs, f"{_prefix}: missing 'resource'"
        assert "scopeSpans" in rs, f"{_prefix}: missing 'scopeSpans'"

        scope_spans = rs["scopeSpans"]
        assert isinstance(scope_spans, list) and len(scope_spans) > 0, (
            f"{_prefix}: scopeSpans must be a non-empty list"
        )

        for ss_idx, ss in enumerate(scope_spans):
            _ss_prefix = f"{_prefix}.scopeSpans[{ss_idx}]"
            assert "scope" in ss, f"{_ss_prefix}: missing 'scope'"
            assert "spans" in ss, f"{_ss_prefix}: missing 'spans'"

            for sp_idx, span in enumerate(ss["spans"]):
                _sp_prefix = f"{_ss_prefix}.spans[{sp_idx}]"
                _assert_otlp_span(span, _sp_prefix)


def _assert_otlp_span(span: dict[str, Any], prefix: str) -> None:
    """Validate a single OTLP span dict."""
    # traceId - 32 hex chars
    trace_id = span.get("traceId", "")
    assert isinstance(trace_id, str) and len(trace_id) == 32 and _HEX_RE.match(trace_id), (
        f"{prefix}: traceId must be 32 hex chars, got {trace_id!r}"
    )

    # spanId - 16 hex chars
    span_id = span.get("spanId", "")
    assert isinstance(span_id, str) and len(span_id) == 16 and _HEX_RE.match(span_id), (
        f"{prefix}: spanId must be 16 hex chars, got {span_id!r}"
    )

    # kind
    kind = span.get("kind")
    assert kind in _VALID_SPAN_KINDS, (
        f"{prefix}: kind must be 1-5, got {kind}"
    )

    # timestamps
    start_ns = int(span.get("startTimeUnixNano", 0))
    end_ns = int(span.get("endTimeUnixNano", 0))
    assert start_ns > 0, f"{prefix}: startTimeUnixNano must be > 0"
    assert end_ns > 0, f"{prefix}: endTimeUnixNano must be > 0"
    assert start_ns < end_ns, (
        f"{prefix}: startTimeUnixNano ({start_ns}) must be < endTimeUnixNano ({end_ns})"
    )

    # attributes encoding
    attrs = span.get("attributes", [])
    assert isinstance(attrs, list), f"{prefix}: attributes must be a list"
    for attr in attrs:
        assert "key" in attr, f"{prefix}: attribute missing 'key': {attr}"
        assert "value" in attr, f"{prefix}: attribute missing 'value' for key {attr.get('key')}"
        val = attr["value"]
        assert isinstance(val, dict), (
            f"{prefix}: attribute value must be a dict, got {type(val).__name__}"
        )
        valid_value_keys = {
            "stringValue", "intValue", "doubleValue", "boolValue", "arrayValue",
        }
        assert val.keys() & valid_value_keys, (
            f"{prefix}: attribute value has no valid type key: {list(val.keys())}"
        )


# --------------------------------------------------------------------------- #
# GenAI semantic attribute validation
# --------------------------------------------------------------------------- #

def assert_genai_attributes(
    span_dict: dict[str, Any],
    provider: str | None = None,
) -> None:
    """Assert a span dict has required ``gen_ai.*`` attributes.

    Checks ``gen_ai.system`` and ``gen_ai.request.model`` exist.
    If *provider* is given, checks ``gen_ai.system`` matches it.
    """
    attrs_list = span_dict.get("attributes", [])
    attrs_map = {a["key"]: a["value"] for a in attrs_list if "key" in a and "value" in a}

    assert "gen_ai.system" in attrs_map, (
        "Span missing required attribute 'gen_ai.system'"
    )
    assert "gen_ai.request.model" in attrs_map, (
        "Span missing required attribute 'gen_ai.request.model'"
    )

    if provider is not None:
        system_val = attrs_map["gen_ai.system"]
        actual = system_val.get("stringValue", system_val)
        assert actual == provider, (
            f"gen_ai.system expected '{provider}', got '{actual}'"
        )


# --------------------------------------------------------------------------- #
# Span tree validation
# --------------------------------------------------------------------------- #

def assert_span_tree(otlp_dict: dict[str, Any]) -> None:
    """Assert spans form a valid parent-child tree.

    Checks:
    * Every ``parentSpanId`` references an existing span.
    * Exactly one root span has no ``parentSpanId``.
    * No circular references.
    """
    all_spans: list[dict[str, Any]] = []
    for rs in otlp_dict.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            all_spans.extend(ss.get("spans", []))

    span_ids = {s["spanId"] for s in all_spans}
    roots: list[str] = []

    for span in all_spans:
        parent = span.get("parentSpanId")
        if parent is None or parent == "":
            roots.append(span["spanId"])
        else:
            assert parent in span_ids, (
                f"Span {span['spanId']} references non-existent parentSpanId {parent}"
            )

    assert len(roots) == 1, (
        f"Expected exactly 1 root span, found {len(roots)}: {roots}"
    )

    # Check for circular references via DFS
    children_map: dict[str, list[str]] = {sid: [] for sid in span_ids}
    for span in all_spans:
        parent = span.get("parentSpanId")
        if parent and parent in children_map:
            children_map[parent].append(span["spanId"])

    visited: set[str] = set()
    stack: set[str] = set()

    def _dfs(node: str) -> None:
        assert node not in stack, f"Circular reference detected at span {node}"
        if node in visited:
            return
        stack.add(node)
        for child in children_map.get(node, []):
            _dfs(child)
        stack.discard(node)
        visited.add(node)

    for root in roots:
        _dfs(root)


# --------------------------------------------------------------------------- #
# Token count validation
# --------------------------------------------------------------------------- #

def assert_token_counts(trace: SimulatedTrace) -> None:
    """Assert token counts are non-negative and total >= prompt + completion.

    Works on a ``SimulatedTrace`` instance.
    """
    for span in trace.spans:
        if span.token_usage is None:
            continue
        tu = span.token_usage
        assert tu.prompt_tokens >= 0, (
            f"Span {span.span_id}: prompt_tokens ({tu.prompt_tokens}) must be >= 0"
        )
        assert tu.completion_tokens >= 0, (
            f"Span {span.span_id}: completion_tokens ({tu.completion_tokens}) must be >= 0"
        )
        assert tu.total_tokens >= 0, (
            f"Span {span.span_id}: total_tokens ({tu.total_tokens}) must be >= 0"
        )
        assert tu.total_tokens >= tu.prompt_tokens + tu.completion_tokens, (
            f"Span {span.span_id}: total_tokens ({tu.total_tokens}) must be >= "
            f"prompt ({tu.prompt_tokens}) + completion ({tu.completion_tokens})"
        )
        if tu.cached_tokens is not None:
            assert tu.cached_tokens >= 0, (
                f"Span {span.span_id}: cached_tokens ({tu.cached_tokens}) must be >= 0"
            )
        if tu.reasoning_tokens is not None:
            assert tu.reasoning_tokens >= 0, (
                f"Span {span.span_id}: reasoning_tokens ({tu.reasoning_tokens}) must be >= 0"
            )


# --------------------------------------------------------------------------- #
# Determinism validation
# --------------------------------------------------------------------------- #

def assert_deterministic(
    generate_fn: Callable[[], Any],
    runs: int = 3,
) -> None:
    """Assert *generate_fn* produces identical output across *runs* invocations.

    Compares the JSON-serialisable output of each run.
    """
    import json

    results: list[str] = []
    for i in range(runs):
        output = generate_fn()
        # Normalise to JSON string for comparison
        if hasattr(output, "model_dump"):
            serialised = json.dumps(output.model_dump(mode="json"), sort_keys=True)
        elif isinstance(output, (dict, list)):
            serialised = json.dumps(output, sort_keys=True)
        else:
            serialised = str(output)
        results.append(serialised)

    first = results[0]
    for i, result in enumerate(results[1:], start=2):
        assert result == first, (
            f"Run {i} produced different output than run 1. "
            f"First divergence found in outputs."
        )


# --------------------------------------------------------------------------- #
# Round-trip validation
# --------------------------------------------------------------------------- #

def assert_round_trip(
    trace: SimulatedTrace,
    source_formatter: Any,
    output_formatter: Any,
) -> None:
    """Assert *trace* survives source enrichment + output formatting.

    Enriches the trace with *source_formatter*, formats it with
    *output_formatter*, then validates the resulting OTLP structure.
    """
    # Apply source enrichment
    profile = source_formatter.get_default_profile()
    trace.resource_attributes = source_formatter.get_resource_attributes()
    scope_name, scope_version = source_formatter.get_scope()
    trace.scope_name = scope_name
    trace.scope_version = scope_version

    for span in trace.spans:
        source_formatter.enrich_span(span, profile, include_content=False)

    # Format output
    output = output_formatter.format_trace(trace)

    # Validate the formatted output
    assert_valid_otlp_trace(output)

    # Validate span tree
    assert_span_tree(output)

    # Validate span count is preserved
    formatted_spans: list[dict[str, Any]] = []
    for rs in output.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            formatted_spans.extend(ss.get("spans", []))

    assert len(formatted_spans) == len(trace.spans), (
        f"Span count mismatch: trace has {len(trace.spans)} spans, "
        f"output has {len(formatted_spans)} spans"
    )

    # Validate trace ID is preserved
    for sp in formatted_spans:
        assert sp["traceId"] == trace.trace_id, (
            f"Trace ID mismatch: expected {trace.trace_id}, got {sp['traceId']}"
        )
