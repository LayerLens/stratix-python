"""L2 — disabling a layer must actually suppress its events (no fail-open).

Historically 13/14 protocol content event types were ABSENT from
``_EVENT_TYPE_MAP``, so ``is_layer_enabled`` fell open (returned True) for them:
disabling ``l6b_protocol_streams`` or using ``minimal()`` suppressed nothing —
message text, tool args, amounts, merchant names still flowed when the customer
asked the layer off (LAY-3578 / L2). These tests assert every content-bearing
protocol type maps to a layer and that disabling that layer suppresses it.
"""

from __future__ import annotations

import pytest

from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import (
    _CONTENT_KEYS,
    _ALWAYS_ENABLED,
    _EVENT_TYPE_MAP,
    CaptureConfig,
)

# Run as a required CI gate via `-m invariant` (see .github/workflows/invariants.yaml).
pytestmark = pytest.mark.invariant

_PROTOCOL_PREFIXES = {"agui", "mcp", "a2a", "payment", "commerce", "protocol"}
_PROTOCOL_CONTENT_TYPES = sorted(
    et for et in _CONTENT_KEYS if et.split(".")[0] in _PROTOCOL_PREFIXES and et not in _ALWAYS_ENABLED
)


@pytest.mark.parametrize("event_type", _PROTOCOL_CONTENT_TYPES)
def test_disabling_mapped_layer_suppresses_event(event_type: str) -> None:
    layer = _EVENT_TYPE_MAP.get(event_type)
    assert layer is not None, f"{event_type} is not mapped to a layer — is_layer_enabled fails OPEN (L2)"
    collector = TraceCollector(object(), CaptureConfig(**{layer: False}))
    collector.emit(event_type, {"marker": 1}, span_id="s")
    assert not collector.events, f"disabling {layer} did NOT suppress {event_type} (fail-open, L2)"


@pytest.mark.parametrize(
    "event_type",
    ["agui.message", "agui.tool_call", "agui.state", "mcp.tool.call", "mcp.elicitation", "mcp.tools.listed"],
)
def test_minimal_suppresses_stream_and_tool_protocol_content(event_type: str) -> None:
    """minimal() turns off l6b (streams) and l5a (tools), so AG-UI + MCP
    content is dropped entirely under the lightweight preset."""
    collector = TraceCollector(object(), CaptureConfig.minimal())
    collector.emit(event_type, {"marker": 1}, span_id="s")
    assert not collector.events, f"minimal() did not suppress {event_type}"


@pytest.mark.parametrize(
    "agui_type",
    ["RUN_STARTED", "RUN_FINISHED", "RUN_ERROR", "STATE_SNAPSHOT", "STATE_DELTA", "MESSAGES_SNAPSHOT"],
)
def test_agui_lifecycle_state_events_are_suppressible(agui_type: str) -> None:
    """AG-UI run/state events must NOT ride the ALWAYS-ENABLED agent.state.change
    (which no layer toggle can suppress). After the remap they map to L6b types
    that minimal()/l6b-off actually suppresses (LAY-3578)."""
    from layerlens.instrument.adapters.protocols.agui.event_mapper import map_agui_to_stratix

    stratix_event = map_agui_to_stratix(agui_type)["stratix_event"]
    assert stratix_event not in _ALWAYS_ENABLED, f"{agui_type} -> {stratix_event} is un-suppressible (always-enabled)"
    collector = TraceCollector(object(), CaptureConfig.minimal())
    collector.emit(stratix_event, {"marker": 1}, span_id="s")
    assert not collector.events, f"{agui_type} -> {stratix_event} not suppressed by minimal()"


def test_agui_state_snapshot_same_type_on_both_paths() -> None:
    """Dual-path reconciliation: a STATE_SNAPSHOT wire event yields the SAME
    stratix type via wrap_stream (adapter) and the ASGI/WSGI middleware."""
    from layerlens.instrument._context import _current_collector
    from layerlens.instrument.adapters.protocols.agui.adapter import AGUIProtocolAdapter
    from layerlens.instrument.adapters.protocols.agui.middleware import _process_sse_chunk

    def _types(run) -> set:
        collector = TraceCollector(object(), CaptureConfig.standard())
        token = _current_collector.set(collector)
        try:
            run()
        finally:
            _current_collector.reset(token)
        return {e["event_type"] for e in collector.events}

    adapter = AGUIProtocolAdapter()
    stream_types = _types(
        lambda: [None for _ in adapter.wrap_stream(iter([{"type": "STATE_SNAPSHOT", "state": {"k": 1}}]))]
    )
    mw_types = _types(lambda: _process_sse_chunk(adapter, b'data: {"type": "STATE_SNAPSHOT", "state": {"k": 1}}\n'))
    assert "agui.state" in stream_types, f"wrap_stream path: {stream_types}"
    assert "agui.state" in mw_types, f"middleware path emitted {mw_types}, not agui.state (dual-path mismatch)"


@pytest.mark.parametrize(
    "event_type",
    ["payment.mandate_signed", "payment.receipt_issued", "commerce.checkout_completed", "a2a.task.created"],
)
def test_minimal_keeps_lifecycle_audit_trail(event_type: str) -> None:
    """Documented decision (2026-06-24): payment/commerce/a2a-task events are
    L6c *lifecycle* — kept by minimal() so the financial/delegation audit trail
    survives a lightweight config, and suppressed only when l6c is explicitly
    disabled (see test above)."""
    collector = TraceCollector(object(), CaptureConfig.minimal())
    collector.emit(event_type, {"marker": 1}, span_id="s")
    assert collector.events, f"minimal() unexpectedly dropped lifecycle event {event_type}"
