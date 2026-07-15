"""ADP-W2 Family-B recorder for the ``agui`` protocol adapter (record-real-once).

AG-UI (the CopilotKit agent<->frontend SSE transport) is an **LLM-free UI
protocol surface**, not an agent framework: the adapter observes the SSE event
stream a CopilotKit runtime emits and reconstructs telemetry from it. So — like
the other protocol adapters (a2a / mcp) — there is no model call, no
``agent.identity`` and no agent DAG. The recorded trace renders as an **honest
empty-state** (Agent column ``—``) OTel-style waterfall of the protocol events,
which is the correct, non-fabricated rendering for a UI transport. We do NOT
invent an agent.

Records TWO real AG-UI sessions by driving the REAL ``AGUIProtocolAdapter``
(``wrap_stream`` — the integration that actually reconstructs message + tool-call
content) inside a real ``TraceCollector``, and writes each as a sealed real-trace
fixture under ``samples/data/traces/industry/``:

* ``generate_agui_single`` -> ``retail_agui_shopping.jsonl``: a CopilotKit
  retail shopping-assistant SSE session — a streamed assistant message, a
  multi-fragment ``product_lookup`` tool call (split ``TOOL_CALL_ARGS`` deltas
  that the adapter accumulates + parses), the tool result, and the assistant's
  streamed recommendation. Emits ``agui.message`` (buffered text) +
  ``agui.tool_call`` (name + parsed args) + lifecycle ``protocol.stream.event``s.

* ``generate_agui_multi`` -> ``retail_agui_cart.jsonl``: a fuller cart-management
  session exercising ALL THREE ``agui.*`` families in one flow — a
  ``STATE_SNAPSHOT`` then several ``STATE_DELTA`` JSON-Patch rounds
  (add / replace / remove, with the adapter's chained before/after SHA-256 state
  hashes), a multi-fragment ``add_to_cart`` tool call, and streamed assistant
  messages. Still single-agent / empty-state — "multi" here means multi-family
  protocol session, NOT multi-agent (there is no agent graph).

The SSE event lists are the genuine AG-UI wire shapes a CopilotKit runtime
produces (the same protocol the Family-A ``samples/adapters/protocols/agui_sse.py``
sample and the live ``run_agui`` harness drive); the scenario content (products,
cart) is synthetic, non-sensitive retail demo data. Nothing about the telemetry
is fabricated: every ``agui.message`` / ``agui.tool_call`` / ``agui.state`` /
``protocol.stream.event`` event is what the real adapter emitted from the stream,
carried through the real collector + attestation hash chain. The trace is
captured (via the ``_generate_fixtures`` observer seam) but never uploaded during
generation; the samples upload the captured fixtures themselves at run time.
"""

from __future__ import annotations

import os
import sys
import uuid

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE


# --------------------------------------------------------------------------
# Protocol-driven capture: unlike a provider/framework run there is no adapter
# self-flush — we drive the real ``wrap_stream`` observer inside a collector we
# own, then flush it through the production seal path. Pushing a root span makes
# every protocol event a child of ONE dangling parent, so the collector's
# flush-time root synthesizer emits a single content-free ``trace.root`` marker
# (companion to atlas-app PR #2042) and the trace renders as ONE clean waterfall
# rooted at ``trace`` (no fabricated agent — honest empty-state).
# --------------------------------------------------------------------------
def _capture_agui_session(client: Stratix, stream: list, tags: list) -> dict:
    from layerlens.instrument.adapters.protocols.agui import AGUIProtocolAdapter
    from layerlens.instrument._collector import TraceCollector
    from layerlens.instrument._context import (
        _current_collector,
        _push_span,
        _pop_span,
    )

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None

    collector = TraceCollector(client, _CAPTURE)
    adapter = AGUIProtocolAdapter(capture_config=_CAPTURE)
    root_span_id = uuid.uuid4().hex[:16]
    tok = _current_collector.set(collector)
    snap = _push_span(root_span_id, "agui-session")
    try:
        for _ in adapter.wrap_stream(iter(stream)):
            pass
    finally:
        _pop_span(snap)
        _current_collector.reset(tok)
        collector.flush()
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for agui session")
    payload["tags"] = list(tags)
    return payload


def _summarize(payload: dict) -> str:
    events = payload.get("events", [])
    by_type: dict = {}
    for e in events:
        by_type[e.get("event_type")] = by_type.get(e.get("event_type"), 0) + 1
    idents = [e for e in events if e.get("event_type") == "agent.identity"]
    return "events=%d types=%s agent.identity=%d (empty-state)" % (
        len(events),
        {k: by_type[k] for k in sorted(by_type)},
        len(idents),
    )


# --------------------------------------------------------------------------
# Single: a CopilotKit retail shopping-assistant SSE session
# (streamed message + multi-fragment product_lookup tool call).
# --------------------------------------------------------------------------
def generate_agui_single(client: Stratix) -> dict:
    """Record a retail shopping-assistant AG-UI session (message + tool call)."""
    stream = [
        {"type": "RUN_STARTED", "threadId": "th-shop-1", "runId": "run-1"},
        {"type": "TEXT_MESSAGE_START", "messageId": "m1", "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "Let me find "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "wireless headphones "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "under $200 for you."},
        {"type": "TEXT_MESSAGE_END", "messageId": "m1"},
        {"type": "TOOL_CALL_START", "toolCallId": "tc1", "toolCallName": "product_lookup"},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": '{"category": "audio", '},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": '"query": "wireless headphones", '},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": '"max_price": 200}'},
        {"type": "TOOL_CALL_END", "toolCallId": "tc1"},
        {
            "type": "TOOL_CALL_RESULT",
            "toolCallId": "tc1",
            "content": (
                '[{"sku": "sony-wh-ch720n", "name": "Sony WH-CH720N", "price": 149.0, '
                '"anc": true}, {"sku": "anker-q30", "name": "Anker Soundcore Q30", '
                '"price": 79.0, "anc": true}, {"sku": "jbl-760nc", "name": "JBL Tune '
                '760NC", "price": 129.0, "anc": true}]'
            ),
        },
        {"type": "TEXT_MESSAGE_START", "messageId": "m2", "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m2", "delta": "I found three great options: "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m2", "delta": "the Sony WH-CH720N ($149), "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m2", "delta": "Anker Soundcore Q30 ($79), and "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m2", "delta": "JBL Tune 760NC ($129) — all with active noise cancellation."},
        {"type": "TEXT_MESSAGE_END", "messageId": "m2"},
        {"type": "RUN_FINISHED", "threadId": "th-shop-1", "runId": "run-1"},
    ]
    payload = _capture_agui_session(
        client,
        stream,
        tags=["layerlens-sample", "industry", "retail", "shopping-assistant", "agui", "protocol"],
    )
    print("  agui single (retail shopping-assistant SSE)  " + _summarize(payload))
    print("  ->", _write([payload], "industry", "retail_agui_shopping"), "\n")
    return payload


# --------------------------------------------------------------------------
# Multi (protocol-realistic): cart management exercising all three agui.* families
# — STATE_SNAPSHOT + STATE_DELTA (add/replace/remove) round-trip + a multi-fragment
# tool call + messages. Still single-agent / empty-state (NOT multi-agent).
# --------------------------------------------------------------------------
def generate_agui_multi(client: Stratix) -> dict:
    """Record a cart-management AG-UI session: STATE snapshot + delta round-trip,
    a multi-fragment add_to_cart tool call, and streamed messages."""
    stream = [
        {"type": "RUN_STARTED", "threadId": "th-cart-1", "runId": "run-2"},
        # Initial cart snapshot (items keyed by SKU so JSON-Patch add/replace/remove
        # operate on object paths the StateDeltaHandler supports).
        {
            "type": "STATE_SNAPSHOT",
            "state": {"cart": {"items": {}, "subtotal": 0.0}, "currency": "USD"},
        },
        {"type": "TEXT_MESSAGE_START", "messageId": "m1", "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "Adding the Sony WH-CH720N "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "and Anker Q30 to your cart."},
        {"type": "TEXT_MESSAGE_END", "messageId": "m1"},
        # Multi-fragment tool call (split JSON args the adapter accumulates + parses).
        {"type": "TOOL_CALL_START", "toolCallId": "tc1", "toolCallName": "add_to_cart"},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": '{"items": ['},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": '{"sku": "sony-wh-ch720n", "qty": 1, "price": 149.0}, '},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": '{"sku": "anker-q30", "qty": 1, "price": 79.0}'},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": "]}"},
        {"type": "TOOL_CALL_END", "toolCallId": "tc1"},
        {
            "type": "TOOL_CALL_RESULT",
            "toolCallId": "tc1",
            "content": '{"added": 2, "subtotal": 228.0}',
        },
        # Cart deltas: add each line item + keep subtotal (replace) in step. Each
        # delta's before_hash chains off the prior after_hash (single cached state).
        {
            "type": "STATE_DELTA",
            "delta": [
                {"op": "add", "path": "/cart/items/sony-wh-ch720n", "value": {"qty": 1, "price": 149.0}},
                {"op": "add", "path": "/cart/items/anker-q30", "value": {"qty": 1, "price": 79.0}},
                {"op": "replace", "path": "/cart/subtotal", "value": 228.0},
            ],
        },
        {"type": "TEXT_MESSAGE_START", "messageId": "m2", "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m2", "delta": "Updating the Sony to quantity 2."},
        {"type": "TEXT_MESSAGE_END", "messageId": "m2"},
        # Update quantity (replace on an existing path) + subtotal.
        {
            "type": "STATE_DELTA",
            "delta": [
                {"op": "replace", "path": "/cart/items/sony-wh-ch720n", "value": {"qty": 2, "price": 149.0}},
                {"op": "replace", "path": "/cart/subtotal", "value": 377.0},
            ],
        },
        # Remove the Anker (remove on an existing path) + subtotal.
        {
            "type": "STATE_DELTA",
            "delta": [
                {"op": "remove", "path": "/cart/items/anker-q30"},
                {"op": "replace", "path": "/cart/subtotal", "value": 298.0},
            ],
        },
        {"type": "TEXT_MESSAGE_START", "messageId": "m3", "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m3", "delta": "Your cart is 2x Sony WH-CH720N — "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m3", "delta": "subtotal $298. Ready to check out?"},
        {"type": "TEXT_MESSAGE_END", "messageId": "m3"},
        {"type": "RUN_FINISHED", "threadId": "th-cart-1", "runId": "run-2"},
    ]
    payload = _capture_agui_session(
        client,
        stream,
        tags=["layerlens-sample", "industry", "retail", "shopping-cart", "agui", "protocol"],
    )
    states = [e for e in payload.get("events", []) if e.get("event_type") == "agui.state"]
    print("  agui multi (retail cart STATE round-trip)  " + _summarize(payload)
          + "  agui.state=%d" % len(states))
    print("  ->", _write([payload], "industry", "retail_agui_cart"), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_agui_single(_client)
    generate_agui_multi(_client)
