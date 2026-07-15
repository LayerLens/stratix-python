"""Offline attestation + reconstruction + state-hash floor for the AG-UI adapter.

Closes the W2 census OFFLINE cells that ``test_agui_middleware.py`` /
``test_protocol_redaction.py`` leave open, driving the *real*
``AGUIProtocolAdapter`` (the ``wrap_stream`` / ``wrap_async_stream`` observer
path — the integration that actually reconstructs message + tool-call content)
so a regression fails in plain CI with no credentials and no network:

* Attestation (``attest`` cell, gap -> solid) — a realistic AG-UI SSE session
                (RUN_STARTED -> text -> tool-call -> STATE_SNAPSHOT/DELTA ->
                RUN_FINISHED) flushed through a REAL ``TraceCollector`` seals a
                trace whose attestation chain reconstructs and ``verify_chain``
                returns valid; a tamper control (broken interior link) proves the
                check is not vacuous, and every wire event carries its own
                ``sha256:`` per-event hash.
* Multi-fragment reconstruction (``streaming`` cell, partial -> solid) — the
                adapter concatenates split ``TEXT_MESSAGE_CONTENT`` deltas into a
                SINGLE ``agui.message`` and accumulates split ``TOOL_CALL_ARGS``
                deltas into a SINGLE ``agui.tool_call`` whose JSON args parse to
                the reconstructed object (bite: last-fragment-only / dropped
                fragment / per-fragment emission all fail).
* ``wrap_async_stream`` (``conc_async`` / ``unit`` cells) — the async iterator
                path reconstructs identically to the sync path AND passes the
                underlying events through untouched.
* State JSON-Patch + before/after SHA-256 (``unit`` cell) — ``StateDeltaHandler``
                applies add/remove/replace and returns deterministic before/after
                hashes that match an independently recomputed digest, and the
                adapter chains the cached state across STATE_SNAPSHOT -> STATE_DELTA
                events (the DELTA's ``before_hash`` == the SNAPSHOT's ``after_hash``).

Redaction is already solid (``test_protocol_redaction.py::TestAGUIRedaction`` +
``TestAGUIFallbackRedaction`` + ``test_no_content_sweep.py``) — referenced, not
duplicated here.

The only mock is the trace-upload client (the network boundary); every AG-UI
event, the adapter's stream observer, the state handler, and the attestation
hash chain are real.

Two former source-bug held findings are now FIXED and asserted here (they were
previously RED on the shipped code):
  * ``error`` — ``RUN_ERROR`` now surfaces as ``agent.error``
    ({error_type, status:"error", error}) instead of a generic
    ``protocol.stream.event`` (``TestRunErrorSurfacesAsAgentError``).
  * state ``replace`` on a MISSING path is now a no-create honest failure
    (``StateDeltaHandler._patch_replace``), per RFC-6902
    (``TestReplaceOnMissingPathDoesNotCreate``).
"""

from __future__ import annotations

import json
import asyncio
import hashlib
from typing import Any, Dict, List

from layerlens.attestation._verify import verify_chain
from layerlens.instrument._context import _current_collector
from layerlens.attestation._envelope import HashScope, AttestationEnvelope
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.protocols.agui.adapter import AGUIProtocolAdapter
from layerlens.instrument.adapters.protocols.agui.state_handler import StateDeltaHandler

_FULL = CaptureConfig(capture_content=True)


# ---------------------------------------------------------------------------
# Realistic Retail shopping-assistant AG-UI SSE session (the census industry
# fit): a streamed assistant message, a multi-fragment product-lookup tool call,
# and a cart STATE_SNAPSHOT + STATE_DELTA — exercising all three agui.* families
# plus lifecycle passthrough in one flow.
# ---------------------------------------------------------------------------
def _retail_session() -> List[Dict[str, Any]]:
    return [
        {"type": "RUN_STARTED"},
        {"type": "TEXT_MESSAGE_START", "messageId": "m1"},
        {"type": "TEXT_MESSAGE_CONTENT", "delta": "Here are three "},
        {"type": "TEXT_MESSAGE_CONTENT", "delta": "wireless "},
        {"type": "TEXT_MESSAGE_CONTENT", "delta": "headphones."},
        {"type": "TEXT_MESSAGE_END"},
        {"type": "TOOL_CALL_START", "toolCallId": "tc1", "toolCallName": "product_lookup"},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": '{"category": "audio", '},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc1", "delta": '"max_price": 200}'},
        {"type": "TOOL_CALL_END", "toolCallId": "tc1"},
        {"type": "STATE_SNAPSHOT", "state": {"cart": {"items": 0}}},
        {"type": "STATE_DELTA", "delta": [{"op": "add", "path": "/cart/items", "value": 2}]},
        {"type": "RUN_FINISHED"},
    ]


def _collect_sync(config: CaptureConfig, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drive ``wrap_stream`` inside an ambient collector; return raw events."""
    collector = TraceCollector(object(), config)
    adapter = AGUIProtocolAdapter(capture_config=config)
    token = _current_collector.set(collector)
    try:
        for _ in adapter.wrap_stream(iter(events)):
            pass
    finally:
        _current_collector.reset(token)
    return collector.events


def _collect_async(config: CaptureConfig, events: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Any]]:
    """Drive ``wrap_async_stream`` inside an ambient collector; return (events, passthrough)."""
    collector = TraceCollector(object(), config)
    adapter = AGUIProtocolAdapter(capture_config=config)

    async def _src() -> Any:
        for e in events:
            yield e

    async def _run() -> List[Any]:
        token = _current_collector.set(collector)
        seen: List[Any] = []
        try:
            async for ev in adapter.wrap_async_stream(_src()):
                seen.append(ev)
        finally:
            _current_collector.reset(token)
        return seen

    passthrough = asyncio.run(_run())
    return collector.events, passthrough


def _by_type(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    return [e["payload"] for e in events if e["event_type"] == event_type]


def _expected_state_hash(state: Dict[str, Any]) -> str:
    """Reproduce StateDeltaHandler._hash_state independently, so a change to the
    hashing algorithm (a silent attestation-of-state break) fails the floor."""
    return "sha256:" + hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real AG-UI session flush
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_agui_session(self, mock_client, capture_trace) -> None:
        # Build a REAL collector bound to the (mocked-network) client, drive the
        # real wrap_stream observer inside it, and flush through the production
        # seal+upload path — capture_trace reads back the uploaded payload.
        collector = TraceCollector(mock_client, CaptureConfig())
        token = _current_collector.set(collector)
        adapter = AGUIProtocolAdapter()
        try:
            for _ in adapter.wrap_stream(iter(_retail_session())):
                pass
        finally:
            _current_collector.reset(token)
        collector.flush()

        events = capture_trace["events"]
        assert events, "the AG-UI session must flush a non-empty trace"

        # Every wire event carries its own per-event attestation hash (bite: the
        # attach-hash wiring in _append_locked regressing drops these).
        assert all(str(e.get("hash", "")).startswith("sha256:") for e in events), (
            "an AG-UI event shipped without its per-event attestation hash"
        )
        assert events[0]["previous_hash"] is None, "the first event must open the chain (previous_hash=None)"

        chain = (capture_trace["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the AG-UI trace"
        assert len(envelopes) == len(events), (
            f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
        )
        assert (capture_trace["attestation"] or {}).get("root_hash") is not None

        result_ok = verify_chain(envelopes)
        assert result_ok.valid, f"attestation chain invalid: {result_ok.error}"

        # Vacuity control: verify_chain must REJECT a broken interior link, proving
        # the pass above is not trivially true.
        assert len(envelopes) >= 2
        tampered = list(envelopes)
        tampered[1] = AttestationEnvelope(
            hash=tampered[1].hash,
            scope=tampered[1].scope,
            previous_hash="sha256:deadbeef-not-the-prior-hash",
        )
        broken = verify_chain(tampered)
        assert not broken.valid and broken.break_index == 1, "verify_chain failed to detect a broken link"


# ---------------------------------------------------------------------------
# Multi-fragment reconstruction (wrap_stream) — the content-completeness contract
# ---------------------------------------------------------------------------
class TestMultiFragmentReconstruction:
    def test_text_message_content_fragments_concatenate_into_one_message(self, mock_client) -> None:
        events = _collect_sync(_FULL, _retail_session())
        messages = _by_type(events, "agui.message")
        # Exactly one buffered message — NOT one per TEXT_MESSAGE_CONTENT fragment.
        assert len(messages) == 1, f"expected 1 reconstructed agui.message, got {len(messages)}"
        # The three deltas are concatenated IN ORDER (bite: last-fragment-only,
        # dropped fragment, or reordered concat all fail this exact equality).
        assert messages[0]["text"] == "Here are three wireless headphones."

    def test_tool_call_args_fragments_accumulate_and_parse(self, mock_client) -> None:
        events = _collect_sync(_FULL, _retail_session())
        calls = _by_type(events, "agui.tool_call")
        assert len(calls) == 1, f"expected 1 reconstructed agui.tool_call, got {len(calls)}"
        call = calls[0]
        assert call["tool_name"] == "product_lookup"
        # The two split JSON-arg fragments are accumulated then parsed to the exact
        # object (bite: a dropped fragment leaves an unparseable partial string).
        assert call["arguments"] == {"category": "audio", "max_price": 200}
        assert isinstance(call["arguments"], dict), "reconstructed tool args must parse to an object"

    def test_split_message_is_not_emitted_per_fragment(self, mock_client) -> None:
        # Guard against a regression to per-fragment emission (the middleware
        # divergence): only the buffered families emit, no raw text fragment leaks
        # as its own agui.message.
        events = _collect_sync(_FULL, _retail_session())
        assert len(_by_type(events, "agui.message")) == 1
        # Sanity that fragments were genuinely split (3 content deltas) — otherwise
        # the concat assertion above would be vacuous.
        assert sum(1 for e in _retail_session() if e["type"] == "TEXT_MESSAGE_CONTENT") == 3


# ---------------------------------------------------------------------------
# Async stream path — reconstructs identically + passes events through untouched
# ---------------------------------------------------------------------------
class TestWrapAsyncStream:
    def test_async_stream_reconstructs_identically_to_sync(self, mock_client) -> None:
        events, passthrough = _collect_async(_FULL, _retail_session())

        messages = _by_type(events, "agui.message")
        calls = _by_type(events, "agui.tool_call")
        assert len(messages) == 1 and messages[0]["text"] == "Here are three wireless headphones."
        assert len(calls) == 1 and calls[0]["arguments"] == {"category": "audio", "max_price": 200}

        # The wrapper is a pass-through: every source event is yielded unchanged
        # (bite: an async observer that swallows or mutates the stream fails here).
        assert passthrough == _retail_session()

    def test_async_stream_flushes_unterminated_message_buffer(self, mock_client) -> None:
        # A stream that ends mid-message (no TEXT_MESSAGE_END) must still flush the
        # buffered text on stream-close — content is not lost.
        events, _ = _collect_async(
            _FULL,
            [
                {"type": "TEXT_MESSAGE_START"},
                {"type": "TEXT_MESSAGE_CONTENT", "delta": "partial "},
                {"type": "TEXT_MESSAGE_CONTENT", "delta": "answer"},
            ],
        )
        messages = _by_type(events, "agui.message")
        assert len(messages) == 1, "unterminated buffer was not flushed on async stream close"
        assert messages[0]["text"] == "partial answer"
        assert messages[0].get("reason") == "stream_closed"


# ---------------------------------------------------------------------------
# State JSON-Patch application + before/after SHA-256 correctness
# ---------------------------------------------------------------------------
class TestStateDeltaHash:
    def test_snapshot_then_delta_hashes_are_correct_and_chained(self, mock_client) -> None:
        handler = StateDeltaHandler()

        # Snapshot from empty: before == hash({}), after == hash(snapshot).
        snap = {"cart": {"items": 0}, "user": "guest"}
        s_before, s_after = handler.apply_snapshot(snap)
        assert s_before == _expected_state_hash({}), "snapshot before_hash must be the empty-state digest"
        assert s_after == _expected_state_hash(snap), "snapshot after_hash must be the digest of the applied snapshot"
        assert handler.current_state == snap

        # Delta chains off the snapshot: before == snapshot.after, after reflects op.
        d_before, d_after = handler.apply_delta([{"op": "add", "path": "/cart/items", "value": 2}])
        assert d_before == s_after, "delta before_hash must equal the prior snapshot after_hash (cached-state chain)"
        expected_after_state = {"cart": {"items": 2}, "user": "guest"}
        assert handler.current_state == expected_after_state
        assert d_after == _expected_state_hash(expected_after_state)
        assert d_after != d_before, "a mutating delta must change the state hash"

    def test_add_remove_replace_on_existing_paths(self, mock_client) -> None:
        handler = StateDeltaHandler()
        handler.apply_snapshot({"cart": {"items": 1}, "coupon": "SAVE10"})
        handler.apply_delta(
            [
                {"op": "add", "path": "/wishlist", "value": ["sku-9"]},  # add new key
                {"op": "replace", "path": "/cart/items", "value": 5},  # replace existing
                {"op": "remove", "path": "/coupon"},  # remove existing
            ]
        )
        assert handler.current_state == {"cart": {"items": 5}, "wishlist": ["sku-9"]}, (
            "add/replace/remove on existing paths did not apply as expected"
        )

    def test_adapter_emits_state_before_after_hashes_across_events(self, mock_client) -> None:
        # Through the real observer: a SNAPSHOT then a DELTA emit agui.state events
        # whose hashes chain (the adapter's single _state_handler keeps cached
        # state across events). Bite: a per-event fresh handler would reset the
        # DELTA before_hash back to the empty-state digest.
        events = _collect_sync(
            _FULL,
            [
                {"type": "STATE_SNAPSHOT", "state": {"cart": {"items": 0}}},
                {"type": "STATE_DELTA", "delta": [{"op": "add", "path": "/cart/items", "value": 3}]},
            ],
        )
        states = _by_type(events, "agui.state")
        assert len(states) == 2, f"expected 2 agui.state events, got {len(states)}"
        snap, delta = states[0], states[1]
        assert snap["after_hash"] == _expected_state_hash({"cart": {"items": 0}})
        assert delta["before_hash"] == snap["after_hash"], "state hash chain broke across SNAPSHOT -> DELTA"
        assert delta["after_hash"] == _expected_state_hash({"cart": {"items": 3}})


# ---------------------------------------------------------------------------
# RUN_ERROR surfaces as agent.error (not ordinary lifecycle stream telemetry)
# ---------------------------------------------------------------------------
class TestRunErrorSurfacesAsAgentError:
    def test_run_error_emits_agent_error_with_error_fields(self, mock_client) -> None:
        # A streamed RUN_ERROR is a run FAILURE. It must surface as agent.error
        # {error_type, status:"error", error} — matching how a2a/mcp surface
        # failures — so the trace's derived status is error, not completed.
        events = _collect_sync(
            _FULL,
            [
                {"type": "RUN_STARTED"},
                {"type": "RUN_ERROR", "message": "model provider timed out", "code": "PROVIDER_TIMEOUT"},
            ],
        )
        errors = _by_type(events, "agent.error")
        assert len(errors) == 1, f"expected exactly 1 agent.error for RUN_ERROR, got {len(errors)}"
        err = errors[0]
        assert err["status"] == "error", "agent.error must carry status='error'"
        assert err["error_type"] == "PROVIDER_TIMEOUT", "error_type must reflect the AG-UI error code"
        assert err["error"] == "model provider timed out", "error must reflect the AG-UI error message"

        # Bite: RUN_ERROR must NOT leak through as a generic protocol.stream.event
        # (the old lifecycle mapping) — a run failure carried as ordinary stream
        # telemetry is read by no downstream engine and mislabels the trace.
        leaked = [
            p for p in _by_type(events, "protocol.stream.event") if p.get("agui_event") == "RUN_ERROR"
        ]
        assert not leaked, "RUN_ERROR still emitted as protocol.stream.event lifecycle telemetry"

    def test_run_error_without_code_uses_honest_default_type(self, mock_client) -> None:
        # No AG-UI error code on the wire -> an honest generic error_type, never a
        # fabricated one; the message still flows through as error.
        events = _collect_sync(
            _FULL,
            [{"type": "RUN_ERROR", "message": "boom"}],
        )
        errors = _by_type(events, "agent.error")
        assert len(errors) == 1
        assert errors[0]["error_type"] == "agui_run_error"
        assert errors[0]["error"] == "boom"
        assert errors[0]["status"] == "error"


# ---------------------------------------------------------------------------
# RFC-6902 replace requires the target to already exist (must NOT create)
# ---------------------------------------------------------------------------
class TestReplaceOnMissingPathDoesNotCreate:
    def test_replace_on_missing_top_level_path_does_not_create(self, mock_client) -> None:
        handler = StateDeltaHandler()
        handler.apply_snapshot({"cart": {"items": 1}})
        before = handler.current_state

        # RFC-6902 'replace' requires the target location to already exist. A
        # replace on an absent path is an error, NOT an implicit add — it must
        # leave the state untouched, not silently materialize the key.
        handler.apply_delta([{"op": "replace", "path": "/wishlist", "value": ["sku-9"]}])
        assert "wishlist" not in handler.current_state, (
            "replace on a nonexistent top-level path silently created it (RFC-6902 violation)"
        )
        assert handler.current_state == before, "a failed replace must not perturb existing state"

    def test_replace_on_missing_nested_path_does_not_create(self, mock_client) -> None:
        handler = StateDeltaHandler()
        handler.apply_snapshot({"cart": {"items": 1}})
        before = handler.current_state

        handler.apply_delta([{"op": "replace", "path": "/cart/color", "value": "red"}])
        assert "color" not in handler.current_state["cart"], (
            "replace on a nonexistent nested path silently created it (RFC-6902 violation)"
        )
        assert handler.current_state == before

    def test_replace_on_existing_path_still_replaces(self, mock_client) -> None:
        # Guard: the no-create fix must NOT break a legitimate replace of an
        # existing target.
        handler = StateDeltaHandler()
        handler.apply_snapshot({"cart": {"items": 1}})
        handler.apply_delta([{"op": "replace", "path": "/cart/items", "value": 9}])
        assert handler.current_state == {"cart": {"items": 9}}
