"""A2UI commerce-surface protocol adapter — BEHAVIORAL unit tests (LAY-3617).

A2UI is the one protocol with no redaction tier: it is built to emit ONLY
ids / counts / hashes, never raw content ("nothing to redact"). These tests
drive the LIVE :class:`A2UIProtocolAdapter` reached through ``connect()`` (the
production monkey-patch path) and through its direct ``record_*`` emit helpers,
and pin that privacy promise:

* ``on_user_action`` / ``record_user_action`` SHA-256-hash the action context
  into ``action_context_hash = "sha256:" + hexdigest`` (``a2ui._sha``) and emit
  ``surface_id`` / ``action_type`` / ``action_context_hash`` — and the raw
  cleartext context string NEVER appears in any emitted payload (mirrors the
  SECRET-leak assertions in ``test_protocol_redaction.py``: a SENTINEL context
  value is embedded, then asserted absent from all payload text while its
  sha256 IS present).
* ``on_surface_created`` / ``record_surface_created`` emit
  ``surface_id`` / ``surface_type`` / ``item_count``.

Every emitted event is fed to the LAY-3583 schema lock (``record_for_schema_lock``)
so the autouse ``_enforce_schema_lock`` fixture validates it — these are real,
schema-valid uploads, not contract stubs.
"""

from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from layerlens.instrument._events import (
    COMMERCE_UI_USER_ACTION,
    COMMERCE_UI_SURFACE_CREATED,
)
from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.protocols.a2ui import A2UIProtocolAdapter

from ...conftest import record_for_schema_lock  # relative: binds the autouse-fixture's module (not a 2nd copy)

# A cleartext action context that must NEVER appear in any emitted payload —
# only its SHA-256 may. Embeds a recognisable, payment-flavoured string so a
# leak is unmistakable.
SECRET_CONTEXT = "SENTINEL-the user tapped buy-now on a private cart {visa:4111-1111}"
SECRET_SURFACE_TYPE = "checkout"

_NO_CONTENT = CaptureConfig(capture_content=False)


def _sha(value: Any) -> str:
    """Reproduce a2ui._sha independently so the test pins the exact format."""
    return "sha256:" + hashlib.sha256(str(value).encode()).hexdigest()


def _run_collected(
    fn: Any,
    collector_config: Optional[CaptureConfig] = None,
) -> List[Dict[str, Any]]:
    """Run *fn* under an ambient collector; return + schema-lock its events.

    Mirrors the redaction suite's helper, but additionally hands every emitted
    event to ``record_for_schema_lock`` so the autouse lock validates the live
    adapter's output after the test body.
    """
    collector = TraceCollector(object(), collector_config or CaptureConfig())
    token = _current_collector.set(collector)
    try:
        fn()
    finally:
        _current_collector.reset(token)
    events = collector.events
    record_for_schema_lock(events)
    return events


def _all_payload_text(events: List[Dict[str, Any]]) -> str:
    return json.dumps([e["payload"] for e in events], default=str)


def _types(events: List[Dict[str, Any]]) -> List[str]:
    return [e["event_type"] for e in events]


def _payloads_of(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    return [e["payload"] for e in events if e["event_type"] == event_type]


# ---------------------------------------------------------------------------
# connect() monkey-patch path — on_user_action
# ---------------------------------------------------------------------------


class TestUserActionViaConnect:
    def _drive(self, adapter_config: Optional[CaptureConfig], collector_config: Optional[CaptureConfig]):
        adapter = A2UIProtocolAdapter(capture_config=adapter_config) if adapter_config else A2UIProtocolAdapter()
        called: Dict[str, Any] = {}

        def _on_user_action(*, surface_id, action_type, context):
            called["surface_id"] = surface_id
            called["action_type"] = action_type
            called["context"] = context
            return {"ok": True}

        target = SimpleNamespace(on_user_action=_on_user_action)
        returned = adapter.connect(target=target)
        # connect() returns the (now-patched) target and rebinds the method.
        assert returned is target

        def go() -> None:
            target.on_user_action(
                surface_id="surf-42",
                action_type="add_to_cart",
                context=SECRET_CONTEXT,
            )

        events = _run_collected(go, collector_config)
        return events, called

    def test_emits_user_action_with_hashed_context(self) -> None:
        events, called = self._drive(None, None)

        # Exactly one commerce.ui.user_action event, and nothing else.
        assert _types(events) == [COMMERCE_UI_USER_ACTION]
        payload = _payloads_of(events, COMMERCE_UI_USER_ACTION)[0]

        # Strong field assertions.
        assert payload["protocol"] == "a2ui"
        assert payload["surface_id"] == "surf-42"
        assert payload["action_type"] == "add_to_cart"
        assert payload["action_context_hash"] == _sha(SECRET_CONTEXT)
        assert payload["action_context_hash"].startswith("sha256:")
        assert len(payload["action_context_hash"]) == len("sha256:") + 64

        # The wrapper still delegates to the original (instrumentation is
        # transparent), with the real cleartext context.
        assert called["context"] == SECRET_CONTEXT
        assert called["surface_id"] == "surf-42"

    def test_cleartext_context_never_leaks_only_its_hash(self) -> None:
        events, _ = self._drive(None, None)
        text = _all_payload_text(events)
        assert SECRET_CONTEXT not in text, "raw action context leaked into a2ui telemetry"
        # Defence: even a fragment (the embedded card number) must be absent.
        assert "4111-1111" not in text
        assert _sha(SECRET_CONTEXT) in text, "hashed context missing — the only retained signal"

    def test_no_content_config_keeps_the_hash(self) -> None:
        # A2UI has no redaction tier: there is nothing to strip because the
        # only sensitive field is pre-hashed. capture_content=False on either
        # side must leave the metadata + hash intact, and never the cleartext.
        events_adapter, _ = self._drive(_NO_CONTENT, None)
        events_collector, _ = self._drive(None, _NO_CONTENT)
        for events in (events_adapter, events_collector):
            text = _all_payload_text(events)
            assert SECRET_CONTEXT not in text
            payload = _payloads_of(events, COMMERCE_UI_USER_ACTION)[0]
            assert payload["action_context_hash"] == _sha(SECRET_CONTEXT)
            assert payload["surface_id"] == "surf-42"
            assert payload["action_type"] == "add_to_cart"


# ---------------------------------------------------------------------------
# connect() monkey-patch path — on_surface_created
# ---------------------------------------------------------------------------


class TestSurfaceCreatedViaConnect:
    def _drive(self):
        adapter = A2UIProtocolAdapter()
        target = SimpleNamespace(on_surface_created=lambda **kw: {"rendered": True})
        adapter.connect(target=target)

        def go() -> None:
            target.on_surface_created(surface_id="surf-7", surface_type=SECRET_SURFACE_TYPE, item_count=3)

        return _run_collected(go)

    def test_emits_surface_created_metadata(self) -> None:
        events = self._drive()
        assert _types(events) == [COMMERCE_UI_SURFACE_CREATED]
        payload = _payloads_of(events, COMMERCE_UI_SURFACE_CREATED)[0]
        assert payload["protocol"] == "a2ui"
        assert payload["surface_id"] == "surf-7"
        assert payload["surface_type"] == SECRET_SURFACE_TYPE
        assert payload["item_count"] == 3
        # surface_created is metadata-only: it carries NO context hash field.
        assert "action_context_hash" not in payload

    def test_surface_type_via_type_alias(self) -> None:
        # The wrapper accepts ``type=`` as an alias for ``surface_type``.
        adapter = A2UIProtocolAdapter()
        target = SimpleNamespace(on_surface_created=lambda **kw: None)
        adapter.connect(target=target)

        def go() -> None:
            target.on_surface_created(surface_id="surf-9", type="product_grid", item_count=12)

        events = _run_collected(go)
        payload = _payloads_of(events, COMMERCE_UI_SURFACE_CREATED)[0]
        assert payload["surface_type"] == "product_grid"
        assert payload["item_count"] == 12


# ---------------------------------------------------------------------------
# Direct emit helpers — record_user_action / record_surface_created
# ---------------------------------------------------------------------------


class TestRecordHelpers:
    def test_record_user_action_hashes_context(self) -> None:
        adapter = A2UIProtocolAdapter()

        def go() -> None:
            adapter.record_user_action(
                surface_id="surf-1",
                action_type="checkout_confirm",
                context=SECRET_CONTEXT,
            )

        events = _run_collected(go)
        assert _types(events) == [COMMERCE_UI_USER_ACTION]
        payload = _payloads_of(events, COMMERCE_UI_USER_ACTION)[0]
        assert payload["surface_id"] == "surf-1"
        assert payload["action_type"] == "checkout_confirm"
        assert payload["action_context_hash"] == _sha(SECRET_CONTEXT)

        text = _all_payload_text(events)
        assert SECRET_CONTEXT not in text, "record_user_action leaked raw context"
        assert _sha(SECRET_CONTEXT) in text

    def test_record_user_action_no_content_keeps_hash(self) -> None:
        adapter = A2UIProtocolAdapter(capture_config=_NO_CONTENT)

        def go() -> None:
            adapter.record_user_action(
                surface_id="surf-1",
                action_type="checkout_confirm",
                context=SECRET_CONTEXT,
            )

        events = _run_collected(go, _NO_CONTENT)
        payload = _payloads_of(events, COMMERCE_UI_USER_ACTION)[0]
        # Nothing to redact: the hash survives a no-content config.
        assert payload["action_context_hash"] == _sha(SECRET_CONTEXT)
        assert SECRET_CONTEXT not in _all_payload_text(events)

    def test_record_surface_created_metadata(self) -> None:
        adapter = A2UIProtocolAdapter()

        def go() -> None:
            adapter.record_surface_created(surface_id="surf-2", surface_type="cart", item_count=5)

        events = _run_collected(go)
        assert _types(events) == [COMMERCE_UI_SURFACE_CREATED]
        payload = _payloads_of(events, COMMERCE_UI_SURFACE_CREATED)[0]
        assert payload["surface_id"] == "surf-2"
        assert payload["surface_type"] == "cart"
        assert payload["item_count"] == 5

    def test_distinct_contexts_hash_distinctly(self) -> None:
        # The hash is a real fingerprint, not a constant — different contexts
        # must produce different hashes (so analytics can correlate identical
        # interactions without ever seeing the cleartext).
        adapter = A2UIProtocolAdapter()

        def go() -> None:
            adapter.record_user_action(surface_id="s", action_type="a", context="context-A")
            adapter.record_user_action(surface_id="s", action_type="a", context="context-B")
            adapter.record_user_action(surface_id="s", action_type="a", context="context-A")

        events = _run_collected(go)
        hashes = [p["action_context_hash"] for p in _payloads_of(events, COMMERCE_UI_USER_ACTION)]
        assert len(hashes) == 3
        assert hashes[0] == hashes[2] == _sha("context-A")
        assert hashes[1] == _sha("context-B")
        assert hashes[0] != hashes[1]
