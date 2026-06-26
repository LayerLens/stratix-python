"""Behavioral unit tests for the UCP (Universal Commerce Protocol) adapter (LAY-3617).

The old protocol unit tier for commerce (``test_certification.py``) was
contract-only. This suite drives the REAL :class:`UCPProtocolAdapter` reached by
``connect()`` and asserts the events it actually emits across the whole commerce
flow — discover → browse → checkout (start + complete) → refund — including the
exact ``session_duration_ms`` the adapter computes per session from a tracked
start time.

Pattern mirrors ``test_protocol_redaction.py``: instantiate the real adapter,
build a ``SimpleNamespace`` target exposing the patched methods, ``connect()``,
invoke the patched methods, and inspect the events captured by an ambient
``TraceCollector``. Captured events are also recorded for the autouse schema
lock so it validates them after each test.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from layerlens.instrument._events import (
    COMMERCE_REFUND_ISSUED,
    COMMERCE_CHECKOUT_COMPLETED,
    COMMERCE_SUPPLIER_DISCOVERED,
)
from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.protocols.ucp import UCPProtocolAdapter

from ...conftest import find_event, find_events, record_for_schema_lock

# String-literal event types the adapter emits without an _events.py constant.
CATALOG_BROWSED = "commerce.catalog.browsed"
CHECKOUT_STARTED = "commerce.checkout.started"


def _run_collected(fn: Any) -> List[Dict[str, Any]]:
    """Run *fn* under an ambient collector and return its raw events.

    Captured events are recorded for the autouse schema lock so the lock has
    real teeth on this suite (the ``_run_collected`` path does not upload, so it
    would otherwise bypass the upload-fed schema buffer).
    """
    collector = TraceCollector(client=None, config=CaptureConfig())
    token = _current_collector.set(collector)
    try:
        fn()
    finally:
        _current_collector.reset(token)
    record_for_schema_lock(collector.events)
    return collector.events


def _make_adapter_and_target(
    *,
    discover_result: Any = None,
    browse_result: Any = None,
    start_result: Any = None,
    complete_result: Any = None,
    refund_result: Any = None,
    capture_config: Optional[CaptureConfig] = None,
):
    """Build a connected UCPProtocolAdapter over a SimpleNamespace target.

    Each lambda returns the configured result and is the *original* the adapter
    wraps, so the emitted payloads reflect both the inbound kwargs and the
    method's return value.
    """
    adapter = UCPProtocolAdapter(capture_config=capture_config)
    target = SimpleNamespace(
        discover_suppliers=lambda **kw: discover_result if discover_result is not None else [],
        browse_catalog=lambda **kw: browse_result if browse_result is not None else [],
        start_checkout=lambda **kw: start_result if start_result is not None else {"ok": True},
        complete_checkout=lambda **kw: complete_result if complete_result is not None else {"ok": True},
        issue_refund=lambda **kw: refund_result if refund_result is not None else {"ok": True},
    )
    adapter.connect(target=target)
    return adapter, target


# ---------------------------------------------------------------------------
# connect() actually patches the commerce surface
# ---------------------------------------------------------------------------


class TestConnect:
    def test_connect_patches_all_five_methods(self) -> None:
        adapter, target = _make_adapter_and_target()
        # Each method must be re-bound to a wrapper (not the bare lambda).
        for method in (
            "discover_suppliers",
            "browse_catalog",
            "start_checkout",
            "complete_checkout",
            "issue_refund",
        ):
            assert method in adapter._originals, f"{method} not wrapped by connect()"
            assert getattr(target, method) is not adapter._originals[method]

    def test_no_collector_is_a_noop(self) -> None:
        # With no ambient collector, invoking patched methods must not raise and
        # must still return the original method's result.
        _adapter, target = _make_adapter_and_target(browse_result=["x", "y"])
        assert target.browse_catalog(supplier_id="sup1", query="q") == ["x", "y"]


# ---------------------------------------------------------------------------
# discover_suppliers -> commerce.supplier_discovered (one per supplier)
# ---------------------------------------------------------------------------


class TestDiscoverSuppliers:
    def test_emits_one_event_per_supplier_with_id_and_name(self) -> None:
        suppliers = [
            {"id": "sup-1", "name": "Acme"},
            {"id": "sup-2", "name": "Globex"},
        ]
        _adapter, target = _make_adapter_and_target(discover_result=suppliers)

        events = _run_collected(lambda: target.discover_suppliers(region="us"))

        discovered = find_events(events, COMMERCE_SUPPLIER_DISCOVERED)
        assert len(discovered) == 2
        by_id = {e["payload"]["supplier_id"]: e["payload"] for e in discovered}
        assert by_id["sup-1"]["name"] == "Acme"
        assert by_id["sup-2"]["name"] == "Globex"
        # protocol marker injected by emit()
        assert all(e["payload"]["protocol"] == "ucp" for e in discovered)

    def test_supports_object_suppliers(self) -> None:
        suppliers = [SimpleNamespace(id="obj-1", name="ObjSupplier")]
        _adapter, target = _make_adapter_and_target(discover_result=suppliers)

        events = _run_collected(lambda: target.discover_suppliers())

        ev = find_event(events, COMMERCE_SUPPLIER_DISCOVERED)
        assert ev["payload"]["supplier_id"] == "obj-1"
        assert ev["payload"]["name"] == "ObjSupplier"

    def test_skips_suppliers_without_id(self) -> None:
        suppliers = [{"name": "NoId"}, {"id": "has-id", "name": "HasId"}]
        _adapter, target = _make_adapter_and_target(discover_result=suppliers)

        events = _run_collected(lambda: target.discover_suppliers())

        discovered = find_events(events, COMMERCE_SUPPLIER_DISCOVERED)
        assert len(discovered) == 1
        assert discovered[0]["payload"]["supplier_id"] == "has-id"

    def test_remembers_known_suppliers_first_seen(self) -> None:
        suppliers = [{"id": "sup-1", "name": "Acme"}]
        adapter, target = _make_adapter_and_target(discover_result=suppliers)

        _run_collected(lambda: target.discover_suppliers())
        assert "sup-1" in adapter._known_suppliers
        first_seen = adapter._known_suppliers["sup-1"]["discovered_at"]

        # Re-discovering the same supplier must not overwrite the first-seen time,
        # but must still emit (each discovery is observable).
        events = _run_collected(lambda: target.discover_suppliers())
        assert adapter._known_suppliers["sup-1"]["discovered_at"] == first_seen
        assert len(find_events(events, COMMERCE_SUPPLIER_DISCOVERED)) == 1


# ---------------------------------------------------------------------------
# browse_catalog -> "commerce.catalog.browsed" (string literal)
# ---------------------------------------------------------------------------


class TestBrowseCatalog:
    def test_emits_browsed_with_query_and_item_count(self) -> None:
        items = [{"sku": "a"}, {"sku": "b"}, {"sku": "c"}]
        _adapter, target = _make_adapter_and_target(browse_result=items)

        events = _run_collected(lambda: target.browse_catalog(supplier_id="sup-1", query="red shoes"))

        ev = find_event(events, CATALOG_BROWSED)
        assert ev["payload"]["supplier_id"] == "sup-1"
        assert ev["payload"]["query"] == "red shoes"
        assert ev["payload"]["item_count"] == 3

    def test_item_count_from_object_with_items_attr(self) -> None:
        result = SimpleNamespace(items=[1, 2])
        _adapter, target = _make_adapter_and_target(browse_result=result)

        events = _run_collected(lambda: target.browse_catalog(supplier_id="sup-1", query="q"))

        ev = find_event(events, CATALOG_BROWSED)
        assert ev["payload"]["item_count"] == 2

    def test_empty_catalog_reports_zero(self) -> None:
        _adapter, target = _make_adapter_and_target(browse_result=[])

        events = _run_collected(lambda: target.browse_catalog(supplier_id="sup-1", query="none"))

        ev = find_event(events, CATALOG_BROWSED)
        assert ev["payload"]["item_count"] == 0


# ---------------------------------------------------------------------------
# start_checkout -> "commerce.checkout.started" (string literal)
# ---------------------------------------------------------------------------


class TestStartCheckout:
    def test_emits_started_with_provided_session_id(self) -> None:
        _adapter, target = _make_adapter_and_target()

        events = _run_collected(lambda: target.start_checkout(supplier_id="sup-1", session_id="sess-xyz"))

        ev = find_event(events, CHECKOUT_STARTED)
        assert ev["payload"]["session_id"] == "sess-xyz"
        assert ev["payload"]["supplier_id"] == "sup-1"

    def test_generates_session_id_when_absent_and_tracks_start(self) -> None:
        adapter, target = _make_adapter_and_target()

        events = _run_collected(lambda: target.start_checkout(supplier_id="sup-1"))

        ev = find_event(events, CHECKOUT_STARTED)
        generated = ev["payload"]["session_id"]
        assert isinstance(generated, str) and len(generated) == 16
        # The adapter must track the start time for this generated session.
        assert generated in adapter._sessions


# ---------------------------------------------------------------------------
# complete_checkout -> commerce.checkout_completed (+ session_duration_ms)
# ---------------------------------------------------------------------------


class TestCompleteCheckout:
    def test_emits_completed_with_amount_and_supplier(self) -> None:
        _adapter, target = _make_adapter_and_target()

        events = _run_collected(
            lambda: target.complete_checkout(session_id="sess-1", supplier_id="sup-1", amount=42.50)
        )

        ev = find_event(events, COMMERCE_CHECKOUT_COMPLETED)
        assert ev["payload"]["session_id"] == "sess-1"
        assert ev["payload"]["supplier_id"] == "sup-1"
        assert ev["payload"]["amount"] == 42.50
        assert isinstance(ev["payload"]["session_duration_ms"], (int, float))

    def test_session_duration_ms_computed_across_start_and_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The adapter tracks start time per session; duration = (now - start) * 1000.

        Drive a deterministic clock: start at t=100.0s, complete at t=100.250s,
        so the emitted session_duration_ms must equal exactly 250.0.
        """
        import layerlens.instrument.adapters.protocols.ucp as ucp_mod

        clock = {"now": 100.0}
        monkeypatch.setattr(ucp_mod.time, "time", lambda: clock["now"])

        _adapter, target = _make_adapter_and_target()

        def flow() -> None:
            target.start_checkout(supplier_id="sup-1", session_id="sess-dur")
            clock["now"] = 100.250  # 250 ms elapse between start and complete
            target.complete_checkout(session_id="sess-dur", supplier_id="sup-1", amount=10.0)

        events = _run_collected(flow)

        completed = find_event(events, COMMERCE_CHECKOUT_COMPLETED)
        assert completed["payload"]["session_duration_ms"] == pytest.approx(250.0)

    def test_completion_pops_session_so_duration_is_not_double_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import layerlens.instrument.adapters.protocols.ucp as ucp_mod

        clock = {"now": 0.0}
        monkeypatch.setattr(ucp_mod.time, "time", lambda: clock["now"])

        adapter, target = _make_adapter_and_target()

        def flow() -> None:
            target.start_checkout(supplier_id="sup-1", session_id="sess-pop")
            clock["now"] = 1.0
            target.complete_checkout(session_id="sess-pop", supplier_id="sup-1")

        _run_collected(flow)
        # The session start time must have been popped on completion.
        assert "sess-pop" not in adapter._sessions

    def test_completion_for_unknown_session_uses_now_as_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A complete with no matching start falls back to now → ~0 ms duration."""
        import layerlens.instrument.adapters.protocols.ucp as ucp_mod

        monkeypatch.setattr(ucp_mod.time, "time", lambda: 500.0)

        _adapter, target = _make_adapter_and_target()

        events = _run_collected(lambda: target.complete_checkout(session_id="never-started", amount=1.0))

        ev = find_event(events, COMMERCE_CHECKOUT_COMPLETED)
        assert ev["payload"]["session_duration_ms"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# issue_refund -> commerce.refund_issued
# ---------------------------------------------------------------------------


class TestIssueRefund:
    def test_emits_refund_with_amount_and_reason(self) -> None:
        _adapter, target = _make_adapter_and_target()

        events = _run_collected(lambda: target.issue_refund(session_id="sess-1", amount=9.99, reason="damaged item"))

        ev = find_event(events, COMMERCE_REFUND_ISSUED)
        assert ev["payload"]["session_id"] == "sess-1"
        assert ev["payload"]["amount"] == 9.99
        assert ev["payload"]["reason"] == "damaged item"


# ---------------------------------------------------------------------------
# Full flow: every commerce event type emitted, in order
# ---------------------------------------------------------------------------


class TestFullCommerceFlow:
    def test_all_event_types_emitted_across_one_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import layerlens.instrument.adapters.protocols.ucp as ucp_mod

        clock = {"now": 1000.0}
        monkeypatch.setattr(ucp_mod.time, "time", lambda: clock["now"])

        _adapter, target = _make_adapter_and_target(
            discover_result=[{"id": "sup-1", "name": "Acme"}],
            browse_result=[{"sku": "a"}, {"sku": "b"}],
        )

        def flow() -> None:
            target.discover_suppliers(region="us")
            target.browse_catalog(supplier_id="sup-1", query="red shoes")
            target.start_checkout(supplier_id="sup-1", session_id="sess-9")
            clock["now"] = 1000.5  # 500 ms
            target.complete_checkout(session_id="sess-9", supplier_id="sup-1", amount=59.99)
            target.issue_refund(session_id="sess-9", amount=59.99, reason="returned")

        events = _run_collected(flow)

        emitted_types = [e["event_type"] for e in events]
        assert emitted_types == [
            COMMERCE_SUPPLIER_DISCOVERED,
            CATALOG_BROWSED,
            CHECKOUT_STARTED,
            COMMERCE_CHECKOUT_COMPLETED,
            COMMERCE_REFUND_ISSUED,
        ]

        completed = find_event(events, COMMERCE_CHECKOUT_COMPLETED)
        assert completed["payload"]["session_duration_ms"] == pytest.approx(500.0)
        assert completed["payload"]["amount"] == 59.99
