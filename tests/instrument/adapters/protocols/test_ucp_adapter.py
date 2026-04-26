"""Unit tests for the UCP (Universal Commerce Protocol) adapter.

UCP emits ``commerce.supplier.*`` / ``commerce.catalog.*`` /
``commerce.checkout.*`` / ``commerce.order.*`` events. As with the
other commerce adapters, these event types are not in the default
:class:`CaptureConfig` layer map, so we replace ``emit_event`` with a
recorder to assert emission.
"""

from __future__ import annotations

from typing import Any, Dict, List, Callable

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters.protocols.ucp import UCPAdapter


def _make_recorder() -> tuple[List[Dict[str, Any]], Callable[..., None]]:
    events: List[Dict[str, Any]] = []

    def _emit(payload: Any, privacy_level: Any = None) -> None:
        events.append(
            {
                "event_type": getattr(payload, "event_type", None),
                "payload": payload,
            }
        )

    return events, _emit


def _make_adapter() -> tuple[UCPAdapter, List[Dict[str, Any]]]:
    events, emit = _make_recorder()
    adapter = UCPAdapter()
    adapter.emit_event = emit  # type: ignore[method-assign]
    adapter.connect()
    return adapter, events


def test_adapter_class_constants() -> None:
    assert UCPAdapter.FRAMEWORK == "ucp"
    assert UCPAdapter.PROTOCOL == "ucp"
    assert UCPAdapter.PROTOCOL_VERSION == "1.0.0"


def test_lifecycle_transitions() -> None:
    adapter = UCPAdapter()
    assert adapter.status == AdapterStatus.DISCONNECTED
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY
    adapter.disconnect()
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_get_adapter_info_shape() -> None:
    adapter, _ = _make_adapter()
    info = adapter.get_adapter_info()
    assert isinstance(info, AdapterInfo)
    assert info.framework == "ucp"
    assert info.name == "UCPAdapter"


def test_probe_health_shape() -> None:
    adapter, _ = _make_adapter()
    health = adapter.probe_health()
    assert health["reachable"] is True
    assert health["protocol_version"] == "1.0.0"


def test_supplier_discovered_emits_event_and_records_state() -> None:
    adapter, events = _make_adapter()
    adapter.on_supplier_discovered(
        supplier_id="sup_1",
        name="Acme",
        profile_url="https://acme.example.com/.well-known/ucp.json",
        org_id="org_1",
        capabilities=["search", "checkout"],
        discovery_method="well_known",
    )
    assert "sup_1" in adapter._suppliers
    assert adapter._suppliers["sup_1"]["discovery_method"] == "well_known"

    types = [e["event_type"] for e in events]
    assert "commerce.supplier.discovered" in types


def test_catalog_browsed_emits_event() -> None:
    adapter, events = _make_adapter()
    adapter.on_catalog_browsed(
        supplier_id="sup_1",
        org_id="org_1",
        items_viewed=10,
        items_selected=2,
    )
    types = [e["event_type"] for e in events]
    assert "commerce.catalog.browsed" in types
    payload = next(e for e in events if e["event_type"] == "commerce.catalog.browsed")["payload"]
    assert payload.items_viewed == 10
    assert payload.items_selected == 2


def test_checkout_lifecycle_tracks_session_duration() -> None:
    adapter, events = _make_adapter()
    adapter.on_checkout_created(
        checkout_session_id="cs_1",
        supplier_id="sup_1",
        line_items=[{"item_id": "sku_1", "quantity": 2, "unit_price": 49.99}],
        total_amount=99.98,
        org_id="org_1",
    )
    assert "cs_1" in adapter._checkout_sessions
    assert "cs_1" in adapter._session_start_times

    adapter.on_checkout_completed("cs_1", org_id="org_1", order_id="ord_1")
    # Session state cleared on completion
    assert "cs_1" not in adapter._checkout_sessions
    assert "cs_1" not in adapter._session_start_times

    types = [e["event_type"] for e in events]
    assert "commerce.checkout.created" in types
    assert "commerce.checkout.completed" in types


def test_checkout_completed_without_create_does_not_crash() -> None:
    """Error path: completing a session that was never opened is non-fatal."""
    adapter, events = _make_adapter()
    adapter.on_checkout_completed("never_started", org_id="org_1", status="cancelled")
    # Still emits the completion event so the audit trail is honest
    assert any(e["event_type"] == "commerce.checkout.completed" for e in events)


def test_order_refunded_emits_event() -> None:
    adapter, events = _make_adapter()
    adapter.on_order_refunded(
        order_id="ord_1",
        refund_amount=12.34,
        org_id="org_1",
        currency="USD",
        reason="defective",
    )
    types = [e["event_type"] for e in events]
    assert "commerce.order.refunded" in types
    payload = next(e for e in events if e["event_type"] == "commerce.order.refunded")["payload"]
    assert payload.refund_amount == 12.34
    assert payload.reason == "defective"


def test_disconnect_clears_state() -> None:
    adapter, _ = _make_adapter()
    adapter.on_supplier_discovered(
        supplier_id="s", name="n", profile_url="u", org_id="o"
    )
    assert adapter._suppliers != {}
    adapter.disconnect()
    assert adapter._suppliers == {}
    assert adapter._checkout_sessions == {}
    assert adapter._session_start_times == {}


def test_serialize_for_replay_shape() -> None:
    adapter, _ = _make_adapter()
    adapter.on_supplier_discovered(supplier_id="s", name="n", profile_url="u", org_id="o")
    rt = adapter.serialize_for_replay()
    assert isinstance(rt, ReplayableTrace)
    assert rt.adapter_name == "UCPAdapter"
    assert rt.framework == "ucp"
    assert "suppliers" in rt.config


def test_line_items_parsed_into_typed_payload() -> None:
    adapter, events = _make_adapter()
    adapter.on_checkout_created(
        checkout_session_id="cs",
        supplier_id="s",
        line_items=[
            {"item_id": "a", "quantity": 1, "unit_price": 10.0},
            {"item_id": "b", "quantity": 2, "unit_price": 5.0, "name": "Widget"},
        ],
        total_amount=20.0,
        org_id="o",
    )
    payload = next(e for e in events if e["event_type"] == "commerce.checkout.created")["payload"]
    assert len(payload.line_items) == 2
    assert payload.line_items[1].name == "Widget"
