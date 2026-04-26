"""Unit tests for the A2UI (Agent-to-User Interface) protocol adapter.

A2UI emits ``commerce.ui.*`` events; like AP2, those event types are
not in the default :class:`CaptureConfig` layer map, so the adapter's
:meth:`emit_event` would silently drop them. The recorder pattern
replaces ``emit_event`` with a hook that captures every payload.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Callable

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters.protocols.a2ui import A2UIAdapter


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


def _make_adapter() -> tuple[A2UIAdapter, List[Dict[str, Any]]]:
    events, emit = _make_recorder()
    adapter = A2UIAdapter()
    adapter.emit_event = emit  # type: ignore[method-assign]
    adapter.connect()
    return adapter, events


def test_adapter_class_constants() -> None:
    assert A2UIAdapter.FRAMEWORK == "a2ui"
    assert A2UIAdapter.PROTOCOL == "a2ui"
    assert A2UIAdapter.PROTOCOL_VERSION == "1.0.0"


def test_lifecycle_transitions() -> None:
    adapter = A2UIAdapter()
    assert adapter.status == AdapterStatus.DISCONNECTED
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY
    adapter.disconnect()
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_disconnect_clears_state() -> None:
    adapter, _ = _make_adapter()
    adapter.on_surface_created(surface_id="s1", org_id="o", component_count=5)
    assert "s1" in adapter._surfaces
    adapter.disconnect()
    assert adapter._surfaces == {}
    assert adapter._component_counts == {}


def test_get_adapter_info_shape() -> None:
    adapter, _ = _make_adapter()
    info = adapter.get_adapter_info()
    assert isinstance(info, AdapterInfo)
    assert info.framework == "a2ui"
    assert info.name == "A2UIAdapter"


def test_probe_health_shape() -> None:
    adapter, _ = _make_adapter()
    health = adapter.probe_health()
    assert health["reachable"] is True
    assert health["protocol_version"] == "1.0.0"
    assert "latency_ms" in health


def test_on_surface_created_emits_event_and_tracks_state() -> None:
    adapter, events = _make_adapter()
    adapter.on_surface_created(
        surface_id="surf_1",
        org_id="org_1",
        root_component_id="cmp_root",
        component_count=7,
    )
    assert "surf_1" in adapter._surfaces
    assert adapter._component_counts["surf_1"] == 7

    types = [e["event_type"] for e in events]
    assert "commerce.ui.surface_created" in types
    payload = next(e for e in events if e["event_type"] == "commerce.ui.surface_created")["payload"]
    assert payload.surface_id == "surf_1"
    assert payload.org_id == "org_1"
    assert payload.component_count == 7


def test_on_user_action_emits_event_with_hashed_context() -> None:
    adapter, events = _make_adapter()
    context = {"cart_total": 99.98, "currency": "USD"}
    adapter.on_user_action(
        surface_id="surf_1",
        action_name="confirm_purchase",
        org_id="org_1",
        component_id="cmp_btn",
        context=context,
    )

    types = [e["event_type"] for e in events]
    assert "commerce.ui.user_action" in types
    payload = next(e for e in events if e["event_type"] == "commerce.ui.user_action")["payload"]
    assert payload.action_name == "confirm_purchase"

    # Cleartext context never appears; hash is used instead
    expected_hash = "sha256:" + hashlib.sha256(str(context).encode()).hexdigest()
    assert payload.context_hash == expected_hash


def test_on_user_action_no_context_no_hash() -> None:
    adapter, events = _make_adapter()
    adapter.on_user_action(
        surface_id="s",
        action_name="cancel",
        org_id="o",
    )
    payload = next(e for e in events if e["event_type"] == "commerce.ui.user_action")["payload"]
    assert payload.context_hash is None


def test_serialize_for_replay_shape() -> None:
    adapter, _ = _make_adapter()
    adapter.on_surface_created(surface_id="s1", org_id="o", component_count=2)
    rt = adapter.serialize_for_replay()
    assert isinstance(rt, ReplayableTrace)
    assert rt.adapter_name == "A2UIAdapter"
    assert rt.framework == "a2ui"
    assert "surfaces" in rt.config


def test_action_context_hash_is_deterministic() -> None:
    """Same context dict produces the same hash (deterministic for replay)."""
    adapter, events = _make_adapter()
    ctx = {"a": 1, "b": "x"}
    adapter.on_user_action(surface_id="s", action_name="a1", org_id="o", context=ctx)
    adapter.on_user_action(surface_id="s", action_name="a2", org_id="o", context=ctx)
    h1 = events[0]["payload"].context_hash
    h2 = events[1]["payload"].context_hash
    assert h1 == h2 and h1 is not None


def test_health_check_reflects_disconnected() -> None:
    adapter = A2UIAdapter()
    # Without connect(), reachable is False
    h = adapter.probe_health()
    assert h["reachable"] is False
