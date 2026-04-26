"""Unit tests for the AP2 (Agent Payments Protocol) adapter.

The AP2 adapter emits ``commerce.payment.*`` events; those event types
are not in the default :class:`CaptureConfig` layer map, so the adapter's
:meth:`emit_event` would silently drop them. To assert event emission
we replace ``emit_event`` with a recorder that bypasses the gate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Callable

import pytest

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters.protocols.ap2 import AP2Adapter


def _make_recorder() -> tuple[List[Dict[str, Any]], Callable[..., None]]:
    """Return (events_list, replacement-emit-callable)."""
    events: List[Dict[str, Any]] = []

    def _emit(payload: Any, privacy_level: Any = None) -> None:
        events.append(
            {
                "event_type": getattr(payload, "event_type", None),
                "payload": payload,
            }
        )

    return events, _emit


def _make_adapter() -> tuple[AP2Adapter, List[Dict[str, Any]]]:
    """Construct a connected AP2Adapter wired with an emit-recorder."""
    events, emit = _make_recorder()
    adapter = AP2Adapter()
    adapter.emit_event = emit  # type: ignore[method-assign]
    adapter.connect()
    return adapter, events


def test_adapter_class_constants() -> None:
    assert AP2Adapter.FRAMEWORK == "ap2"
    assert AP2Adapter.PROTOCOL == "ap2"
    assert AP2Adapter.PROTOCOL_VERSION == "0.1.0"


def test_lifecycle_transitions() -> None:
    adapter = AP2Adapter()
    assert adapter.status == AdapterStatus.DISCONNECTED
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY
    assert adapter.is_connected is True
    adapter.disconnect()
    assert adapter.status == AdapterStatus.DISCONNECTED
    assert adapter.is_connected is False


def test_disconnect_clears_state() -> None:
    adapter, _ = _make_adapter()
    adapter.configure_policy("org-1", max_single_tx=1.0)
    adapter._spending_totals["org-1"] = 99.0
    assert adapter._policies != {}
    adapter.disconnect()
    assert adapter._policies == {}
    assert adapter._spending_totals == {}
    assert adapter._mandates == {}


def test_get_adapter_info_shape() -> None:
    adapter, _ = _make_adapter()
    info = adapter.get_adapter_info()
    assert isinstance(info, AdapterInfo)
    assert info.framework == "ap2"
    assert info.name == "AP2Adapter"


def test_probe_health_shape() -> None:
    adapter, _ = _make_adapter()
    health = adapter.probe_health()
    assert health["reachable"] is True
    assert health["protocol_version"] == "0.1.0"
    assert "active_mandates" in health
    assert health["active_mandates"] == 0


def test_intent_mandate_no_policy_no_violations() -> None:
    adapter, events = _make_adapter()
    violations = adapter.on_intent_mandate_created(
        mandate_id="m1",
        description="Buy supplies",
        org_id="org-x",
        max_amount=999_999.0,
    )
    assert violations == []
    types = [e["event_type"] for e in events]
    assert "commerce.payment.intent_created" in types
    assert "commerce.payment.intent_validated" in types


def test_intent_mandate_amount_violation_emits_guardrail() -> None:
    adapter, events = _make_adapter()
    adapter.configure_policy("org-1", max_single_tx=100.0)
    violations = adapter.on_intent_mandate_created(
        mandate_id="m2",
        description="Big buy",
        org_id="org-1",
        max_amount=500.0,
    )
    assert any("exceeds" in v for v in violations)
    types = [e["event_type"] for e in events]
    assert "commerce.payment.guardrail_violation" in types


def test_intent_mandate_merchant_violation() -> None:
    adapter, events = _make_adapter()
    adapter.configure_policy("org-1", allowed_merchants=["FreshCo"])
    violations = adapter.on_intent_mandate_created(
        mandate_id="m3",
        description="Buy from random merchant",
        org_id="org-1",
        merchants=["RandoCorp"],
        max_amount=10.0,
    )
    assert any("not in allowed" in v for v in violations)
    assert any(e["event_type"] == "commerce.payment.guardrail_violation" for e in events)


def test_intent_mandate_refundability_violation() -> None:
    adapter, _ = _make_adapter()
    adapter.configure_policy("org-1", require_refundability=True)
    violations = adapter.on_intent_mandate_created(
        mandate_id="m4",
        description="Non-refundable",
        org_id="org-1",
        requires_refundability=False,
    )
    assert any("Refundability" in v for v in violations)


def test_payment_mandate_signed_emits_event_and_tracks_spend() -> None:
    adapter, events = _make_adapter()
    adapter.on_payment_mandate_signed(
        mandate_id="m1",
        payment_details_id="pd-1",
        total_amount=42.0,
        merchant_agent="FreshCo",
        org_id="org-1",
        signature="raw-jwt",
    )
    assert adapter._spending_totals["org-1"] == 42.0
    types = [e["event_type"] for e in events]
    assert "commerce.payment.mandate_signed" in types

    # Signature is hashed (sha256), never stored raw
    sig_event = next(e for e in events if e["event_type"] == "commerce.payment.mandate_signed")
    assert "raw-jwt" not in str(sig_event["payload"].mandate.signature_hash)


def test_spending_threshold_event_emitted_when_exceeded() -> None:
    adapter, events = _make_adapter()
    adapter.configure_policy("org-1", daily_limit=100.0)
    # Push spending total over threshold
    adapter.on_payment_mandate_signed(
        mandate_id="m1",
        payment_details_id="pd-1",
        total_amount=150.0,
        merchant_agent="FreshCo",
        org_id="org-1",
    )
    types = [e["event_type"] for e in events]
    assert "commerce.payment.threshold_exceeded" in types


def test_payment_receipt_success_clears_mandate() -> None:
    adapter, _ = _make_adapter()
    # Seed an intent mandate so we can verify removal
    adapter.on_intent_mandate_created(
        mandate_id="m1",
        description="t",
        org_id="org-1",
    )
    assert "m1" in adapter._mandates

    adapter.on_payment_receipt_issued(
        mandate_id="m1",
        payment_id="PAY-1",
        amount=42.0,
        org_id="org-1",
        status="success",
    )
    assert "m1" not in adapter._mandates


def test_payment_receipt_failure_keeps_mandate() -> None:
    adapter, _ = _make_adapter()
    adapter.on_intent_mandate_created(
        mandate_id="m1",
        description="t",
        org_id="org-1",
    )
    adapter.on_payment_receipt_issued(
        mandate_id="m1",
        payment_id="PAY-2",
        amount=42.0,
        org_id="org-1",
        status="failed",
    )
    assert "m1" in adapter._mandates


def test_serialize_for_replay_shape() -> None:
    adapter, _ = _make_adapter()
    adapter.configure_policy("org-1", max_single_tx=10.0)
    adapter.on_intent_mandate_created(mandate_id="m1", description="t", org_id="org-1")
    rt = adapter.serialize_for_replay()
    assert isinstance(rt, ReplayableTrace)
    assert rt.adapter_name == "AP2Adapter"
    assert rt.framework == "ap2"
    assert "policies" in rt.config
    assert rt.state_snapshots and "mandates" in rt.state_snapshots[0]


def test_commerce_events_bypass_capture_gate() -> None:
    """Commerce events are cross-cutting and MUST bypass ``CaptureConfig``.

    Earlier in the port, ``commerce.*`` events fell off the layer-gate
    map and were silently dropped (a CLAUDE.md rule-2 violation). The
    fix is in ``_base/capture.py`` — ``ALWAYS_ENABLED_EVENT_TYPES`` now
    includes the commerce family heads AND ``is_layer_enabled`` has a
    prefix bypass for ``commerce.*`` so subtypes also pass.

    This test pins that fix in place: a fresh adapter (no recorder
    override) emits a commerce event and the trace records it.
    """
    adapter = AP2Adapter()
    adapter.connect()
    adapter.on_intent_mandate_created(mandate_id="m1", description="t", org_id="o1")
    # _trace_events is populated by _post_emit_success — proves the
    # event flowed through the gate, the null-Stratix call succeeded,
    # and the post-success path ran.
    assert adapter._trace_events, (
        "commerce events must not be dropped by the default CaptureConfig"
    )
    assert adapter._trace_events[0]["event_type"].startswith("commerce.")


def test_invalid_keyword_argument_raises() -> None:
    """Constructor rejects unknown kwargs — guards against typo regressions."""
    with pytest.raises(TypeError):
        AP2Adapter(nonsense_arg=True)  # type: ignore[call-arg]
