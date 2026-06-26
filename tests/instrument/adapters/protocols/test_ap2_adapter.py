"""AP2 payment-mandate adapter — BEHAVIORAL unit tier (LAY-3617).

Payment mandates are the highest-stakes protocol content (money moves), so the
unit tier must drive the *live* :class:`AP2ProtocolAdapter` reached by
``connect()`` and assert the events it actually emits — not a contract double.

Coverage
========

* The three-stage mandate **chain** — ``create_intent_mandate`` ->
  ``sign_payment_mandate`` -> ``issue_receipt`` emits
  ``payment.intent_mandate`` / ``payment.mandate_signed`` (``status="signed"``)
  / ``payment.receipt_issued``, with the ids/amounts the adapter records.
* **Every** :class:`AP2Guardrails` field (``max_transaction`` /
  ``merchant_whitelist`` / ``cumulative_threshold`` / ``mandate_ttl_seconds``):
  for each blocked case ``sign_payment_mandate`` BOTH emits a
  ``payment.mandate_signed`` event with ``status="blocked"`` AND raises
  ``PermissionError`` (``_evaluate_guardrails`` short-circuits the sign).
* ``cumulative_threshold`` is **stateful** (running ``_cumulative_spend``), so
  the order of sign calls matters — multiple under-threshold signs accumulate
  until one crosses the line and is blocked.

These mirror the connect()+invoke shape of ``test_protocol_redaction.py``: a
real adapter, a ``SimpleNamespace`` target exposing the patched methods,
``adapter.connect(target=...)``, then ``target.method(...)``. Emitted events are
read off the ambient :class:`TraceCollector` and recorded for the autouse
schema lock.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from layerlens.instrument._events import (
    PAYMENT_INTENT_MANDATE,
    PAYMENT_MANDATE_SIGNED,
    PAYMENT_RECEIPT_ISSUED,
)
from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.protocols.ap2 import (
    AP2Guardrails,
    AP2ProtocolAdapter,
)

# The conftest schema-lock buffer lives in tests/instrument/conftest.py; feed it
# so the autouse _enforce_schema_lock genuinely validates the events we emit
# (the collector pattern bypasses the upload path that normally records them).
from ...conftest import record_for_schema_lock


def _events_of(collector: TraceCollector, event_type: str) -> List[Dict[str, Any]]:
    return [e for e in collector.events if e["event_type"] == event_type]


def _one(collector: TraceCollector, event_type: str) -> Dict[str, Any]:
    matches = _events_of(collector, event_type)
    assert matches, f"no {event_type!r} event in {[e['event_type'] for e in collector.events]}"
    assert len(matches) == 1, f"expected exactly one {event_type!r}, got {len(matches)}"
    return matches[0]


def _drive(
    adapter: AP2ProtocolAdapter,
    body: Any,
    *,
    target: Optional[SimpleNamespace] = None,
) -> TraceCollector:
    """connect() the live adapter to *target*, run *body(target)* under an
    ambient collector, then hand the emitted events to the schema lock."""
    if target is None:
        target = SimpleNamespace(
            create_intent_mandate=lambda **kw: {"ok": "intent"},
            sign_payment_mandate=lambda **kw: {"ok": "signed"},
            issue_receipt=lambda **kw: {"ok": "receipt"},
        )
    adapter.connect(target=target)

    collector = TraceCollector(client=object(), config=CaptureConfig())
    token = _current_collector.set(collector)
    try:
        body(target)
    finally:
        _current_collector.reset(token)
    record_for_schema_lock(collector.events)
    return collector


# ---------------------------------------------------------------------------
# The three-stage mandate chain: intent -> sign -> receipt
# ---------------------------------------------------------------------------


class TestMandateChain:
    def test_intent_sign_receipt_chain_emits_in_order(self) -> None:
        adapter = AP2ProtocolAdapter()

        def body(target: SimpleNamespace) -> None:
            target.create_intent_mandate(mandate_id="m-chain", amount=42.5, merchant="ACME")
            target.sign_payment_mandate(mandate_id="m-chain")
            target.issue_receipt(receipt_id="r-1", mandate_id="m-chain", amount=42.5, merchant="ACME")

        collector = _drive(adapter, body)

        # Exactly the three chain events, in chain order.
        types = [e["event_type"] for e in collector.events]
        assert types == [
            PAYMENT_INTENT_MANDATE,
            PAYMENT_MANDATE_SIGNED,
            PAYMENT_RECEIPT_ISSUED,
        ]

        intent = _one(collector, PAYMENT_INTENT_MANDATE)
        assert intent["payload"]["mandate_id"] == "m-chain"
        assert intent["payload"]["amount"] == 42.5
        assert intent["payload"]["merchant"] == "ACME"
        assert intent["payload"]["protocol"] == "ap2"

        signed = _one(collector, PAYMENT_MANDATE_SIGNED)
        assert signed["payload"]["status"] == "signed"
        assert signed["payload"]["mandate_id"] == "m-chain"
        # amount falls back to the intent-mandate record when sign omits it.
        assert signed["payload"]["amount"] == 42.5
        assert signed["payload"]["cumulative_spend"] == 42.5

        receipt = _one(collector, PAYMENT_RECEIPT_ISSUED)
        assert receipt["payload"]["receipt_id"] == "r-1"
        assert receipt["payload"]["mandate_id"] == "m-chain"
        assert receipt["payload"]["amount"] == 42.5

    def test_chain_returns_underlying_results(self) -> None:
        """connect() patches in place; the wrapped methods still return the
        original callable's result (instrumentation is transparent)."""
        adapter = AP2ProtocolAdapter()
        target = SimpleNamespace(
            create_intent_mandate=lambda **kw: {"stage": "intent"},
            sign_payment_mandate=lambda **kw: {"stage": "signed"},
            issue_receipt=lambda **kw: {"stage": "receipt"},
        )
        results: Dict[str, Any] = {}

        def body(t: SimpleNamespace) -> None:
            results["intent"] = t.create_intent_mandate(mandate_id="m", amount=1)
            results["sign"] = t.sign_payment_mandate(mandate_id="m")
            results["receipt"] = t.issue_receipt(mandate_id="m")

        _drive(adapter, body, target=target)
        assert results["intent"] == {"stage": "intent"}
        assert results["sign"] == {"stage": "signed"}
        assert results["receipt"] == {"stage": "receipt"}

    def test_sign_without_intent_uses_supplied_amount(self) -> None:
        """A bare sign (no prior intent record) accumulates the amount passed
        directly to sign_payment_mandate."""
        adapter = AP2ProtocolAdapter()

        def body(target: SimpleNamespace) -> None:
            target.sign_payment_mandate(mandate_id="m-direct", amount=10.0)

        collector = _drive(adapter, body)
        signed = _one(collector, PAYMENT_MANDATE_SIGNED)
        assert signed["payload"]["status"] == "signed"
        assert signed["payload"]["amount"] == 10.0
        assert signed["payload"]["cumulative_spend"] == 10.0


# ---------------------------------------------------------------------------
# Guardrail: max_transaction
# ---------------------------------------------------------------------------


class TestMaxTransactionGuardrail:
    def test_over_limit_blocks_and_raises(self) -> None:
        adapter = AP2ProtocolAdapter(guardrails=AP2Guardrails(max_transaction=100.0))

        def body(target: SimpleNamespace) -> None:
            with pytest.raises(PermissionError) as exc:
                target.sign_payment_mandate(mandate_id="m-big", amount=150.0)
            assert "max_transaction" in str(exc.value)

        collector = _drive(adapter, body)
        signed = _one(collector, PAYMENT_MANDATE_SIGNED)
        assert signed["payload"]["status"] == "blocked"
        assert signed["payload"]["mandate_id"] == "m-big"
        assert "exceeds max_transaction 100.0" in signed["payload"]["reason"]
        # A blocked sign never reaches the original, so spend stays at zero.
        assert adapter._cumulative_spend == 0.0

    def test_at_limit_is_allowed(self) -> None:
        """The boundary is strict ``>`` — an amount equal to the cap signs."""
        adapter = AP2ProtocolAdapter(guardrails=AP2Guardrails(max_transaction=100.0))

        def body(target: SimpleNamespace) -> None:
            target.sign_payment_mandate(mandate_id="m-edge", amount=100.0)

        collector = _drive(adapter, body)
        signed = _one(collector, PAYMENT_MANDATE_SIGNED)
        assert signed["payload"]["status"] == "signed"
        assert signed["payload"]["amount"] == 100.0


# ---------------------------------------------------------------------------
# Guardrail: merchant_whitelist
# ---------------------------------------------------------------------------


class TestMerchantWhitelistGuardrail:
    def test_off_whitelist_merchant_blocks_and_raises(self) -> None:
        adapter = AP2ProtocolAdapter(guardrails=AP2Guardrails(merchant_whitelist=["GoodCorp"]))

        def body(target: SimpleNamespace) -> None:
            with pytest.raises(PermissionError) as exc:
                target.sign_payment_mandate(mandate_id="m-bad", amount=5.0, merchant="EvilCorp")
            assert "whitelist" in str(exc.value)

        collector = _drive(adapter, body)
        signed = _one(collector, PAYMENT_MANDATE_SIGNED)
        assert signed["payload"]["status"] == "blocked"
        assert "'EvilCorp' not in whitelist" in signed["payload"]["reason"]
        assert adapter._cumulative_spend == 0.0

    def test_whitelisted_merchant_signs(self) -> None:
        adapter = AP2ProtocolAdapter(guardrails=AP2Guardrails(merchant_whitelist=["GoodCorp"]))

        def body(target: SimpleNamespace) -> None:
            target.sign_payment_mandate(mandate_id="m-ok", amount=5.0, merchant="GoodCorp")

        collector = _drive(adapter, body)
        signed = _one(collector, PAYMENT_MANDATE_SIGNED)
        assert signed["payload"]["status"] == "signed"
        assert adapter._cumulative_spend == 5.0


# ---------------------------------------------------------------------------
# Guardrail: cumulative_threshold (STATEFUL — order of signs matters)
# ---------------------------------------------------------------------------


class TestCumulativeThresholdGuardrail:
    def test_running_spend_crosses_threshold_blocks_late_sign(self) -> None:
        """Two under-threshold signs accumulate; the third would cross the
        threshold and is blocked. The earlier signs stay applied."""
        adapter = AP2ProtocolAdapter(guardrails=AP2Guardrails(cumulative_threshold=100.0))
        signs: List[Any] = []

        def body(target: SimpleNamespace) -> None:
            target.sign_payment_mandate(mandate_id="m1", amount=40.0)  # cumulative -> 40
            target.sign_payment_mandate(mandate_id="m2", amount=40.0)  # cumulative -> 80
            with pytest.raises(PermissionError) as exc:
                target.sign_payment_mandate(mandate_id="m3", amount=40.0)  # 80+40=120 > 100
            signs.append(str(exc.value))

        collector = _drive(adapter, body)

        events = _events_of(collector, PAYMENT_MANDATE_SIGNED)
        assert len(events) == 3
        statuses = [e["payload"]["status"] for e in events]
        assert statuses == ["signed", "signed", "blocked"]

        # The two accepted signs accumulated; the blocked one did NOT add to spend.
        assert events[0]["payload"]["cumulative_spend"] == 40.0
        assert events[1]["payload"]["cumulative_spend"] == 80.0
        assert adapter._cumulative_spend == 80.0

        blocked = events[2]
        assert blocked["payload"]["mandate_id"] == "m3"
        assert "would exceed threshold 100.0" in blocked["payload"]["reason"]
        assert "cumulative spend 120.0" in blocked["payload"]["reason"]
        assert "would exceed threshold" in signs[0]

    def test_single_sign_under_threshold_is_allowed(self) -> None:
        adapter = AP2ProtocolAdapter(guardrails=AP2Guardrails(cumulative_threshold=100.0))

        def body(target: SimpleNamespace) -> None:
            target.sign_payment_mandate(mandate_id="m-lone", amount=99.0)

        collector = _drive(adapter, body)
        signed = _one(collector, PAYMENT_MANDATE_SIGNED)
        assert signed["payload"]["status"] == "signed"
        assert signed["payload"]["cumulative_spend"] == 99.0


# ---------------------------------------------------------------------------
# Guardrail: mandate_ttl_seconds (expiry of a known intent mandate)
# ---------------------------------------------------------------------------


class TestMandateTtlGuardrail:
    def test_expired_intent_mandate_blocks_sign_and_raises(self) -> None:
        """The TTL check fires only for a mandate the adapter recorded at
        intent time. We create the intent, then backdate its recorded
        ``created_at`` past the TTL to simulate an aged mandate (no sleep)."""
        adapter = AP2ProtocolAdapter(guardrails=AP2Guardrails(mandate_ttl_seconds=60.0))

        def body(target: SimpleNamespace) -> None:
            target.create_intent_mandate(mandate_id="m-stale", amount=10.0, merchant="ACME")
            # Age it: pretend the intent was created 1000s ago (> 60s TTL).
            adapter._intent_mandates["m-stale"]["created_at"] -= 1000.0
            with pytest.raises(PermissionError) as exc:
                target.sign_payment_mandate(mandate_id="m-stale", amount=10.0)
            assert "ttl" in str(exc.value)

        collector = _drive(adapter, body)

        # intent emitted, then a blocked sign.
        assert [e["event_type"] for e in collector.events] == [
            PAYMENT_INTENT_MANDATE,
            PAYMENT_MANDATE_SIGNED,
        ]
        signed = _one(collector, PAYMENT_MANDATE_SIGNED)
        assert signed["payload"]["status"] == "blocked"
        assert signed["payload"]["mandate_id"] == "m-stale"
        assert "exceeds ttl 60.0s" in signed["payload"]["reason"]
        assert adapter._cumulative_spend == 0.0

    def test_fresh_intent_mandate_signs_within_ttl(self) -> None:
        adapter = AP2ProtocolAdapter(guardrails=AP2Guardrails(mandate_ttl_seconds=60.0))

        def body(target: SimpleNamespace) -> None:
            target.create_intent_mandate(mandate_id="m-fresh", amount=10.0, merchant="ACME")
            target.sign_payment_mandate(mandate_id="m-fresh", amount=10.0)

        collector = _drive(adapter, body)
        signed = _one(collector, PAYMENT_MANDATE_SIGNED)
        assert signed["payload"]["status"] == "signed"
        assert signed["payload"]["cumulative_spend"] == 10.0
