"""UCP commerce-autonomy invariants — PII fail-open + per-run dollar ceiling.

Two corners, both bite-proven (revert the guarded source line → RED):

* **A15 / UCP-Q2 PII fail-open.** A ``commerce.checkout_completed`` (and the
  other commerce.* spend events) can carry the buyer's billing/shipping address,
  card / PAN / CVC, tokenized payment instrument, and email/phone. Under
  ``capture_content=False`` those MUST be stripped (ACP rfc.delegate_payment §277
  "logs MUST NOT contain full PAN or CVC"; UCP "Never log raw credentials"). The
  pre-fix content-key set listed only ``amount``, so every PII field LEAKED.
  Defense in depth: the card PAN/CVC are ALSO scrubbed at the collector
  chokepoint regardless of capture_content (see test_secret_scrub.py).

* **A13 per-run dollar ceiling.** ``complete_checkout`` feeds the run-scoped
  ledger (:mod:`layerlens.instrument._spend_ledger`) — the SAME ledger AP2 feeds
  — checking the ceiling BEFORE the charge and accruing after. A checkout over
  the ceiling is BLOCKED (the real UCP client is never called, no spend accrues)
  and emits ``policy.violation``. Distinct from the collector's MAX_EVENTS count
  cap.

The tests drive the adapter's REAL ``connect()`` wrap path with a stand-in UCP
client (a true unit double — the FULL real UCP SDK rewrite is ticketed and there
is no installable real UCP client; the PII/ceiling fixes here are on the current
shape). The PAYLOADS routed through the real emit path (collector + redact
backstop + secret-scrub chokepoint) are what is under test.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from layerlens.instrument import _spend_ledger
from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.protocols.ucp import UCPProtocolAdapter

# Commerce PII + ceiling guards belong in the fast Invariant Gates job
# (shift-left): a leaked PAN or an un-ceilinged autonomous spend is a
# release-blocking defect.
pytestmark = pytest.mark.invariant

_NO_CONTENT = CaptureConfig(capture_content=False)
_CONTENT = CaptureConfig(capture_content=True)

# A real Luhn-valid test PAN + CVC + an address, the kind a real checkout hook
# carries (ACP PaymentMethodCard.number/cvc + Address). The PAN is a sentinel we
# assert NEVER reaches an uploaded payload.
PAN = "4111 1111 1111 1111"
CVC = "737"
BILLING_LINE = "SENTINEL-42 Privacy Lane"
BUYER_EMAIL = "buyer-SENTINEL@example.com"


@pytest.fixture(autouse=True)
def _fresh_ledger():
    """Each test gets a fresh run-scoped ledger (a clean run), restored after, so
    the ContextVar ledger never leaks spend across tests."""
    token = _spend_ledger.set_ledger(None)
    try:
        yield
    finally:
        _spend_ledger.reset_ledger(token)


def _collect(fn: Any, collector_config: Optional[CaptureConfig] = None) -> List[Dict[str, Any]]:
    collector = TraceCollector(object(), collector_config or CaptureConfig())
    token = _current_collector.set(collector)
    try:
        fn()
    finally:
        _current_collector.reset(token)
    return collector.events


def _by_type(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    return [e["payload"] for e in events if e["event_type"] == event_type]


def _client() -> SimpleNamespace:
    """A stand-in UCP client whose complete_checkout echoes whatever it is
    handed (a real checkout client receives the buyer's address/card on the
    completion call — we model that PII-carrying shape)."""
    return SimpleNamespace(
        discover_suppliers=lambda **kw: [],
        browse_catalog=lambda **kw: [],
        start_checkout=lambda **kw: {"session_id": kw.get("session_id", "s1")},
        complete_checkout=lambda **kw: {"ok": True},
        issue_refund=lambda **kw: {"ok": True},
    )


# ===========================================================================
# A15 / UCP-Q2 — PII FAIL-OPEN: a commerce.checkout_completed carrying
# billing_address + card under capture_content=False must have them STRIPPED
# (ids/currency survive). The CURRENT adapter hooks emit only session/supplier/
# amount metadata (the FULL real-UCP rewrite that forwards the buyer's
# address/card is TICKETED), so the leak is LATENT: the moment a real checkout
# hook (or a **kwargs splat against the real complete_checkout signature)
# carries this PII, the pre-fix content-key set ({amount} only) lets it through.
# The fix is the collector-side redaction BACKSTOP — the real upload chokepoint
# (TraceCollector.emit -> redact_payload) — which is exactly what _CONTENT_KEYS
# governs. We drive a PII-carrying payload through the REAL collector.emit
# chokepoint (not a hand-rolled redactor). BITE: remove the PII keys from
# _CONTENT_KEYS["commerce.checkout_completed"] -> they survive -> RED.
# ===========================================================================

# The realistic PII-carrying checkout payload a real UCP/ACP completion hook
# emits: ACP PaymentData.billing_address + PaymentMethodCard (number/cvc) +
# Buyer email, alongside the existing financial-flow metadata.
_PII_CHECKOUT_PAYLOAD: Dict[str, Any] = {
    "session_id": "sess-1",
    "supplier_id": "sup-1",
    "currency": "USD",
    "amount": 49.99,
    "billing_address": {"line_one": BILLING_LINE, "city": "Townsville", "postal_code": "90210"},
    "email": BUYER_EMAIL,
    "card": {"number": PAN, "cvc": CVC, "exp_month": 12, "exp_year": 2030},
}


def _emit_checkout(payload: Dict[str, Any], config: CaptureConfig) -> Dict[str, Any]:
    """Route a commerce.checkout_completed payload through the REAL collector
    chokepoint (redact + scrub) and return the uploaded payload."""
    collector = TraceCollector(object(), config)
    collector.emit("commerce.checkout_completed", dict(payload), span_id="s1")
    return collector.events[0]["payload"]


class TestCheckoutPIIRedaction:
    def test_content_present_by_default(self) -> None:
        # Sanity: under capture_content=True the PII IS captured, so the
        # no-content assertion below is meaningful (not vacuous). (The card PAN
        # is still scrubbed by the secret net even here — see test_secret_scrub —
        # so we assert the address/email survive, which the scrub never touches.)
        p = _emit_checkout(_PII_CHECKOUT_PAYLOAD, _CONTENT)
        blob = json.dumps(p, default=str)
        assert BILLING_LINE in blob, "billing line not captured under content-on — test would be vacuous"
        assert BUYER_EMAIL in blob, "buyer email not captured under content-on — test would be vacuous"

    def test_no_content_strips_pii_keeps_metadata(self) -> None:
        p = _emit_checkout(_PII_CHECKOUT_PAYLOAD, _NO_CONTENT)
        blob = json.dumps(p, default=str)
        # billing_address / email / card PAN are CONTENT -> stripped.
        assert BILLING_LINE not in blob, "billing_address leaked under capture_content=False (A15 fail-open)"
        assert BUYER_EMAIL not in blob, "buyer email leaked under capture_content=False (A15 fail-open)"
        assert PAN not in blob, "card PAN leaked under capture_content=False (A15/PCI)"
        # the whole card / billing_address / email keys are gone…
        assert "card" not in p and "billing_address" not in p and "email" not in p, "PII key survived redaction"
        # …but the financial-flow skeleton survives (ids + currency code).
        assert p.get("session_id") == "sess-1", "session_id over-stripped (metadata lost)"
        assert p.get("supplier_id") == "sup-1", "supplier_id over-stripped (metadata lost)"
        assert p.get("currency") == "USD", "currency code over-stripped (it is a code, not a sum)"

    def test_amount_value_still_stripped_under_no_content(self) -> None:
        # Regression: the pre-existing amount-is-content invariant must not weaken
        # — a distinctive amount value must not survive no-content (the currency
        # code does).
        p = _emit_checkout({"session_id": "s2", "currency": "USD", "amount": 4242.42}, _NO_CONTENT)
        assert "4242.42" not in json.dumps(p, default=str), "checkout amount value leaked under capture_content=False"
        assert p.get("currency") == "USD"

    def test_started_event_strips_pii_too(self) -> None:
        # The commerce.checkout.started event (empty content-key set pre-fix —
        # fully fail-open) now strips PII as well.
        collector = TraceCollector(object(), _NO_CONTENT)
        collector.emit(
            "commerce.checkout.started",
            {"session_id": "s1", "supplier_id": "sup-1", "shipping_address": {"line_one": BILLING_LINE}},
            span_id="s1",
        )
        blob = json.dumps(collector.events[0]["payload"], default=str)
        assert BILLING_LINE not in blob, "commerce.checkout.started leaked shipping_address under no-content"
        assert collector.events[0]["payload"].get("supplier_id") == "sup-1", "metadata over-stripped"


# ===========================================================================
# A13 — PER-RUN DOLLAR CEILING: complete_checkout feeds the run-scoped ledger;
# a checkout over the ceiling is BLOCKED before the charge (real client never
# called, no spend accrues) and emits policy.violation. BITE: remove the
# `if not verdict.allowed:` ceiling block in ucp._on_complete_checkout -> the
# over-ceiling checkout proceeds, spend accrues, no policy.violation -> RED.
# ===========================================================================


class TestCheckoutCeiling:
    def _adapter_and_calls(self):
        """Build an adapter wrapping a client that COUNTS real completion calls,
        so 'the real client was never called' is directly assertable."""
        calls: List[Dict[str, Any]] = []

        def _complete(**kw: Any) -> Any:
            calls.append(kw)
            return {"ok": True}

        target = _client()
        target.complete_checkout = _complete
        adapter = UCPProtocolAdapter()
        adapter.connect(target=target)
        return adapter, target, calls

    def test_checkout_over_ceiling_blocked_no_spend(self) -> None:
        _spend_ledger.configure_ledger(ceiling_usd=100.0, currency="USD")
        _adapter, target, calls = self._adapter_and_calls()

        def go() -> None:
            with pytest.raises(PermissionError):
                target.complete_checkout(session_id="s1", supplier_id="sup-1", amount=150.0, currency="USD")

        events = _collect(go)
        assert calls == [], "real complete_checkout was called despite the ceiling block (charge moved money)"
        assert _spend_ledger.current_spend("USD") == 0.0, "over-ceiling checkout still accrued spend"
        violations = _by_type(events, "policy.violation")
        assert violations, "no policy.violation emitted on ceiling block"
        assert violations[0]["reason_code"] == "RUN_CEILING_EXCEEDED"
        assert violations[0]["stage"] == "checkout"
        blocked = _by_type(events, "commerce.checkout_completed")
        assert blocked and blocked[0].get("status") == "blocked", "blocked checkout not marked blocked"

    def test_checkout_under_ceiling_accrues_and_completes(self) -> None:
        _spend_ledger.configure_ledger(ceiling_usd=100.0, currency="USD")
        _adapter, target, calls = self._adapter_and_calls()

        def go() -> None:
            target.complete_checkout(session_id="s1", supplier_id="sup-1", amount=40.0, currency="USD")

        events = _collect(go)
        assert len(calls) == 1, "under-ceiling checkout did not call through to the real client"
        assert _spend_ledger.current_spend("USD") == 40.0, "under-ceiling checkout did not accrue spend"
        completed = _by_type(events, "commerce.checkout_completed")
        assert completed and completed[0].get("status") == "completed"
        assert not _by_type(events, "policy.violation"), "policy.violation emitted for an in-budget checkout"

    def test_ceiling_spans_ap2_and_ucp_in_one_run(self) -> None:
        """The whole point of the run-scoped ledger: AP2 spend + UCP checkout
        spend share ONE ceiling. Seed the ledger with prior (AP2-style) spend via
        add_spend, then a UCP checkout that alone is under the ceiling but pushes
        the RUN total over it is blocked."""
        _spend_ledger.configure_ledger(ceiling_usd=100.0, currency="USD")
        _spend_ledger.add_spend(80.0, "USD")  # prior spend in this run (e.g. an AP2 charge)
        _adapter, target, calls = self._adapter_and_calls()

        def go() -> None:
            with pytest.raises(PermissionError):
                # 30 alone < 100, but 80 + 30 = 110 > 100 -> blocked by the RUN total.
                target.complete_checkout(session_id="s1", supplier_id="sup-1", amount=30.0, currency="USD")

        events = _collect(go)
        assert calls == [], "checkout charged despite breaching the shared run ceiling"
        assert _spend_ledger.current_spend("USD") == 80.0, "blocked UCP checkout still accrued onto the run total"
        assert any(v["reason_code"] == "RUN_CEILING_EXCEEDED" for v in _by_type(events, "policy.violation"))

    def test_check_is_before_charge_not_after(self) -> None:
        """Ordering invariant: the ceiling is checked BEFORE the charge. If the
        real client raised, an after-the-charge check could still have moved
        money. We prove the block path never reaches the client AND no spend
        accrues (the post-charge add_spend is skipped)."""
        _spend_ledger.configure_ledger(ceiling_usd=10.0, currency="USD")
        _adapter, target, calls = self._adapter_and_calls()

        def go() -> None:
            with pytest.raises(PermissionError):
                target.complete_checkout(session_id="s1", supplier_id="sup-1", amount=11.0, currency="USD")

        _collect(go)
        assert calls == [], "client called before ceiling enforced"
        assert _spend_ledger.current_spend("USD") == 0.0
