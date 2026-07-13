"""AP2 v0.2 payment-autonomy invariants — real ap2 SDK fixtures, bite-proven.

Every fixture here is a REAL pydantic object from the pinned ``ap2`` SDK
(google-agentic-commerce/AP2 @ v0.2.0) — NOT a hand-rolled SimpleNamespace. A
library upgrade that changes the mandate schema fails these fixtures loudly, so
the test can never drift from the real wire shape (brief §3.5). The whole module
``importorskip("ap2")``, so it SKIPS in the base py3.9 venv (which has no ap2)
and runs in ``.audit-venvs/ap2`` (py3.11 + ap2 + the repo editable).

Each invariant below has a BITING test: revert/weaken the guard in ap2.py (or the
run-scoped ceiling primitive) and the test goes RED. The bite for each was
confirmed by hand (temporarily breaking the guard, watching it fail, restoring)
— noted per-test.

The tests drive the adapter's REAL emit path (collector + redact backstop) and,
for the cart-swap invariant, the REAL ``connect()``/``MandateClient.create``
wrap chokepoint with a stand-in client (the client is a true unit double; the
MANDATE objects passed through it are real ap2 models).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone, timedelta

import pytest

pytest.importorskip("ap2")

from ap2.models.cart import Cart  # noqa: E402,F401  (proves the convenience model imports)
from ap2.models.mandate import (  # noqa: E402
    CartMandate,
    CartContents,
    IntentMandate,
    PaymentMandate,
    PaymentMandateContents,
)
from ap2.models.payment_request import (  # noqa: E402
    PaymentItem,
    PaymentRequest,
    PaymentResponse,
    PaymentMethodData,
    PaymentDetailsInit,
    PaymentCurrencyAmount,
)

from layerlens.instrument import _spend_ledger  # noqa: E402
from layerlens.instrument._context import _current_collector  # noqa: E402
from layerlens.instrument._collector import TraceCollector  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.protocols.ap2 import (  # noqa: E402
    AP2Guardrails,
    AP2ProtocolAdapter,
)

# Payment-control guards belong in the fast Invariant Gates job (shift-left): a
# guard that fails to block a charge is a release-blocking defect.
pytestmark = pytest.mark.invariant

_NO_CONTENT = CaptureConfig(capture_content=False)
_CONTENT = CaptureConfig(capture_content=True)
_FAR_FUTURE = "2999-01-01T00:00:00Z"


# ── real-ap2 fixture builders (the library's OWN typings) ──────────────────


def make_intent(
    *,
    merchants: List[str] | None = None,
    expiry: str = _FAR_FUTURE,
    confirmation_required: bool = True,
    description: str = "red basketball shoes",
) -> IntentMandate:
    return IntentMandate(
        natural_language_description=description,
        intent_expiry=expiry,
        merchants=merchants,
        user_cart_confirmation_required=confirmation_required,
    )


def make_cart(
    *,
    cart_id: str = "cart1",
    value: float = 49.99,
    currency: str = "USD",
    merchant: str = "ACME",
    expiry: str = _FAR_FUTURE,
    signature: str | None = "eyJhbGciOiJSUzI1NiJ9.merchant-cart-sig.zzz",
) -> CartMandate:
    item = PaymentItem(label="Shoes", amount=PaymentCurrencyAmount(currency=currency, value=value))
    details = PaymentDetailsInit(id="pd1", display_items=[item], total=item)
    pr = PaymentRequest(method_data=[PaymentMethodData(supported_methods="card")], details=details)
    contents = CartContents(
        id=cart_id,
        user_cart_confirmation_required=True,
        payment_request=pr,
        cart_expiry=expiry,
        merchant_name=merchant,
    )
    return CartMandate(contents=contents, merchant_authorization=signature)


def make_payment(
    *,
    payment_mandate_id: str = "pay1",
    value: float = 49.99,
    currency: str = "USD",
    signature: str | None = "eyJhbGciOiJFUzI1NksifQ.user-payment-vp.zzz",
) -> PaymentMandate:
    contents = PaymentMandateContents(
        payment_mandate_id=payment_mandate_id,
        payment_details_id="pd1",
        payment_details_total=PaymentItem(label="Total", amount=PaymentCurrencyAmount(currency=currency, value=value)),
        payment_response=PaymentResponse(request_id="pd1", method_name="card"),
        merchant_agent="ACME",
    )
    return PaymentMandate(payment_mandate_contents=contents, user_authorization=signature)


# ── harness: fresh run-scoped ledger + collector per test ──────────────────


@pytest.fixture(autouse=True)
def _fresh_ledger():
    """Each test gets a fresh run-scoped ledger (a clean run), restored after."""
    token = _spend_ledger.set_ledger(None)
    try:
        yield
    finally:
        _spend_ledger.reset_ledger(token)


def _adapter(
    guardrails: AP2Guardrails | None = None,
    capture_config: CaptureConfig | None = None,
) -> AP2ProtocolAdapter:
    return AP2ProtocolAdapter(guardrails=guardrails, capture_config=capture_config)


def _collect(fn: Any, collector_config: CaptureConfig | None = None) -> List[Dict[str, Any]]:
    collector = TraceCollector(object(), collector_config or CaptureConfig())
    token = _current_collector.set(collector)
    try:
        fn()
    finally:
        _current_collector.reset(token)
    return collector.events


def _by_type(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    return [e["payload"] for e in events if e["event_type"] == event_type]


def _ceiling(value: float | None, currency: str = "USD") -> None:
    _spend_ledger.configure_ledger(ceiling_usd=value, currency=currency)


# ===========================================================================
# INVARIANT 1 — CART-BINDING AMOUNT: the guard reads the BINDING cart total,
# not the (non-binding) intent. A cart over the limit is BLOCKED. (cart-swap)
# BITE: delete the `g.max_transaction ... amount > g.max_transaction` clause in
# `_guard_cart` (or make it read the intent instead of the cart) -> RED.
# ===========================================================================


class TestCartBindingAmount:
    def test_cart_total_over_max_transaction_is_blocked(self) -> None:
        adapter = _adapter(guardrails=AP2Guardrails(max_transaction=100.0), capture_config=_CONTENT)
        # The intent is silent on price (AP2 v0.2 intent is non-binding); the
        # BINDING price is the cart's $1000 total — it must be what the guard reads.
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")

        def go() -> None:
            with pytest.raises(PermissionError):
                adapter.record_cart_mandate(make_cart(value=1000.0), intent_mandate_id="m1")

        events = _collect(go, _CONTENT)
        # No spend accrued, no signed/accepted cart recorded.
        assert _spend_ledger.current_spend("USD") == 0.0
        assert "cart1" not in adapter._carts
        violations = _by_type(events, "policy.violation")
        assert violations and violations[0]["reason_code"] == "MAX_TRANSACTION_EXCEEDED"
        blocked_cart = [p for p in _by_type(events, "payment.cart_mandate") if p.get("status") == "blocked"]
        assert blocked_cart and blocked_cart[0]["reason_code"] == "MAX_TRANSACTION_EXCEEDED"

    def test_cart_total_over_intent_max_is_blocked(self) -> None:
        """A cart total above the user's approved intent max (an integrator that
        sets `max_amount` on the intent) is blocked — the cart-swap proper.
        BITE: delete the `CART_EXCEEDS_INTENT_MAX` clause -> RED."""
        adapter = _adapter(capture_config=_CONTENT)
        intent = make_intent(merchants=["ACME"])
        # carry an approved max on the real IntentMandate object (extra attr)
        object.__setattr__(intent, "max_amount", 100.0)
        adapter.record_intent_mandate(intent, mandate_id="m1")

        def go() -> None:
            with pytest.raises(PermissionError):
                adapter.record_cart_mandate(make_cart(value=250.0), intent_mandate_id="m1")

        events = _collect(go, _CONTENT)
        violations = _by_type(events, "policy.violation")
        assert violations and violations[0]["reason_code"] == "CART_EXCEEDS_INTENT_MAX"

    def test_cart_within_limit_is_accepted(self) -> None:
        adapter = _adapter(guardrails=AP2Guardrails(max_transaction=100.0), capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        events = _collect(lambda: adapter.record_cart_mandate(make_cart(value=49.99), intent_mandate_id="m1"), _CONTENT)
        accepted = [p for p in _by_type(events, "payment.cart_mandate") if p.get("status") == "accepted"]
        assert accepted and accepted[0]["cart_id"] == "cart1"
        assert "cart1" in adapter._carts

    def test_offwhitelist_merchant_blocked_via_intent_whitelist(self) -> None:
        """The intent's allowed-merchant whitelist blocks an off-list cart.
        BITE: delete the `MERCHANT_NOT_WHITELISTED` clause -> RED."""
        adapter = _adapter(capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")

        def go() -> None:
            with pytest.raises(PermissionError):
                adapter.record_cart_mandate(make_cart(merchant="EVIL-CORP"), intent_mandate_id="m1")

        events = _collect(go, _CONTENT)
        violations = _by_type(events, "policy.violation")
        assert violations and violations[0]["reason_code"] == "MERCHANT_NOT_WHITELISTED"

    def test_binding_total_is_read_from_cart_not_payment_mandate(self) -> None:
        """When a payment mandate is linked to a cart, the CART total is charged
        even if the payment mandate claims a different (lower) total — the cart is
        the merchant-signed binding price.
        BITE: in `record_payment_mandate`, change `amount = cart.amount` to read
        the payment mandate's own total -> RED."""
        adapter = _adapter(guardrails=AP2Guardrails(max_transaction=500.0), capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        adapter.record_cart_mandate(make_cart(value=300.0), intent_mandate_id="m1")

        # payment mandate LIES that the total is 1.00; the cart binding is 300.
        events = _collect(lambda: adapter.record_payment_mandate(make_payment(value=1.00), cart_id="cart1"), _CONTENT)
        signed = [p for p in _by_type(events, "payment.mandate_signed") if p.get("status") == "signed"]
        assert signed and signed[0]["amount"] == 300.0, "charged the payment-mandate total, not the binding cart total"
        assert _spend_ledger.current_spend("USD") == 300.0

    def test_merchant_agent_is_emitted_and_survives_redaction(self) -> None:
        """S15/F8: PaymentMandateContents.merchant_agent (the merchant the payment
        is bound to) is emitted on payment.mandate_signed as an identifier that
        survives capture_content=False — mirrors task_id, not free-text content.
        BITE: drop the merchant_agent stamp -> RED."""
        adapter = _adapter(guardrails=AP2Guardrails(max_transaction=500.0), capture_config=_NO_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        adapter.record_cart_mandate(make_cart(value=100.0), intent_mandate_id="m1")

        events = _collect(
            lambda: adapter.record_payment_mandate(make_payment(value=100.0), cart_id="cart1"), _NO_CONTENT
        )
        signed = [p for p in _by_type(events, "payment.mandate_signed") if p.get("status") == "signed"]
        assert signed, "no signed payment.mandate_signed event"
        assert signed[0].get("merchant_agent") == "ACME", "merchant_agent missing or stripped under no-content"


# ===========================================================================
# INVARIANT 2 — IDEMPOTENCY: the same mandate signed/charged twice accrues ONCE.
# BITE: delete the `already`/`_charged_*` dedup in `record_payment_mandate` -> RED.
# ===========================================================================


class TestIdempotency:
    def test_same_payment_mandate_charges_once(self) -> None:
        adapter = _adapter(capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        adapter.record_cart_mandate(make_cart(value=49.99), intent_mandate_id="m1")

        def go() -> None:
            adapter.record_payment_mandate(make_payment(value=49.99), cart_id="cart1")
            # retry-on-timeout / agent loop re-presents the SAME mandate
            adapter.record_payment_mandate(make_payment(value=49.99), cart_id="cart1")

        events = _collect(go, _CONTENT)
        assert _spend_ledger.current_spend("USD") == 49.99, "re-presented mandate double-counted spend"
        signed = [p for p in _by_type(events, "payment.mandate_signed") if p.get("status") == "signed"]
        replayed = [p for p in _by_type(events, "payment.mandate_signed") if p.get("status") == "replayed"]
        assert len(signed) == 1, "more than one signed charge for one logical payment"
        assert len(replayed) == 1, "no idempotent-replay marker emitted for the retry"

    def test_receipt_is_idempotent(self) -> None:
        adapter = _adapter(capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        adapter.record_cart_mandate(make_cart(value=49.99), intent_mandate_id="m1")
        adapter.record_payment_mandate(make_payment(value=49.99), cart_id="cart1")

        def go() -> None:
            adapter.issue_receipt(cart_id="cart1")
            adapter.issue_receipt(cart_id="cart1")  # retry

        events = _collect(go, _CONTENT)
        issued = [p for p in _by_type(events, "payment.receipt_issued") if p.get("status") == "issued"]
        replayed = [p for p in _by_type(events, "payment.receipt_issued") if p.get("status") == "replayed"]
        assert len(issued) == 1 and len(replayed) == 1


# ===========================================================================
# INVARIANT 3 — RUN-SCOPED CUMULATIVE CAP: cumulative spend is shared across the
# WHOLE run (TWO adapter instances in one run share the running total), NOT reset
# per adapter instance.
# BITE: move the cumulative total back onto an instance attribute (revert the
# ledger to per-instance _cumulative_spend) -> the second adapter starts at 0 and
# the over-cap charge is wrongly allowed -> RED.
# ===========================================================================


class TestRunScopedCumulativeCap:
    def test_two_adapters_share_one_run_total(self) -> None:
        _ceiling(150.0)  # per-run dollar ceiling shared across instances
        a1 = _adapter(capture_config=_CONTENT)
        a2 = _adapter(capture_config=_CONTENT)  # a SECOND adapter in the SAME run

        a1.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        a1.record_cart_mandate(make_cart(cart_id="c1", value=100.0), intent_mandate_id="m1")
        a2.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m2")
        a2.record_cart_mandate(make_cart(cart_id="c2", value=100.0), intent_mandate_id="m2")

        def go() -> None:
            a1.record_payment_mandate(make_payment(payment_mandate_id="p1", value=100.0), cart_id="c1")  # 100 ok
            # second adapter, same run: 100 + 100 = 200 > 150 ceiling -> blocked
            with pytest.raises(PermissionError):
                a2.record_payment_mandate(make_payment(payment_mandate_id="p2", value=100.0), cart_id="c2")

        events = _collect(go, _CONTENT)
        assert _spend_ledger.current_spend("USD") == 100.0, "second instance reset the run total (per-instance bug)"
        violations = _by_type(events, "policy.violation")
        assert any(v["reason_code"] == "RUN_CEILING_EXCEEDED" for v in violations)

    def test_cumulative_threshold_blocks_second_charge_same_instance(self) -> None:
        adapter = _adapter(guardrails=AP2Guardrails(cumulative_threshold=100.0), capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        adapter.record_cart_mandate(make_cart(cart_id="c1", value=60.0), intent_mandate_id="m1")
        adapter.record_cart_mandate(make_cart(cart_id="c2", value=60.0), intent_mandate_id="m1")

        def go() -> None:
            adapter.record_payment_mandate(make_payment(payment_mandate_id="p1", value=60.0), cart_id="c1")
            with pytest.raises(PermissionError):
                adapter.record_payment_mandate(make_payment(payment_mandate_id="p2", value=60.0), cart_id="c2")

        events = _collect(go, _CONTENT)
        assert _spend_ledger.current_spend("USD") == 60.0
        violations = _by_type(events, "policy.violation")
        assert any(v["reason_code"] == "CUMULATIVE_THRESHOLD_EXCEEDED" for v in violations)


# ===========================================================================
# INVARIANT 4 — PER-RUN DOLLAR CEILING (A13): a run-scoped dollar ceiling trips a
# guard / emits policy.violation BEFORE the charge, distinct from MAX_EVENTS.
# BITE: in `check_ceiling`, drop the `projected > ledger.ceiling` test (return
# allowed) -> the over-ceiling charge accrues -> RED.
# ===========================================================================


class TestPerRunCeiling:
    def test_charge_over_ceiling_blocked_before_accrual(self) -> None:
        _ceiling(50.0)
        adapter = _adapter(capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        adapter.record_cart_mandate(make_cart(value=75.0), intent_mandate_id="m1")

        def go() -> None:
            with pytest.raises(PermissionError):
                adapter.record_payment_mandate(make_payment(value=75.0), cart_id="cart1")

        events = _collect(go, _CONTENT)
        assert _spend_ledger.current_spend("USD") == 0.0, "ceiling-breaching charge still accrued (checked AFTER?)"
        violations = _by_type(events, "policy.violation")
        assert violations and violations[0]["reason_code"] == "RUN_CEILING_EXCEEDED"
        # The block is a DOLLAR ceiling, not the MAX_EVENTS count cap.
        assert violations[0]["policy"] == "payment_guardrail"

    def test_ceiling_default_is_generous(self) -> None:
        """With no explicit ceiling the default is generous (a routine charge is
        allowed) — it is a backstop, not a default-deny that breaks integrators."""
        # do NOT configure a ceiling -> default DEFAULT_CEILING applies
        adapter = _adapter(capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        adapter.record_cart_mandate(make_cart(value=49.99), intent_mandate_id="m1")
        adapter.record_payment_mandate(make_payment(value=49.99), cart_id="cart1")
        assert _spend_ledger.current_spend("USD") == 49.99

    def test_primitive_check_ceiling_api(self) -> None:
        """The reusable primitive's API: check_ceiling is side-effect-free and
        add_spend accrues. (UCP will feed this same primitive.)"""
        _ceiling(100.0)
        v = _spend_ledger.check_ceiling(40.0, "USD")
        assert v.allowed and v.projected == 40.0
        assert _spend_ledger.current_spend("USD") == 0.0, "check_ceiling must not accrue"
        _spend_ledger.add_spend(40.0, "USD")
        over = _spend_ledger.check_ceiling(70.0, "USD")
        assert not over.allowed and over.reason_code == "RUN_CEILING_EXCEEDED"


# ===========================================================================
# INVARIANT 5 — EXPIRY GATES RECEIPT: an expired or never-signed mandate must NOT
# produce a receipt (gated on signed + unexpired, not the sign path alone).
# BITE: in `issue_receipt`, delete the `cart.expiry and clock > cart.expiry` test
# (or the `not cart.signed` test) -> a receipt is issued for an expired/unsigned
# mandate -> RED.
# ===========================================================================


class TestExpiryGatesReceipt:
    def test_expired_mandate_yields_no_receipt(self) -> None:
        adapter = _adapter(capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        # cart already expired (yesterday)
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        adapter.record_cart_mandate(make_cart(value=49.99, expiry=past), intent_mandate_id="m1")
        adapter.record_payment_mandate(make_payment(value=49.99), cart_id="cart1")

        def go() -> None:
            with pytest.raises(PermissionError):
                adapter.issue_receipt(cart_id="cart1")

        events = _collect(go, _CONTENT)
        assert not _by_type(events, "payment.receipt_issued"), "receipt issued for an EXPIRED mandate"
        violations = _by_type(events, "policy.violation")
        assert violations and violations[0]["reason_code"] == "MANDATE_EXPIRED"

    def test_never_signed_mandate_yields_no_receipt(self) -> None:
        adapter = _adapter(capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        adapter.record_cart_mandate(make_cart(value=49.99), intent_mandate_id="m1")
        # NOTE: no record_payment_mandate -> cart.signed stays False

        def go() -> None:
            with pytest.raises(PermissionError):
                adapter.issue_receipt(cart_id="cart1")

        events = _collect(go, _CONTENT)
        assert not _by_type(events, "payment.receipt_issued"), "receipt issued for a NEVER-SIGNED mandate"
        violations = _by_type(events, "policy.violation")
        assert violations and violations[0]["reason_code"] == "RECEIPT_FOR_UNSIGNED_MANDATE"

    def test_signed_unexpired_mandate_receipts(self) -> None:
        adapter = _adapter(capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        adapter.record_cart_mandate(make_cart(value=49.99), intent_mandate_id="m1")
        adapter.record_payment_mandate(make_payment(value=49.99), cart_id="cart1")
        events = _collect(lambda: adapter.issue_receipt(cart_id="cart1"), _CONTENT)
        issued = [p for p in _by_type(events, "payment.receipt_issued") if p.get("status") == "issued"]
        assert issued and issued[0]["cart_id"] == "cart1"


# ===========================================================================
# INVARIANT 6 — CURRENCY: amounts carry currency; comparisons never treat
# different currencies as equal.
# BITE: in `check_ceiling`, remove the `ccy != ledger.currency` branch (treat all
# currencies as the home currency) -> a ¥ charge is summed under a $ ceiling -> RED.
# ===========================================================================


class TestCurrency:
    def test_foreign_currency_charge_refused_under_home_ceiling(self) -> None:
        _ceiling(100.0, currency="USD")
        adapter = _adapter(capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        # cart total ¥5000 — cannot be proven under a $100 ceiling -> refused
        adapter.record_cart_mandate(make_cart(value=5000.0, currency="JPY"), intent_mandate_id="m1")

        def go() -> None:
            with pytest.raises(PermissionError):
                adapter.record_payment_mandate(make_payment(value=5000.0, currency="JPY"), cart_id="cart1")

        events = _collect(go, _CONTENT)
        assert _spend_ledger.current_spend("USD") == 0.0, "JPY charge folded into the USD home total"
        violations = _by_type(events, "policy.violation")
        assert violations and violations[0]["reason_code"] == "CURRENCY_MISMATCH"

    def test_currency_survives_redaction(self) -> None:
        """The currency CODE is metadata (not a sum) and survives no-content."""
        adapter = _adapter(guardrails=AP2Guardrails(max_transaction=1000.0), capture_config=_NO_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        events = _collect(
            lambda: adapter.record_cart_mandate(make_cart(value=49.99, currency="EUR"), intent_mandate_id="m1"),
            _NO_CONTENT,
        )
        accepted = [p for p in _by_type(events, "payment.cart_mandate") if p.get("status") == "accepted"]
        assert accepted and accepted[0]["currency"] == "EUR"

    def test_max_transaction_does_not_cross_currency(self) -> None:
        """A max_transaction in USD does not block a same-numeric JPY cart by
        comparing across currencies (¥80 is not blocked by max_transaction=100
        USD via the per-transaction clause — that clause is currency-guarded)."""
        adapter = _adapter(guardrails=AP2Guardrails(max_transaction=100.0, currency="USD"), capture_config=_CONTENT)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        # ¥8000 would exceed 100 *numerically* but is JPY, not the USD cap currency.
        events = _collect(
            lambda: adapter.record_cart_mandate(make_cart(value=8000.0, currency="JPY"), intent_mandate_id="m1"),
            _CONTENT,
        )
        # not blocked by the USD per-transaction clause (currency-guarded)
        accepted = [p for p in _by_type(events, "payment.cart_mandate") if p.get("status") == "accepted"]
        assert accepted, "USD max_transaction wrongly applied to a JPY cart total"


# ===========================================================================
# INTEGRATION SURFACE — the REAL MandateClient.create wrap chokepoint dispatches
# on the real mandate type. The client is a unit double; the MANDATES are real.
# ===========================================================================


class TestMandateClientWrap:
    def _wrapped(self, guardrails: AP2Guardrails | None = None) -> Tuple[AP2ProtocolAdapter, Any]:
        adapter = _adapter(guardrails=guardrails, capture_config=_CONTENT)
        # a stand-in MandateClient: create() just echoes a token. The adapter
        # observes payloads[0] (a REAL mandate) before calling through.
        client = SimpleNamespace(create=lambda payloads, issuer_key=None, sd=None: "sd-jwt-token")
        adapter.connect(target=client)
        return adapter, client

    def test_create_intent_then_cart_emits_via_wrap(self) -> None:
        adapter, client = self._wrapped(guardrails=AP2Guardrails(max_transaction=1000.0))

        def go() -> None:
            client.create(payloads=[make_intent(merchants=["ACME"])], issuer_key=object())
            client.create(payloads=[make_cart(value=49.99)], issuer_key=object())

        events = _collect(go, _CONTENT)
        assert _by_type(events, "payment.intent_mandate"), "intent not observed via MandateClient.create wrap"
        assert _by_type(events, "payment.cart_mandate"), "cart not observed via MandateClient.create wrap"

    def test_wrap_blocks_over_cap_cart_before_sdk_mints(self) -> None:
        """A blocked cart raises BEFORE the underlying create() runs — so a
        blocked charge never reaches the SDK / the network."""
        minted: List[Any] = []
        adapter = _adapter(guardrails=AP2Guardrails(max_transaction=100.0), capture_config=_CONTENT)
        client = SimpleNamespace(create=lambda payloads, issuer_key=None, sd=None: minted.append(payloads) or "tok")
        adapter.connect(target=client)
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")

        def go() -> None:
            with pytest.raises(PermissionError):
                client.create(payloads=[make_cart(value=1000.0)], issuer_key=object())

        _collect(go, _CONTENT)
        assert minted == [], "over-cap cart was minted by the SDK despite the guard"


# ===========================================================================
# REDACTION (ported from TestAP2Redaction to real shapes) — financial details
# are CONTENT; ids/status/reason_code/currency/signature-fingerprint survive.
# BITE: remove the payment.* entries from _CONTENT_KEYS -> amount/merchant leak.
# ===========================================================================


class TestAP2Redaction:
    def _drive_full_flow(self, adapter_cfg: CaptureConfig, collector_cfg: CaptureConfig) -> List[Dict[str, Any]]:
        adapter = _adapter(guardrails=AP2Guardrails(max_transaction=2000.0), capture_config=adapter_cfg)
        SECRET_MERCHANT = "SENTINEL-ACME-Corp"

        def go() -> None:
            adapter.record_intent_mandate(
                make_intent(merchants=[SECRET_MERCHANT], description="SENTINEL-intent-text"), mandate_id="m1"
            )
            adapter.record_cart_mandate(make_cart(value=1234.56, merchant=SECRET_MERCHANT), intent_mandate_id="m1")
            adapter.record_payment_mandate(make_payment(value=1234.56), cart_id="cart1")
            adapter.issue_receipt(cart_id="cart1")

        return _collect(go, collector_cfg)

    def test_content_present_by_default_when_capture_on(self) -> None:
        import json

        events = self._drive_full_flow(_CONTENT, _CONTENT)
        blob = json.dumps([e["payload"] for e in events], default=str)
        assert "SENTINEL-ACME-Corp" in blob and "1234.56" in blob

    def test_adapter_side_no_content_strips_amount_and_merchant(self) -> None:
        import json

        events = self._drive_full_flow(_NO_CONTENT, _CONTENT)
        blob = json.dumps([e["payload"] for e in events], default=str)
        assert "SENTINEL-ACME-Corp" not in blob, "merchant leaked despite adapter capture_content=False"
        assert "1234.56" not in blob, "binding amount leaked despite adapter capture_content=False"
        # observability not blinded: ids + currency + signature presence/fp survive
        carts = _by_type(events, "payment.cart_mandate")
        assert carts and carts[0]["cart_id"] == "cart1"
        assert carts[0]["currency"] == "USD"
        assert carts[0]["merchant_signature_present"] is True
        assert carts[0]["merchant_signature_fp"].startswith("sha256:")

    def test_collector_side_no_content_strips_amount_and_merchant(self) -> None:
        import json

        events = self._drive_full_flow(_CONTENT, _NO_CONTENT)
        blob = json.dumps([e["payload"] for e in events], default=str)
        assert "SENTINEL-ACME-Corp" not in blob, "collector backstop missed merchant"
        assert "1234.56" not in blob, "collector backstop missed binding amount"

    def test_raw_merchant_signature_never_placed_in_payload(self) -> None:
        """The ADAPTER never puts the raw merchant_authorization into a payload —
        only PRESENCE + a keyed-HMAC fingerprint. We assert on the adapter's emit
        BEFORE the collector's secret-scrubber runs, using a NON-JWT-shaped
        signature so the scrubber's `eyJ...` JWT pattern can't be what hides it —
        isolating the adapter's own non-emission. (The collector JWT scrub is a
        defense-in-depth backstop on top, proven separately.)
        BITE: emit the raw `merchant_authorization` in record_cart_mandate -> RED."""
        import json

        # deliberately NOT JWT-shaped so the secret-scrubber would NOT catch it.
        raw_sig = "MERCHANT-SIG-RAW-not-a-jwt-0xDEADBEEFCAFEBABE"
        captured: List[Dict[str, Any]] = []
        adapter = _adapter(guardrails=AP2Guardrails(max_transaction=1000.0), capture_config=_CONTENT)
        # intercept the adapter's own emit (pre-collector, pre-scrub)
        orig_emit = adapter.emit

        def _spy(event_name: str, payload: Dict[str, Any], **kw: Any) -> None:
            captured.append({"event_type": event_name, "payload": payload})
            return orig_emit(event_name, payload, **kw)

        adapter.emit = _spy  # type: ignore[method-assign]
        adapter.record_intent_mandate(make_intent(merchants=["ACME"]), mandate_id="m1")
        _collect(
            lambda: adapter.record_cart_mandate(make_cart(value=49.99, signature=raw_sig), intent_mandate_id="m1"),
            _CONTENT,
        )
        blob = json.dumps(captured, default=str)
        assert raw_sig not in blob, "ADAPTER placed the raw merchant signature into a payload"
        carts = [e["payload"] for e in captured if e["event_type"] == "payment.cart_mandate"]
        assert carts and "merchant_authorization" not in carts[0], "raw merchant_authorization key emitted"
        assert carts[0]["merchant_signature_present"] is True
        assert carts[0]["merchant_signature_fp"].startswith("sha256:")

    def test_fingerprint_is_keyed_not_plain_sha(self) -> None:
        """Two adapter instances produce DIFFERENT fingerprints for the same
        signature (keyed HMAC, not a plain reversible SHA-256)."""
        sig = "eyJ.same-sig.zzz"
        a1 = _adapter(capture_config=_CONTENT)
        a2 = _adapter(capture_config=_CONTENT)
        fp1 = a1._fingerprint(sig)
        fp2 = a2._fingerprint(sig)
        assert fp1 != fp2, "fingerprint is an unkeyed SHA (brute-forceable) — must be keyed HMAC"
        assert fp1 == a1._fingerprint(sig), "fingerprint not stable within one instance"
