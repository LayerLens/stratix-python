"""AP2 (Agent Payments Protocol) adapter — real v0.2.0 mandate chain.

Instruments the real Agent Payments Protocol three-mandate lifecycle:

    IntentMandate  →  CartMandate  →  PaymentMandate  →  receipt

against the installed, version-pinned ``ap2`` SDK (google-agentic-commerce/AP2
@ v0.2.0). The real public surface is :class:`ap2.sdk.mandate.MandateClient`,
whose ``create(payloads, issuer_key, sd=None)`` / ``present(...)`` mint an
SD-JWT for ``payloads[0]`` — a real pydantic mandate object
(:class:`ap2.models.mandate.IntentMandate` / ``CartMandate`` / ``PaymentMandate``).
That ``create``/``present`` call is the single chokepoint where every mandate is
minted, so we wrap it and dispatch on the type of ``payloads[0]``. Callers who
observe mandate objects without minting can drive the same telemetry + guards via
``record_intent_mandate`` / ``record_cart_mandate`` / ``record_payment_mandate``
/ ``issue_receipt``.

We do NOT re-verify the SD-JWT cryptography (that is the SDK's job and the
network/issuer's). We OBSERVE the mandate flow, emit redaction-safe telemetry,
and enforce the payment-autonomy guardrails (A12 charge-safety + A13 ceiling):

* **Cart-binding amount.** The guardrail reads the BINDING cart total
  ``CartMandate.contents.payment_request.details.total.amount.value`` (+ its
  currency) — NOT the non-binding intent amount. A cart total over the intent's
  declared max / a guardrail limit / the per-run ceiling is BLOCKED before any
  charge accrues. This is the central AP2 v0.2 invariant (the "cart-swap").
* **Idempotency.** Re-observing the SAME cart (``cart_id``) / payment mandate
  (``payment_mandate_id``) accrues spend ONCE; a replay emits an idempotent
  marker, never a second charge.
* **Run-scoped cumulative cap + per-run ceiling.** Cumulative spend lives in a
  run-scoped ledger (:mod:`layerlens.instrument._spend_ledger`), shared across
  adapter instances in one run; a charge that would breach the per-run dollar
  ceiling is refused before it accrues.
* **Expiry gates the receipt.** A receipt is issued only for a signed,
  unexpired mandate (per ``cart_expiry`` / ``intent_expiry`` ISO-8601) — not the
  sign path alone.
* **Currency.** Amounts carry an ISO-4217 currency; comparisons never treat
  different currencies as equal (the ledger fails closed on a mismatch).

Privacy: the merchant signature (``CartMandate.merchant_authorization``, a JWT)
is NEVER emitted — only its PRESENCE and a redaction-surviving keyed-HMAC
fingerprint (the a2ui.py pattern), so provenance stays auditable under
``capture_content=False``.
"""

from __future__ import annotations

import hmac
import uuid
import hashlib
import logging
import secrets
from typing import Any, Set, Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import field, dataclass

from ..._events import (
    POLICY_VIOLATION,
    PAYMENT_CART_MANDATE,
    PAYMENT_INTENT_MANDATE,
    PAYMENT_MANDATE_SIGNED,
    PAYMENT_RECEIPT_ISSUED,
)
from ._base_protocol import BaseProtocolAdapter
from ..._spend_ledger import add_spend, check_ceiling, current_spend

log = logging.getLogger(__name__)


@dataclass
class AP2Guardrails:
    """Caller-declared budget controls, layered ON TOP of the per-run ceiling.

    ``max_transaction`` caps a single binding cart total; ``merchant_whitelist``
    constrains the merchant; ``cumulative_threshold`` is a soft per-RUN cap that
    feeds the same run-scoped ledger as the hard ceiling. All amounts are in the
    guardrail's ``currency`` (default USD) — a cart in a different currency is
    handled conservatively by the ledger (never summed across currencies).
    """

    max_transaction: float | None = None
    merchant_whitelist: List[str] = field(default_factory=list)
    cumulative_threshold: float | None = None
    currency: str = "USD"


class _BlockedCharge(PermissionError):
    """Raised when a guardrail blocks a mandate before any charge accrues."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


@dataclass
class _IntentRecord:
    """What we remember about an observed IntentMandate, to link + bound a cart."""

    mandate_id: str
    max_amount: float | None
    currency: str | None
    merchants: List[str]
    expiry: Optional[datetime]
    confirmation_required: bool


@dataclass
class _CartRecord:
    """A cart we have accepted (passed guardrails) — the receipt gate reads this."""

    cart_id: str
    amount: float
    currency: str
    merchant_name: str
    expiry: Optional[datetime]
    intent_mandate_id: Optional[str]
    signed: bool = False


class AP2ProtocolAdapter(BaseProtocolAdapter):
    PROTOCOL = "ap2"
    PROTOCOL_VERSION = "0.2.0"

    def __init__(self, guardrails: AP2Guardrails | None = None, *, capture_config: Any = None) -> None:
        super().__init__(capture_config=capture_config)
        self._guardrails = guardrails or AP2Guardrails()
        # Per-instance HMAC key: the merchant-signature fingerprint is a keyed
        # HMAC (a2ui.py P3) so a low-entropy JWT can't be brute-forced from the
        # emitted fingerprint, and the key is never emitted.
        self._hash_key = secrets.token_bytes(32)
        # Linkage + idempotency state. NOTE: cumulative spend is NOT here — it
        # lives in the run-scoped ledger so two adapter instances in one run
        # share one total (A12). These maps are per-instance LINKAGE only.
        self._intents: Dict[str, _IntentRecord] = {}
        self._carts: Dict[str, _CartRecord] = {}
        self._charged_carts: Set[str] = set()  # idempotency: cart ids already accrued
        self._charged_payments: Set[str] = set()  # idempotency: payment_mandate_ids accrued
        self._receipted: Set[str] = set()  # idempotency: cart/mandate ids already receipted

    # ── connection: wrap the real MandateClient.create / present chokepoint ──

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        """Instrument a ``MandateClient`` (or any object exposing ``create`` /
        ``present``). Each call mints an SD-JWT for a real mandate object in
        ``payloads[0]`` — we observe it, guard it, and emit before calling
        through (so a blocked charge never reaches the SDK / the network)."""
        self._client = target
        for method in ("create", "present"):
            if hasattr(target, method):
                orig = getattr(target, method)
                self._originals[method] = orig
                setattr(target, method, self._wrap(orig))
        return target

    def _wrap(self, original: Any) -> Any:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            payloads = kwargs.get("payloads")
            if payloads is None and args:
                payloads = args[0]
            mandate = payloads[0] if isinstance(payloads, (list, tuple)) and payloads else None
            if mandate is not None:
                # observe + guard BEFORE the SDK mints anything; a blocked charge
                # raises and the SDK call is never made.
                adapter._observe_mandate(mandate)
            return original(*args, **kwargs)

        return wrapped

    # ── dispatch on the real mandate type (duck-typed; no hard ap2 import) ──

    def _observe_mandate(self, mandate: Any) -> None:
        kind = type(mandate).__name__
        if kind == "IntentMandate":
            self.record_intent_mandate(mandate)
        elif kind == "CartMandate":
            self.record_cart_mandate(mandate)
        elif kind == "PaymentMandate":
            self.record_payment_mandate(mandate)
        # other payload types (CheckoutMandate, etc.) flow through untouched

    # ── IntentMandate: non-binding desire + hard constraints (whitelist/max) ──

    def record_intent_mandate(self, intent: Any, *, mandate_id: str | None = None) -> str:
        """Observe an :class:`ap2.models.mandate.IntentMandate`.

        The intent is the user's non-binding desire + constraints (allowed
        merchants, expiry, confirmation policy). It carries NO binding amount in
        AP2 v0.2, so it never accrues spend — it only records constraints a later
        cart is checked against. ``max_amount``/``currency`` may be supplied by an
        integrator via attributes if their intent variant carries them.
        """
        mid = mandate_id or _attr(intent, "mandate_id") or uuid.uuid4().hex[:16]
        merchants = _attr(intent, "merchants") or []
        record = _IntentRecord(
            mandate_id=mid,
            max_amount=_as_float(_attr(intent, "max_amount")),
            currency=_attr(intent, "currency"),
            merchants=list(merchants) if isinstance(merchants, (list, tuple)) else [],
            expiry=_parse_iso(_attr(intent, "intent_expiry")),
            confirmation_required=bool(_attr(intent, "user_cart_confirmation_required", default=True)),
        )
        self._intents[mid] = record
        self.emit(
            PAYMENT_INTENT_MANDATE,
            {
                "mandate_id": mid,
                "status": "created",
                "confirmation_required": record.confirmation_required,
                "intent_expiry": _attr(intent, "intent_expiry"),
                # content (stripped under no-content): merchant whitelist + intent text
                "merchants": record.merchants,
                "description": _attr(intent, "natural_language_description"),
                # an integrator-supplied max (rare) is a financial detail -> content
                "amount": record.max_amount,
                "currency": record.currency,
            },
        )
        return mid

    # ── CartMandate: the BINDING price the merchant signed (guard reads THIS) ──

    def record_cart_mandate(self, cart: Any, *, intent_mandate_id: str | None = None) -> _CartRecord:
        """Observe an :class:`ap2.models.mandate.CartMandate` and guard the
        BINDING total. This is the cart-swap-defeating invariant: a cart whose
        total exceeds the intent max / guardrail / per-run ceiling is BLOCKED."""
        contents = _attr(cart, "contents")
        cart_id = _attr(contents, "id") or uuid.uuid4().hex[:16]
        merchant_name = _attr(contents, "merchant_name")
        expiry = _parse_iso(_attr(contents, "cart_expiry"))
        amount, currency = _binding_total(contents)
        sig = _attr(cart, "merchant_authorization")
        intent = self._intents.get(intent_mandate_id) if intent_mandate_id else None

        base_payload: Dict[str, Any] = {
            "cart_id": cart_id,
            "currency": currency,
            "confirmation_required": bool(_attr(contents, "user_cart_confirmation_required", default=True)),
            "cart_expiry": _attr(contents, "cart_expiry"),
            "intent_mandate_id": intent_mandate_id,
            # merchant-signature PRESENCE + keyed-HMAC fingerprint (provenance,
            # auditable under no-content) — NEVER the raw JWT.
            "merchant_signature_present": sig is not None,
            "merchant_signature_fp": self._fingerprint(sig) if sig is not None else None,
            # content (stripped under no-content):
            "merchant_name": merchant_name,
            "amount": amount,
        }

        try:
            self._guard_cart(amount, currency, merchant_name, intent)
        except _BlockedCharge as blocked:
            self._emit_violation(
                stage="cart_mandate",
                cart_id=cart_id,
                reason_code=blocked.reason_code,
                detail=blocked.detail,
                currency=currency,
                amount=amount,
                merchant_name=merchant_name,
            )
            self.emit(PAYMENT_CART_MANDATE, {**base_payload, "status": "blocked", "reason_code": blocked.reason_code})
            raise

        record = _CartRecord(
            cart_id=cart_id,
            amount=amount,
            currency=currency,
            merchant_name=merchant_name or "",
            expiry=expiry,
            intent_mandate_id=intent_mandate_id,
        )
        self._carts[cart_id] = record
        self.emit(PAYMENT_CART_MANDATE, {**base_payload, "status": "accepted"})
        return record

    # ── PaymentMandate: the user authorizes payment -> the actual charge ──

    def record_payment_mandate(self, payment: Any, *, cart_id: str | None = None) -> None:
        """Observe an :class:`ap2.models.mandate.PaymentMandate`.

        This is where money moves: it accrues spend (ONCE per payment mandate /
        cart — idempotent) into the run-scoped ledger and emits
        ``payment.mandate_signed``. The charge amount is the cart's binding total
        when we can link the cart, else the payment mandate's own total."""
        contents = _attr(payment, "payment_mandate_contents")
        payment_mandate_id = _attr(contents, "payment_mandate_id") or uuid.uuid4().hex[:16]
        user_auth = _attr(payment, "user_authorization")
        # The merchant the payment is bound to (PaymentMandateContents.merchant_agent).
        # An identifier, not content — survives redaction like task_id (S15/F8).
        merchant_agent = _attr(contents, "merchant_agent")

        cart = self._carts.get(cart_id) if cart_id else None
        if cart is not None:
            amount, currency = cart.amount, cart.currency
        else:
            amount, currency = _payment_total(contents)

        dedup_key = payment_mandate_id if cart is None else cart.cart_id
        already = payment_mandate_id in self._charged_payments or (
            cart is not None and cart.cart_id in self._charged_carts
        )
        if already:
            # Idempotent replay: do NOT accrue or double-count; emit a marker.
            self.emit(
                PAYMENT_MANDATE_SIGNED,
                {
                    "payment_mandate_id": payment_mandate_id,
                    "cart_id": cart.cart_id if cart else None,
                    "status": "replayed",
                    "currency": currency,
                    "user_authorization_present": user_auth is not None,
                    "user_authorization_fp": self._fingerprint(user_auth) if user_auth is not None else None,
                },
            )
            return

        # Per-run ceiling check BEFORE accrual (A13).
        verdict = check_ceiling(amount, currency)
        cumulative_threshold = self._guardrails.cumulative_threshold
        if not verdict.allowed:
            detail = (
                f"charge {amount} {currency} would bring run spend to {verdict.projected} "
                f"(ceiling {verdict.ceiling} {currency})"
            )
            self._emit_violation(
                stage="payment_mandate",
                cart_id=cart.cart_id if cart else None,
                reason_code=verdict.reason_code or "RUN_CEILING_EXCEEDED",
                detail=detail,
                currency=currency,
                amount=amount,
                merchant_name=cart.merchant_name if cart else None,
            )
            self.emit(
                PAYMENT_MANDATE_SIGNED,
                {
                    "payment_mandate_id": payment_mandate_id,
                    "cart_id": cart.cart_id if cart else None,
                    "status": "blocked",
                    "reason_code": verdict.reason_code,
                    "currency": currency,
                },
            )
            raise _BlockedCharge(verdict.reason_code or "RUN_CEILING_EXCEEDED", detail)

        # Soft per-run cumulative threshold (same run-scoped total, home ccy only).
        if cumulative_threshold is not None and currency == self._guardrails.currency:
            projected = current_spend(currency) + amount
            if projected > cumulative_threshold:
                detail = f"cumulative spend {projected} {currency} would exceed threshold {cumulative_threshold}"
                self._emit_violation(
                    stage="payment_mandate",
                    cart_id=cart.cart_id if cart else None,
                    reason_code="CUMULATIVE_THRESHOLD_EXCEEDED",
                    detail=detail,
                    currency=currency,
                    amount=amount,
                    merchant_name=cart.merchant_name if cart else None,
                )
                self.emit(
                    PAYMENT_MANDATE_SIGNED,
                    {
                        "payment_mandate_id": payment_mandate_id,
                        "cart_id": cart.cart_id if cart else None,
                        "status": "blocked",
                        "reason_code": "CUMULATIVE_THRESHOLD_EXCEEDED",
                        "currency": currency,
                    },
                )
                raise _BlockedCharge("CUMULATIVE_THRESHOLD_EXCEEDED", detail)

        # Allowed: accrue ONCE into the run-scoped ledger and mark consumed.
        new_total = add_spend(amount, currency)
        self._charged_payments.add(payment_mandate_id)
        if cart is not None:
            self._charged_carts.add(cart.cart_id)
            cart.signed = True
        signed_payload: Dict[str, Any] = {
            "payment_mandate_id": payment_mandate_id,
            "cart_id": cart.cart_id if cart else None,
            "status": "signed",
            "currency": currency,
            "user_authorization_present": user_auth is not None,
            "user_authorization_fp": self._fingerprint(user_auth) if user_auth is not None else None,
            # content (stripped under no-content):
            "amount": amount,
            "cumulative_spend": new_total,
        }
        if merchant_agent is not None:
            signed_payload["merchant_agent"] = str(merchant_agent)
        self.emit(PAYMENT_MANDATE_SIGNED, signed_payload)

    # ── receipt: gated on signed + unexpired (NOT the sign path alone) ──

    def issue_receipt(
        self,
        *,
        cart_id: str,
        receipt_id: str | None = None,
        now: datetime | None = None,
    ) -> str:
        """Issue a receipt ONLY for a signed, unexpired, not-already-receipted
        cart. An expired or never-signed mandate produces a ``policy.violation``
        and NO ``payment.receipt_issued`` (A12 — expiry gates the receipt, not
        only the sign path)."""
        rid = receipt_id or uuid.uuid4().hex[:16]
        cart = self._carts.get(cart_id)
        clock = now or datetime.now(timezone.utc)

        if cart is None or not cart.signed:
            self._emit_violation(
                stage="receipt",
                cart_id=cart_id,
                reason_code="RECEIPT_FOR_UNSIGNED_MANDATE",
                detail=f"no signed mandate for cart {cart_id}; receipt refused",
                currency=cart.currency if cart else None,
                amount=cart.amount if cart else None,
                merchant_name=cart.merchant_name if cart else None,
            )
            raise _BlockedCharge("RECEIPT_FOR_UNSIGNED_MANDATE", f"cart {cart_id} not signed")

        if cart.expiry is not None and clock > cart.expiry:
            self._emit_violation(
                stage="receipt",
                cart_id=cart_id,
                reason_code="MANDATE_EXPIRED",
                detail=f"cart {cart_id} expired at {cart.expiry.isoformat()}; receipt refused",
                currency=cart.currency,
                amount=cart.amount,
                merchant_name=cart.merchant_name,
            )
            raise _BlockedCharge("MANDATE_EXPIRED", f"cart {cart_id} expired")

        if cart_id in self._receipted:
            self.emit(
                PAYMENT_RECEIPT_ISSUED,
                {"receipt_id": rid, "cart_id": cart_id, "status": "replayed", "currency": cart.currency},
            )
            return rid

        self._receipted.add(cart_id)
        self.emit(
            PAYMENT_RECEIPT_ISSUED,
            {
                "receipt_id": rid,
                "cart_id": cart_id,
                "status": "issued",
                "currency": cart.currency,
                # content (stripped under no-content):
                "amount": cart.amount,
                "merchant_name": cart.merchant_name,
            },
        )
        return rid

    # ── guard helpers ──

    def _guard_cart(
        self,
        amount: float,
        currency: str,
        merchant_name: str | None,
        intent: _IntentRecord | None,
    ) -> None:
        """Block a cart whose BINDING total / merchant violates a constraint.

        Reads the binding cart total — never the intent amount — so a cart-swap
        (a cart total != the approved intent) cannot slip past. Raises
        :class:`_BlockedCharge` (a PermissionError) on a violation."""
        g = self._guardrails

        # 1. Merchant whitelist: intent's allowed-merchant list AND/OR the caller
        #    guardrail. An off-whitelist merchant is blocked.
        whitelist = list(g.merchant_whitelist)
        if intent is not None and intent.merchants:
            # the union must be satisfied by BOTH where both are set; here we
            # treat the intent whitelist as authoritative if present.
            whitelist = intent.merchants if not whitelist else whitelist
        if whitelist and merchant_name is not None and merchant_name not in whitelist:
            raise _BlockedCharge("MERCHANT_NOT_WHITELISTED", f"merchant {merchant_name!r} not in whitelist")

        # 2. Intent max (if the intent declared one) — binding cart over the
        #    user's approved max is blocked (cart-swap / overcharge).
        if intent is not None and intent.max_amount is not None:
            if _same_currency(currency, intent.currency, g.currency) and amount > intent.max_amount:
                raise _BlockedCharge(
                    "CART_EXCEEDS_INTENT_MAX",
                    f"cart total {amount} {currency} exceeds intent max {intent.max_amount}",
                )

        # 3. Per-transaction guardrail on the BINDING cart total.
        if g.max_transaction is not None and currency == g.currency and amount > g.max_transaction:
            raise _BlockedCharge(
                "MAX_TRANSACTION_EXCEEDED",
                f"cart total {amount} {currency} exceeds max_transaction {g.max_transaction}",
            )

    def _emit_violation(
        self,
        *,
        stage: str,
        cart_id: str | None,
        reason_code: str,
        detail: str,
        currency: str | None,
        amount: float | None,
        merchant_name: str | None,
    ) -> None:
        """Emit a ``policy.violation`` for a blocked charge. ``reason_code`` +
        ``status`` + ids + currency survive redaction; the free-text ``detail``
        (which interpolates amount/merchant) is stripped under no-content."""
        self.emit(
            POLICY_VIOLATION,
            {
                "policy": "payment_guardrail",
                "stage": stage,
                "status": "blocked",
                "reason_code": reason_code,
                "cart_id": cart_id,
                "currency": currency,
                # content (stripped under no-content):
                "detail": detail,
                "amount": amount,
                "merchant_name": merchant_name,
            },
        )

    def _fingerprint(self, value: Any) -> str:
        """Keyed-HMAC fingerprint of a signature/JWT (a2ui.py P3 pattern). One-way
        without the per-instance key, which is never emitted; the raw signature is
        NEVER emitted (it is also JWT-shaped and would be secret-scrubbed)."""
        return "sha256:" + hmac.new(self._hash_key, str(value).encode(), hashlib.sha256).hexdigest()


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Duck-typed attribute/key read (works on a pydantic model or a dict)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _binding_total(contents: Any) -> tuple[float, str]:
    """Read the BINDING cart total: contents.payment_request.details.total.amount.

    The single authoritative amount a guardrail must read in AP2 v0.2. Falls back
    to 0.0/"?" only if the shape is missing (a malformed cart is treated as a
    zero-value unknown-currency charge, which the conservative currency policy
    will not silently wave under a home-currency ceiling)."""
    pr = _attr(contents, "payment_request")
    details = _attr(pr, "details")
    total = _attr(details, "total")
    amount_obj = _attr(total, "amount")
    value = _as_float(_attr(amount_obj, "value"))
    currency = _attr(amount_obj, "currency")
    return (value if value is not None else 0.0, str(currency).upper() if currency else "?")


def _payment_total(contents: Any) -> tuple[float, str]:
    """Read a PaymentMandateContents total (payment_details_total.amount)."""
    total = _attr(contents, "payment_details_total")
    amount_obj = _attr(total, "amount")
    value = _as_float(_attr(amount_obj, "value"))
    currency = _attr(amount_obj, "currency")
    return (value if value is not None else 0.0, str(currency).upper() if currency else "?")


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 expiry into a tz-aware datetime (UTC if naive)."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _same_currency(a: str | None, *others: str | None) -> bool:
    """True iff every set currency among the args matches (case-insensitive)."""
    seen = {str(c).upper() for c in (a, *others) if c}
    return len(seen) <= 1


def instrument_ap2(target: Any, guardrails: AP2Guardrails | None = None) -> AP2ProtocolAdapter:
    from .._registry import get, register

    existing = get("ap2")
    if existing is not None:
        existing.disconnect()
    adapter = AP2ProtocolAdapter(guardrails=guardrails)
    adapter.connect(target)
    register("ap2", adapter)
    return adapter


def uninstrument_ap2() -> None:
    from .._registry import unregister

    unregister("ap2")
