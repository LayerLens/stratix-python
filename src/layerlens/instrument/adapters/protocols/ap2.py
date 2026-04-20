"""AP2 (Agent Payments Protocol) adapter.

Instruments the three-stage mandate chain:

    Intent Mandate → Payment Mandate → Payment Receipt

Also exposes a simple guardrail evaluator for caller-declared budget controls
(per-transaction limit, merchant whitelist, cumulative threshold, expiry).
The evaluator emits ``payment.mandate_signed`` on success or an error event
describing the blocked transaction.
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, List
from dataclasses import field, dataclass

from ..._events import (
    PAYMENT_INTENT_MANDATE,
    PAYMENT_MANDATE_SIGNED,
    PAYMENT_RECEIPT_ISSUED,
)
from ._base_protocol import BaseProtocolAdapter

log = logging.getLogger(__name__)


@dataclass
class AP2Guardrails:
    max_transaction: float | None = None
    merchant_whitelist: List[str] = field(default_factory=list)
    cumulative_threshold: float | None = None
    mandate_ttl_seconds: float | None = None


class AP2ProtocolAdapter(BaseProtocolAdapter):
    PROTOCOL = "ap2"
    PROTOCOL_VERSION = "0.1.0"

    def __init__(self, guardrails: AP2Guardrails | None = None) -> None:
        super().__init__()
        self._guardrails = guardrails or AP2Guardrails()
        self._cumulative_spend: float = 0.0
        self._intent_mandates: Dict[str, Dict[str, Any]] = {}

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target
        for method in ("create_intent_mandate", "sign_payment_mandate", "issue_receipt"):
            if hasattr(target, method):
                orig = getattr(target, method)
                self._originals[method] = orig
                setattr(target, method, self._wrap(orig, method))
        return target

    def _wrap(self, original: Any, method: str) -> Any:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if method == "create_intent_mandate":
                return adapter._handle_intent(original, args, kwargs)
            if method == "sign_payment_mandate":
                return adapter._handle_sign(original, args, kwargs)
            if method == "issue_receipt":
                return adapter._handle_receipt(original, args, kwargs)
            return original(*args, **kwargs)

        return wrapped

    # --- mandate handlers ---

    def _handle_intent(self, original: Any, args: Any, kwargs: Any) -> Any:
        mandate_id = kwargs.get("mandate_id") or uuid.uuid4().hex[:16]
        payload = {
            "mandate_id": mandate_id,
            "amount": kwargs.get("amount"),
            "merchant": kwargs.get("merchant"),
            "expires_at": kwargs.get("expires_at"),
            "ttl_seconds": self._guardrails.mandate_ttl_seconds,
        }
        self._intent_mandates[mandate_id] = {
            "created_at": time.time(),
            "amount": kwargs.get("amount") or 0,
            "merchant": kwargs.get("merchant"),
        }
        self.emit(PAYMENT_INTENT_MANDATE, payload)
        return original(*args, **kwargs)

    def _handle_sign(self, original: Any, args: Any, kwargs: Any) -> Any:
        mandate_id = kwargs.get("mandate_id") or (args[0] if args else None)
        verdict = self._evaluate_guardrails(mandate_id, kwargs)
        if verdict is not None:
            self.emit(
                PAYMENT_MANDATE_SIGNED,
                {"mandate_id": mandate_id, "status": "blocked", "reason": verdict},
            )
            raise PermissionError(f"AP2 guardrail blocked mandate {mandate_id}: {verdict}")
        result = original(*args, **kwargs)
        amount = kwargs.get("amount") or self._intent_mandates.get(mandate_id, {}).get("amount") or 0
        self._cumulative_spend += float(amount or 0)
        self.emit(
            PAYMENT_MANDATE_SIGNED,
            {
                "mandate_id": mandate_id,
                "status": "signed",
                "amount": amount,
                "cumulative_spend": self._cumulative_spend,
            },
        )
        return result

    def _handle_receipt(self, original: Any, args: Any, kwargs: Any) -> Any:
        receipt_id = kwargs.get("receipt_id") or uuid.uuid4().hex[:16]
        result = original(*args, **kwargs)
        self.emit(
            PAYMENT_RECEIPT_ISSUED,
            {
                "receipt_id": receipt_id,
                "mandate_id": kwargs.get("mandate_id"),
                "amount": kwargs.get("amount"),
                "merchant": kwargs.get("merchant"),
            },
        )
        return result

    # --- guardrail evaluator ---

    def _evaluate_guardrails(self, mandate_id: str | None, kwargs: Dict[str, Any]) -> str | None:
        g = self._guardrails
        amount = float(kwargs.get("amount") or 0)
        merchant = kwargs.get("merchant")

        if g.max_transaction is not None and amount > g.max_transaction:
            return f"amount {amount} exceeds max_transaction {g.max_transaction}"
        if g.merchant_whitelist and merchant is not None and merchant not in g.merchant_whitelist:
            return f"merchant {merchant!r} not in whitelist"
        if g.cumulative_threshold is not None and (self._cumulative_spend + amount) > g.cumulative_threshold:
            return f"cumulative spend {self._cumulative_spend + amount} would exceed threshold {g.cumulative_threshold}"
        if g.mandate_ttl_seconds is not None and mandate_id in self._intent_mandates:
            age = time.time() - self._intent_mandates[mandate_id]["created_at"]
            if age > g.mandate_ttl_seconds:
                return f"mandate age {age:.1f}s exceeds ttl {g.mandate_ttl_seconds}s"
        return None


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
