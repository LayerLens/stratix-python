"""UCP (Universal Commerce Protocol) adapter.

Instruments the high-level commerce flow: supplier discovery, catalog browse,
checkout sessions, and refunds. Session duration is tracked from session
start → completion and reported in ``commerce.checkout_completed``.
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict

from ..._events import (
    POLICY_VIOLATION,
    COMMERCE_REFUND_ISSUED,
    COMMERCE_CATALOG_BROWSED,
    COMMERCE_CHECKOUT_STARTED,
    COMMERCE_CHECKOUT_COMPLETED,
    COMMERCE_SUPPLIER_DISCOVERED,
)
from ._base_protocol import BaseProtocolAdapter
from ..._spend_ledger import add_spend, check_ceiling

log = logging.getLogger(__name__)


class _CheckoutBlocked(PermissionError):
    """Raised when the per-run dollar ceiling blocks a checkout before the charge."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


class UCPProtocolAdapter(BaseProtocolAdapter):
    PROTOCOL = "ucp"
    PROTOCOL_VERSION = "0.1.0"

    def __init__(self, *, capture_config: Any = None) -> None:
        super().__init__(capture_config=capture_config)
        self._sessions: Dict[str, float] = {}
        self._known_suppliers: Dict[str, Dict[str, Any]] = {}

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target
        for method, handler in (
            ("discover_suppliers", self._on_discover),
            ("browse_catalog", self._on_browse),
            ("start_checkout", self._on_start_checkout),
            ("complete_checkout", self._on_complete_checkout),
            ("issue_refund", self._on_refund),
        ):
            if hasattr(target, method):
                orig = getattr(target, method)
                self._originals[method] = orig
                setattr(target, method, handler(orig))
        return target

    # --- hooks ---

    def _on_discover(self, original: Any) -> Any:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            suppliers = result if isinstance(result, list) else getattr(result, "suppliers", None) or []
            for supplier in suppliers:
                supplier_id = getattr(supplier, "id", None) or (
                    supplier.get("id") if isinstance(supplier, dict) else None
                )
                if supplier_id is None:
                    continue
                if supplier_id not in adapter._known_suppliers:
                    adapter._known_suppliers[supplier_id] = {"discovered_at": time.time()}
                adapter.emit(
                    COMMERCE_SUPPLIER_DISCOVERED,
                    {
                        "supplier_id": supplier_id,
                        "name": getattr(supplier, "name", None)
                        or (supplier.get("name") if isinstance(supplier, dict) else None),
                    },
                )
            return result

        return wrapped

    def _on_browse(self, original: Any) -> Any:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            items = result if isinstance(result, list) else getattr(result, "items", None) or []
            adapter.emit(
                COMMERCE_CATALOG_BROWSED,
                {
                    "supplier_id": kwargs.get("supplier_id"),
                    "query": kwargs.get("query"),
                    "item_count": len(items),
                },
            )
            return result

        return wrapped

    def _on_start_checkout(self, original: Any) -> Any:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            session_id = kwargs.get("session_id") or uuid.uuid4().hex[:16]
            adapter._sessions[session_id] = time.time()
            kwargs.setdefault("session_id", session_id)
            adapter.emit(
                COMMERCE_CHECKOUT_STARTED,
                {"session_id": session_id, "supplier_id": kwargs.get("supplier_id")},
            )
            return original(*args, **kwargs)

        return wrapped

    def _on_complete_checkout(self, original: Any) -> Any:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            session_id = kwargs.get("session_id") or (args[0] if args else None)
            start = adapter._sessions.pop(session_id, time.time())
            amount = _as_float(kwargs.get("amount"))
            currency = kwargs.get("currency")

            # A13 — per-run dollar ceiling: aggregate UCP checkout spend into the
            # SAME run-scoped ledger AP2 feeds, so one run's ceiling spans AP2 +
            # UCP. Check BEFORE the charge; a checkout that would breach the
            # ceiling is BLOCKED (the real UCP client is never called, so no
            # money moves) and a policy.violation is emitted. This is distinct
            # from the collector's MAX_EVENTS count cap.
            if amount is not None:
                verdict = check_ceiling(amount, currency)
                if not verdict.allowed:
                    detail = (
                        f"checkout {amount} {verdict.currency} would bring run spend to "
                        f"{verdict.projected} (ceiling {verdict.ceiling} {verdict.currency})"
                    )
                    reason_code = verdict.reason_code or "RUN_CEILING_EXCEEDED"
                    adapter.emit(
                        POLICY_VIOLATION,
                        {
                            "policy": "payment_guardrail",
                            "stage": "checkout",
                            "status": "blocked",
                            "reason_code": reason_code,
                            "session_id": session_id,
                            "supplier_id": kwargs.get("supplier_id"),
                            "currency": currency,
                            # content (stripped under no-content):
                            "detail": detail,
                            "amount": amount,
                        },
                    )
                    adapter.emit(
                        COMMERCE_CHECKOUT_COMPLETED,
                        {
                            "session_id": session_id,
                            "supplier_id": kwargs.get("supplier_id"),
                            "status": "blocked",
                            "reason_code": reason_code,
                            "currency": currency,
                            "amount": amount,
                            "session_duration_ms": (time.time() - start) * 1000,
                        },
                    )
                    raise _CheckoutBlocked(reason_code, detail)

            result = original(*args, **kwargs)

            # Allowed (and the charge executed): accrue ONCE into the run ledger.
            if amount is not None:
                add_spend(amount, currency)
            adapter.emit(
                COMMERCE_CHECKOUT_COMPLETED,
                {
                    "session_id": session_id,
                    "supplier_id": kwargs.get("supplier_id"),
                    "status": "completed",
                    "currency": currency,
                    "amount": amount,
                    "session_duration_ms": (time.time() - start) * 1000,
                },
            )
            return result

        return wrapped

    def _on_refund(self, original: Any) -> Any:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            adapter.emit(
                COMMERCE_REFUND_ISSUED,
                {
                    "session_id": kwargs.get("session_id"),
                    "amount": kwargs.get("amount"),
                    "reason": kwargs.get("reason"),
                },
            )
            return result

        return wrapped


def _as_float(value: Any) -> float | None:
    """Coerce a checkout amount to float; a non-numeric/None amount is treated as
    'no amount' (the ceiling is only checked + accrued for a real numeric sum)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def instrument_ucp(target: Any) -> UCPProtocolAdapter:
    from .._registry import get, register

    existing = get("ucp")
    if existing is not None:
        existing.disconnect()
    adapter = UCPProtocolAdapter()
    adapter.connect(target)
    register("ucp", adapter)
    return adapter


def uninstrument_ucp() -> None:
    from .._registry import unregister

    unregister("ucp")
