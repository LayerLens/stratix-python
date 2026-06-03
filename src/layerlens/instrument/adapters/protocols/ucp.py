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
    COMMERCE_REFUND_ISSUED,
    COMMERCE_CHECKOUT_COMPLETED,
    COMMERCE_SUPPLIER_DISCOVERED,
)
from ._base_protocol import BaseProtocolAdapter

log = logging.getLogger(__name__)


class UCPProtocolAdapter(BaseProtocolAdapter):
    PROTOCOL = "ucp"
    PROTOCOL_VERSION = "0.1.0"

    def __init__(self) -> None:
        super().__init__()
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
                "commerce.catalog.browsed",
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
                "commerce.checkout.started",
                {"session_id": session_id, "supplier_id": kwargs.get("supplier_id")},
            )
            return original(*args, **kwargs)

        return wrapped

    def _on_complete_checkout(self, original: Any) -> Any:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            session_id = kwargs.get("session_id") or (args[0] if args else None)
            start = adapter._sessions.pop(session_id, time.time())
            result = original(*args, **kwargs)
            adapter.emit(
                COMMERCE_CHECKOUT_COMPLETED,
                {
                    "session_id": session_id,
                    "supplier_id": kwargs.get("supplier_id"),
                    "amount": kwargs.get("amount"),
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
