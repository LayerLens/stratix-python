"""
UCP Protocol Adapter — Universal Commerce Protocol adapter.

Instruments UCP protocol interactions including supplier discovery, catalog
browsing, checkout session lifecycle, and order refunds. Emits L7b commerce
events from ``stratix.core.events.commerce``.
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters.protocols.base import BaseProtocolAdapter

logger = logging.getLogger(__name__)


class UCPAdapter(BaseProtocolAdapter):
    """
    LayerLens adapter for the UCP (Universal Commerce Protocol).

    Instruments the full UCP commerce lifecycle:

    - Supplier discovery via well-known endpoint, registry, or referral
    - Catalog browse activity (high-frequency, item-count granularity)
    - Checkout session creation and completion
    - Order refunds

    All monetary events emit L7b events from ``stratix.core.events.commerce``.
    Checkout session durations are tracked from ``on_checkout_created`` to
    ``on_checkout_completed`` and logged for observability.

    Usage::

        adapter = UCPAdapter()
        adapter.connect()

        adapter.on_supplier_discovered(
            supplier_id="sup_abc",
            name="Acme Supplies",
            profile_url="https://acme.example.com/.well-known/ucp.json",
            org_id="org_123",
        )

        adapter.on_checkout_created(
            checkout_session_id="cs_xyz",
            supplier_id="sup_abc",
            line_items=[{"item_id": "sku_1", "quantity": 2, "unit_price": 49.99}],
            total_amount=99.98,
            org_id="org_123",
        )

        adapter.on_checkout_completed("cs_xyz", org_id="org_123", order_id="ord_456")
        adapter.disconnect()
    """

    FRAMEWORK = "ucp"
    PROTOCOL = "ucp"
    PROTOCOL_VERSION = "1.0.0"
    VERSION = "0.1.0"

    def __init__(self, memory_service: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._suppliers: dict[str, dict[str, Any]] = {}
        self._checkout_sessions: dict[str, dict[str, Any]] = {}
        self._session_start_times: dict[str, float] = {}
        self._memory_service = memory_service

    # --- Lifecycle ---

    def connect(self) -> None:
        """Connect the adapter and mark it healthy."""
        self._connected = True
        self._status = AdapterStatus.HEALTHY
        logger.debug("UCPAdapter connected (protocol=%s v%s)", self.PROTOCOL, self.PROTOCOL_VERSION)

    def disconnect(self) -> None:
        """Disconnect the adapter and release all tracked state."""
        self._suppliers.clear()
        self._checkout_sessions.clear()
        self._session_start_times.clear()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        self._close_sinks()
        logger.debug("UCPAdapter disconnected")

    def get_adapter_info(self) -> AdapterInfo:
        """Return static metadata describing this adapter's identity and capabilities."""
        return AdapterInfo(
            name="UCPAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self.PROTOCOL_VERSION,
            capabilities=[
                AdapterCapability.TRACE_PROTOCOL_EVENTS,
                AdapterCapability.REPLAY,
            ],
            description="LayerLens adapter for the UCP (Universal Commerce Protocol)",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        """Serialize accumulated trace events and adapter state for replay."""
        return ReplayableTrace(
            adapter_name="UCPAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={
                "capture_config": self._capture_config.model_dump(),
                "suppliers": dict(self._suppliers.items()),
            },
        )

    def probe_health(self, endpoint: str | None = None) -> dict[str, Any]:
        """
        Probe adapter health.

        Args:
            endpoint: Optional UCP well-known endpoint URL to probe. If None,
                      returns local adapter connectivity status only.

        Returns:
            Dict with ``reachable`` (bool), ``latency_ms`` (float), and
            ``protocol_version`` (str).
        """
        return {
            "reachable": self._connected,
            "latency_ms": 0.0,
            "protocol_version": self.PROTOCOL_VERSION,
        }

    # --- Supplier discovery ---

    def on_supplier_discovered(
        self,
        supplier_id: str,
        name: str,
        profile_url: str,
        org_id: str,
        *,
        capabilities: list[str] | None = None,
        discovery_method: str = "well_known",
    ) -> None:
        """
        Record the discovery of a UCP supplier and emit a
        ``commerce.supplier.discovered`` event.

        Args:
            supplier_id: Unique supplier identifier.
            name: Human-readable supplier name.
            profile_url: URL of the supplier's UCP profile document.
            org_id: Organization performing the discovery.
            capabilities: Declared UCP capability identifiers, if known.
            discovery_method: How the supplier was found:
                ``well_known`` | ``registry`` | ``referral``.
        """
        from layerlens.instrument.adapters.protocols._commerce import (
            SupplierInfo,
            SupplierDiscoveredEvent,
        )

        supplier_info = SupplierInfo(
            supplier_id=supplier_id,
            name=name,
            profile_url=profile_url,
            capabilities=capabilities or [],
        )
        self._suppliers[supplier_id] = {
            "name": name,
            "profile_url": profile_url,
            "capabilities": capabilities or [],
            "discovery_method": discovery_method,
        }

        event = SupplierDiscoveredEvent.create(
            supplier=supplier_info,
            org_id=org_id,
            discovery_method=discovery_method,
        )
        logger.debug(
            "UCPAdapter: supplier discovered supplier_id=%s method=%s org_id=%s",
            supplier_id,
            discovery_method,
            org_id,
        )
        self.emit_event(event)

        # Store supplier info as semantic memory
        if self._memory_service is not None:
            self._store_supplier_memory(
                supplier_id, name, profile_url, capabilities or [], discovery_method, org_id
            )

    def _store_supplier_memory(
        self,
        supplier_id: str,
        name: str,
        profile_url: str,
        capabilities: list[str],
        discovery_method: str,
        org_id: str,
    ) -> None:
        """Store supplier information as semantic memory.

        Failures are logged and swallowed.
        """
        try:
            from layerlens.instrument._vendored.memory_models import MemoryEntry

            content = (
                f"Supplier '{name}' (id={supplier_id}), profile={profile_url}, "
                f"capabilities={capabilities}, discovered via {discovery_method}"
            )
            entry = MemoryEntry(
                org_id=org_id,
                agent_id=f"ucp_{org_id}",
                memory_type="semantic",
                key=f"supplier_{supplier_id}",
                content=content,
                importance=0.6,
                metadata={
                    "source": "ucp_adapter",
                    "supplier_id": supplier_id,
                    "discovery_method": discovery_method,
                },
            )
            self._memory_service.store(entry)  # type: ignore[union-attr]
        except Exception:
            logger.debug(
                "UCP: failed to store supplier memory for %s",
                supplier_id,
                exc_info=True,
            )

    # --- Catalog browsing ---

    def on_catalog_browsed(
        self,
        supplier_id: str,
        org_id: str,
        *,
        items_viewed: int = 0,
        items_selected: int = 0,
    ) -> None:
        """
        Record a catalog browse session against a supplier and emit a
        ``commerce.catalog.browsed`` event.

        This is a high-frequency event; individual item details are not captured
        to keep payload size minimal.

        Args:
            supplier_id: Supplier whose catalog was browsed.
            org_id: Organization performing the browse.
            items_viewed: Number of catalog items viewed during the session.
            items_selected: Number of items added to the checkout basket.
        """
        from layerlens.instrument.adapters.protocols._commerce import CatalogBrowsedEvent

        event = CatalogBrowsedEvent.create(
            supplier_id=supplier_id,
            org_id=org_id,
            items_viewed=items_viewed,
            items_selected=items_selected,
        )
        logger.debug(
            "UCPAdapter: catalog browsed supplier_id=%s viewed=%d selected=%d org_id=%s",
            supplier_id,
            items_viewed,
            items_selected,
            org_id,
        )
        self.emit_event(event)

    # --- Checkout lifecycle ---

    def on_checkout_created(
        self,
        checkout_session_id: str,
        supplier_id: str,
        line_items: list[dict[str, Any]],
        total_amount: float,
        org_id: str,
        *,
        currency: str = "USD",
        idempotency_key: str | None = None,
    ) -> None:
        """
        Record the creation of a UCP checkout session and emit a
        ``commerce.checkout.created`` event.

        Starts an internal timer for this session so that duration can be
        calculated when ``on_checkout_completed`` is called.

        Args:
            checkout_session_id: Unique checkout session identifier.
            supplier_id: Supplier hosting the checkout.
            line_items: List of line-item dicts. Each dict may contain
                ``item_id``, ``name``, ``quantity``, ``unit_price``, and
                ``currency`` keys; extra keys are ignored.
            total_amount: Pre-tax total of all line items.
            org_id: Organization initiating the checkout.
            currency: ISO 4217 currency code (default ``"USD"``).
            idempotency_key: Optional client-supplied idempotency key to allow
                safe retries.
        """
        from layerlens.instrument.adapters.protocols._commerce import (
            LineItemInfo,
            CheckoutCreatedEvent,
        )

        parsed_items = [
            LineItemInfo(
                item_id=item.get("item_id", ""),
                name=item.get("name"),
                quantity=item.get("quantity", 1),
                unit_price=item.get("unit_price"),
                currency=item.get("currency", currency),
            )
            for item in line_items
        ]

        self._checkout_sessions[checkout_session_id] = {
            "supplier_id": supplier_id,
            "total_amount": total_amount,
            "currency": currency,
            "item_count": len(parsed_items),
        }
        self._session_start_times[checkout_session_id] = time.monotonic()

        event = CheckoutCreatedEvent.create(
            checkout_session_id=checkout_session_id,
            supplier_id=supplier_id,
            total_amount=total_amount,
            org_id=org_id,
            line_items=parsed_items,
            currency=currency,
            idempotency_key=idempotency_key,
        )
        logger.debug(
            "UCPAdapter: checkout created session_id=%s supplier_id=%s total=%.2f %s org_id=%s",
            checkout_session_id,
            supplier_id,
            total_amount,
            currency,
            org_id,
        )
        self.emit_event(event)

    def on_checkout_completed(
        self,
        checkout_session_id: str,
        org_id: str,
        *,
        order_id: str | None = None,
        payment_reference: str | None = None,
        status: str = "completed",
    ) -> None:
        """
        Record the terminal state of a UCP checkout session and emit a
        ``commerce.checkout.completed`` event.

        Calculates and logs the session duration if a corresponding
        ``on_checkout_created`` call was recorded.

        Args:
            checkout_session_id: Checkout session reaching a terminal state.
            org_id: Organization that initiated the checkout.
            order_id: Supplier-issued order identifier, if available.
            payment_reference: Reference to the AP2 payment that settled the
                order, if applicable.
            status: Terminal status: ``completed`` | ``failed`` | ``cancelled``.
        """
        from layerlens.instrument.adapters.protocols._commerce import CheckoutCompletedEvent

        duration_ms: float | None = None
        if checkout_session_id in self._session_start_times:
            duration_ms = (
                time.monotonic() - self._session_start_times.pop(checkout_session_id)
            ) * 1000
        self._checkout_sessions.pop(checkout_session_id, None)

        event = CheckoutCompletedEvent.create(
            checkout_session_id=checkout_session_id,
            org_id=org_id,
            order_id=order_id,
            payment_reference=payment_reference,
            status=status,
        )
        logger.debug(
            "UCPAdapter: checkout completed session_id=%s status=%s duration_ms=%s org_id=%s",
            checkout_session_id,
            status,
            f"{duration_ms:.1f}" if duration_ms is not None else "n/a",
            org_id,
        )
        self.emit_event(event)

    # --- Order refunds ---

    def on_order_refunded(
        self,
        order_id: str,
        refund_amount: float,
        org_id: str,
        *,
        currency: str = "USD",
        reason: str | None = None,
    ) -> None:
        """
        Record a full or partial order refund and emit a
        ``commerce.order.refunded`` event.

        Args:
            order_id: Supplier-issued order identifier being refunded.
            refund_amount: Amount refunded (may be partial).
            org_id: Organization that placed the original order.
            currency: ISO 4217 currency code (default ``"USD"``).
            reason: Optional human-readable explanation for the refund.
        """
        from layerlens.instrument.adapters.protocols._commerce import OrderRefundedEvent

        event = OrderRefundedEvent.create(
            order_id=order_id,
            refund_amount=refund_amount,
            org_id=org_id,
            currency=currency,
            reason=reason,
        )
        logger.debug(
            "UCPAdapter: order refunded order_id=%s amount=%.2f %s org_id=%s",
            order_id,
            refund_amount,
            currency,
            org_id,
        )
        self.emit_event(event)
