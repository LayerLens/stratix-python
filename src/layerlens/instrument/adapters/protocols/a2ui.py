"""
A2UI Protocol Adapter — Agent-to-User Interface adapter.

Instruments A2UI protocol interactions: surface lifecycle and user action
events. Emits L7c commerce events from ``stratix.core.events.commerce``.

Action context is always hashed (sha256) before emission to prevent PII
from appearing in the event stream.
"""

from __future__ import annotations

import uuid
import hashlib
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


class A2UIAdapter(BaseProtocolAdapter):
    """
    LayerLens adapter for the A2UI (Agent-to-User Interface) protocol.

    Instruments the A2UI surface and interaction lifecycle:

    - Surface creation (top-level UI contexts such as checkout widgets or
      confirmation dialogs)
    - User action events (button clicks, form submissions, selections)

    Action context passed to ``on_user_action`` is hashed with sha256 before
    being stored in the emitted event. Cleartext context is never written to
    the event stream, making this adapter safe for PII-sensitive deployments.

    Usage::

        adapter = A2UIAdapter()
        adapter.connect()

        adapter.on_surface_created(
            surface_id="surf_abc",
            org_id="org_123",
            root_component_id="cmp_header",
            component_count=5,
        )

        adapter.on_user_action(
            surface_id="surf_abc",
            action_name="confirm_purchase",
            org_id="org_123",
            component_id="cmp_confirm_btn",
            context={"cart_total": 99.98, "currency": "USD"},
        )

        adapter.disconnect()
    """

    FRAMEWORK = "a2ui"
    PROTOCOL = "a2ui"
    PROTOCOL_VERSION = "1.0.0"
    VERSION = "0.1.0"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._surfaces: dict[str, dict[str, Any]] = {}
        self._component_counts: dict[str, int] = {}

    # --- Lifecycle ---

    def connect(self) -> None:
        """Connect the adapter and mark it healthy."""
        self._connected = True
        self._status = AdapterStatus.HEALTHY
        logger.debug(
            "A2UIAdapter connected (protocol=%s v%s)", self.PROTOCOL, self.PROTOCOL_VERSION
        )

    def disconnect(self) -> None:
        """Disconnect the adapter and release all tracked surface state."""
        self._surfaces.clear()
        self._component_counts.clear()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        self._close_sinks()
        logger.debug("A2UIAdapter disconnected")

    def get_adapter_info(self) -> AdapterInfo:
        """Return static metadata describing this adapter's identity and capabilities."""
        return AdapterInfo(
            name="A2UIAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self.PROTOCOL_VERSION,
            capabilities=[
                AdapterCapability.TRACE_PROTOCOL_EVENTS,
                AdapterCapability.REPLAY,
            ],
            description="LayerLens adapter for the A2UI (Agent-to-User Interface) protocol",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        """Serialize accumulated trace events and adapter state for replay."""
        return ReplayableTrace(
            adapter_name="A2UIAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={
                "capture_config": self._capture_config.model_dump(),
                "surfaces": dict(self._surfaces.items()),
            },
        )

    def probe_health(self, endpoint: str | None = None) -> dict[str, Any]:
        """
        Probe adapter health.

        Args:
            endpoint: Optional endpoint URL. If None, returns local adapter
                      connectivity status only.

        Returns:
            Dict with ``reachable`` (bool), ``latency_ms`` (float), and
            ``protocol_version`` (str).
        """
        return {
            "reachable": self._connected,
            "latency_ms": 0.0,
            "protocol_version": self.PROTOCOL_VERSION,
        }

    # --- Surface lifecycle ---

    def on_surface_created(
        self,
        surface_id: str,
        org_id: str,
        *,
        root_component_id: str | None = None,
        component_count: int = 0,
    ) -> None:
        """
        Record the instantiation of an A2UI surface and emit a
        ``commerce.ui.surface_created`` event.

        A surface is a top-level UI context (e.g., a checkout widget, a
        confirmation dialog, or an inline agent panel). The surface tree
        structure is captured at a summary level via ``component_count``
        rather than enumerating individual component definitions.

        Args:
            surface_id: Unique surface instance identifier.
            org_id: Organization that owns this surface.
            root_component_id: Identifier of the root component in the surface
                tree, if known at creation time.
            component_count: Total number of components in the rendered surface.
        """
        from layerlens.instrument.adapters.protocols._commerce import SurfaceCreatedEvent

        self._surfaces[surface_id] = {
            "org_id": org_id,
            "root_component_id": root_component_id,
            "component_count": component_count,
        }
        self._component_counts[surface_id] = component_count

        event = SurfaceCreatedEvent.create(
            surface_id=surface_id,
            org_id=org_id,
            root_component_id=root_component_id,
            component_count=component_count,
        )
        logger.debug(
            "A2UIAdapter: surface created surface_id=%s components=%d org_id=%s",
            surface_id,
            component_count,
            org_id,
        )
        self.emit_event(event)

    # --- User actions ---

    def on_user_action(
        self,
        surface_id: str,
        action_name: str,
        org_id: str,
        *,
        component_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a user interaction on an A2UI surface and emit a
        ``commerce.ui.user_action`` event.

        The ``context`` dict is hashed with sha256 before being stored in the
        emitted event. Cleartext context values are never written to the event
        stream, ensuring PII safety.

        Args:
            surface_id: Surface on which the interaction occurred.
            action_name: Semantic action name (e.g. ``confirm_purchase``,
                ``cancel_order``, ``select_payment_method``).
            org_id: Organization that owns the surface.
            component_id: Optional identifier of the component that received
                the interaction (e.g. a button or form field ID).
            context: Optional dict of action context data. This is hashed and
                never stored in cleartext. Use for deduplication and replay
                correlation only.
        """
        from layerlens.instrument.adapters.protocols._commerce import UserActionTriggeredEvent

        context_hash: str | None = None
        if context is not None:
            ctx_str = str(context)
            context_hash = f"sha256:{hashlib.sha256(ctx_str.encode()).hexdigest()}"

        event = UserActionTriggeredEvent.create(
            surface_id=surface_id,
            action_name=action_name,
            org_id=org_id,
            component_id=component_id,
            context_hash=context_hash,
        )
        logger.debug(
            "A2UIAdapter: user action surface_id=%s action=%s component_id=%s org_id=%s",
            surface_id,
            action_name,
            component_id or "n/a",
            org_id,
        )
        self.emit_event(event)
