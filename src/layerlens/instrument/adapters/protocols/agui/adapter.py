"""
AG-UI Protocol Adapter — Main adapter class.

Instruments AG-UI (Agent-User Interaction) protocol events via SSE
middleware wrapping. Captures lifecycle events, text messages, tool
calls, state management, and special events.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Any

from layerlens.instrument.adapters._base import (
    AdapterCapability,
    AdapterInfo,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters.protocols.base import BaseProtocolAdapter
from layerlens.instrument.adapters.protocols.agui.event_mapper import (
    AGUIEventType,
    map_agui_to_stratix,
)

logger = logging.getLogger(__name__)


class AGUIAdapter(BaseProtocolAdapter):
    """
    Stratix adapter for the AG-UI (Agent-User Interaction) protocol.

    Provides SSE middleware that intercepts the event stream between
    an agent and its frontend without modifying either side.
    """

    FRAMEWORK = "agui"
    PROTOCOL = "agui"
    PROTOCOL_VERSION = "1.0.0"
    VERSION = "0.1.0"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._framework_version: str | None = None
        self._stream_sequence = 0
        self._state_cache: dict[str, Any] = {}
        self._text_buffer: list[str] = []
        self._in_text_message = False

    # --- Lifecycle ---

    def connect(self) -> None:
        try:
            import ag_ui  # type: ignore[import-untyped]
            self._framework_version = getattr(ag_ui, "__version__", "unknown")
        except ImportError:
            self._framework_version = None
            logger.debug("ag-ui-protocol not installed; adapter operates in standalone mode")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        self._state_cache.clear()
        self._text_buffer.clear()
        self._stream_sequence = 0
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        self._close_sinks()

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="AGUIAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_PROTOCOL_EVENTS,
                AdapterCapability.STREAMING,
            ],
            description="Stratix adapter for the AG-UI protocol",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="AGUIAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
        )

    def probe_health(self, endpoint: str | None = None) -> dict[str, Any]:
        from layerlens.instrument.adapters.protocols.health import probe_http_endpoint
        if endpoint:
            result = probe_http_endpoint(endpoint)
            return result.to_dict()
        return {"reachable": self._connected, "latency_ms": 0.0, "protocol_version": self._framework_version}

    # --- AG-UI event processing ---

    def on_agui_event(
        self,
        agui_event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """
        Process a single AG-UI SSE event.

        Maps the AG-UI event type to appropriate Stratix events and emits them.
        High-frequency TEXT_MESSAGE_CONTENT events are gated by l6b_protocol_streams.

        Args:
            agui_event_type: AG-UI event type string (e.g. TEXT_MESSAGE_CONTENT)
            payload: Event payload dict
        """
        payload = payload or {}
        mapping = map_agui_to_stratix(agui_event_type)

        # Handle text message buffering
        if agui_event_type == "TEXT_MESSAGE_START":
            self._in_text_message = True
            self._text_buffer.clear()
        elif agui_event_type == "TEXT_MESSAGE_CONTENT":
            if self._in_text_message:
                self._text_buffer.append(payload.get("content", ""))
            # Gate high-frequency content events
            if not self._capture_config.l6b_protocol_streams:
                self._stream_sequence += 1
                return
        elif agui_event_type == "TEXT_MESSAGE_END":
            self._in_text_message = False
            if self._text_buffer:
                payload["full_text"] = "".join(self._text_buffer)
                self._text_buffer.clear()

        # Emit protocol.stream.event
        self._emit_stream_event(agui_event_type, payload)

        # Emit mapped Stratix events
        if mapping.get("stratix_event") == "agent.state.change":
            self._emit_state_change(agui_event_type, payload)
        elif mapping.get("stratix_event") == "tool.call":
            self._emit_tool_call(agui_event_type, payload)

    def _emit_stream_event(
        self,
        agui_event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Emit a protocol.stream.event for an AG-UI event."""
        from layerlens.instrument.schema.events.protocol import ProtocolStreamEvent

        payload_str = str(payload)
        payload_hash = f"sha256:{hashlib.sha256(payload_str.encode()).hexdigest()}"

        event = ProtocolStreamEvent.create(
            protocol="agui",
            sequence_in_stream=self._stream_sequence,
            payload_hash=payload_hash,
            agui_event_type=agui_event_type,
            payload_summary=payload_str[:200] if len(payload_str) > 200 else payload_str,
        )
        self.emit_event(event)
        self._stream_sequence += 1

    def _emit_state_change(
        self,
        agui_event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Emit an agent.state.change event for AG-UI lifecycle/state events."""
        from layerlens.instrument.schema.events.cross_cutting import AgentStateChangeEvent, StateType

        state_str = str(payload)
        after_hash = f"sha256:{hashlib.sha256(state_str.encode()).hexdigest()}"
        before_hash = f"sha256:{hashlib.sha256(str(self._state_cache).encode()).hexdigest()}"

        if agui_event_type in ("STATE_SNAPSHOT", "STATE_DELTA"):
            self._state_cache.update(payload)

        event = AgentStateChangeEvent.create(
            state_type=StateType.INTERNAL,
            before_hash=before_hash,
            after_hash=after_hash,
        )
        self.emit_event(event)

    def _emit_tool_call(
        self,
        agui_event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Emit a tool.call event for AG-UI tool call events."""
        from layerlens.instrument.schema.events.l5_tools import ToolCallEvent, IntegrationType

        if agui_event_type == "TOOL_CALL_START":
            event = ToolCallEvent.create(
                name=payload.get("tool_name", payload.get("name", "unknown")),
                integration=IntegrationType.SERVICE,
                input_data=payload.get("args", {}),
            )
            self.emit_event(event)
        elif agui_event_type == "TOOL_CALL_RESULT":
            event = ToolCallEvent.create(
                name=payload.get("tool_name", payload.get("name", "unknown")),
                integration=IntegrationType.SERVICE,
                output_data=payload.get("result", {}),
            )
            self.emit_event(event)
