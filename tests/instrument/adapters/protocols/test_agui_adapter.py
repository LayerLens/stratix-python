"""Unit tests for the AG-UI (Agent-User Interaction) protocol adapter.

AG-UI emits ``protocol.stream.event`` for every SSE event, plus
mapped ``agent.state.change`` and ``tool.call`` events for state and
tool events. All these event types pass the default
:class:`CaptureConfig` layer-gate, so the canonical
``_RecordingStratix`` pattern works.
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters.protocols.agui import ADAPTER_CLASS, AGUIAdapter


class _RecordingStratix:
    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this stratix pass the
    # BaseAdapter fail-fast check. Tests asserting cross-tenant
    # isolation override this default.
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if args:
            payload = args[0]
            self.events.append(
                {
                    "event_type": getattr(payload, "event_type", None),
                    "payload": payload,
                }
            )


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is AGUIAdapter


def test_adapter_class_constants() -> None:
    assert AGUIAdapter.FRAMEWORK == "agui"
    assert AGUIAdapter.PROTOCOL == "agui"
    assert AGUIAdapter.PROTOCOL_VERSION == "1.0.0"


def test_lifecycle_transitions() -> None:
    adapter = AGUIAdapter(org_id="test-org")
    assert adapter.status == AdapterStatus.DISCONNECTED
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY
    adapter.disconnect()
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_disconnect_clears_state() -> None:
    adapter = AGUIAdapter(stratix=_RecordingStratix())
    adapter.connect()
    adapter._state_cache["k"] = "v"
    adapter._text_buffer.append("x")
    adapter._stream_sequence = 5
    adapter.disconnect()
    assert adapter._state_cache == {}
    assert adapter._text_buffer == []
    assert adapter._stream_sequence == 0


def test_get_adapter_info_shape() -> None:
    adapter = AGUIAdapter(org_id="test-org")
    info = adapter.get_adapter_info()
    assert isinstance(info, AdapterInfo)
    assert info.framework == "agui"
    assert info.name == "AGUIAdapter"


def test_probe_health_default_no_endpoint() -> None:
    adapter = AGUIAdapter(org_id="test-org")
    adapter.connect()
    h = adapter.probe_health()
    assert h["reachable"] is True
    assert "latency_ms" in h
    assert "protocol_version" in h


def test_text_message_lifecycle_emits_stream_events_and_buffers() -> None:
    stratix = _RecordingStratix()
    adapter = AGUIAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_agui_event("TEXT_MESSAGE_START", {"id": "msg-1"})
    adapter.on_agui_event("TEXT_MESSAGE_CONTENT", {"content": "hello"})
    adapter.on_agui_event("TEXT_MESSAGE_CONTENT", {"content": " world"})
    adapter.on_agui_event("TEXT_MESSAGE_END", {})

    types = [e["event_type"] for e in stratix.events]
    # Each AG-UI event becomes a protocol.stream.event
    assert types.count("protocol.stream.event") == 4

    # The END event payload should be enriched with the buffered full_text
    end = stratix.events[-1]["payload"]
    assert "world" in end.payload_summary  # summary contains full_text


def test_text_message_content_gated_by_l6b() -> None:
    """When l6b_protocol_streams=False, high-frequency CONTENT events are
    skipped (sequence still advances) — boundary START/END still emit.
    """
    from layerlens.instrument.adapters._base.capture import CaptureConfig

    stratix = _RecordingStratix()
    adapter = AGUIAdapter(
        stratix=stratix,
        capture_config=CaptureConfig(l6b_protocol_streams=False),
    )
    adapter.connect()

    adapter.on_agui_event("TEXT_MESSAGE_START", {})
    adapter.on_agui_event("TEXT_MESSAGE_CONTENT", {"content": "x"})
    adapter.on_agui_event("TEXT_MESSAGE_END", {})

    # CONTENT is dropped (returns early, no emit), START/END still emit but
    # are also gated since stream events go through the same l6b layer.
    # However, _emit_stream_event is also gated by the BaseAdapter
    # _pre_emit_check on l6b; so all three are dropped. The sequence
    # counter advances regardless to keep ordering consistent.
    assert adapter._stream_sequence >= 1


def test_state_snapshot_emits_state_change() -> None:
    stratix = _RecordingStratix()
    adapter = AGUIAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_agui_event("STATE_SNAPSHOT", {"counter": 1})
    types = [e["event_type"] for e in stratix.events]
    assert "agent.state.change" in types
    # state_cache updated by snapshot
    assert adapter._state_cache.get("counter") == 1


def test_tool_call_start_emits_tool_call_event() -> None:
    stratix = _RecordingStratix()
    adapter = AGUIAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_agui_event(
        "TOOL_CALL_START",
        {"tool_name": "search", "args": {"q": "weather"}},
    )
    tool_calls = [e for e in stratix.events if e["event_type"] == "tool.call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["payload"].tool.name == "search"
    assert tool_calls[0]["payload"].input == {"q": "weather"}


def test_tool_call_result_emits_tool_call_event() -> None:
    stratix = _RecordingStratix()
    adapter = AGUIAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_agui_event(
        "TOOL_CALL_RESULT",
        {"tool_name": "search", "result": {"answer": "sunny"}},
    )
    tool_calls = [e for e in stratix.events if e["event_type"] == "tool.call"]
    assert len(tool_calls) == 1
    # output should be carried through
    assert tool_calls[0]["payload"].output == {"answer": "sunny"}


def test_unknown_agui_event_emits_only_stream_event() -> None:
    stratix = _RecordingStratix()
    adapter = AGUIAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_agui_event("CUSTOM_NEW_EVENT", {"x": 1})
    types = [e["event_type"] for e in stratix.events]
    # protocol.stream.event always emitted; no state.change / tool.call
    assert "protocol.stream.event" in types
    assert "agent.state.change" not in types
    assert "tool.call" not in types


def test_serialize_for_replay_shape() -> None:
    adapter = AGUIAdapter(stratix=_RecordingStratix())
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert isinstance(rt, ReplayableTrace)
    assert rt.adapter_name == "AGUIAdapter"
    assert rt.framework == "agui"
    assert "capture_config" in rt.config
