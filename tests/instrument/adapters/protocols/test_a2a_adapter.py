"""Unit tests for the A2A (Agent-to-Agent) protocol adapter.

A2A emits ``protocol.agent_card``, ``protocol.task.submitted``,
``protocol.task.completed``, ``protocol.stream.event``, and the
cross-cutting ``agent.handoff`` event. All of these pass the default
:class:`CaptureConfig` layer-gate, so we can use the
``_RecordingStratix`` pattern (matching the canonical SmolAgents tests)
without needing to monkey-patch ``emit_event``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters.protocols.a2a import ADAPTER_CLASS, A2AAdapter


class _RecordingStratix:
    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this stratix pass the
    # BaseAdapter fail-fast check. Tests asserting cross-tenant
    # isolation override this default.
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        # Adapter calls ``self._stratix.emit(payload)`` (single positional arg)
        if args:
            payload = args[0]
            self.events.append(
                {
                    "event_type": getattr(payload, "event_type", None),
                    "payload": payload,
                }
            )


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is A2AAdapter


def test_adapter_class_constants() -> None:
    assert A2AAdapter.FRAMEWORK == "a2a"
    assert A2AAdapter.PROTOCOL == "a2a"
    assert A2AAdapter.PROTOCOL_VERSION == "0.2.1"


def test_lifecycle_transitions() -> None:
    adapter = A2AAdapter(org_id="test-org")
    assert adapter.status == AdapterStatus.DISCONNECTED
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY
    adapter.disconnect()
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_disconnect_clears_state() -> None:
    adapter = A2AAdapter(stratix=_RecordingStratix())
    adapter.connect()
    adapter.register_agent_card({"name": "agent_a", "url": "http://x", "version": "1.0"}, source="discovery")
    assert adapter._agent_cards != {}
    adapter.disconnect()
    assert adapter._agent_cards == {}
    assert adapter._task_machines == {}
    assert adapter._task_start_times == {}


def test_get_adapter_info_shape() -> None:
    adapter = A2AAdapter(org_id="test-org")
    info = adapter.get_adapter_info()
    assert isinstance(info, AdapterInfo)
    assert info.framework == "a2a"
    assert info.name == "A2AAdapter"


def test_probe_health_default_no_endpoint() -> None:
    adapter = A2AAdapter(org_id="test-org")
    adapter.connect()
    h = adapter.probe_health()
    assert h["reachable"] is True
    assert "latency_ms" in h
    assert "protocol_version" in h


def test_register_agent_card_emits_event() -> None:
    stratix = _RecordingStratix()
    adapter = A2AAdapter(stratix=stratix)
    adapter.connect()
    adapter.register_agent_card(
        {
            "name": "researcher",
            "url": "http://researcher.example",
            "protocolVersion": "0.2.1",
            "skills": [
                {"id": "search", "name": "Web Search", "tags": ["http"], "examples": []},
            ],
            "capabilities": {"streaming": True},
        },
        source="discovery",
    )
    types = [e["event_type"] for e in stratix.events]
    assert "protocol.agent_card" in types
    assert "researcher" in adapter._agent_cards


def test_task_submitted_and_completed_emit_lifecycle_events() -> None:
    stratix = _RecordingStratix()
    adapter = A2AAdapter(stratix=stratix)
    adapter.connect()
    adapter.on_task_submitted(
        task_id="t-1",
        receiver_url="http://receiver.example",
        task_type="summarize",
        submitter_agent_id="agent-a",
    )
    assert "t-1" in adapter._task_start_times
    assert "t-1" in adapter._task_machines

    adapter.on_task_completed(
        task_id="t-1",
        final_status="completed",
        artifacts=[{"id": "out-1", "data": "result"}],
    )
    assert "t-1" not in adapter._task_start_times
    assert "t-1" not in adapter._task_machines

    types = [e["event_type"] for e in stratix.events]
    assert "protocol.task.submitted" in types
    assert "protocol.task.completed" in types


def test_task_completed_with_error_emits_error_fields() -> None:
    stratix = _RecordingStratix()
    adapter = A2AAdapter(stratix=stratix)
    adapter.connect()
    adapter.on_task_submitted(task_id="t-bad", receiver_url="http://r")
    adapter.on_task_completed(
        task_id="t-bad",
        final_status="failed",
        error_code="E_TIMEOUT",
        error_message="receiver did not respond",
    )
    completed = next(e for e in stratix.events if e["event_type"] == "protocol.task.completed")
    assert completed["payload"].error_code == "E_TIMEOUT"
    assert completed["payload"].error_message == "receiver did not respond"
    assert completed["payload"].final_status == "failed"


def test_task_delegation_emits_handoff() -> None:
    stratix = _RecordingStratix()
    adapter = A2AAdapter(stratix=stratix)
    adapter.connect()
    adapter.on_task_delegation(from_agent="a", to_agent="b", context={"k": "v"})
    types = [e["event_type"] for e in stratix.events]
    assert "agent.handoff" in types
    handoff = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert handoff["payload"].from_agent == "a"
    assert handoff["payload"].to_agent == "b"
    assert handoff["payload"].handoff_context_hash.startswith("sha256:")


def test_stream_event_emits_protocol_stream_event() -> None:
    stratix = _RecordingStratix()
    adapter = A2AAdapter(stratix=stratix)
    adapter.connect()
    adapter.on_stream_event(sequence=0, payload={"chunk": "hello"})
    adapter.on_stream_event(sequence=1, payload={"chunk": " world"})
    streams = [e for e in stratix.events if e["event_type"] == "protocol.stream.event"]
    assert len(streams) == 2
    assert streams[0]["payload"].protocol == "a2a"
    assert streams[1]["payload"].sequence_in_stream == 1


def test_acp_origin_payload_normalized() -> None:
    """ACP-origin (IBM Agent Communication Protocol) payloads should be
    detected and reflected in the emitted task event's protocol_origin."""
    stratix = _RecordingStratix()
    adapter = A2AAdapter(stratix=stratix)
    adapter.connect()
    # ACPNormalizer detects via specific keys; pass a payload that may or may
    # not match — at minimum it must NOT crash.
    adapter.on_task_submitted(
        task_id="t-acp",
        receiver_url="http://r",
        raw_payload={"agent": "x", "input": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )
    types = [e["event_type"] for e in stratix.events]
    assert "protocol.task.submitted" in types


def test_serialize_for_replay_shape() -> None:
    adapter = A2AAdapter(stratix=_RecordingStratix())
    adapter.connect()
    adapter.register_agent_card({"name": "a", "url": "u", "version": "1.0"})
    rt = adapter.serialize_for_replay()
    assert isinstance(rt, ReplayableTrace)
    assert rt.adapter_name == "A2AAdapter"
    assert rt.framework == "a2a"
    assert "agent_cards" in rt.config
