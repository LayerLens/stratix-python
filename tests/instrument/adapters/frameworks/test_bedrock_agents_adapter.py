"""Unit tests for the AWS Bedrock Agents framework adapter.

Mocked at the SDK shape level — no real ``boto3`` runtime needed.
The adapter integrates via boto3 event hooks: ``client.meta.events.register(...)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Callable

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.bedrock_agents import (
    ADAPTER_CLASS,
    BedrockAgentsAdapter,
    instrument_client,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _FakeEventSystem:
    """Mimics boto3 client.meta.events register/unregister."""

    def __init__(self) -> None:
        self.handlers: Dict[str, List[Callable[..., Any]]] = {}
        self.unregistered: List[Tuple[str, Callable[..., Any]]] = []

    def register(self, event: str, handler: Callable[..., Any]) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def unregister(self, event: str, handler: Callable[..., Any]) -> None:
        self.unregistered.append((event, handler))
        if event in self.handlers and handler in self.handlers[event]:
            self.handlers[event].remove(handler)


class _FakeClient:
    """Mimics a boto3 bedrock-agent-runtime client."""

    def __init__(self) -> None:
        self.meta = _FakeMeta()


class _FakeMeta:
    def __init__(self) -> None:
        self.events = _FakeEventSystem()


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is BedrockAgentsAdapter


def test_lifecycle() -> None:
    a = BedrockAgentsAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = BedrockAgentsAdapter()
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "bedrock_agents"
    assert info.name == "BedrockAgentsAdapter"
    health = a.health_check()
    assert health.framework_name == "bedrock_agents"


def test_adapter_info_declares_streaming_and_replay_capabilities() -> None:
    """Bedrock Agents responses are EventStream payloads (``invoke_agent``
    returns a streaming completion), so STREAMING is supported. The
    adapter also implements ``serialize_for_replay``.
    """
    from layerlens.instrument.adapters._base.adapter import AdapterCapability

    info = BedrockAgentsAdapter().get_adapter_info()
    assert AdapterCapability.STREAMING in info.capabilities
    assert AdapterCapability.REPLAY in info.capabilities


def test_instrument_client_registers_event_hooks() -> None:
    adapter = BedrockAgentsAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()

    client = _FakeClient()
    adapter.instrument_client(client)

    handlers = client.meta.events.handlers
    assert "provide-client-params.bedrock-agent-runtime.InvokeAgent" in handlers
    assert "after-call.bedrock-agent-runtime.InvokeAgent" in handlers


def test_disconnect_unregisters_event_hooks() -> None:
    adapter = BedrockAgentsAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()
    client = _FakeClient()
    adapter.instrument_client(client)

    adapter.disconnect()
    assert len(client.meta.events.unregistered) == 2


def test_before_invoke_emits_input_event() -> None:
    stratix = _RecordingStratix()
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _FakeClient()
    adapter.instrument_client(client)

    # Simulate the boto3 'provide-client-params' event firing.
    adapter._before_invoke_agent(
        params={
            "agentId": "agent-123",
            "agentAliasId": "alias-1",
            "sessionId": "sess-1",
            "inputText": "hello",
            "enableTrace": True,
        }
    )

    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    assert "agent.input" in types

    inp = next(e for e in stratix.events if e["event_type"] == "agent.input")
    assert inp["payload"]["agent_id"] == "agent-123"
    assert inp["payload"]["input"] == "hello"


def test_after_invoke_emits_output_and_processes_trace() -> None:
    stratix = _RecordingStratix()
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    # Simulate the after-call event with a parsed response.
    adapter._after_invoke_agent(
        parsed={
            "outputText": "the answer is 42",
            "sessionId": "sess-1",
            "trace": {
                "steps": [
                    {
                        "type": "ACTION_GROUP",
                        "actionGroupName": "calc",
                        "actionGroupInput": {"x": 1},
                        "actionGroupInvocationOutput": {"output": "ok"},
                    },
                    {
                        "type": "MODEL_INVOCATION",
                        "foundationModel": "anthropic.claude-v2",
                        "modelInvocationOutput": {
                            "usage": {"inputTokens": 100, "outputTokens": 50}
                        },
                    },
                    {
                        "type": "AGENT_COLLABORATOR",
                        "supervisorAgentId": "sup-1",
                        "collaboratorAgentId": "col-1",
                    },
                ]
            },
        }
    )

    types = [e["event_type"] for e in stratix.events]
    assert "agent.output" in types
    assert "tool.call" in types
    assert "model.invoke" in types
    assert "cost.record" in types
    assert "agent.handoff" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["output"] == "the answer is 42"

    model = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert model["payload"]["model"] == "anthropic.claude-v2"
    assert model["payload"]["tokens_prompt"] == 100


def test_on_tool_use_emits_event() -> None:
    stratix = _RecordingStratix()
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2, latency_ms=12.3)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert evt["payload"]["tool_name"] == "calc"
    assert evt["payload"]["latency_ms"] == 12.3


def test_on_handoff_emits_event_with_context_hash() -> None:
    stratix = _RecordingStratix()
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context="some context")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "a"
    assert evt["payload"]["to_agent"] == "b"
    assert evt["payload"]["context_hash"] is not None


def test_capture_config_gates_l5a_tool_calls() -> None:
    """When l5a_tool_calls is disabled, tool.call events do NOT fire (handoff still does)."""
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l5a_tool_calls=False)
    adapter = BedrockAgentsAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
    adapter.on_handoff(from_agent="a", to_agent="b", context="x")

    types = [e["event_type"] for e in stratix.events]
    assert "tool.call" not in types
    assert "agent.handoff" in types


def test_instrument_client_helper() -> None:
    """Top-level convenience function returns a connected adapter."""
    client = _FakeClient()
    adapter = instrument_client(client)
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY
    # Hooks were registered.
    assert "provide-client-params.bedrock-agent-runtime.InvokeAgent" in client.meta.events.handlers


def test_serialize_for_replay() -> None:
    adapter = BedrockAgentsAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    rt = adapter.serialize_for_replay()
    assert rt.framework == "bedrock_agents"
    assert rt.adapter_name == "BedrockAgentsAdapter"
    assert "capture_config" in rt.config
