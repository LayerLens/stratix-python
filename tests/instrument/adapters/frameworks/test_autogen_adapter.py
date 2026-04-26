"""Unit tests for the AutoGen framework adapter.

Mocked at the SDK shape level — no real ``autogen`` runtime needed.
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.autogen import (
    ADAPTER_CLASS,
    AutoGenAdapter,
    instrument_agents,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _FakeAgent:
    """Minimal duck-typed AutoGen ConversableAgent for tests."""

    def __init__(
        self,
        name: str = "agent",
        system_message: Any = None,
        llm_config: Any = None,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config

    def send(self, message: Any, recipient: Any, **kwargs: Any) -> Any:
        return None

    def receive(self, message: Any, sender: Any, **kwargs: Any) -> Any:
        return None

    def generate_reply(self, messages: Any = None, sender: Any = None, **kwargs: Any) -> Any:
        return "reply"

    def execute_code_blocks(self, code_blocks: Any) -> Any:
        return "exec result"


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is AutoGenAdapter


def test_lifecycle() -> None:
    a = AutoGenAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = AutoGenAdapter()
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "autogen"
    assert info.name == "AutoGenAdapter"
    health = a.health_check()
    assert health.framework_name == "autogen"


def test_connect_agents_wraps_methods_and_emits_config() -> None:
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="alice", llm_config={"model": "gpt-5"})
    adapter.connect_agents(agent)

    # Methods replaced.
    assert agent.send.__name__ == "traced_send"
    assert agent.receive.__name__ == "traced_receive"
    assert agent.generate_reply.__name__ == "traced_generate_reply"
    assert agent.execute_code_blocks.__name__ == "traced_execute_code"

    # environment.config emitted once for this agent.
    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1

    adapter.disconnect()
    # Original methods restored.
    assert agent.send.__name__ == "send"


def test_connect_agents_idempotent() -> None:
    """Calling connect_agents twice with the same agent does not double-wrap."""
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    agent = _FakeAgent(name="alice")
    adapter.connect_agents(agent)
    adapter.connect_agents(agent)

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1


def test_on_send_emits_handoff() -> None:
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    sender = _FakeAgent(name="alice")
    recipient = _FakeAgent(name="bob")
    adapter.on_send(sender=sender, message="hi there", recipient=recipient)

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "alice"
    assert evt["payload"]["to_agent"] == "bob"
    assert evt["payload"]["message_seq"] == 1


def test_on_receive_emits_state_change() -> None:
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    receiver = _FakeAgent(name="bob")
    sender = _FakeAgent(name="alice")
    adapter.on_receive(receiver=receiver, message={"content": "hello"}, sender=sender)

    evt = next(e for e in stratix.events if e["event_type"] == "agent.state.change")
    assert evt["payload"]["agent"] == "bob"
    assert evt["payload"]["from_agent"] == "alice"


def test_on_generate_reply_emits_model_invoke() -> None:
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="alice", llm_config={"model": "gpt-5"})
    reply = type("Reply", (), {"usage": {"prompt_tokens": 10, "completion_tokens": 5}})()
    adapter.on_generate_reply(agent=agent, messages=[{"role": "user", "content": "hi"}], reply=reply, latency_ms=42.0)

    evt = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert evt["payload"]["agent"] == "alice"
    assert evt["payload"]["model"] == "gpt-5"
    assert evt["payload"]["latency_ms"] == 42.0
    assert evt["payload"]["tokens_prompt"] == 10


def test_on_execute_code_emits_tool_call_and_environment() -> None:
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="alice")
    adapter.on_execute_code(agent=agent, code_blocks=[("python", "print(1)")], result="1\n", latency_ms=5.0)

    types = [e["event_type"] for e in stratix.events]
    assert "tool.call" in types
    assert "tool.environment" in types
    tool = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert tool["payload"]["tool_name"] == "code_execution"
    assert tool["payload"]["code_blocks_count"] == 1


def test_on_conversation_start_end_emits_input_output() -> None:
    stratix = _RecordingStratix()
    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    initiator = _FakeAgent(name="alice")
    adapter.on_conversation_start(initiator=initiator, message="start")
    adapter.on_conversation_end(final_message="bye", termination_reason="max_rounds")

    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["termination_reason"] == "max_rounds"
    assert out["payload"]["duration_ns"] >= 0


def test_capture_config_gates_l3_model_metadata() -> None:
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l3_model_metadata=False)
    adapter = AutoGenAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    agent = _FakeAgent(name="alice", llm_config={"model": "gpt-5"})
    adapter.on_generate_reply(agent=agent, reply="hi")
    sender = _FakeAgent(name="alice")
    recipient = _FakeAgent(name="bob")
    adapter.on_send(sender=sender, message="x", recipient=recipient)

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" not in types
    # handoff (from on_send) is cross-cutting / always enabled.
    assert "agent.handoff" in types


def test_instrument_agents_helper() -> None:
    """Top-level convenience wraps multiple agents at once."""
    a = _FakeAgent(name="a")
    b = _FakeAgent(name="b")
    result = instrument_agents(a, b)
    assert isinstance(result, list)
    assert len(result) == 2
    # Both wrapped.
    assert a.send.__name__ == "traced_send"
    assert b.send.__name__ == "traced_send"


def test_serialize_for_replay() -> None:
    adapter = AutoGenAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "autogen"
    assert rt.adapter_name == "AutoGenAdapter"
    assert "capture_config" in rt.config
