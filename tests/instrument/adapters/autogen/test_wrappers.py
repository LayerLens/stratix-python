"""Tests for AutoGen Method Wrappers."""

import pytest

from layerlens.instrument.adapters.autogen.lifecycle import AutoGenAdapter
from layerlens.instrument.adapters.autogen.wrappers import (
    create_traced_send,
    create_traced_receive,
    create_traced_generate_reply,
    create_traced_execute_code,
)


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self):
        self.events = []

    def emit(self, event_type: str, payload: dict):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str = None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class MockAgent:
    """Mock AutoGen agent."""

    def __init__(self, name="test_agent"):
        self.name = name
        self.llm_config = {"model": "gpt-4"}

    def send(self, message, recipient, **kwargs):
        return "sent"

    def receive(self, message, sender, **kwargs):
        return "received"

    def generate_reply(self, messages=None, sender=None, **kwargs):
        return "reply"

    def execute_code_blocks(self, code_blocks, **kwargs):
        return (0, "success")


class TestCreateTracedSend:
    """Tests for create_traced_send wrapper."""

    def test_calls_original(self):
        """Test wrapper calls original send."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockAgent()
        recipient = MockAgent(name="recipient")
        original = agent.send

        traced = create_traced_send(adapter, agent, original)
        result = traced("hello", recipient)

        assert result == "sent"

    def test_emits_handoff_event(self):
        """Test wrapper emits agent.handoff event."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockAgent(name="sender")
        recipient = MockAgent(name="recipient")

        traced = create_traced_send(adapter, agent, agent.send)
        traced("hello", recipient)

        events = stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "sender"

    def test_preserves_original_reference(self):
        """Test wrapper stores reference to original."""
        adapter = AutoGenAdapter()
        agent = MockAgent()
        original = agent.send

        traced = create_traced_send(adapter, agent, original)
        assert traced._stratix_original is original

    def test_adapter_error_does_not_propagate(self):
        """Test adapter errors don't break the original call."""
        class FailingSTRATIX:
            def emit(self, *args, **kwargs):
                raise RuntimeError("emit failed")

        adapter = AutoGenAdapter(stratix=FailingSTRATIX())
        adapter.connect()

        agent = MockAgent()
        recipient = MockAgent(name="other")

        traced = create_traced_send(adapter, agent, agent.send)
        result = traced("hello", recipient)

        assert result == "sent"  # original still executed


class TestCreateTracedReceive:
    """Tests for create_traced_receive wrapper."""

    def test_calls_original(self):
        """Test wrapper calls original receive."""
        adapter = AutoGenAdapter()
        agent = MockAgent()
        sender = MockAgent(name="sender")

        traced = create_traced_receive(adapter, agent, agent.receive)
        result = traced("hello", sender)

        assert result == "received"

    def test_emits_state_change_event(self):
        """Test wrapper emits agent.state.change event."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockAgent(name="receiver")
        sender = MockAgent(name="sender")

        traced = create_traced_receive(adapter, agent, agent.receive)
        traced("hello", sender)

        events = stratix.get_events("agent.state.change")
        assert len(events) == 1

    def test_preserves_original_reference(self):
        """Test wrapper stores reference to original."""
        adapter = AutoGenAdapter()
        agent = MockAgent()
        original = agent.receive

        traced = create_traced_receive(adapter, agent, original)
        assert traced._stratix_original is original


class TestCreateTracedGenerateReply:
    """Tests for create_traced_generate_reply wrapper."""

    def test_calls_original(self):
        """Test wrapper calls original generate_reply."""
        adapter = AutoGenAdapter()
        agent = MockAgent()

        traced = create_traced_generate_reply(adapter, agent, agent.generate_reply)
        result = traced(messages=[{"role": "user", "content": "hello"}])

        assert result == "reply"

    def test_emits_model_invoke_event(self):
        """Test wrapper emits model.invoke event."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockAgent(name="assistant")

        traced = create_traced_generate_reply(adapter, agent, agent.generate_reply)
        traced(messages=[{"role": "user", "content": "hello"}])

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["agent"] == "assistant"

    def test_captures_latency(self):
        """Test wrapper captures latency."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockAgent()

        traced = create_traced_generate_reply(adapter, agent, agent.generate_reply)
        traced()

        events = stratix.get_events("model.invoke")
        assert "latency_ms" in events[0]["payload"]
        assert events[0]["payload"]["latency_ms"] >= 0

    def test_exception_passthrough(self):
        """Test exceptions from original propagate."""
        adapter = AutoGenAdapter()
        agent = MockAgent()

        def failing_reply(**kwargs):
            raise ValueError("LLM error")

        traced = create_traced_generate_reply(adapter, agent, failing_reply)

        with pytest.raises(ValueError, match="LLM error"):
            traced()


class TestCreateTracedExecuteCode:
    """Tests for create_traced_execute_code wrapper."""

    def test_calls_original(self):
        """Test wrapper calls original execute_code_blocks."""
        adapter = AutoGenAdapter()
        agent = MockAgent()

        traced = create_traced_execute_code(adapter, agent, agent.execute_code_blocks)
        result = traced([("python", "print('hi')")])

        assert result == (0, "success")

    def test_emits_tool_call_event(self):
        """Test wrapper emits tool.call event."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockAgent()

        traced = create_traced_execute_code(adapter, agent, agent.execute_code_blocks)
        traced([("python", "x=1")])

        events = stratix.get_events("tool.call")
        assert len(events) == 1

    def test_exception_passthrough(self):
        """Test exceptions from original propagate."""
        adapter = AutoGenAdapter()
        agent = MockAgent()

        def failing_exec(code_blocks, **kwargs):
            raise RuntimeError("Code execution failed")

        traced = create_traced_execute_code(adapter, agent, failing_exec)

        with pytest.raises(RuntimeError, match="Code execution failed"):
            traced([])

    def test_adapter_error_does_not_propagate(self):
        """Test adapter errors don't break the original call."""
        class FailingSTRATIX:
            def emit(self, *args, **kwargs):
                raise RuntimeError("emit failed")

        adapter = AutoGenAdapter(stratix=FailingSTRATIX())
        adapter.connect()

        agent = MockAgent()

        traced = create_traced_execute_code(adapter, agent, agent.execute_code_blocks)
        result = traced([("python", "x=1")])

        assert result == (0, "success")
