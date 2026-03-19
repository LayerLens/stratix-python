"""Tests for STRATIX AutoGen Lifecycle Hooks."""

import pytest

from layerlens.instrument.adapters._base import (
    AdapterCapability,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters._capture import CaptureConfig
from layerlens.instrument.adapters.autogen.lifecycle import AutoGenAdapter


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


class MockConversableAgent:
    """Mock AutoGen ConversableAgent."""

    def __init__(
        self,
        name="assistant",
        system_message="You are a helpful assistant.",
        human_input_mode="NEVER",
        llm_config=None,
        max_consecutive_auto_reply=10,
        code_execution_config=None,
    ):
        self.name = name
        self.system_message = system_message
        self.human_input_mode = human_input_mode
        self.llm_config = llm_config or {"model": "gpt-4"}
        self.max_consecutive_auto_reply = max_consecutive_auto_reply
        self.code_execution_config = code_execution_config
        self._sent = []
        self._received = []

    def send(self, message, recipient, **kwargs):
        self._sent.append({"message": message, "recipient": recipient})
        return None

    def receive(self, message, sender, **kwargs):
        self._received.append({"message": message, "sender": sender})
        return None

    def generate_reply(self, messages=None, sender=None, **kwargs):
        return "I understand. Here is my reply."

    def execute_code_blocks(self, code_blocks, **kwargs):
        return (0, "Execution successful")


class TestAutoGenAdapter:
    """Tests for AutoGenAdapter."""

    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)

        assert adapter._stratix is stratix
        assert adapter.FRAMEWORK == "autogen"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_with_legacy_param(self):
        """Test adapter accepts legacy stratix_instance param."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix_instance=stratix)

        assert adapter._stratix is stratix

    def test_adapter_without_stratix(self):
        """Test adapter works without STRATIX instance."""
        adapter = AutoGenAdapter()
        assert not adapter.has_stratix

    def test_connect_sets_healthy(self):
        """Test connect sets adapter to HEALTHY state."""
        adapter = AutoGenAdapter()
        adapter.connect()

        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self):
        """Test disconnect sets adapter to DISCONNECTED state."""
        adapter = AutoGenAdapter()
        adapter.connect()
        adapter.disconnect()

        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check(self):
        """Test health_check returns correct health info."""
        adapter = AutoGenAdapter()
        adapter.connect()

        health = adapter.health_check()

        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "autogen"
        assert health.error_count == 0

    def test_get_adapter_info(self):
        """Test get_adapter_info returns correct metadata."""
        adapter = AutoGenAdapter()

        info = adapter.get_adapter_info()

        assert info.name == "AutoGenAdapter"
        assert info.framework == "autogen"
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_STATE in info.capabilities
        assert AdapterCapability.TRACE_HANDOFFS in info.capabilities

    def test_serialize_for_replay(self):
        """Test serialize_for_replay returns valid trace."""
        adapter = AutoGenAdapter()

        trace = adapter.serialize_for_replay()

        assert isinstance(trace, ReplayableTrace)
        assert trace.adapter_name == "AutoGenAdapter"
        assert trace.framework == "autogen"

    def test_connect_agents_wraps_methods(self):
        """Test connect_agents wraps agent methods."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockConversableAgent()
        result = adapter.connect_agents(agent)

        assert len(result) == 1
        assert result[0] is agent
        assert hasattr(agent.send, "_stratix_original")
        assert hasattr(agent.receive, "_stratix_original")
        assert hasattr(agent.generate_reply, "_stratix_original")
        assert hasattr(agent.execute_code_blocks, "_stratix_original")

    def test_connect_agents_emits_config(self):
        """Test connect_agents emits environment.config."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockConversableAgent(name="assistant")
        adapter.connect_agents(agent)

        config_events = stratix.get_events("environment.config")
        assert len(config_events) == 1
        assert config_events[0]["payload"]["name"] == "assistant"

    def test_connect_agents_config_emitted_once(self):
        """Test agent config is only emitted once per agent name."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockConversableAgent(name="assistant")
        adapter.connect_agents(agent)
        adapter._emit_agent_config(agent)  # try again

        config_events = stratix.get_events("environment.config")
        assert len(config_events) == 1

    def test_connect_agents_idempotent(self):
        """Test connecting same agent twice is idempotent."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockConversableAgent()
        adapter.connect_agents(agent)
        adapter.connect_agents(agent)

        # Should only have one set of originals
        assert len(adapter._originals) == 1

    def test_disconnect_unwraps_agents(self):
        """Test disconnect restores original methods."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockConversableAgent()
        adapter.connect_agents(agent)

        # After wrapping, send has _stratix_original marker
        assert hasattr(agent.send, "_stratix_original")

        adapter.disconnect()

        # After unwrapping, send should no longer have the marker
        assert not hasattr(agent.send, "_stratix_original")

    def test_on_send_emits_handoff(self):
        """Test on_send emits agent.handoff event."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        sender = MockConversableAgent(name="user_proxy")
        recipient = MockConversableAgent(name="assistant")

        adapter.on_send(sender, "Hello", recipient)

        events = stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "user_proxy"
        assert events[0]["payload"]["to_agent"] == "assistant"

    def test_on_send_increments_message_seq(self):
        """Test message sequence counter increments."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        sender = MockConversableAgent()
        recipient = MockConversableAgent(name="other")

        adapter.on_send(sender, "msg1", recipient)
        adapter.on_send(sender, "msg2", recipient)

        events = stratix.get_events("agent.handoff")
        assert events[0]["payload"]["message_seq"] == 1
        assert events[1]["payload"]["message_seq"] == 2

    def test_on_receive_emits_state_change(self):
        """Test on_receive emits agent.state.change event."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        receiver = MockConversableAgent(name="assistant")
        sender = MockConversableAgent(name="user_proxy")

        adapter.on_receive(receiver, "Hello", sender)

        events = stratix.get_events("agent.state.change")
        assert len(events) == 1
        assert events[0]["payload"]["agent"] == "assistant"
        assert events[0]["payload"]["from_agent"] == "user_proxy"

    def test_on_generate_reply_emits_model_invoke(self):
        """Test on_generate_reply emits model.invoke event."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockConversableAgent(name="assistant", llm_config={"model": "gpt-4"})

        adapter.on_generate_reply(
            agent=agent,
            reply="Here is my reply",
            latency_ms=100.0,
        )

        events = stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["model"] == "gpt-4"
        assert events[0]["payload"]["latency_ms"] == 100.0

    def test_on_execute_code_emits_tool_call(self):
        """Test on_execute_code emits tool.call and tool.environment."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        agent = MockConversableAgent()
        code_blocks = [("python", "print('hello')")]

        adapter.on_execute_code(agent, code_blocks, result=(0, "hello"))

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "code_execution"

    def test_on_execute_code_emits_tool_environment(self):
        """Test on_execute_code emits tool.environment (L5c)."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        agent = MockConversableAgent()
        adapter.on_execute_code(agent, [("python", "x=1")])

        env_events = stratix.get_events("tool.environment")
        assert len(env_events) == 1

    def test_on_conversation_start_emits_agent_input(self):
        """Test on_conversation_start emits agent.input."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        initiator = MockConversableAgent(name="user_proxy")
        adapter.on_conversation_start(initiator, "Hello, help me code")

        events = stratix.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["initiator"] == "user_proxy"

    def test_on_conversation_end_emits_agent_output(self):
        """Test on_conversation_end emits agent.output."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        adapter.on_conversation_start(MockConversableAgent(), "Hi")
        adapter.on_conversation_end("Done", termination_reason="max_turns")

        events = stratix.get_events("agent.output")
        assert len(events) == 1
        assert events[0]["payload"]["termination_reason"] == "max_turns"

    def test_capture_config_minimal_gates_l3_l5(self):
        """Test minimal config disables L3 and L5 events."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()

        agent = MockConversableAgent()
        adapter.on_generate_reply(agent=agent, reply="test")
        adapter.on_execute_code(agent, [])

        assert len(stratix.get_events("model.invoke")) == 0
        assert len(stratix.get_events("tool.call")) == 0

    def test_capture_config_minimal_allows_cross_cutting(self):
        """Test minimal config allows cross-cutting events."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()

        sender = MockConversableAgent()
        recipient = MockConversableAgent(name="other")
        adapter.on_send(sender, "hi", recipient)

        assert len(stratix.get_events("agent.handoff")) == 1

    def test_message_content_extraction(self):
        """Test _message_content handles various types."""
        adapter = AutoGenAdapter()

        assert adapter._message_content("hello") == "hello"
        assert adapter._message_content({"content": "world"}) == "world"
        assert adapter._message_content(None) == ""

    def test_truncate(self):
        """Test _truncate truncates long text."""
        adapter = AutoGenAdapter()

        short = "short text"
        assert adapter._truncate(short) == short

        long = "x" * 1000
        result = adapter._truncate(long)
        assert len(result) == 503  # 500 + "..."
        assert result.endswith("...")

    def test_extract_model_name(self):
        """Test _extract_model_name from llm_config."""
        adapter = AutoGenAdapter()

        agent = MockConversableAgent(llm_config={"model": "gpt-4"})
        assert adapter._extract_model_name(agent) == "gpt-4"

    def test_extract_model_name_from_config_list(self):
        """Test _extract_model_name from config_list."""
        adapter = AutoGenAdapter()

        agent = MockConversableAgent(llm_config={
            "config_list": [{"model": "gpt-3.5-turbo"}]
        })
        assert adapter._extract_model_name(agent) == "gpt-3.5-turbo"

    def test_extract_model_name_missing(self):
        """Test _extract_model_name returns None when missing."""
        adapter = AutoGenAdapter()

        class NoLLMAgent:
            name = "no_llm"
            llm_config = None

        assert adapter._extract_model_name(NoLLMAgent()) is None

    def test_replay_trace_accumulates_events(self):
        """Test events accumulate for replay."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()

        adapter.on_conversation_start(MockConversableAgent(), "Hi")
        adapter.on_conversation_end("Bye")

        trace = adapter.serialize_for_replay()
        assert len(trace.events) == 2
