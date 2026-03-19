"""Tests for AutoGen GroupChat Tracing."""

import pytest

from layerlens.instrument.adapters.autogen.lifecycle import AutoGenAdapter
from layerlens.instrument.adapters.autogen.groupchat import GroupChatTracer
from layerlens.instrument.adapters._capture import CaptureConfig


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


class MockGroupChatManager:
    """Mock AutoGen GroupChatManager."""

    def __init__(self):
        self._run_count = 0

    def run_chat(self, *args, **kwargs):
        self._run_count += 1
        return {"messages": ["msg1", "msg2"]}


class TestGroupChatTracer:
    """Tests for GroupChatTracer."""

    def test_tracer_initialization(self):
        """Test tracer initializes correctly."""
        adapter = AutoGenAdapter()
        tracer = GroupChatTracer(adapter)

        assert tracer.message_seq == 0

    def test_wrap_manager(self):
        """Test wrap_manager wraps run_chat."""
        adapter = AutoGenAdapter()
        tracer = GroupChatTracer(adapter)
        manager = MockGroupChatManager()

        result = tracer.wrap_manager(manager)

        assert result is manager
        assert hasattr(manager, "_stratix_tracer")
        assert hasattr(manager.run_chat, "_stratix_original")

    def test_wrapped_run_chat_calls_original(self):
        """Test wrapped run_chat calls original."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()
        tracer = GroupChatTracer(adapter)

        manager = MockGroupChatManager()
        tracer.wrap_manager(manager)

        result = manager.run_chat()

        assert result == {"messages": ["msg1", "msg2"]}
        assert manager._run_count == 1

    def test_wrapped_run_chat_emits_events(self):
        """Test wrapped run_chat emits start and end events."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()
        tracer = GroupChatTracer(adapter)

        manager = MockGroupChatManager()
        tracer.wrap_manager(manager)
        manager.run_chat()

        input_events = stratix.get_events("agent.input")
        assert len(input_events) == 1
        assert input_events[0]["payload"]["event_subtype"] == "groupchat_start"

        output_events = stratix.get_events("agent.output")
        assert len(output_events) == 1

    def test_on_speaker_selected(self):
        """Test on_speaker_selected emits agent.code event."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()
        tracer = GroupChatTracer(adapter)

        tracer.on_speaker_selected(
            method="round_robin",
            candidates=["agent_a", "agent_b"],
            chosen="agent_a",
        )

        events = stratix.get_events("agent.code")
        assert len(events) == 1
        assert events[0]["payload"]["chosen"] == "agent_a"
        assert events[0]["payload"]["method"] == "round_robin"

    def test_on_message_routed_emits_handoff(self):
        """Test on_message_routed emits agent.handoff event."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()
        tracer = GroupChatTracer(adapter)

        tracer.on_message_routed("agent_a", "agent_b")

        events = stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "agent_a"
        assert events[0]["payload"]["to_agent"] == "agent_b"
        assert events[0]["payload"]["reason"] == "groupchat_routing"

    def test_on_message_routed_increments_seq(self):
        """Test message routing increments sequence counter."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()
        tracer = GroupChatTracer(adapter)

        tracer.on_message_routed("a", "b")
        tracer.on_message_routed("b", "c")

        assert tracer.message_seq == 2

        events = stratix.get_events("agent.handoff")
        assert events[0]["payload"]["message_seq"] == 1
        assert events[1]["payload"]["message_seq"] == 2

    def test_on_termination_emits_output(self):
        """Test on_termination emits agent.output event."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()
        tracer = GroupChatTracer(adapter)

        tracer.on_termination(reason="max_turns", final_speaker="agent_a")

        events = stratix.get_events("agent.output")
        assert len(events) == 1
        assert events[0]["payload"]["termination_reason"] == "max_turns"
        assert events[0]["payload"]["final_speaker"] == "agent_a"

    def test_speaker_selection_gated_by_capture_config(self):
        """Test L2 events gated by CaptureConfig."""
        stratix = MockStratix()
        # Default config has l2_agent_code=False
        adapter = AutoGenAdapter(stratix=stratix)
        adapter.connect()
        tracer = GroupChatTracer(adapter)

        tracer.on_speaker_selected(method="auto", chosen="agent_a")

        events = stratix.get_events("agent.code")
        assert len(events) == 0

    def test_message_routing_always_emitted(self):
        """Test handoff events are always emitted (cross-cutting)."""
        stratix = MockStratix()
        adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()
        tracer = GroupChatTracer(adapter)

        tracer.on_message_routed("a", "b")

        assert len(stratix.get_events("agent.handoff")) == 1
