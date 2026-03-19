"""Tests for STRATIX Python SDK Emit API."""

import pytest

from layerlens.instrument import (
    STRATIX,
    emit,
    emit_input,
    emit_output,
    emit_tool_call,
    emit_model_invoke,
    emit_handoff,
)
from layerlens.instrument._context import set_current_context, reset_context


class TestEmitFunction:
    """Tests for the emit function."""

    def test_emit_requires_context(self):
        """Test that emit raises without context."""
        from layerlens.instrument.schema.events import AgentInputEvent
        from layerlens.instrument.schema.events.l1_io import MessageRole

        # Ensure no context
        token = set_current_context(None)
        try:
            payload = AgentInputEvent.create(message="Hello", role=MessageRole.HUMAN)
            with pytest.raises(RuntimeError, match="No active STRATIX context"):
                emit(payload)
        finally:
            reset_context(token)


class TestEmitInput:
    """Tests for emit_input function."""

    def test_emit_input_basic(self):
        """Test basic input emission."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        emit_input("Hello, agent!")

        events = stratix.get_events()
        assert len(events) == 1
        assert events[0].payload.event_type == "agent.input"
        assert events[0].payload.content.message == "Hello, agent!"

    def test_emit_input_with_role(self):
        """Test input emission with custom role."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        emit_input("System instruction", role="system")

        events = stratix.get_events()
        assert events[0].payload.content.role.value == "system"


class TestEmitOutput:
    """Tests for emit_output function."""

    def test_emit_output_basic(self):
        """Test basic output emission."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        emit_output("Here's my response!")

        events = stratix.get_events()
        assert len(events) == 1
        assert events[0].payload.event_type == "agent.output"
        assert events[0].payload.content.message == "Here's my response!"


class TestEmitToolCall:
    """Tests for emit_tool_call function."""

    def test_emit_tool_call_basic(self):
        """Test basic tool call emission."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        emit_tool_call(
            name="get_weather",
            input_data={"city": "Seattle"},
            output_data={"temp": 55, "conditions": "cloudy"},
            latency_ms=150.5,
        )

        events = stratix.get_events()
        assert len(events) == 1

        event = events[0]
        assert event.payload.event_type == "tool.call"
        assert event.payload.tool.name == "get_weather"
        assert event.payload.input == {"city": "Seattle"}
        assert event.payload.output == {"temp": 55, "conditions": "cloudy"}
        assert event.payload.latency_ms == 150.5

    def test_emit_tool_call_with_error(self):
        """Test tool call emission with error."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        emit_tool_call(
            name="failing_tool",
            input_data={"param": "value"},
            error="Connection timeout",
            latency_ms=5000,
        )

        events = stratix.get_events()
        event = events[0]

        assert event.payload.error == "Connection timeout"
        assert event.payload.output is None

    def test_emit_tool_call_integration_type(self):
        """Test tool call emission with integration type."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        emit_tool_call(
            name="external_api",
            integration="service",
        )

        events = stratix.get_events()
        assert events[0].payload.tool.integration == "service"


class TestEmitModelInvoke:
    """Tests for emit_model_invoke function."""

    def test_emit_model_invoke_basic(self):
        """Test basic model invoke emission."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        emit_model_invoke(
            provider="openai",
            name="gpt-4",
            version="turbo",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=1500,
        )

        events = stratix.get_events()
        assert len(events) == 1

        event = events[0]
        assert event.payload.event_type == "model.invoke"
        assert event.payload.model.provider == "openai"
        assert event.payload.model.name == "gpt-4"
        assert event.payload.model.version == "turbo"
        assert event.payload.prompt_tokens == 100
        assert event.payload.completion_tokens == 50
        assert event.payload.total_tokens == 150

    def test_emit_model_invoke_with_parameters(self):
        """Test model invoke emission with parameters."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        emit_model_invoke(
            provider="anthropic",
            name="claude-3",
            parameters={"temperature": 0.7, "max_tokens": 1000},
        )

        events = stratix.get_events()
        event = events[0]

        params = event.payload.model.parameters
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 1000


class TestEmitHandoff:
    """Tests for emit_handoff function."""

    def test_emit_handoff_basic(self):
        """Test basic handoff emission."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        emit_handoff(
            source_agent="agent_a",
            target_agent="agent_b",
        )

        events = stratix.get_events()
        assert len(events) == 1

        event = events[0]
        assert event.payload.event_type == "agent.handoff"
        assert event.payload.from_agent == "agent_a"
        assert event.payload.to_agent == "agent_b"

    def test_emit_handoff_with_context(self):
        """Test handoff emission with context data."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        ctx = stratix.start_trial()

        emit_handoff(
            source_agent="support_agent",
            target_agent="billing_agent",
            context_passed={"user_id": "123", "ticket_id": "456"},
        )

        events = stratix.get_events()
        event = events[0]

        # Context should be hashed
        assert event.payload.handoff_context_hash.startswith("sha256:")

    def test_emit_handoff_requires_context(self):
        """Test that emit_handoff raises without context."""
        # Ensure no context
        token = set_current_context(None)
        try:
            with pytest.raises(RuntimeError, match="No active STRATIX context"):
                emit_handoff(
                    source_agent="agent_a",
                    target_agent="agent_b",
                )
        finally:
            reset_context(token)
