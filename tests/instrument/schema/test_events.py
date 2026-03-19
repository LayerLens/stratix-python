"""Tests for STRATIX event types."""

import pytest

from layerlens.instrument.schema.events import (
    # L1
    AgentInputEvent,
    AgentOutputEvent,
    MessageRole,
    # L2
    AgentCodeEvent,
    # L3
    ModelInvokeEvent,
    # L4
    EnvironmentConfigEvent,
    EnvironmentMetricsEvent,
    EnvironmentType,
    # L5
    ToolCallEvent,
    ToolLogicEvent,
    ToolEnvironmentEvent,
    IntegrationType,
    # Cross-cutting
    AgentStateChangeEvent,
    CostRecordEvent,
    PolicyViolationEvent,
    AgentHandoffEvent,
    StateType,
    ViolationType,
)


class TestL1Events:
    """Tests for Layer 1 (Agent I/O) events."""

    def test_agent_input_create(self):
        """Test creating an agent input event."""
        event = AgentInputEvent.create(
            message="Hello, how can you help?",
            role=MessageRole.HUMAN,
        )
        assert event.event_type == "agent.input"
        assert event.layer == "L1"
        assert event.content.message == "Hello, how can you help?"
        assert event.content.role == MessageRole.HUMAN

    def test_agent_output_create(self):
        """Test creating an agent output event."""
        event = AgentOutputEvent.create(
            message="I can help you with that!",
        )
        assert event.event_type == "agent.output"
        assert event.layer == "L1"
        assert event.content.message == "I can help you with that!"
        assert event.content.role == MessageRole.AGENT


class TestL2Events:
    """Tests for Layer 2 (Agent Code) events."""

    def test_agent_code_create(self):
        """Test creating an agent code event."""
        event = AgentCodeEvent.create(
            repo="https://github.com/example/agent",
            commit="abc123def456",
            artifact_hash="sha256:" + "a" * 64,
            config_hash="sha256:" + "b" * 64,
            branch="main",
        )
        assert event.event_type == "agent.code"
        assert event.layer == "L2"
        assert event.code.repo == "https://github.com/example/agent"
        assert event.code.branch == "main"

    def test_agent_code_hash_validation(self):
        """Test that hashes must be valid."""
        with pytest.raises(ValueError):
            AgentCodeEvent.create(
                repo="https://github.com/example/agent",
                commit="abc123",
                artifact_hash="invalid",  # Invalid
                config_hash="sha256:" + "b" * 64,
            )


class TestL3Events:
    """Tests for Layer 3 (Model) events."""

    def test_model_invoke_create(self):
        """Test creating a model invoke event."""
        event = ModelInvokeEvent.create(
            provider="openai",
            name="gpt-4",
            version="2024-01-01",
            parameters={"temperature": 0.7},
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert event.event_type == "model.invoke"
        assert event.layer == "L3"
        assert event.model.provider == "openai"
        assert event.model.parameters["temperature"] == 0.7
        assert event.prompt_tokens == 100

    def test_model_invoke_unavailable_version(self):
        """Test that version can be 'unavailable'."""
        event = ModelInvokeEvent.create(
            provider="anthropic",
            name="claude-3",
        )
        assert event.model.version == "unavailable"


class TestL4Events:
    """Tests for Layer 4 (Environment) events."""

    def test_environment_config_create(self):
        """Test creating an environment config event."""
        event = EnvironmentConfigEvent.create(
            env_type=EnvironmentType.CLOUD,
            region="us-east-1",
            attributes={"instance_type": "ml.p3.2xlarge"},
        )
        assert event.event_type == "environment.config"
        assert event.layer == "L4a"
        assert event.environment.type == EnvironmentType.CLOUD
        assert event.environment.region == "us-east-1"

    def test_environment_metrics_create(self):
        """Test creating an environment metrics event."""
        event = EnvironmentMetricsEvent.create(
            cpu_pct=45.2,
            gpu_pct=78.5,
            latency_ms=150.0,
        )
        assert event.event_type == "environment.metrics"
        assert event.layer == "L4b"
        assert event.metrics.cpu_pct == 45.2


class TestL5Events:
    """Tests for Layer 5 (Tool) events."""

    def test_tool_call_create(self):
        """Test creating a tool call event."""
        event = ToolCallEvent.create(
            name="lookup_order",
            version="1.2.3",
            integration=IntegrationType.SERVICE,
            input_data={"order_id": "12345"},
            output_data={"status": "shipped"},
            latency_ms=250.0,
        )
        assert event.event_type == "tool.call"
        assert event.layer == "L5a"
        assert event.tool.name == "lookup_order"
        assert event.tool.integration == IntegrationType.SERVICE
        assert event.input == {"order_id": "12345"}

    def test_tool_call_with_error(self):
        """Test creating a tool call event with error."""
        event = ToolCallEvent.create(
            name="failing_tool",
            version="1.0.0",
            integration=IntegrationType.LIBRARY,
            input_data={"param": "value"},
            error="Connection timeout",
        )
        assert event.error == "Connection timeout"
        assert event.output is None

    def test_tool_logic_create(self):
        """Test creating a tool logic event."""
        event = ToolLogicEvent.create(
            description="Apply discount rules",
            rules=["10% off orders > $100", "Free shipping > $50"],
        )
        assert event.event_type == "tool.logic"
        assert event.layer == "L5b"
        assert len(event.logic.rules) == 2

    def test_tool_environment_create(self):
        """Test creating a tool environment event."""
        event = ToolEnvironmentEvent.create(
            api="https://api.example.com/v1",
            permissions=["read:orders", "write:orders"],
        )
        assert event.event_type == "tool.environment"
        assert event.layer == "L5c"
        assert "read:orders" in event.environment.permissions


class TestCrossCuttingEvents:
    """Tests for cross-cutting events."""

    def test_state_change_create(self):
        """Test creating a state change event."""
        event = AgentStateChangeEvent.create(
            state_type=StateType.INTERNAL,
            before_hash="sha256:" + "a" * 64,
            after_hash="sha256:" + "b" * 64,
        )
        assert event.event_type == "agent.state.change"
        assert event.state.type == StateType.INTERNAL

    def test_cost_record_create(self):
        """Test creating a cost record event."""
        event = CostRecordEvent.create(
            tokens=1500,
            prompt_tokens=1000,
            completion_tokens=500,
            api_cost_usd=0.045,
        )
        assert event.event_type == "cost.record"
        assert event.cost.tokens == 1500
        assert event.cost.api_cost_usd == 0.045

    def test_cost_record_unavailable(self):
        """Test creating a cost record with unavailable costs."""
        event = CostRecordEvent.create(
            tokens=1500,
            api_cost_usd="unavailable",
            infra_cost_usd="unavailable",
        )
        assert event.cost.api_cost_usd == "unavailable"

    def test_policy_violation_create(self):
        """Test creating a policy violation event."""
        event = PolicyViolationEvent.create(
            violation_type=ViolationType.PRIVACY,
            root_cause="PII detected in output",
            remediation="Enable PII redaction in policy",
            failed_layer="L1",
            failed_sequence_id=17,
        )
        assert event.event_type == "policy.violation"
        assert event.violation.type == ViolationType.PRIVACY
        assert event.violation.failed_sequence_id == 17

    def test_agent_handoff_create(self):
        """Test creating an agent handoff event."""
        event = AgentHandoffEvent.create(
            from_agent="supervisor",
            to_agent="specialist",
            handoff_context_hash="sha256:" + "c" * 64,
        )
        assert event.event_type == "agent.handoff"
        assert event.from_agent == "supervisor"
        assert event.to_agent == "specialist"
