"""
Tests for schema v1.2.0 — all 9 new protocol event types.

Validates that each event type:
1. Can be instantiated with required fields
2. Has correct default event_type string
3. Generates valid JSON Schema
4. Uses the .create() classmethod correctly
"""

import pytest

try:
    from stratix.schemas.generator import EVENT_TYPE_MAP, generate_event_schema
except ImportError:
    pytest.skip("Requires server-side stratix.schemas package", allow_module_level=True)

from layerlens.instrument.schema.events.protocol import (  # noqa: E402
    AgentCardEvent,
    AgentCardInfo,
    SkillInfo,
    TaskSubmittedEvent,
    TaskCompletedEvent,
    ProtocolStreamEvent,
    ElicitationRequestEvent,
    ElicitationResponseEvent,
    StructuredToolOutputEvent,
    McpAppInvocationEvent,
    AsyncTaskEvent,
)


class TestAgentCardEvent:
    def test_create(self):
        event = AgentCardEvent.create(
            agent_id="agent-1",
            name="Test Agent",
            url="http://agent.example.com",
            version="0.2.1",
        )
        assert event.event_type == "protocol.agent_card"
        assert event.layer == "L6a"
        assert event.card.agent_id == "agent-1"
        assert event.card.name == "Test Agent"

    def test_with_skills(self):
        event = AgentCardEvent.create(
            agent_id="agent-1",
            name="Skilled Agent",
            url="http://agent.example.com",
            version="0.2.1",
            skills=[SkillInfo(id="s1", name="Search", description="Web search")],
        )
        assert len(event.card.skills) == 1
        assert event.card.skills[0].name == "Search"

    def test_schema_generation(self):
        schema = generate_event_schema("protocol.agent_card")
        assert schema is not None
        assert "properties" in schema


class TestTaskSubmittedEvent:
    def test_create(self):
        event = TaskSubmittedEvent.create(
            task_id="task-001",
            receiver_agent_url="http://agent.example.com",
        )
        assert event.event_type == "protocol.task.submitted"
        assert event.task_id == "task-001"
        assert event.protocol_origin == "a2a"

    def test_acp_origin(self):
        event = TaskSubmittedEvent.create(
            task_id="task-002",
            receiver_agent_url="http://agent.example.com",
            protocol_origin="acp",
        )
        assert event.protocol_origin == "acp"

    def test_in_event_type_map(self):
        assert "protocol.task.submitted" in EVENT_TYPE_MAP


class TestTaskCompletedEvent:
    def test_create_completed(self):
        event = TaskCompletedEvent.create(
            task_id="task-001",
            final_status="completed",
            artifact_count=2,
        )
        assert event.event_type == "protocol.task.completed"
        assert event.final_status == "completed"
        assert event.artifact_count == 2

    def test_create_failed(self):
        event = TaskCompletedEvent.create(
            task_id="task-002",
            final_status="failed",
            error_code="-32001",
            error_message="Task not found",
        )
        assert event.final_status == "failed"
        assert event.error_code == "-32001"

    def test_with_duration(self):
        event = TaskCompletedEvent.create(
            task_id="task-003",
            final_status="completed",
            duration_ms=1500.0,
        )
        assert event.duration_ms == 1500.0


class TestProtocolStreamEvent:
    def test_create_agui(self):
        event = ProtocolStreamEvent.create(
            protocol="agui",
            sequence_in_stream=0,
            payload_hash="sha256:abc123",
            agui_event_type="TEXT_MESSAGE_CONTENT",
        )
        assert event.event_type == "protocol.stream.event"
        assert event.layer == "L6b"
        assert event.protocol == "agui"
        assert event.agui_event_type == "TEXT_MESSAGE_CONTENT"

    def test_create_a2a(self):
        event = ProtocolStreamEvent.create(
            protocol="a2a",
            sequence_in_stream=5,
            payload_hash="sha256:def456",
        )
        assert event.protocol == "a2a"
        assert event.agui_event_type is None


class TestElicitationRequestEvent:
    def test_create(self):
        event = ElicitationRequestEvent.create(
            elicitation_id="elic-001",
            server_name="mcp-server",
            schema_hash="sha256:abc",
            request_title="Confirm deletion",
        )
        assert event.event_type == "protocol.elicitation.request"
        assert event.layer == "L5a"
        assert event.elicitation_id == "elic-001"
        assert event.request_title == "Confirm deletion"


class TestElicitationResponseEvent:
    def test_create_submit(self):
        event = ElicitationResponseEvent.create(
            elicitation_id="elic-001",
            action="submit",
            response_hash="sha256:user_response_hash",
            latency_ms=2500.0,
        )
        assert event.event_type == "protocol.elicitation.response"
        assert event.action == "submit"
        assert event.latency_ms == 2500.0

    def test_create_cancel(self):
        event = ElicitationResponseEvent.create(
            elicitation_id="elic-002",
            action="cancel",
            response_hash="sha256:empty",
        )
        assert event.action == "cancel"


class TestStructuredToolOutputEvent:
    def test_create_valid(self):
        event = StructuredToolOutputEvent.create(
            tool_name="search",
            schema_hash="sha256:schema_hash",
            validation_passed=True,
            output_hash="sha256:output_hash",
        )
        assert event.event_type == "protocol.tool.structured_output"
        assert event.validation_passed is True
        assert event.validation_errors == []

    def test_create_invalid(self):
        event = StructuredToolOutputEvent.create(
            tool_name="search",
            schema_hash="sha256:schema_hash",
            validation_passed=False,
            output_hash="sha256:output_hash",
            validation_errors=["Missing field: name"],
        )
        assert event.validation_passed is False
        assert len(event.validation_errors) == 1


class TestMcpAppInvocationEvent:
    def test_create(self):
        event = McpAppInvocationEvent.create(
            app_id="app-form-1",
            component_type="form",
            interaction_result="submitted",
            parameters_hash="sha256:params",
            result_hash="sha256:result",
        )
        assert event.event_type == "protocol.mcp_app.invocation"
        assert event.component_type == "form"
        assert event.interaction_result == "submitted"


class TestAsyncTaskEvent:
    def test_create(self):
        event = AsyncTaskEvent.create(
            async_task_id="async-001",
            status="created",
            protocol="mcp",
            timeout_ms=300000,
        )
        assert event.event_type == "protocol.async_task"
        assert event.protocol == "mcp"
        assert event.status == "created"

    def test_with_progress(self):
        event = AsyncTaskEvent.create(
            async_task_id="async-002",
            status="running",
            protocol="mcp",
            progress_pct=45.0,
            elapsed_ms=5000.0,
        )
        assert event.progress_pct == 45.0
        assert event.elapsed_ms == 5000.0


class TestAllProtocolEventsInMap:
    """Verify all 9 protocol event types are in EVENT_TYPE_MAP."""

    @pytest.mark.parametrize("event_type", [
        "protocol.agent_card",
        "protocol.task.submitted",
        "protocol.task.completed",
        "protocol.stream.event",
        "protocol.elicitation.request",
        "protocol.elicitation.response",
        "protocol.tool.structured_output",
        "protocol.mcp_app.invocation",
        "protocol.async_task",
    ])
    def test_event_type_in_map(self, event_type):
        assert event_type in EVENT_TYPE_MAP

    @pytest.mark.parametrize("event_type", [
        "protocol.agent_card",
        "protocol.task.submitted",
        "protocol.task.completed",
        "protocol.stream.event",
        "protocol.elicitation.request",
        "protocol.elicitation.response",
        "protocol.tool.structured_output",
        "protocol.mcp_app.invocation",
        "protocol.async_task",
    ])
    def test_schema_generation(self, event_type):
        schema = generate_event_schema(event_type)
        assert schema is not None
        assert "properties" in schema
