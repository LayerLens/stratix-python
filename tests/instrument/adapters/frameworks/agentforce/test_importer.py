"""Tests for AgentForce import adapter and normalizer.

Ported from ``ateam/tests/adapters/agentforce/test_importer.py``.

Exercises ``AgentForceImporter``, ``AgentForceNormalizer``, ``ImportResult``,
and SOQL input validation. The importer is independent of BaseAdapter, so
no ``org_id`` is required here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from layerlens.instrument.adapters.frameworks.agentforce.auth import (
    SalesforceConnection,
    SalesforceCredentials,
)
from layerlens.instrument.adapters.frameworks.agentforce.importer import (
    ImportResult,
    AgentForceImporter,
)
from layerlens.instrument.adapters.frameworks.agentforce.normalizer import (
    AgentForceNormalizer,
)

# --- Test Data ---------------------------------------------------------------


def _session(
    sid: str = "a0B5f0000Sess01",
    start: str = "2026-02-21T10:00:00Z",
    end: str | None = None,
    channel: str = "Chat",
) -> dict[str, Any]:
    return {
        "Id": sid,
        "StartTimestamp": start,
        "EndTimestamp": end,
        "AiAgentChannelTypeId": channel,
        "AiAgentSessionEndType": "Resolved" if end else None,
        "VoiceCallId": None,
        "MessagingSessionId": None,
        "PreviousSessionId": None,
    }


def _participant(
    pid: str = "a0B5f0000Part01",
    session_id: str = "a0B5f0000Sess01",
    agent_type: str = "EinsteinServiceAgent",
) -> dict[str, Any]:
    return {
        "Id": pid,
        "AiAgentSessionId": session_id,
        "AiAgentTypeId": agent_type,
        "AiAgentApiName": "my_agent",
        "AiAgentVersionApiName": "v1.0",
        "ParticipantId": "user-001",
        "AiAgentSessionParticipantRoleId": "Owner",
    }


def _interaction(
    iid: str = "a0B5f0000Intr01",
    session_id: str = "a0B5f0000Sess01",
) -> dict[str, Any]:
    return {
        "Id": iid,
        "AiAgentSessionId": session_id,
        "AiAgentInteractionTypeId": "Turn",
        "TelemetryTraceId": "trace-abc",
        "TelemetryTraceSpanId": "span-001",
        "TopicApiName": "Order_Status",
        "AttributeText": '{"intent": "check_order"}',
        "PrevInteractionId": None,
    }


def _step_user_input(
    step_id: str = "a0B5f0000Step01",
    interaction_id: str = "a0B5f0000Intr01",
) -> dict[str, Any]:
    return {
        "Id": step_id,
        "AiAgentInteractionId": interaction_id,
        "AiAgentInteractionStepTypeId": "UserInputStep",
        "InputValueText": "What is my order status?",
        "OutputValueText": None,
        "ErrorMessageText": None,
        "GenerationId": None,
        "GenAiGatewayRequestId": None,
        "GenAiGatewayResponseId": None,
        "Name": "user_input",
        "TelemetryTraceSpanId": "span-a0B5f0000Step01",
    }


def _step_llm(
    step_id: str = "a0B5f0000Step02",
    interaction_id: str = "a0B5f0000Intr01",
) -> dict[str, Any]:
    return {
        "Id": step_id,
        "AiAgentInteractionId": interaction_id,
        "AiAgentInteractionStepTypeId": "LLMExecutionStep",
        "InputValueText": "Find order status for user",
        "OutputValueText": "Your order is being shipped",
        "ErrorMessageText": None,
        "GenerationId": "gen-001",
        "GenAiGatewayRequestId": "req-001",
        "GenAiGatewayResponseId": "resp-001",
        "Name": "einstein_model",
        "TelemetryTraceSpanId": "span-a0B5f0000Step02",
    }


def _step_function(
    step_id: str = "a0B5f0000Step03",
    interaction_id: str = "a0B5f0000Intr01",
) -> dict[str, Any]:
    return {
        "Id": step_id,
        "AiAgentInteractionId": interaction_id,
        "AiAgentInteractionStepTypeId": "FunctionStep",
        "InputValueText": '{"order_id": "ORD-123"}',
        "OutputValueText": '{"status": "shipped", "eta": "2026-03-01"}',
        "ErrorMessageText": None,
        "GenerationId": None,
        "GenAiGatewayRequestId": None,
        "GenAiGatewayResponseId": None,
        "Name": "lookup_order",
        "TelemetryTraceSpanId": "span-a0B5f0000Step03",
    }


def _message_input(
    mid: str = "a0B5f0000Msg001",
    interaction_id: str = "a0B5f0000Intr01",
) -> dict[str, Any]:
    return {
        "Id": mid,
        "AiAgentInteractionId": interaction_id,
        "AiAgentInteractionMessageTypeId": "Input",
        "ContentText": "What is my order status?",
        "AiAgentInteractionMsgContentTypeId": "text/plain",
        "MessageSentTimestamp": "2026-02-21T10:00:05Z",
        "ParentMessageId": None,
    }


def _message_output(
    mid: str = "a0B5f0000Msg002",
    interaction_id: str = "a0B5f0000Intr01",
) -> dict[str, Any]:
    return {
        "Id": mid,
        "AiAgentInteractionId": interaction_id,
        "AiAgentInteractionMessageTypeId": "Output",
        "ContentText": "Your order ORD-123 is being shipped and will arrive by March 1st.",
        "AiAgentInteractionMsgContentTypeId": "text/plain",
        "MessageSentTimestamp": "2026-02-21T10:00:10Z",
        "ParentMessageId": "a0B5f0000Msg001",
    }


def _make_mock_connection(
    query_results: dict[str, list[dict[str, Any]]] | None = None,
) -> MagicMock:
    """Build a :class:`MagicMock` that quacks like ``SalesforceConnection``."""
    conn = MagicMock(spec=SalesforceConnection)
    conn.queries_executed = []  # type: ignore[attr-defined]
    results = query_results or {}

    def query(soql: str) -> list[dict[str, Any]]:
        conn.queries_executed.append(soql)  # type: ignore[attr-defined]
        # Match on object name in the FROM clause (longest match first to avoid
        # substring collision, e.g. AIAgentInteraction vs AIAgentInteractionStep)
        sorted_keys = sorted(results.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if f"FROM {key}" in soql:
                return results[key]
        return []

    conn.query.side_effect = query
    return conn


# --- Normalizer Tests --------------------------------------------------------


class TestAgentForceNormalizer:
    """Tests for AgentForce DMO to STRATIX event normalization."""

    def setup_method(self) -> None:
        self.normalizer = AgentForceNormalizer()

    def test_normalize_session_start(self) -> None:
        """Test that a session produces a start lifecycle event."""
        session = _session()
        events = self.normalizer.normalize_session(session)

        assert len(events) == 1  # No end event since EndTimestamp is None
        assert events[0]["event_type"] == "agent.lifecycle"
        assert events[0]["payload"]["lifecycle_action"] == "start"
        assert events[0]["payload"]["session_id"] == "a0B5f0000Sess01"
        assert events[0]["payload"]["channel_type"] == "Chat"
        assert events[0]["timestamp"] == "2026-02-21T10:00:00Z"

    def test_normalize_session_start_and_end(self) -> None:
        """Test that a completed session produces start and end events."""
        session = _session(end="2026-02-21T10:15:00Z")
        events = self.normalizer.normalize_session(session)

        assert len(events) == 2
        assert events[0]["payload"]["lifecycle_action"] == "start"
        assert events[1]["payload"]["lifecycle_action"] == "end"
        assert events[1]["payload"]["session_end_type"] == "Resolved"
        assert events[1]["timestamp"] == "2026-02-21T10:15:00Z"

    def test_normalize_participant_ai(self) -> None:
        """Test normalizing an AI participant."""
        participant = _participant(agent_type="EinsteinServiceAgent")
        event = self.normalizer.normalize_participant(participant)

        assert event["event_type"] == "agent.identity"
        assert event["payload"]["participant_type"] == "ai"
        assert event["payload"]["agent_type"] == "EinsteinServiceAgent"
        assert event["payload"]["agent_api_name"] == "my_agent"
        assert event["payload"]["agent_version"] == "v1.0"
        assert event["payload"]["role"] == "Owner"

    def test_normalize_participant_human(self) -> None:
        """Test normalizing a human (Employee) participant."""
        participant = _participant(agent_type="Employee")
        event = self.normalizer.normalize_participant(participant)

        assert event["payload"]["participant_type"] == "human"
        assert event["payload"]["agent_type"] == "Employee"

    def test_normalize_interaction(self) -> None:
        """Test normalizing an interaction turn."""
        interaction = _interaction()
        event = self.normalizer.normalize_interaction(interaction)

        assert event["event_type"] == "agent.interaction"
        assert event["identity"]["trace_id"] == "trace-abc"
        assert event["identity"]["span_id"] == "span-001"
        assert event["payload"]["interaction_type"] == "Turn"
        assert event["payload"]["topic"] == "Order_Status"
        assert event["payload"]["attributes"] == {"intent": "check_order"}

    def test_normalize_interaction_invalid_json_attributes(self) -> None:
        """Test normalizing interaction with invalid JSON in AttributeText."""
        interaction = _interaction()
        interaction["AttributeText"] = "not valid json"
        event = self.normalizer.normalize_interaction(interaction)

        assert event["payload"]["attributes"] == {"raw": "not valid json"}

    def test_normalize_interaction_no_attributes(self) -> None:
        """Test normalizing interaction with no AttributeText."""
        interaction = _interaction()
        interaction["AttributeText"] = None
        event = self.normalizer.normalize_interaction(interaction)

        assert event["payload"]["attributes"] == {}

    def test_normalize_step_user_input(self) -> None:
        """Test normalizing a UserInputStep to agent.input."""
        step = _step_user_input()
        event = self.normalizer.normalize_step(step)

        assert event["event_type"] == "agent.input"
        assert event["payload"]["content"]["role"] == "human"
        assert event["payload"]["content"]["message"] == "What is my order status?"

    def test_normalize_step_llm_execution(self) -> None:
        """Test normalizing an LLMExecutionStep to model.invoke."""
        step = _step_llm()
        event = self.normalizer.normalize_step(step)

        assert event["event_type"] == "model.invoke"
        assert event["payload"]["model"]["provider"] == "salesforce"
        assert event["payload"]["model"]["name"] == "einstein_model"
        assert event["payload"]["input_messages"][0]["content"] == "Find order status for user"
        assert event["payload"]["output_message"]["content"] == "Your order is being shipped"
        assert event["payload"]["metadata"]["generation_id"] == "gen-001"
        assert event["payload"]["metadata"]["gateway_request_id"] == "req-001"

    def test_normalize_step_function(self) -> None:
        """Test normalizing a FunctionStep to tool.call."""
        step = _step_function()
        event = self.normalizer.normalize_step(step)

        assert event["event_type"] == "tool.call"
        assert event["payload"]["tool"]["name"] == "lookup_order"
        assert event["payload"]["tool"]["integration"] == "salesforce_agentforce"
        assert event["payload"]["input"] == {"order_id": "ORD-123"}
        assert event["payload"]["output"]["status"] == "shipped"

    def test_normalize_step_function_non_json_input(self) -> None:
        """Test normalizing a FunctionStep with non-JSON input."""
        step = _step_function()
        step["InputValueText"] = "plain text input"
        event = self.normalizer.normalize_step(step)

        assert event["payload"]["input"] == {"raw": "plain text input"}

    def test_normalize_step_with_error(self) -> None:
        """Test normalizing a step with an error message."""
        step = _step_llm()
        step["ErrorMessageText"] = "Model timeout"
        event = self.normalizer.normalize_step(step)

        assert event["payload"]["error"] == "Model timeout"

    def test_normalize_step_with_timing(self) -> None:
        """Test that step timing is extracted when timestamps are present."""
        step = _step_llm()
        step["StartTimestamp"] = "2026-02-21T10:00:01Z"
        step["EndTimestamp"] = "2026-02-21T10:00:03Z"
        event = self.normalizer.normalize_step(step)

        assert event["timestamp"] == "2026-02-21T10:00:01Z"
        assert event["duration_ms"] == pytest.approx(2000.0)

    def test_normalize_step_without_timing(self) -> None:
        """Test that step without timestamps has no timing fields."""
        step = _step_llm()
        event = self.normalizer.normalize_step(step)

        assert "timestamp" not in event
        assert "duration_ms" not in event

    def test_normalize_step_action_invocation_maps_to_tool_call(self) -> None:
        """Test that ActionInvocationStep maps to tool.call."""
        step = _step_function()
        step["AiAgentInteractionStepTypeId"] = "ActionInvocationStep"
        event = self.normalizer.normalize_step(step)

        assert event["event_type"] == "tool.call"

    def test_normalize_step_unknown_type_defaults_to_tool_call(self) -> None:
        """Test that unknown step types default to tool.call."""
        step = _step_user_input()
        step["AiAgentInteractionStepTypeId"] = "CustomStep"
        event = self.normalizer.normalize_step(step)

        assert event["event_type"] == "tool.call"

    def test_normalize_step_includes_sf_metadata(self) -> None:
        """Test that normalized steps include sf.* metadata passthrough."""
        step = _step_llm()
        event = self.normalizer.normalize_step(step)

        assert event["metadata"]["sf.step.name"] == "einstein_model"
        assert event["metadata"]["sf.step.id"] == "a0B5f0000Step02"
        assert event["metadata"]["sf.generation.id"] == "gen-001"

    def test_normalize_session_includes_sf_metadata(self) -> None:
        """Test that normalized sessions include sf.* metadata passthrough."""
        session = _session(end="2026-02-21T10:15:00Z")
        events = self.normalizer.normalize_session(session)

        assert events[0]["metadata"]["sf.session.id"] == "a0B5f0000Sess01"
        assert events[0]["metadata"]["sf.session.channel"] == "Chat"

    def test_normalize_interaction_includes_sf_metadata(self) -> None:
        """Test that normalized interactions include sf.* metadata passthrough."""
        interaction = _interaction()
        event = self.normalizer.normalize_interaction(interaction)

        assert event["metadata"]["sf.topic.name"] == "Order_Status"

    def test_normalize_message_input(self) -> None:
        """Test normalizing an input message."""
        msg = _message_input()
        event = self.normalizer.normalize_message(msg)

        assert event["event_type"] == "agent.input"
        assert event["payload"]["content"]["role"] == "human"
        assert event["payload"]["content"]["message"] == "What is my order status?"
        assert event["timestamp"] == "2026-02-21T10:00:05Z"

    def test_normalize_message_output(self) -> None:
        """Test normalizing an output message."""
        msg = _message_output()
        event = self.normalizer.normalize_message(msg)

        assert event["event_type"] == "agent.output"
        assert event["payload"]["content"]["role"] == "agent"
        assert "ORD-123" in event["payload"]["content"]["message"]
        assert event["payload"]["content"]["metadata"]["parent_message_id"] == "a0B5f0000Msg001"
        assert event["timestamp"] == "2026-02-21T10:00:10Z"


# --- ImportResult Tests ------------------------------------------------------


class TestImportResult:
    """Tests for ImportResult dataclass."""

    def test_default_values(self) -> None:
        """Test default import result values."""
        result = ImportResult()
        assert result.sessions_imported == 0
        assert result.events_generated == 0
        assert result.total_records == 0
        assert result.errors == []

    def test_total_records(self) -> None:
        """Test total_records property sums all record types."""
        result = ImportResult(
            sessions_imported=5,
            participants_imported=10,
            interactions_imported=20,
            steps_imported=50,
            messages_imported=40,
        )
        assert result.total_records == 125


# --- Importer Tests (with mock connection) -----------------------------------


class TestAgentForceImporter:
    """Tests for AgentForceImporter."""

    def test_import_no_sessions(self) -> None:
        """Test importing when no sessions match the filter."""
        connection = _make_mock_connection({"AIAgentSession": []})
        importer = AgentForceImporter(connection)

        events, result = importer.import_sessions(start_date="2026-02-21")

        assert len(events) == 0
        assert result.sessions_imported == 0

    def test_import_single_session(self) -> None:
        """Test importing a single session with no related records."""
        connection = _make_mock_connection(
            {
                "AIAgentSession": [_session()],
                "AIAgentSessionParticipant": [],
                "AIAgentInteraction": [],
            }
        )
        importer = AgentForceImporter(connection)

        events, result = importer.import_sessions()

        assert result.sessions_imported == 1
        assert result.events_generated >= 1
        # Session start event
        lifecycle_events = [e for e in events if e["event_type"] == "agent.lifecycle"]
        assert len(lifecycle_events) == 1
        assert lifecycle_events[0]["payload"]["lifecycle_action"] == "start"

    def test_import_full_session(self) -> None:
        """Test importing a complete session with all related records."""
        connection = _make_mock_connection(
            {
                "AIAgentSession": [_session(end="2026-02-21T10:15:00Z")],
                "AIAgentSessionParticipant": [_participant()],
                "AIAgentInteraction": [_interaction()],
                "AIAgentInteractionStep": [_step_user_input(), _step_llm(), _step_function()],
                "AIAgentInteractionMessage": [_message_input(), _message_output()],
            }
        )
        importer = AgentForceImporter(connection)

        events, result = importer.import_sessions()

        assert result.sessions_imported == 1
        assert result.participants_imported == 1
        assert result.interactions_imported == 1
        assert result.steps_imported == 3
        assert result.messages_imported == 2

        # Verify event types present
        event_types = [e["event_type"] for e in events]
        assert "agent.lifecycle" in event_types
        assert "agent.identity" in event_types
        assert "agent.interaction" in event_types
        assert "agent.input" in event_types
        assert "model.invoke" in event_types
        assert "tool.call" in event_types
        assert "agent.output" in event_types

    def test_import_with_date_filter(self) -> None:
        """Test that date filters are included in the SOQL query."""
        connection = _make_mock_connection({"AIAgentSession": []})
        importer = AgentForceImporter(connection)

        importer.import_sessions(
            start_date="2026-02-21",
            end_date="2026-02-28",
        )

        assert len(connection.queries_executed) >= 1
        query = connection.queries_executed[0]
        assert "StartTimestamp >= 2026-02-21T00:00:00Z" in query
        assert "StartTimestamp <= 2026-02-28T23:59:59Z" in query

    def test_import_with_incremental_sync(self) -> None:
        """Test incremental sync using last_import_timestamp."""
        connection = _make_mock_connection({"AIAgentSession": []})
        importer = AgentForceImporter(connection)

        importer.import_sessions(
            last_import_timestamp="2026-02-25T15:30:00Z",
        )

        query = connection.queries_executed[0]
        assert "StartTimestamp > 2026-02-25T15:30:00Z" in query

    def test_import_with_limit(self) -> None:
        """Test that limit is included in the query."""
        connection = _make_mock_connection({"AIAgentSession": []})
        importer = AgentForceImporter(connection)

        importer.import_sessions(limit=10)

        query = connection.queries_executed[0]
        assert "LIMIT 10" in query

    def test_import_default_batch_limit(self) -> None:
        """Test that default batch_size limit is applied."""
        connection = _make_mock_connection({"AIAgentSession": []})
        importer = AgentForceImporter(connection, batch_size=50)

        importer.import_sessions()

        query = connection.queries_executed[0]
        assert "LIMIT 50" in query

    def test_import_query_error_captured(self) -> None:
        """Test that query errors are captured in result."""
        failing_connection = MagicMock(spec=SalesforceConnection)
        failing_connection.query.side_effect = ConnectionError("Network timeout")

        importer = AgentForceImporter(failing_connection)

        events, result = importer.import_sessions()

        assert len(events) == 0
        assert len(result.errors) == 1
        assert "Session query failed" in result.errors[0]

    def test_import_events_generated_count(self) -> None:
        """Test that events_generated count matches actual events."""
        connection = _make_mock_connection(
            {
                "AIAgentSession": [_session(end="2026-02-21T10:15:00Z")],
                "AIAgentSessionParticipant": [_participant()],
                "AIAgentInteraction": [_interaction()],
                "AIAgentInteractionStep": [_step_llm()],
                "AIAgentInteractionMessage": [_message_input()],
            }
        )
        importer = AgentForceImporter(connection)

        events, result = importer.import_sessions()

        assert result.events_generated == len(events)


# --- SOQL Validation Tests ---------------------------------------------------


class TestImporterSOQLValidation:
    """Tests for SOQL input validation and injection prevention."""

    def test_validate_date_accepts_valid(self) -> None:
        """Valid ISO 8601 dates pass validation."""
        AgentForceImporter._validate_date("2026-02-21")
        AgentForceImporter._validate_date("2025-01-01")

    def test_validate_date_rejects_invalid(self) -> None:
        """Invalid date formats raise ValueError."""
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            AgentForceImporter._validate_date("02-21-2026")

        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            AgentForceImporter._validate_date("2026/02/21")

        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            AgentForceImporter._validate_date("not-a-date")

    def test_validate_date_rejects_soql_injection(self) -> None:
        """Date validation blocks SOQL injection attempts."""
        with pytest.raises(ValueError):
            AgentForceImporter._validate_date("2026-01-01' OR 1=1 --")

    def test_validate_timestamp_accepts_valid(self) -> None:
        """Valid ISO 8601 timestamps pass validation."""
        AgentForceImporter._validate_timestamp("2026-02-21T10:00:00Z")
        AgentForceImporter._validate_timestamp("2026-02-21T10:00:00.123Z")

    def test_validate_timestamp_rejects_invalid(self) -> None:
        """Invalid timestamps raise ValueError."""
        with pytest.raises(ValueError, match="ISO 8601"):
            AgentForceImporter._validate_timestamp("2026-02-21")  # date only

        with pytest.raises(ValueError, match="ISO 8601"):
            AgentForceImporter._validate_timestamp("not-a-timestamp")

    def test_validate_sf_id_rejects_quotes(self) -> None:
        """SOQL ID validation rejects IDs with quotes."""
        with pytest.raises(ValueError, match="Invalid Salesforce ID"):
            AgentForceImporter._validate_sf_id("abc'123")

    def test_validate_sf_id_rejects_backslashes(self) -> None:
        """SOQL ID validation rejects IDs with backslashes."""
        with pytest.raises(ValueError, match="Invalid Salesforce ID"):
            AgentForceImporter._validate_sf_id("abc\\123")

    def test_validate_sf_id_accepts_15_char(self) -> None:
        """Valid 15-char Salesforce IDs pass through unchanged."""
        assert AgentForceImporter._validate_sf_id("001000000000001") == "001000000000001"

    def test_validate_sf_id_accepts_18_char(self) -> None:
        """Valid 18-char Salesforce IDs pass through unchanged."""
        assert AgentForceImporter._validate_sf_id("001000000000001AAA") == "001000000000001AAA"

    def test_validate_sf_id_rejects_wrong_length(self) -> None:
        """SOQL ID validation rejects IDs with wrong length."""
        with pytest.raises(ValueError, match="Invalid Salesforce ID"):
            AgentForceImporter._validate_sf_id("too-short")

    def test_import_invalid_date_raises(self) -> None:
        """Import with invalid date raises ValueError."""
        connection = _make_mock_connection({"AIAgentSession": []})
        importer = AgentForceImporter(connection)

        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            importer.import_sessions(start_date="bad-date")

    def test_import_invalid_timestamp_raises(self) -> None:
        """Import with invalid timestamp raises ValueError."""
        connection = _make_mock_connection({"AIAgentSession": []})
        importer = AgentForceImporter(connection)

        with pytest.raises(ValueError, match="ISO 8601"):
            importer.import_sessions(last_import_timestamp="not-a-timestamp")

    def test_related_query_error_propagated_to_result(self) -> None:
        """Errors from related record queries appear in ImportResult.errors."""
        partial_fail = MagicMock(spec=SalesforceConnection)

        def query(soql: str) -> list[dict[str, Any]]:
            # Check longer object names first to avoid substring matches
            if "FROM AIAgentSessionParticipant" in soql:
                raise ConnectionError("participant query failed")
            if "FROM AIAgentSession" in soql:
                return [_session()]
            return []

        partial_fail.query.side_effect = query

        importer = AgentForceImporter(partial_fail)
        events, result = importer.import_sessions()

        assert result.sessions_imported == 1
        assert len(result.errors) >= 1
        assert "AIAgentSessionParticipant" in result.errors[0]


# --- Auth Tests --------------------------------------------------------------


class TestSalesforceCredentials:
    """Tests for SalesforceCredentials dataclass."""

    def test_token_not_expired_initially(self) -> None:
        """Test that a credential with future expiry is not expired."""
        import time

        creds = SalesforceCredentials(
            client_id="test",
            username="test@example.com",
            private_key="fake-key",
            token_expiry=time.time() + 3600,
        )
        assert creds.is_expired is False

    def test_token_expired(self) -> None:
        """Test that a credential with past expiry is expired."""
        creds = SalesforceCredentials(
            client_id="test",
            username="test@example.com",
            private_key="fake-key",
            token_expiry=0.0,
        )
        assert creds.is_expired is True

    def test_default_instance_url(self) -> None:
        """Test default Salesforce instance URL."""
        creds = SalesforceCredentials(
            client_id="test",
            username="test@example.com",
            private_key="fake-key",
        )
        assert creds.instance_url == "https://login.salesforce.com"
