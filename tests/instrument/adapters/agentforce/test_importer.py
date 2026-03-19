"""Tests for AgentForce import adapter and normalizer."""

import pytest

from layerlens.instrument.adapters.agentforce.normalizer import AgentForceNormalizer
from layerlens.instrument.adapters.agentforce.importer import (
    AgentForceImporter,
    ImportResult,
)
from layerlens.instrument.adapters.agentforce.auth import (
    SalesforceCredentials,
    SalesforceConnection,
)


# --- Test Data ---


def _session(sid="sess-001", start="2026-02-21T10:00:00Z", end=None, channel="Chat"):
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


def _participant(pid="part-001", session_id="sess-001", agent_type="EinsteinServiceAgent"):
    return {
        "Id": pid,
        "AiAgentSessionId": session_id,
        "AiAgentTypeId": agent_type,
        "AiAgentApiName": "my_agent",
        "AiAgentVersionApiName": "v1.0",
        "ParticipantId": "user-001",
        "AiAgentSessionParticipantRoleId": "Owner",
    }


def _interaction(iid="int-001", session_id="sess-001"):
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


def _step_user_input(step_id="step-001", interaction_id="int-001"):
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
        "TelemetryTraceSpanId": "span-step-001",
    }


def _step_llm(step_id="step-002", interaction_id="int-001"):
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
        "TelemetryTraceSpanId": "span-step-002",
    }


def _step_function(step_id="step-003", interaction_id="int-001"):
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
        "TelemetryTraceSpanId": "span-step-003",
    }


def _message_input(mid="msg-001", interaction_id="int-001"):
    return {
        "Id": mid,
        "AiAgentInteractionId": interaction_id,
        "AiAgentInteractionMessageTypeId": "Input",
        "ContentText": "What is my order status?",
        "AiAgentInteractionMsgContentTypeId": "text/plain",
        "MessageSentTimestamp": "2026-02-21T10:00:05Z",
        "ParentMessageId": None,
    }


def _message_output(mid="msg-002", interaction_id="int-001"):
    return {
        "Id": mid,
        "AiAgentInteractionId": interaction_id,
        "AiAgentInteractionMessageTypeId": "Output",
        "ContentText": "Your order ORD-123 is being shipped and will arrive by March 1st.",
        "AiAgentInteractionMsgContentTypeId": "text/plain",
        "MessageSentTimestamp": "2026-02-21T10:00:10Z",
        "ParentMessageId": "msg-001",
    }


# --- Normalizer Tests ---


class TestAgentForceNormalizer:
    """Tests for AgentForce DMO to STRATIX event normalization."""

    def setup_method(self):
        self.normalizer = AgentForceNormalizer()

    def test_normalize_session_start(self):
        """Test that a session produces a start lifecycle event."""
        session = _session()
        events = self.normalizer.normalize_session(session)

        assert len(events) == 1  # No end event since EndTimestamp is None
        assert events[0]["event_type"] == "agent.lifecycle"
        assert events[0]["payload"]["lifecycle_action"] == "start"
        assert events[0]["payload"]["session_id"] == "sess-001"
        assert events[0]["payload"]["channel_type"] == "Chat"
        assert events[0]["timestamp"] == "2026-02-21T10:00:00Z"

    def test_normalize_session_start_and_end(self):
        """Test that a completed session produces start and end events."""
        session = _session(end="2026-02-21T10:15:00Z")
        events = self.normalizer.normalize_session(session)

        assert len(events) == 2
        assert events[0]["payload"]["lifecycle_action"] == "start"
        assert events[1]["payload"]["lifecycle_action"] == "end"
        assert events[1]["payload"]["session_end_type"] == "Resolved"
        assert events[1]["timestamp"] == "2026-02-21T10:15:00Z"

    def test_normalize_participant_ai(self):
        """Test normalizing an AI participant."""
        participant = _participant(agent_type="EinsteinServiceAgent")
        event = self.normalizer.normalize_participant(participant)

        assert event["event_type"] == "agent.identity"
        assert event["payload"]["participant_type"] == "ai"
        assert event["payload"]["agent_type"] == "EinsteinServiceAgent"
        assert event["payload"]["agent_api_name"] == "my_agent"
        assert event["payload"]["agent_version"] == "v1.0"
        assert event["payload"]["role"] == "Owner"

    def test_normalize_participant_human(self):
        """Test normalizing a human (Employee) participant."""
        participant = _participant(agent_type="Employee")
        event = self.normalizer.normalize_participant(participant)

        assert event["payload"]["participant_type"] == "human"
        assert event["payload"]["agent_type"] == "Employee"

    def test_normalize_interaction(self):
        """Test normalizing an interaction turn."""
        interaction = _interaction()
        event = self.normalizer.normalize_interaction(interaction)

        assert event["event_type"] == "agent.interaction"
        assert event["identity"]["trace_id"] == "trace-abc"
        assert event["identity"]["span_id"] == "span-001"
        assert event["payload"]["interaction_type"] == "Turn"
        assert event["payload"]["topic"] == "Order_Status"
        assert event["payload"]["attributes"] == {"intent": "check_order"}

    def test_normalize_interaction_invalid_json_attributes(self):
        """Test normalizing interaction with invalid JSON in AttributeText."""
        interaction = _interaction()
        interaction["AttributeText"] = "not valid json"
        event = self.normalizer.normalize_interaction(interaction)

        assert event["payload"]["attributes"] == {"raw": "not valid json"}

    def test_normalize_interaction_no_attributes(self):
        """Test normalizing interaction with no AttributeText."""
        interaction = _interaction()
        interaction["AttributeText"] = None
        event = self.normalizer.normalize_interaction(interaction)

        assert event["payload"]["attributes"] == {}

    def test_normalize_step_user_input(self):
        """Test normalizing a UserInputStep to agent.input."""
        step = _step_user_input()
        event = self.normalizer.normalize_step(step)

        assert event["event_type"] == "agent.input"
        assert event["payload"]["content"]["role"] == "human"
        assert event["payload"]["content"]["message"] == "What is my order status?"

    def test_normalize_step_llm_execution(self):
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

    def test_normalize_step_function(self):
        """Test normalizing a FunctionStep to tool.call."""
        step = _step_function()
        event = self.normalizer.normalize_step(step)

        assert event["event_type"] == "tool.call"
        assert event["payload"]["tool"]["name"] == "lookup_order"
        assert event["payload"]["tool"]["integration"] == "salesforce_agentforce"
        assert event["payload"]["input"] == {"order_id": "ORD-123"}
        assert event["payload"]["output"]["status"] == "shipped"

    def test_normalize_step_function_non_json_input(self):
        """Test normalizing a FunctionStep with non-JSON input."""
        step = _step_function()
        step["InputValueText"] = "plain text input"
        event = self.normalizer.normalize_step(step)

        assert event["payload"]["input"] == {"raw": "plain text input"}

    def test_normalize_step_with_error(self):
        """Test normalizing a step with an error message."""
        step = _step_llm()
        step["ErrorMessageText"] = "Model timeout"
        event = self.normalizer.normalize_step(step)

        assert event["payload"]["error"] == "Model timeout"

    def test_normalize_step_with_timing(self):
        """Test that step timing is extracted when timestamps are present."""
        step = _step_llm()
        step["StartTimestamp"] = "2026-02-21T10:00:01Z"
        step["EndTimestamp"] = "2026-02-21T10:00:03Z"
        event = self.normalizer.normalize_step(step)

        assert event["timestamp"] == "2026-02-21T10:00:01Z"
        assert event["duration_ms"] == pytest.approx(2000.0)

    def test_normalize_step_without_timing(self):
        """Test that step without timestamps has no timing fields."""
        step = _step_llm()
        event = self.normalizer.normalize_step(step)

        assert "timestamp" not in event
        assert "duration_ms" not in event

    def test_normalize_step_action_invocation_maps_to_tool_call(self):
        """Test that ActionInvocationStep maps to tool.call."""
        step = _step_function()
        step["AiAgentInteractionStepTypeId"] = "ActionInvocationStep"
        event = self.normalizer.normalize_step(step)

        assert event["event_type"] == "tool.call"

    def test_normalize_step_unknown_type_defaults_to_tool_call(self):
        """Test that unknown step types default to tool.call."""
        step = _step_user_input()
        step["AiAgentInteractionStepTypeId"] = "CustomStep"
        event = self.normalizer.normalize_step(step)

        assert event["event_type"] == "tool.call"

    def test_normalize_step_includes_sf_metadata(self):
        """Test that normalized steps include sf.* metadata passthrough."""
        step = _step_llm()
        event = self.normalizer.normalize_step(step)

        assert event["metadata"]["sf.step.name"] == "einstein_model"
        assert event["metadata"]["sf.step.id"] == "step-002"
        assert event["metadata"]["sf.generation.id"] == "gen-001"

    def test_normalize_session_includes_sf_metadata(self):
        """Test that normalized sessions include sf.* metadata passthrough."""
        session = _session(end="2026-02-21T10:15:00Z")
        events = self.normalizer.normalize_session(session)

        assert events[0]["metadata"]["sf.session.id"] == "sess-001"
        assert events[0]["metadata"]["sf.session.channel"] == "Chat"

    def test_normalize_interaction_includes_sf_metadata(self):
        """Test that normalized interactions include sf.* metadata passthrough."""
        interaction = _interaction()
        event = self.normalizer.normalize_interaction(interaction)

        assert event["metadata"]["sf.topic.name"] == "Order_Status"

    def test_normalize_message_input(self):
        """Test normalizing an input message."""
        msg = _message_input()
        event = self.normalizer.normalize_message(msg)

        assert event["event_type"] == "agent.input"
        assert event["payload"]["content"]["role"] == "human"
        assert event["payload"]["content"]["message"] == "What is my order status?"
        assert event["timestamp"] == "2026-02-21T10:00:05Z"

    def test_normalize_message_output(self):
        """Test normalizing an output message."""
        msg = _message_output()
        event = self.normalizer.normalize_message(msg)

        assert event["event_type"] == "agent.output"
        assert event["payload"]["content"]["role"] == "agent"
        assert "ORD-123" in event["payload"]["content"]["message"]
        assert event["payload"]["content"]["metadata"]["parent_message_id"] == "msg-001"
        assert event["timestamp"] == "2026-02-21T10:00:10Z"


# --- ImportResult Tests ---


class TestImportResult:
    """Tests for ImportResult dataclass."""

    def test_default_values(self):
        """Test default import result values."""
        result = ImportResult()
        assert result.sessions_imported == 0
        assert result.events_generated == 0
        assert result.total_records == 0
        assert result.errors == []

    def test_total_records(self):
        """Test total_records property sums all record types."""
        result = ImportResult(
            sessions_imported=5,
            participants_imported=10,
            interactions_imported=20,
            steps_imported=50,
            messages_imported=40,
        )
        assert result.total_records == 125


# --- Importer Tests (with mock connection) ---


class MockConnection:
    """Mock Salesforce connection for testing."""

    def __init__(self, query_results=None):
        self._query_results = query_results or {}
        self.queries_executed = []

    def query(self, soql):
        self.queries_executed.append(soql)
        # Match on object name in the FROM clause (longest match first to avoid
        # substring collision, e.g. AIAgentInteraction vs AIAgentInteractionStep)
        sorted_keys = sorted(self._query_results.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if f"FROM {key}" in soql:
                return self._query_results[key]
        return []


class TestAgentForceImporter:
    """Tests for AgentForceImporter."""

    def test_import_no_sessions(self):
        """Test importing when no sessions match the filter."""
        connection = MockConnection({"AIAgentSession": []})
        importer = AgentForceImporter(connection)

        events, result = importer.import_sessions(start_date="2026-02-21")

        assert len(events) == 0
        assert result.sessions_imported == 0

    def test_import_single_session(self):
        """Test importing a single session with no related records."""
        connection = MockConnection({
            "AIAgentSession": [_session()],
            "AIAgentSessionParticipant": [],
            "AIAgentInteraction": [],
        })
        importer = AgentForceImporter(connection)

        events, result = importer.import_sessions()

        assert result.sessions_imported == 1
        assert result.events_generated >= 1
        # Session start event
        lifecycle_events = [e for e in events if e["event_type"] == "agent.lifecycle"]
        assert len(lifecycle_events) == 1
        assert lifecycle_events[0]["payload"]["lifecycle_action"] == "start"

    def test_import_full_session(self):
        """Test importing a complete session with all related records."""
        connection = MockConnection({
            "AIAgentSession": [_session(end="2026-02-21T10:15:00Z")],
            "AIAgentSessionParticipant": [_participant()],
            "AIAgentInteraction": [_interaction()],
            "AIAgentInteractionStep": [_step_user_input(), _step_llm(), _step_function()],
            "AIAgentInteractionMessage": [_message_input(), _message_output()],
        })
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

    def test_import_with_date_filter(self):
        """Test that date filters are included in the SOQL query."""
        connection = MockConnection({"AIAgentSession": []})
        importer = AgentForceImporter(connection)

        importer.import_sessions(
            start_date="2026-02-21",
            end_date="2026-02-28",
        )

        assert len(connection.queries_executed) >= 1
        query = connection.queries_executed[0]
        assert "StartTimestamp >= 2026-02-21T00:00:00Z" in query
        assert "StartTimestamp <= 2026-02-28T23:59:59Z" in query

    def test_import_with_incremental_sync(self):
        """Test incremental sync using last_import_timestamp."""
        connection = MockConnection({"AIAgentSession": []})
        importer = AgentForceImporter(connection)

        importer.import_sessions(
            last_import_timestamp="2026-02-25T15:30:00Z",
        )

        query = connection.queries_executed[0]
        assert "StartTimestamp > 2026-02-25T15:30:00Z" in query

    def test_import_with_limit(self):
        """Test that limit is included in the query."""
        connection = MockConnection({"AIAgentSession": []})
        importer = AgentForceImporter(connection)

        importer.import_sessions(limit=10)

        query = connection.queries_executed[0]
        assert "LIMIT 10" in query

    def test_import_default_batch_limit(self):
        """Test that default batch_size limit is applied."""
        connection = MockConnection({"AIAgentSession": []})
        importer = AgentForceImporter(connection, batch_size=50)

        importer.import_sessions()

        query = connection.queries_executed[0]
        assert "LIMIT 50" in query

    def test_import_query_error_captured(self):
        """Test that query errors are captured in result."""
        class FailingConnection:
            def query(self, soql):
                raise ConnectionError("Network timeout")

        importer = AgentForceImporter(FailingConnection())

        events, result = importer.import_sessions()

        assert len(events) == 0
        assert len(result.errors) == 1
        assert "Session query failed" in result.errors[0]

    def test_import_events_generated_count(self):
        """Test that events_generated count matches actual events."""
        connection = MockConnection({
            "AIAgentSession": [_session(end="2026-02-21T10:15:00Z")],
            "AIAgentSessionParticipant": [_participant()],
            "AIAgentInteraction": [_interaction()],
            "AIAgentInteractionStep": [_step_llm()],
            "AIAgentInteractionMessage": [_message_input()],
        })
        importer = AgentForceImporter(connection)

        events, result = importer.import_sessions()

        assert result.events_generated == len(events)


# --- SOQL Validation Tests ---


class TestImporterSOQLValidation:
    """Tests for SOQL input validation and injection prevention."""

    def test_validate_date_accepts_valid(self):
        """Valid ISO 8601 dates pass validation."""
        AgentForceImporter._validate_date("2026-02-21")
        AgentForceImporter._validate_date("2025-01-01")

    def test_validate_date_rejects_invalid(self):
        """Invalid date formats raise ValueError."""
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            AgentForceImporter._validate_date("02-21-2026")

        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            AgentForceImporter._validate_date("2026/02/21")

        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            AgentForceImporter._validate_date("not-a-date")

    def test_validate_date_rejects_soql_injection(self):
        """Date validation blocks SOQL injection attempts."""
        with pytest.raises(ValueError):
            AgentForceImporter._validate_date("2026-01-01' OR 1=1 --")

    def test_validate_timestamp_accepts_valid(self):
        """Valid ISO 8601 timestamps pass validation."""
        AgentForceImporter._validate_timestamp("2026-02-21T10:00:00Z")
        AgentForceImporter._validate_timestamp("2026-02-21T10:00:00.123Z")

    def test_validate_timestamp_rejects_invalid(self):
        """Invalid timestamps raise ValueError."""
        with pytest.raises(ValueError, match="ISO 8601"):
            AgentForceImporter._validate_timestamp("2026-02-21")  # date only

        with pytest.raises(ValueError, match="ISO 8601"):
            AgentForceImporter._validate_timestamp("not-a-timestamp")

    def test_escape_soql_id_strips_quotes(self):
        """SOQL ID escaping strips single quotes."""
        assert AgentForceImporter._escape_soql_id("abc'123") == "abc123"

    def test_escape_soql_id_strips_backslashes(self):
        """SOQL ID escaping strips backslashes."""
        assert AgentForceImporter._escape_soql_id("abc\\123") == "abc123"

    def test_escape_soql_id_normal_id(self):
        """Normal Salesforce IDs pass through unchanged."""
        assert AgentForceImporter._escape_soql_id("001000000000001") == "001000000000001"

    def test_import_invalid_date_raises(self):
        """Import with invalid date raises ValueError."""
        connection = MockConnection({"AIAgentSession": []})
        importer = AgentForceImporter(connection)

        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            importer.import_sessions(start_date="bad-date")

    def test_import_invalid_timestamp_raises(self):
        """Import with invalid timestamp raises ValueError."""
        connection = MockConnection({"AIAgentSession": []})
        importer = AgentForceImporter(connection)

        with pytest.raises(ValueError, match="ISO 8601"):
            importer.import_sessions(last_import_timestamp="not-a-timestamp")

    def test_related_query_error_propagated_to_result(self):
        """Errors from related record queries appear in ImportResult.errors."""
        class PartialFailConnection:
            def query(self, soql):
                # Check longer object names first to avoid substring matches
                if "FROM AIAgentSessionParticipant" in soql:
                    raise ConnectionError("participant query failed")
                if "FROM AIAgentSession" in soql:
                    return [_session()]
                return []

        importer = AgentForceImporter(PartialFailConnection())
        events, result = importer.import_sessions()

        assert result.sessions_imported == 1
        assert len(result.errors) >= 1
        assert "AIAgentSessionParticipant" in result.errors[0]


# --- Auth Tests ---


class TestSalesforceCredentials:
    """Tests for SalesforceCredentials dataclass."""

    def test_token_not_expired_initially(self):
        """Test that a credential with future expiry is not expired."""
        import time
        creds = SalesforceCredentials(
            client_id="test",
            username="test@example.com",
            private_key="fake-key",
            token_expiry=time.time() + 3600,
        )
        assert creds.is_expired is False

    def test_token_expired(self):
        """Test that a credential with past expiry is expired."""
        creds = SalesforceCredentials(
            client_id="test",
            username="test@example.com",
            private_key="fake-key",
            token_expiry=0.0,
        )
        assert creds.is_expired is True

    def test_default_instance_url(self):
        """Test default Salesforce instance URL."""
        creds = SalesforceCredentials(
            client_id="test",
            username="test@example.com",
            private_key="fake-key",
        )
        assert creds.instance_url == "https://login.salesforce.com"
