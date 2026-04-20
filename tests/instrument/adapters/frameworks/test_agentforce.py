"""Tests for the Agentforce adapter (batch import from Salesforce Data Cloud).

Mocks httpx and Salesforce connection since they are not available in tests.
Drives import via ``_import_session`` and asserts flat event emission.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import Mock

import pytest

import layerlens.instrument.adapters.frameworks.agentforce as _mod
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks._utils import truncate as _truncate
from layerlens.instrument.adapters.frameworks.agentforce import (
    AgentforceAdapter,
    _int_or_zero,
    _sf_datetime,
    _SalesforceCredentials,
)

from .conftest import find_event, find_events, capture_framework_trace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_httpx(monkeypatch: Any) -> None:
    monkeypatch.setattr(_mod, "_HAS_HTTPX", True)


def _make_session(session_id: str = "sess-001", agent_id: str = "agent-001", **overrides: Any) -> dict:
    base = {
        "Id": session_id,
        "AgentId": agent_id,
        "AgentName": "TestAgent",
        "ParticipantId": "user-001",
        "ParticipantName": "Gary",
        "Channel": "web",
        "Status": "Completed",
        "Outcome": "Resolved",
        "StartTime": "2026-03-01T10:00:00Z",
        "EndTime": "2026-03-01T10:05:00Z",
    }
    base.update(overrides)
    return base


def _make_interaction(step_type: str = "llm", **overrides: Any) -> dict:
    base = {
        "Id": "int-001",
        "SessionId": "sess-001",
        "StepType": step_type,
        "StepName": "generate_response",
        "Sequence": 1,
        "Input": "What is the weather?",
        "Output": "It's sunny today.",
        "ModelName": "gpt-4",
        "PromptTokens": 50,
        "CompletionTokens": 25,
        "ToolName": None,
        "ToolInput": None,
        "ToolOutput": None,
        "EscalationTarget": None,
        "ErrorMessage": None,
    }
    base.update(overrides)
    return base


def _make_agent_config(agent_id: str = "agent-001") -> dict:
    return {
        "Id": "cfg-001",
        "AgentId": agent_id,
        "AgentName": "TestAgent",
        "Description": "A helpful test agent",
        "ModelName": "gpt-4",
        "Instructions": "Be helpful.",
        "TopicCount": 3,
        "ActionCount": 5,
    }


def _make_mock_conn(
    sessions: Optional[list] = None,
    interactions: Optional[list] = None,
    agent_config: Optional[list] = None,
) -> Mock:
    sessions = sessions or []
    interactions = interactions or []
    agent_config = agent_config or []

    def _query(soql: str) -> list:
        if "AIAgentSession__dlm" in soql:
            return sessions
        elif "AIAgentInteraction__dlm" in soql:
            return interactions
        elif "AIAgentConfiguration__dlm" in soql:
            return agent_config
        return []

    conn = Mock(spec=[])
    conn.authenticate = Mock()
    conn.close = Mock()
    conn.query = Mock(side_effect=_query)
    return conn


def _setup(
    mock_client: Any,
    capture_config: Optional[CaptureConfig] = None,
    **conn_kwargs: Any,
) -> tuple:
    uploaded = capture_framework_trace(mock_client)
    adapter = AgentforceAdapter(mock_client, capture_config=capture_config)
    mock_conn = _make_mock_conn(**conn_kwargs)
    adapter._connection = mock_conn
    adapter._connected = True
    adapter._credentials = _SalesforceCredentials(
        client_id="test",
        client_secret="test",
        instance_url="https://test.salesforce.com",
        access_token="fake-token",
    )
    adapter._metadata["instance_url"] = "https://test.salesforce.com"
    return adapter, uploaded, mock_conn


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_adapter_info(self, mock_client):
        adapter = AgentforceAdapter(mock_client)
        info = adapter.adapter_info()
        assert info.name == "agentforce"
        assert not info.connected

    def test_raises_when_httpx_missing(self, mock_client, monkeypatch):
        monkeypatch.setattr(_mod, "_HAS_HTTPX", False)
        with pytest.raises(ImportError, match="httpx"):
            AgentforceAdapter(mock_client).connect(
                credentials={
                    "client_id": "x",
                    "client_secret": "y",
                    "instance_url": "https://test.salesforce.com",
                }
            )

    def test_raises_when_credentials_missing(self, mock_client):
        with pytest.raises(ValueError, match="credentials are required"):
            AgentforceAdapter(mock_client).connect()

    def test_raises_when_instance_url_missing(self, mock_client):
        with pytest.raises(ValueError, match="instance_url is required"):
            AgentforceAdapter(mock_client).connect(
                credentials={
                    "client_id": "x",
                    "client_secret": "y",
                }
            )

    def test_disconnect_closes_connection(self, mock_client):
        adapter, _, mock_conn = _setup(mock_client)
        adapter.disconnect()
        mock_conn.close.assert_called_once()
        assert not adapter.is_connected

    def test_raises_when_not_connected(self, mock_client):
        adapter = AgentforceAdapter(mock_client)
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.import_sessions()

    def test_metadata_includes_instance_url(self, mock_client):
        adapter, _, _ = _setup(mock_client)
        assert adapter.adapter_info().metadata["instance_url"] == "https://test.salesforce.com"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


class TestCredentials:
    def test_normalizes_instance_url(self):
        creds = _SalesforceCredentials(
            client_id="x",
            client_secret="y",
            instance_url="https://test.salesforce.com/",
        )
        assert creds.instance_url == "https://test.salesforce.com"

    def test_builds_token_url(self):
        creds = _SalesforceCredentials(
            client_id="x",
            client_secret="y",
            instance_url="https://test.salesforce.com",
        )
        assert creds.token_url == "https://test.salesforce.com/services/oauth2/token"


# ---------------------------------------------------------------------------
# Session import — summary
# ---------------------------------------------------------------------------


class TestImportSessions:
    def test_returns_correct_counts(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            sessions=[_make_session()],
            interactions=[_make_interaction()],
            agent_config=[_make_agent_config()],
        )
        summary = adapter.import_sessions()
        assert summary["sessions_imported"] == 1
        assert summary["events_emitted"] > 0
        assert summary["errors"] == 0

    def test_no_sessions_returns_zeros(self, mock_client):
        adapter, _, _ = _setup(mock_client, sessions=[])
        summary = adapter.import_sessions()
        assert summary["sessions_imported"] == 0
        assert summary["events_emitted"] == 0
        assert summary["errors"] == 0
        # No cursor advancement when there's nothing to import.
        assert summary["next_cursor"] is None


# ---------------------------------------------------------------------------
# Session processing
# ---------------------------------------------------------------------------


class TestSessionProcessing:
    def test_emits_agent_input(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client, sessions=[_make_session()], interactions=[])
        adapter.import_sessions()
        inp = find_event(uploaded["events"], "agent.input")
        assert inp["payload"]["session_id"] == "sess-001"
        assert inp["payload"]["agent_name"] == "TestAgent"
        assert inp["payload"]["participant_name"] == "Gary"
        assert inp["payload"]["framework"] == "agentforce"

    def test_emits_agent_output(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client, sessions=[_make_session()], interactions=[])
        adapter.import_sessions()
        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["status"] == "Completed"
        assert out["payload"]["outcome"] == "Resolved"

    def test_emits_environment_config(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            sessions=[_make_session()],
            interactions=[],
            agent_config=[_make_agent_config()],
        )
        adapter.import_sessions()
        cfg = find_event(uploaded["events"], "environment.config")
        assert cfg["payload"]["agent_id"] == "agent-001"
        assert cfg["payload"]["model"] == "gpt-4"
        assert cfg["payload"]["description"] == "A helpful test agent"

    def test_skips_config_when_no_records(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client, sessions=[_make_session()], interactions=[], agent_config=[])
        adapter.import_sessions()
        assert len(find_events(uploaded["events"], "environment.config")) == 0

    def test_per_session_trace(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            sessions=[_make_session("s1"), _make_session("s2")],
            interactions=[],
        )
        adapter.import_sessions()
        inputs = find_events(uploaded["events"], "agent.input")
        assert len(inputs) == 2


# ---------------------------------------------------------------------------
# Interaction steps — LLM
# ---------------------------------------------------------------------------


class TestLLMStep:
    def test_model_invoke_emitted(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            sessions=[_make_session()],
            interactions=[_make_interaction(step_type="llm")],
        )
        adapter.import_sessions()
        me = find_event(uploaded["events"], "model.invoke")
        assert me["payload"]["model"] == "gpt-4"
        assert me["payload"]["tokens_prompt"] == 50
        assert me["payload"]["tokens_completion"] == 25
        assert me["payload"]["tokens_total"] == 75

    def test_cost_record_emitted(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            sessions=[_make_session()],
            interactions=[_make_interaction(step_type="model", PromptTokens=100, CompletionTokens=50)],
        )
        adapter.import_sessions()
        cost = find_event(uploaded["events"], "cost.record")
        assert cost["payload"]["tokens_total"] == 150
        assert cost["payload"]["model"] == "gpt-4"

    def test_content_gating(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=False),
            sessions=[_make_session()],
            interactions=[_make_interaction()],
        )
        adapter.import_sessions()
        me = find_event(uploaded["events"], "model.invoke")
        assert "messages" not in me["payload"]
        assert "output_message" not in me["payload"]


# ---------------------------------------------------------------------------
# Interaction steps — tool
# ---------------------------------------------------------------------------


class TestToolStep:
    def test_tool_call_emitted(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            sessions=[_make_session()],
            interactions=[
                _make_interaction(
                    step_type="action",
                    ToolName="get_weather",
                    ToolInput='{"city": "SF"}',
                    ToolOutput='{"temp": 72}',
                )
            ],
        )
        adapter.import_sessions()
        tc = find_event(uploaded["events"], "tool.call")
        assert tc["payload"]["tool_name"] == "get_weather"
        assert tc["payload"]["input"] == '{"city": "SF"}'
        assert tc["payload"]["output"] == '{"temp": 72}'

    def test_tool_content_gating(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=False),
            sessions=[_make_session()],
            interactions=[
                _make_interaction(
                    step_type="action",
                    ToolName="t",
                    ToolInput="secret",
                    ToolOutput="classified",
                )
            ],
        )
        adapter.import_sessions()
        tc = find_event(uploaded["events"], "tool.call")
        assert "input" not in tc["payload"]
        assert "output" not in tc["payload"]


# ---------------------------------------------------------------------------
# Interaction steps — handoff
# ---------------------------------------------------------------------------


class TestHandoffStep:
    def test_handoff_emitted(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            sessions=[_make_session()],
            interactions=[
                _make_interaction(
                    step_type="escalation",
                    StepName="escalate_to_human",
                    EscalationTarget="support-queue-1",
                    Input="Customer needs help",
                )
            ],
        )
        adapter.import_sessions()
        h = find_event(uploaded["events"], "agent.handoff")
        assert h["payload"]["escalation_target"] == "support-queue-1"
        assert h["payload"]["step_name"] == "escalate_to_human"
        assert h["payload"]["reason"] == "Customer needs help"


# ---------------------------------------------------------------------------
# Unknown step types
# ---------------------------------------------------------------------------


class TestUnknownStep:
    def test_unknown_emits_agent_interaction(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            sessions=[_make_session()],
            interactions=[_make_interaction(step_type="custom_routing", StepName="route_to_topic")],
        )
        adapter.import_sessions()
        evt = find_event(uploaded["events"], "agent.interaction")
        assert evt["payload"]["step_type"] == "custom_routing"
        assert evt["payload"]["step_name"] == "route_to_topic"


# ---------------------------------------------------------------------------
# Full invocation
# ---------------------------------------------------------------------------


class TestFullInvocation:
    def test_complete_session(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            sessions=[_make_session()],
            interactions=[
                _make_interaction(step_type="llm"),
                _make_interaction(step_type="action", ToolName="search", ToolInput="{}", ToolOutput="found"),
            ],
            agent_config=[_make_agent_config()],
        )
        adapter.import_sessions()
        events = uploaded["events"]

        assert len(find_events(events, "environment.config")) == 1
        assert len(find_events(events, "agent.input")) == 1
        assert len(find_events(events, "agent.output")) == 1
        assert len(find_events(events, "model.invoke")) == 1
        assert len(find_events(events, "cost.record")) == 1
        assert len(find_events(events, "tool.call")) == 1


# ---------------------------------------------------------------------------
# Trace integrity
# ---------------------------------------------------------------------------


class TestTraceIntegrity:
    def test_shared_trace_id_within_session(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            sessions=[_make_session()],
            interactions=[_make_interaction()],
        )
        adapter.import_sessions()
        trace_ids = {e["trace_id"] for e in uploaded["events"]}
        assert len(trace_ids) == 1

    def test_monotonic_sequence_ids(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            sessions=[_make_session()],
            interactions=[_make_interaction(), _make_interaction(step_type="action", ToolName="t")],
        )
        adapter.import_sessions()
        seq = [e["sequence_id"] for e in uploaded["events"]]
        assert seq == sorted(seq)


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    def test_session_error_counted(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = AgentforceAdapter(mock_client)
        mock_conn = Mock(spec=[])
        mock_conn.authenticate = Mock()
        mock_conn.close = Mock()
        adapter._connection = mock_conn
        adapter._connected = True
        adapter._credentials = _SalesforceCredentials(
            client_id="test",
            client_secret="test",
            instance_url="https://test.salesforce.com",
            access_token="fake-token",
        )

        call_count = [0]

        def _query(soql: str) -> list:
            if "AIAgentSession__dlm" in soql:
                return [_make_session("s1"), _make_session("s2")]
            elif "AIAgentConfiguration__dlm" in soql:
                return []
            elif "AIAgentInteraction__dlm" in soql:
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("API error")
                return []
            return []

        mock_conn.query = Mock(side_effect=_query)
        summary = adapter.import_sessions()
        # Both sessions still get imported (interaction error is caught inside _import_session)
        assert summary["sessions_imported"] == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_int_or_zero(self):
        assert _int_or_zero(42) == 42
        assert _int_or_zero(None) == 0
        assert _int_or_zero("abc") == 0
        assert _int_or_zero("123") == 123

    def test_sf_datetime_date(self):
        assert _sf_datetime("2026-03-01") == "2026-03-01T00:00:00Z"

    def test_sf_datetime_datetime(self):
        assert _sf_datetime("2026-03-01T10:30:00") == "2026-03-01T10:30:00Z"

    def test_sf_datetime_passthrough(self):
        assert _sf_datetime("not-a-date") == "not-a-date"

    def test_truncate(self):
        assert _truncate(None) is None
        assert _truncate("hello") == "hello"
        long_str = "x" * 5000
        result = _truncate(long_str, 4000)
        assert len(result) <= 4010
        assert _truncate(42) == "42"
