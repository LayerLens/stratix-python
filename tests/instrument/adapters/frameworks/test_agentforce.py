"""Tests for the Agentforce adapter (batch import from the Salesforce
Session Tracing Data Model — STDM).

The doubles here are rebuilt from the REAL ``ssot__AiAgent*__dlm`` shapes
captured by a live describe of a provisioned Agentforce + Data Cloud org
(LAY-3599). The previous version of this file mocked SOQL keyed on a
*fictional* schema (``AIAgentSession__dlm`` etc.) that the adapter shared, so
the suite was green against a world that does not exist. Everything below is
keyed on the real object/field names and the real session → interaction →
(step + message) decomposition.

Key real-world facts these doubles encode (verified via live describe):

* Join keys are the **business UUIDs** in ``ssot__Id__c`` / ``ssot__*Id__c``,
  NOT the Salesforce surrogate ``Id``. A message references its interaction by
  the interaction's ``ssot__Id__c``.
* ``"NOT_SET"`` is a sentinel string Salesforce uses instead of null.
* The STDM has **no token-count fields** — the adapter emits **no
  ``cost.record``** and carries the generation / gateway ids as ``model.invoke``
  metadata instead (future token-pull hook).
* Agent identity is on the **participant** (``ssot__AiAgentApiName__c``).
"""

from __future__ import annotations

import re
from typing import Any, Optional
from unittest.mock import Mock

import pytest

import layerlens.instrument.adapters.frameworks.agentforce as _mod
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks._utils import truncate as _truncate
from layerlens.instrument.adapters.frameworks.agentforce import (
    AgentforceAdapter,
    _sf_datetime,
    _SalesforceCredentials,
)

from .conftest import find_event, find_events, capture_framework_trace

# ---------------------------------------------------------------------------
# Real STDM identifiers (from live describe of orgfarm-ed439d65be-dev-ed)
# ---------------------------------------------------------------------------

SESSION_UUID = "019ed365-eb7b-73f6-bb95-15e93f8ed2f0"
INTERACTION_UUID = "4c47dd00-390a-4173-9b22-bc476d894bff"


@pytest.fixture(autouse=True)
def _enable_httpx(monkeypatch: Any) -> None:
    monkeypatch.setattr(_mod, "_HAS_HTTPX", True)


# ---------------------------------------------------------------------------
# Row factories — shaped like the real DMO rows
# ---------------------------------------------------------------------------


def _make_session(session_uuid: str = SESSION_UUID, **overrides: Any) -> dict:
    base = {
        "Id": "q0w000SurrogateAAA",  # SF surrogate row id (NOT the join key)
        "ssot__Id__c": session_uuid,  # business UUID == trace / session id
        "ssot__AiAgentChannelType__c": "EmployeeAgent",
        "ssot__AiAgentSessionEndType__c": "Completed",
        "ssot__StartTimestamp__c": "2026-06-17T02:25:42.878+0000",
        "ssot__EndTimestamp__c": "2026-06-17T02:25:50.000+0000",
    }
    base.update(overrides)
    return base


def _make_participant(
    session_uuid: str = SESSION_UUID,
    role: str = "USER",
    agent_api_name: str = "Agentforce_Employee_Agent",
    **overrides: Any,
) -> dict:
    base = {
        "Id": "q0w000ParticipantA",
        "ssot__Id__c": "e986e990-a5e2-44fb-b327-33562c4a0161",
        "ssot__AiAgentSessionId__c": session_uuid,
        "ssot__AiAgentApiName__c": agent_api_name,
        "ssot__AiAgentVersionApiName__c": "v1",
        "ssot__AiAgentType__c": "AgentforceEmployeeAgent",
        "ssot__AiAgentSessionParticipantRole__c": role,
        "ssot__ParticipantId__c": "005fj00000Gz7jJAAR",
    }
    base.update(overrides)
    return base


def _make_interaction(
    session_uuid: str = SESSION_UUID,
    interaction_uuid: str = INTERACTION_UUID,
    **overrides: Any,
) -> dict:
    base = {
        "Id": "q0t000SurrogateI",
        "ssot__Id__c": interaction_uuid,  # business UUID — step/message join key
        "ssot__AiAgentSessionId__c": session_uuid,
        "ssot__AiAgentInteractionType__c": "TURN",
        "ssot__TopicApiName__c": "GeneralFAQ_16jfj000001nP7x",
        "ssot__TelemetryTraceId__c": "ad91cc668ac1db7b",
        "ssot__PrevInteractionId__c": "NOT_SET",
        "ssot__StartTimestamp__c": "2026-06-17T02:25:42.878+0000",
        "ssot__EndTimestamp__c": "2026-06-17T02:25:43.688+0000",
    }
    base.update(overrides)
    return base


def _make_step(
    interaction_uuid: str = INTERACTION_UUID,
    step_type: str = "LLM_STEP",  # real observed step type (→ model.invoke)
    **overrides: Any,
) -> dict:
    base = {
        "Id": "q0s000SurrogateS",
        "ssot__Id__c": "step-uuid-0001",
        "ssot__AiAgentInteractionId__c": interaction_uuid,
        "ssot__AiAgentInteractionStepType__c": step_type,
        "SubType__c": None,
        "ssot__Name__c": "generate_response",
        "ssot__InputValueText__c": "What is the weather?",
        "ssot__OutputValueText__c": "It's sunny today.",
        "ssot__GenerationId__c": "gen-abc-123",
        "ssot__GenAiGatewayRequestId__c": "req-abc-123",
        "ssot__GenAiGatewayResponseId__c": "resp-abc-123",
        "ssot__ErrorMessageText__c": None,
        "ssot__StartTimestamp__c": "2026-06-17T02:25:42.900+0000",
        "ssot__EndTimestamp__c": "2026-06-17T02:25:43.600+0000",
    }
    base.update(overrides)
    return base


def _make_message(
    interaction_uuid: str = INTERACTION_UUID,
    message_type: str = "Input",
    content: str = "Hello who are you",
    **overrides: Any,
) -> dict:
    base = {
        "Id": "q0r000SurrogateM",
        "ssot__Id__c": "msg-uuid-0001",
        "ssot__AiAgentInteractionId__c": interaction_uuid,
        "ssot__AiAgentSessionId__c": SESSION_UUID,
        "ssot__AiAgentInteractionMessageType__c": message_type,
        "ssot__ContentText__c": content,
        "ssot__MessageSentTimestamp__c": "2026-06-17T02:25:42.878+0000",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Mock connection — routes SOQL on the REAL object names and honours the
# WHERE-clause id so per-interaction children resolve correctly.
# ---------------------------------------------------------------------------


def _make_mock_conn(
    sessions: Optional[list] = None,
    participants: Optional[list] = None,
    interactions: Optional[list] = None,
    steps: Optional[list] = None,
    messages: Optional[list] = None,
) -> Mock:
    sessions = sessions or []
    participants = participants or []
    interactions = interactions or []
    steps = steps or []
    messages = messages or []

    def _quoted_id(soql: str) -> Optional[str]:
        m = re.search(r"=\s*'([^']*)'", soql)
        return m.group(1) if m else None

    def _by(rows: list, field: str, value: Optional[str]) -> list:
        if value is None:
            return rows
        return [r for r in rows if r.get(field) == value]

    def _query(soql: str) -> list:
        # Full ``__dlm`` names are collision-free (the token after the object
        # base differs: ...Interaction__dlm vs ...InteractionStep__dlm etc.).
        if "ssot__AiAgentSessionParticipant__dlm" in soql:
            return _by(participants, "ssot__AiAgentSessionId__c", _quoted_id(soql))
        if "ssot__AiAgentInteractionStep__dlm" in soql:
            return _by(steps, "ssot__AiAgentInteractionId__c", _quoted_id(soql))
        if "ssot__AiAgentInteractionMessage__dlm" in soql:
            return _by(messages, "ssot__AiAgentInteractionId__c", _quoted_id(soql))
        if "ssot__AiAgentInteraction__dlm" in soql:
            return _by(interactions, "ssot__AiAgentSessionId__c", _quoted_id(soql))
        if "ssot__AiAgentSession__dlm" in soql:
            return sessions
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


def _one_turn_session(**conn_overrides: Any) -> dict:
    """A complete single-turn session: participant + 1 interaction + 1 LLM
    step + Input/Output messages."""
    kwargs: dict = {
        "sessions": [_make_session()],
        "participants": [_make_participant()],
        "interactions": [_make_interaction()],
        "steps": [_make_step()],
        "messages": [
            _make_message(message_type="Input", content="Hello who are you"),
            _make_message(message_type="Output", content="I'm your Agentforce assistant."),
        ],
    }
    kwargs.update(conn_overrides)
    return kwargs


def _two_turn_session() -> dict:
    """A two-turn session: two interactions, each with an LLM step + Input/Output
    messages. Exercises the per-interaction loop, child-query routing by
    interaction id, and the cross-turn first-input / last-output edge messages."""
    return {
        "sessions": [_make_session()],
        "participants": [_make_participant()],
        "interactions": [
            _make_interaction(interaction_uuid="int-1", ssot__StartTimestamp__c="2026-06-17T02:25:42.000+0000"),
            _make_interaction(interaction_uuid="int-2", ssot__StartTimestamp__c="2026-06-17T02:26:00.000+0000"),
        ],
        "steps": [
            _make_step(interaction_uuid="int-1", ssot__Id__c="step-1"),
            _make_step(interaction_uuid="int-2", ssot__Id__c="step-2"),
        ],
        "messages": [
            _make_message(interaction_uuid="int-1", message_type="Input", content="first question"),
            _make_message(interaction_uuid="int-1", message_type="Output", content="first answer"),
            _make_message(interaction_uuid="int-2", message_type="Input", content="second question"),
            _make_message(interaction_uuid="int-2", message_type="Output", content="final answer"),
        ],
    }


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
            AgentforceAdapter(mock_client).connect(credentials={"client_id": "x", "client_secret": "y"})

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
        adapter, uploaded, _ = _setup(mock_client, **_one_turn_session())
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
        assert summary["next_cursor"] is None

    def test_queries_real_stdm_objects(self, mock_client):
        """The adapter must query the real ssot__ object names."""
        adapter, _, mock_conn = _setup(mock_client, **_one_turn_session())
        adapter.import_sessions()
        all_soql = " ".join(call.args[0] for call in mock_conn.query.call_args_list)
        assert "ssot__AiAgentSession__dlm" in all_soql
        assert "ssot__AiAgentInteraction__dlm" in all_soql
        assert "ssot__AiAgentInteractionStep__dlm" in all_soql
        assert "ssot__AiAgentInteractionMessage__dlm" in all_soql
        # The fictional names must be gone.
        assert "AIAgentSession__dlm" not in all_soql
        assert "AIAgentConfiguration__dlm" not in all_soql


# ---------------------------------------------------------------------------
# Session processing — envelope + lifecycle + identity
# ---------------------------------------------------------------------------


class TestSessionProcessing:
    def test_emits_lifecycle_start(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client, **_one_turn_session())
        adapter.import_sessions()
        life = find_event(uploaded["events"], "agent.lifecycle")
        assert life["payload"]["lifecycle_action"] == "start"
        assert life["payload"]["session_id"] == SESSION_UUID

    def test_emits_agent_input_with_identity(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client, **_one_turn_session())
        adapter.import_sessions()
        inp = find_event(uploaded["events"], "agent.input")
        assert inp["payload"]["session_id"] == SESSION_UUID
        assert inp["payload"]["agent_name"] == "Agentforce_Employee_Agent"
        assert inp["payload"]["channel"] == "EmployeeAgent"
        assert inp["payload"]["framework"] == "agentforce"

    def test_emits_agent_output(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client, **_one_turn_session())
        adapter.import_sessions()
        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["session_id"] == SESSION_UUID
        assert out["payload"]["outcome"] == "Completed"

    def test_agent_input_content_from_first_input_message(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=True),
            **_one_turn_session(),
        )
        adapter.import_sessions()
        inp = find_event(uploaded["events"], "agent.input")
        assert inp["payload"]["content"] == "Hello who are you"

    def test_per_session_trace(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            sessions=[_make_session("s1"), _make_session("s2")],
            participants=[_make_participant("s1"), _make_participant("s2")],
            interactions=[],
        )
        adapter.import_sessions()
        assert len(find_events(uploaded["events"], "agent.input")) == 2
        assert len(find_events(uploaded["events"], "agent.lifecycle")) == 2


# ---------------------------------------------------------------------------
# Interaction steps — LLM / model.invoke (NO cost, NO tokens)
# ---------------------------------------------------------------------------


class TestLLMStep:
    def test_model_invoke_emitted_with_gen_ids(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=True),
            **_one_turn_session(),
        )
        adapter.import_sessions()
        me = find_event(uploaded["events"], "model.invoke")
        assert me["payload"]["messages"] == "What is the weather?"
        assert me["payload"]["output_message"] == "It's sunny today."
        # Generation / gateway ids are carried as metadata (the token-pull hook).
        assert me["payload"]["generation_id"] == "gen-abc-123"
        assert me["payload"]["gateway_request_id"] == "req-abc-123"
        assert me["payload"]["gateway_response_id"] == "resp-abc-123"

    def test_no_cost_record_emitted(self, mock_client):
        """STDM has no token fields — the adapter must NOT emit cost.record."""
        adapter, uploaded, _ = _setup(mock_client, **_one_turn_session())
        adapter.import_sessions()
        assert len(find_events(uploaded["events"], "cost.record")) == 0

    def test_no_token_fields_on_model_invoke(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client, **_one_turn_session())
        adapter.import_sessions()
        me = find_event(uploaded["events"], "model.invoke")
        assert "tokens_prompt" not in me["payload"]
        assert "tokens_completion" not in me["payload"]
        assert "tokens_total" not in me["payload"]

    def test_content_gating(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=False),
            **_one_turn_session(),
        )
        adapter.import_sessions()
        me = find_event(uploaded["events"], "model.invoke")
        assert "messages" not in me["payload"]
        assert "output_message" not in me["payload"]
        # Metadata still flows.
        assert me["payload"]["generation_id"] == "gen-abc-123"


# ---------------------------------------------------------------------------
# Interaction steps — tool / action
# ---------------------------------------------------------------------------


class TestToolStep:
    def test_tool_call_emitted(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=True),
            **_one_turn_session(
                steps=[
                    _make_step(
                        step_type="ACTION_STEP",
                        ssot__Name__c="get_weather",
                        ssot__InputValueText__c='{"city": "SF"}',
                        ssot__OutputValueText__c='{"temp": 72}',
                        ssot__GenerationId__c=None,
                        ssot__GenAiGatewayRequestId__c=None,
                        ssot__GenAiGatewayResponseId__c=None,
                    )
                ],
            ),
        )
        adapter.import_sessions()
        tc = find_event(uploaded["events"], "tool.call")
        assert tc["payload"]["tool_name"] == "get_weather"
        assert tc["payload"]["input"] == '{"city": "SF"}'
        assert tc["payload"]["output"] == '{"temp": 72}'
        assert len(find_events(uploaded["events"], "model.invoke")) == 0

    def test_tool_content_gating(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=False),
            **_one_turn_session(
                steps=[
                    _make_step(
                        step_type="ACTION_STEP",
                        ssot__Name__c="t",
                        ssot__InputValueText__c="secret",
                        ssot__OutputValueText__c="classified",
                    )
                ],
            ),
        )
        adapter.import_sessions()
        tc = find_event(uploaded["events"], "tool.call")
        assert "input" not in tc["payload"]
        assert "output" not in tc["payload"]


# ---------------------------------------------------------------------------
# Interaction messages — the human-readable conversation turns
# ---------------------------------------------------------------------------


class TestMessages:
    def test_messages_emitted_as_interaction_events(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=True),
            **_one_turn_session(),
        )
        adapter.import_sessions()
        interactions = find_events(uploaded["events"], "agent.interaction")
        roles = {e["payload"].get("role") for e in interactions}
        assert "user" in roles
        assert "agent" in roles
        user_msg = next(e for e in interactions if e["payload"].get("role") == "user")
        assert user_msg["payload"]["content"] == "Hello who are you"

    def test_message_content_gating(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=False),
            **_one_turn_session(),
        )
        adapter.import_sessions()
        for e in find_events(uploaded["events"], "agent.interaction"):
            assert "content" not in e["payload"]
            # role/session metadata still present
            assert e["payload"]["session_id"] == SESSION_UUID


# ---------------------------------------------------------------------------
# Handoff / escalation steps
# ---------------------------------------------------------------------------


class TestHandoffStep:
    def test_handoff_emitted(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            **_one_turn_session(
                steps=[
                    _make_step(
                        step_type="Escalation",
                        ssot__Name__c="escalate_to_human",
                        ssot__InputValueText__c="Customer needs help",
                    )
                ],
            ),
        )
        adapter.import_sessions()
        h = find_event(uploaded["events"], "agent.handoff")
        assert h["payload"]["step_name"] == "escalate_to_human"


# ---------------------------------------------------------------------------
# Error step
# ---------------------------------------------------------------------------


class TestErrorStep:
    def test_error_step_emits_agent_error(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=True),
            **_one_turn_session(
                steps=[
                    _make_step(
                        step_type="LLM_STEP",
                        ssot__ErrorMessageText__c="model timeout",
                    )
                ],
            ),
        )
        adapter.import_sessions()
        err = find_event(uploaded["events"], "agent.error")
        assert err["payload"]["error_message"] == "model timeout"


# ---------------------------------------------------------------------------
# Unknown step types
# ---------------------------------------------------------------------------


class TestUnknownStep:
    def test_unknown_without_gen_id_emits_agent_interaction(self, mock_client):
        # TOPIC_STEP is a real observed step type that maps to the generic
        # agent.interaction (topic routing — not an LLM call or a tool).
        adapter, uploaded, _ = _setup(
            mock_client,
            **_one_turn_session(
                steps=[
                    _make_step(
                        step_type="TOPIC_STEP",
                        ssot__Name__c="GeneralFAQ_topic",
                        ssot__GenerationId__c=None,
                        ssot__GenAiGatewayRequestId__c=None,
                        ssot__GenAiGatewayResponseId__c=None,
                    )
                ],
                messages=[],  # isolate the step event
            ),
        )
        adapter.import_sessions()
        evt = find_event(uploaded["events"], "agent.interaction")
        assert evt["payload"]["step_type"] == "TOPIC_STEP"

    def test_unknown_with_gen_id_defaults_to_model_invoke(self, mock_client):
        """A step carrying a GenerationId is an LLM call even if the type is
        unrecognised."""
        adapter, uploaded, _ = _setup(
            mock_client,
            **_one_turn_session(
                steps=[_make_step(step_type="SomethingNew")],
                messages=[],
            ),
        )
        adapter.import_sessions()
        assert len(find_events(uploaded["events"], "model.invoke")) == 1


# ---------------------------------------------------------------------------
# "NOT_SET" sentinel handling
# ---------------------------------------------------------------------------


class TestNotSetSentinel:
    def test_not_set_content_not_emitted(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            **_one_turn_session(
                steps=[
                    _make_step(
                        ssot__InputValueText__c="NOT_SET",
                        ssot__OutputValueText__c="NOT_SET",
                    )
                ],
                messages=[],
            ),
        )
        adapter.import_sessions()
        me = find_event(uploaded["events"], "model.invoke")
        assert "messages" not in me["payload"]
        assert "output_message" not in me["payload"]

    def test_not_set_topic_not_emitted(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            **_one_turn_session(
                interactions=[_make_interaction(ssot__TopicApiName__c="NOT_SET")],
            ),
        )
        adapter.import_sessions()
        # No payload should carry a literal "NOT_SET" value.
        for e in uploaded["events"]:
            assert "NOT_SET" not in e["payload"].values()


# ---------------------------------------------------------------------------
# Full invocation
# ---------------------------------------------------------------------------


class TestFullInvocation:
    def test_complete_session(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            **_one_turn_session(
                steps=[
                    _make_step(step_type="LLM_STEP"),
                    _make_step(
                        step_type="ACTION_STEP",
                        ssot__Name__c="search",
                        ssot__InputValueText__c="{}",
                        ssot__OutputValueText__c="found",
                        ssot__GenerationId__c=None,
                        ssot__GenAiGatewayRequestId__c=None,
                        ssot__GenAiGatewayResponseId__c=None,
                    ),
                ],
            ),
        )
        adapter.import_sessions()
        events = uploaded["events"]

        assert len(find_events(events, "agent.lifecycle")) == 1
        assert len(find_events(events, "agent.input")) == 1
        assert len(find_events(events, "agent.output")) == 1
        assert len(find_events(events, "model.invoke")) == 1
        assert len(find_events(events, "tool.call")) == 1
        # The STDM has no tokens — never a cost.record, never environment.config
        # (there is no AgentConfiguration DMO).
        assert len(find_events(events, "cost.record")) == 0
        assert len(find_events(events, "environment.config")) == 0


# ---------------------------------------------------------------------------
# Trace integrity + migration contract
# ---------------------------------------------------------------------------


class TestTraceIntegrity:
    def test_shared_trace_id_within_session(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client, **_one_turn_session())
        adapter.import_sessions()
        trace_ids = {e["trace_id"] for e in uploaded["events"]}
        assert len(trace_ids) == 1

    def test_session_id_on_every_payload(self, mock_client):
        """Migration contract: the session id is recoverable from every event
        (1 trace per session + session_id carried in the payload)."""
        adapter, uploaded, _ = _setup(mock_client, **_one_turn_session())
        adapter.import_sessions()
        assert uploaded["events"]
        # Collector-synthesized trace-level structural markers (agent.identity,
        # trace.root) are content-free and trace-scoped, not per-session events —
        # the session_id migration contract applies to the adapter's own events.
        structural = {"agent.identity", "trace.root"}
        for e in uploaded["events"]:
            if e["event_type"] in structural:
                continue
            assert e["payload"].get("session_id") == SESSION_UUID

    def test_monotonic_sequence_ids(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client, **_one_turn_session())
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
            if "ssot__AiAgentSession__dlm" in soql:
                return [_make_session("s1"), _make_session("s2")]
            if "ssot__AiAgentSessionParticipant__dlm" in soql:
                return []
            if "ssot__AiAgentInteraction__dlm" in soql and "Step" not in soql and "Message" not in soql:
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("API error")
                return []
            return []

        mock_conn.query = Mock(side_effect=_query)
        summary = adapter.import_sessions()
        # Both sessions still imported (interaction error caught inside _import_session)
        assert summary["sessions_imported"] == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
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


# ---------------------------------------------------------------------------
# Disconnect leave-no-trace (LAY-3577 / T3)
# ---------------------------------------------------------------------------


class TestDisconnectLeaveNoTrace:
    def test_disconnect_closes_and_clears_connection_state(self, mock_client):
        adapter, _, mock_conn = _setup(mock_client)
        adapter.disconnect()

        mock_conn.close.assert_called_once()
        assert adapter._connection is None
        assert adapter._credentials is None
        assert not adapter.is_connected
        assert adapter.adapter_info().metadata == {}

    def test_disconnected_adapter_rejects_imports(self, mock_client):
        adapter, _, _ = _setup(mock_client, **_one_turn_session())
        adapter.disconnect()
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.import_sessions()

    def test_double_disconnect_is_safe(self, mock_client):
        adapter, _, mock_conn = _setup(mock_client)
        adapter.disconnect()
        adapter.disconnect()

        mock_conn.close.assert_called_once()
        assert not adapter.is_connected
        assert adapter._connection is None

    def test_reconnect_after_disconnect_works(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client, **_one_turn_session(sessions=[_make_session("sess-001")]))
        assert adapter.import_sessions()["sessions_imported"] == 1
        adapter.disconnect()

        new_conn = _make_mock_conn(sessions=[_make_session("sess-002")])
        adapter._connection = new_conn
        adapter._connected = True
        adapter._credentials = _SalesforceCredentials(
            client_id="test",
            client_secret="test",
            instance_url="https://test.salesforce.com",
            access_token="fake-token",
        )

        summary = adapter.import_sessions()
        assert summary["sessions_imported"] == 1
        inputs = find_events(uploaded["events"], "agent.input")
        assert {e["payload"]["session_id"] for e in inputs} == {"sess-001", "sess-002"}

        adapter.disconnect()
        new_conn.close.assert_called_once()
        assert adapter._connection is None


# ---------------------------------------------------------------------------
# Multi-turn sessions (cross-interaction + edge messages)
# ---------------------------------------------------------------------------


class TestMultiTurnSession:
    def test_two_interactions_emit_two_model_invokes(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client, **_two_turn_session())
        adapter.import_sessions()
        invokes = find_events(uploaded["events"], "model.invoke")
        assert len(invokes) == 2
        assert {e["payload"]["interaction_id"] for e in invokes} == {"int-1", "int-2"}

    def test_input_content_is_first_input_across_turns(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=True),
            **_two_turn_session(),
        )
        adapter.import_sessions()
        assert find_event(uploaded["events"], "agent.input")["payload"]["content"] == "first question"

    def test_output_content_is_last_output_across_turns(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=True),
            **_two_turn_session(),
        )
        adapter.import_sessions()
        assert find_event(uploaded["events"], "agent.output")["payload"]["content"] == "final answer"

    def test_all_messages_emitted_across_turns(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=True),
            **_two_turn_session(),
        )
        adapter.import_sessions()
        interactions = find_events(uploaded["events"], "agent.interaction")
        assert len(interactions) == 4
        assert {e["payload"].get("content") for e in interactions} == {
            "first question",
            "first answer",
            "second question",
            "final answer",
        }


# ---------------------------------------------------------------------------
# Staggered DMO ingestion (messages present, steps not yet ingested)
# ---------------------------------------------------------------------------


class TestStaggeredIngestion:
    def test_messages_without_steps_still_produce_readable_trace(self, mock_client):
        # The real org ingests DMOs on staggered streams: interactions + messages
        # land before steps. A session whose steps have not arrived yet must still
        # yield a complete, readable conversation trace (just no model.invoke).
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=True),
            **_one_turn_session(steps=[]),
        )
        adapter.import_sessions()
        events = uploaded["events"]
        assert len(find_events(events, "model.invoke")) == 0
        assert len(find_events(events, "tool.call")) == 0
        assert find_event(events, "agent.lifecycle")["payload"]["lifecycle_action"] == "start"
        assert find_event(events, "agent.input")["payload"]["content"] == "Hello who are you"
        roles = {e["payload"].get("role") for e in find_events(events, "agent.interaction")}
        assert {"user", "agent"} <= roles
        assert find_event(events, "agent.output")["payload"]["session_id"] == SESSION_UUID


# ---------------------------------------------------------------------------
# Session-envelope content gating + step-dispatch edges
# ---------------------------------------------------------------------------


class TestSessionEnvelopeContentGating:
    def test_agent_input_and_output_content_gated(self, mock_client):
        adapter, uploaded, _ = _setup(
            mock_client,
            capture_config=CaptureConfig(capture_content=False),
            **_one_turn_session(),
        )
        adapter.import_sessions()
        inp = find_event(uploaded["events"], "agent.input")
        out = find_event(uploaded["events"], "agent.output")
        assert "content" not in inp["payload"]
        assert "content" not in out["payload"]
        # Non-content metadata still flows.
        assert inp["payload"]["agent_name"] == "Agentforce_Employee_Agent"
        assert out["payload"]["outcome"] == "Completed"


class TestStepDispatchEdges:
    def test_error_step_does_not_also_emit_model_invoke(self, mock_client):
        # An errored LLM step surfaces as agent.error only — not also model.invoke.
        adapter, uploaded, _ = _setup(
            mock_client,
            **_one_turn_session(
                steps=[_make_step(step_type="LLM_STEP", ssot__ErrorMessageText__c="boom")],
                messages=[],
            ),
        )
        adapter.import_sessions()
        assert len(find_events(uploaded["events"], "agent.error")) == 1
        assert len(find_events(uploaded["events"], "model.invoke")) == 0
