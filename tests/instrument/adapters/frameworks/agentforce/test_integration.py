"""Integration tests for the AgentForce adapter using REAL Pydantic models.

Ported from ``ateam/tests/adapters/agentforce/test_integration.py``.

These tests verify that AgentForceAdapter, AgentForceNormalizer, and the
Pydantic DMO models work together correctly with real types — not mocks.

The adapter does not require a live Salesforce connection for these tests;
we exercise the normalizer, models, adapter lifecycle, and event emission
against real adapter internals.

Source-of-truth API parity: the agentforce adapter exports the same public
surface in ``layerlens.instrument.adapters.frameworks.agentforce`` as it
did under ``stratix.sdk.python.adapters.agentforce``. The only behavioural
delta is the new BaseAdapter multi-tenancy contract — every
``AgentForceAdapter`` constructor here passes ``org_id="test-org"``.

Tests are skipped if pydantic is not installed (it always is, but the
importorskip pattern is used for consistency with the adapter test pattern).
"""

from __future__ import annotations  # noqa: E402

import time  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

pydantic = pytest.importorskip("pydantic", reason="pydantic not installed")

from layerlens.instrument.adapters._base.adapter import (  # noqa: E402
    AdapterStatus,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.agentforce.auth import (  # noqa: E402
    SalesforceAuthError,
    SalesforceConnection,
    SalesforceCredentials,
)
from layerlens.instrument.adapters.frameworks.agentforce.models import (  # noqa: E402
    StepType,
    CaptureMode,
    AgentSession,
    SessionEndType,
    AgentChannelType,
    AgentInteraction,
    AgentParticipant,
    EvaluationResult,
    TrustLayerConfig,
    AgentSessionEvent,
    EvaluationRequest,
    TrustLayerGuardrail,
    AgentInteractionStep,
    AgentInteractionMessage,
)
from layerlens.instrument.adapters.frameworks.agentforce.adapter import (  # noqa: E402
    AgentForceAdapter,
)
from layerlens.instrument.adapters.frameworks.agentforce.normalizer import (  # noqa: E402
    AgentForceNormalizer,
)

_TEST_ORG_ID = "test-org"


# ---------------------------------------------------------------------------
# Event collector (same pattern as langchain test_integration.py)
# ---------------------------------------------------------------------------


class EventCollector:
    """Real event collector that accumulates events for assertions.

    Provides ``org_id`` (so it can stand in as a LayerLens client for the
    BaseAdapter tenancy guard) plus a no-op ``emit`` and trace lifecycle
    counters.
    """

    def __init__(self, org_id: str = _TEST_ORG_ID) -> None:
        self.org_id = org_id
        self.events: list[dict[str, Any]] = []
        self.traces_started: int = 0
        self.traces_ended: int = 0

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def start_trace(self, **kwargs: Any) -> str:
        self.traces_started += 1
        return f"trace-{self.traces_started}"

    def end_trace(self, **kwargs: Any) -> None:
        self.traces_ended += 1

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


# ---------------------------------------------------------------------------
# Pydantic DMO models with real Salesforce field aliases
# ---------------------------------------------------------------------------


class TestPydanticDMOModels:
    """Verify Pydantic DMO models parse Salesforce-style data correctly."""

    def test_agent_session_from_sf_dict(self) -> None:
        """AgentSession should parse a Salesforce record dict with field aliases."""
        sf_record = {
            "Id": "0Xx000000000001",
            "StartTimestamp": "2026-02-21T10:00:00Z",
            "EndTimestamp": "2026-02-21T10:05:00Z",
            "AiAgentChannelTypeId": "Web",
            "AiAgentSessionEndType": "Completed",
        }
        session = AgentSession.model_validate(sf_record)
        assert session.id == "0Xx000000000001"
        assert session.start_timestamp == "2026-02-21T10:00:00Z"
        assert session.end_timestamp == "2026-02-21T10:05:00Z"
        assert session.channel_type == "Web"
        assert session.session_end_type == "Completed"

    def test_agent_participant_from_sf_dict(self) -> None:
        """AgentParticipant should parse Salesforce participant records."""
        sf_record = {
            "Id": "0Yy000000000001",
            "AiAgentSessionId": "0Xx000000000001",
            "AiAgentTypeId": "EinsteinServiceAgent",
            "AiAgentApiName": "order_agent",
            "AiAgentVersionApiName": "v2.1",
            "ParticipantId": "005000000000001",
            "AiAgentSessionParticipantRoleId": "Owner",
        }
        participant = AgentParticipant.model_validate(sf_record)
        assert participant.session_id == "0Xx000000000001"
        assert participant.agent_api_name == "order_agent"
        assert participant.agent_version == "v2.1"

    def test_agent_interaction_step_llm_type(self) -> None:
        """AgentInteractionStep should correctly parse LLM execution steps."""
        sf_record = {
            "Id": "0Zz000000000001",
            "AiAgentInteractionId": "0Yy000000000001",
            "AiAgentInteractionStepTypeId": "LLMExecutionStep",
            "InputValueText": "What is the status of order 12345?",
            "OutputValueText": "Your order is being shipped.",
            "Name": "gpt-4o-mini",
            "GenerationId": "gen-001",
            "StartTimestamp": "2026-02-21T10:01:00Z",
            "EndTimestamp": "2026-02-21T10:01:02Z",
        }
        step = AgentInteractionStep.model_validate(sf_record)
        assert step.step_type == "LLMExecutionStep"
        assert step.input_value == "What is the status of order 12345?"
        assert step.output_value == "Your order is being shipped."
        assert step.generation_id == "gen-001"

    def test_agent_interaction_message(self) -> None:
        """AgentInteractionMessage should parse message records."""
        sf_record = {
            "Id": "0Mm000000000001",
            "AiAgentInteractionId": "0Yy000000000001",
            "AiAgentInteractionMessageTypeId": "Output",
            "ContentText": "Your order has been shipped.",
            "AiAgentInteractionMsgContentTypeId": "Text",
            "MessageSentTimestamp": "2026-02-21T10:01:05Z",
        }
        msg = AgentInteractionMessage.model_validate(sf_record)
        assert msg.message_type == "Output"
        assert msg.content_text == "Your order has been shipped."

    def test_agent_interaction_from_sf_dict(self) -> None:
        """AgentInteraction should parse interaction records with trace IDs."""
        sf_record = {
            "Id": "0Ii000000000001",
            "AiAgentSessionId": "0Xx000000000001",
            "AiAgentInteractionTypeId": "Turn",
            "TelemetryTraceId": "trace-abc-123",
            "TelemetryTraceSpanId": "span-001",
            "TopicApiName": "Order_Status",
            "AttributeText": '{"intent": "check_order"}',
        }
        interaction = AgentInteraction.model_validate(sf_record)
        assert interaction.telemetry_trace_id == "trace-abc-123"
        assert interaction.topic_api_name == "Order_Status"


class TestEnumModels:
    """Verify Salesforce enum models work correctly."""

    def test_channel_types(self) -> None:
        """AgentChannelType should have all Salesforce channel values."""
        assert AgentChannelType.WEB == "Web"
        assert AgentChannelType.MESSAGING == "Messaging"
        assert AgentChannelType.VOICE == "Voice"
        assert AgentChannelType.SLACK == "Slack"
        assert AgentChannelType.API == "Api"

    def test_step_types(self) -> None:
        """StepType should map to Salesforce DMO step type IDs."""
        assert StepType.USER_INPUT == "UserInputStep"
        assert StepType.LLM_EXECUTION == "LLMExecutionStep"
        assert StepType.FUNCTION == "FunctionStep"
        assert StepType.ACTION_INVOCATION == "ActionInvocationStep"

    def test_session_end_types(self) -> None:
        """SessionEndType should cover all Agentforce session endings."""
        assert SessionEndType.COMPLETED == "Completed"
        assert SessionEndType.ESCALATED == "Escalated"
        assert SessionEndType.TIMED_OUT == "TimedOut"
        assert SessionEndType.ERROR == "Error"

    def test_capture_mode_enum(self) -> None:
        """CaptureMode should support polling, realtime, and hybrid."""
        assert CaptureMode.POLLING == "polling"
        assert CaptureMode.REALTIME == "realtime"
        assert CaptureMode.HYBRID == "hybrid"


# ---------------------------------------------------------------------------
# Trust Layer models
# ---------------------------------------------------------------------------


class TestTrustLayerModels:
    """Verify Einstein Trust Layer configuration models."""

    def test_guardrail_defaults(self) -> None:
        """TrustLayerGuardrail should have sensible defaults."""
        guardrail = TrustLayerGuardrail(name="toxicity", type="toxicity")
        assert guardrail.enabled is True
        assert guardrail.action == "block"
        assert guardrail.threshold is None

    def test_trust_config_with_guardrails(self) -> None:
        """TrustLayerConfig should accept a list of guardrails."""
        config = TrustLayerConfig(
            guardrails=[
                TrustLayerGuardrail(name="toxicity", type="toxicity", threshold=0.7),
                TrustLayerGuardrail(name="pii", type="pii", action="warn"),
            ],
            data_masking_enabled=True,
            zero_data_retention=True,
        )
        assert len(config.guardrails) == 2
        assert config.data_masking_enabled is True
        assert config.guardrails[0].threshold == 0.7


# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------


class TestEvaluationModels:
    """Verify Agentforce evaluation request/result models."""

    def test_evaluation_request_defaults(self) -> None:
        """EvaluationRequest should have default graders."""
        req = EvaluationRequest(session_ids=["sess-001", "sess-002"])
        assert len(req.session_ids) == 2
        assert "relevance" in req.graders
        assert "faithfulness" in req.graders
        assert req.include_ground_truth is False

    def test_evaluation_result(self) -> None:
        """EvaluationResult should hold scores and composite score."""
        result = EvaluationResult(
            session_id="sess-001",
            scores={"relevance": 0.92, "faithfulness": 0.85},
            composite_score=0.88,
        )
        assert result.scores["relevance"] == 0.92
        assert result.composite_score == 0.88


# ---------------------------------------------------------------------------
# Platform Event models
# ---------------------------------------------------------------------------


class TestPlatformEventModels:
    """Verify Platform Event payload models."""

    def test_agent_session_event(self) -> None:
        """AgentSessionEvent should parse platform event payloads."""
        event = AgentSessionEvent(
            session_id="sess-001",
            agent_name="OrderBot",
            topic_name="Order_Status",
            response_text="Your order has shipped.",
        )
        assert event.session_id == "sess-001"
        assert event.agent_name == "OrderBot"


# ---------------------------------------------------------------------------
# Normalizer with real Salesforce-style data
# ---------------------------------------------------------------------------


class TestNormalizerWithRealData:
    """Verify normalizer produces correct STRATIX events from SF records."""

    def test_normalize_session_start_end(self) -> None:
        """A completed session should produce start + end lifecycle events."""
        normalizer = AgentForceNormalizer()
        events = normalizer.normalize_session(
            {
                "Id": "0Xx000000000001",
                "StartTimestamp": "2026-02-21T10:00:00Z",
                "EndTimestamp": "2026-02-21T10:05:00Z",
                "AiAgentChannelTypeId": "Web",
                "AiAgentSessionEndType": "Completed",
            }
        )
        assert len(events) == 2
        assert events[0]["event_type"] == "agent.lifecycle"
        assert events[0]["payload"]["lifecycle_action"] == "start"
        assert events[1]["payload"]["lifecycle_action"] == "end"

    def test_normalize_session_in_progress(self) -> None:
        """An in-progress session (no EndTimestamp) should produce only start."""
        normalizer = AgentForceNormalizer()
        events = normalizer.normalize_session(
            {
                "Id": "0Xx000000000002",
                "StartTimestamp": "2026-02-21T10:00:00Z",
                "EndTimestamp": None,
                "AiAgentChannelTypeId": "Messaging",
            }
        )
        assert len(events) == 1
        assert events[0]["payload"]["lifecycle_action"] == "start"

    def test_normalize_llm_step(self) -> None:
        """LLMExecutionStep should produce a model.invoke event."""
        normalizer = AgentForceNormalizer()
        event = normalizer.normalize_step(
            {
                "Id": "step-001",
                "AiAgentInteractionStepTypeId": "LLMExecutionStep",
                "InputValueText": "What is the status of order 12345?",
                "OutputValueText": "Your order is being shipped.",
                "Name": "gpt-4o-mini",
                "GenerationId": "gen-001",
                "StartTimestamp": "2026-02-21T10:01:00Z",
                "EndTimestamp": "2026-02-21T10:01:02Z",
            }
        )
        assert event["event_type"] == "model.invoke"
        assert event["payload"]["model"]["provider"] == "salesforce"
        assert "Your order is being shipped." in event["payload"]["output_message"]["content"]

    def test_normalize_function_step(self) -> None:
        """FunctionStep should produce a tool.call event."""
        normalizer = AgentForceNormalizer()
        event = normalizer.normalize_step(
            {
                "Id": "step-002",
                "AiAgentInteractionStepTypeId": "FunctionStep",
                "InputValueText": '{"order_id": "12345"}',
                "OutputValueText": '{"status": "shipped"}',
                "Name": "lookup_order",
            }
        )
        assert event["event_type"] == "tool.call"
        assert event["payload"]["tool"]["name"] == "lookup_order"
        assert event["payload"]["input"]["order_id"] == "12345"
        assert event["payload"]["output"]["status"] == "shipped"

    def test_normalize_user_input_step(self) -> None:
        """UserInputStep should produce an agent.input event."""
        normalizer = AgentForceNormalizer()
        event = normalizer.normalize_step(
            {
                "Id": "step-003",
                "AiAgentInteractionStepTypeId": "UserInputStep",
                "InputValueText": "Where is my order?",
            }
        )
        assert event["event_type"] == "agent.input"
        assert event["payload"]["content"]["role"] == "human"

    def test_normalize_participant_ai(self) -> None:
        """AI participant should have participant_type='ai'."""
        normalizer = AgentForceNormalizer()
        event = normalizer.normalize_participant(
            {
                "Id": "part-001",
                "AiAgentSessionId": "sess-001",
                "AiAgentTypeId": "EinsteinServiceAgent",
                "AiAgentApiName": "order_agent",
                "AiAgentVersionApiName": "v2.1",
                "ParticipantId": "agent-001",
                "AiAgentSessionParticipantRoleId": "Owner",
            }
        )
        assert event["event_type"] == "agent.identity"
        assert event["payload"]["participant_type"] == "ai"
        assert event["payload"]["agent_api_name"] == "order_agent"

    def test_normalize_participant_human(self) -> None:
        """Human participant (Employee) should have participant_type='human'."""
        normalizer = AgentForceNormalizer()
        event = normalizer.normalize_participant(
            {
                "Id": "part-002",
                "AiAgentSessionId": "sess-001",
                "AiAgentTypeId": "Employee",
                "ParticipantId": "user-001",
            }
        )
        assert event["payload"]["participant_type"] == "human"

    def test_normalize_output_message(self) -> None:
        """Output message should produce agent.output event with agent role."""
        normalizer = AgentForceNormalizer()
        event = normalizer.normalize_message(
            {
                "Id": "msg-001",
                "AiAgentInteractionId": "int-001",
                "AiAgentInteractionMessageTypeId": "Output",
                "ContentText": "Your order has shipped.",
                "AiAgentInteractionMsgContentTypeId": "Text",
                "MessageSentTimestamp": "2026-02-21T10:01:05Z",
            }
        )
        assert event["event_type"] == "agent.output"
        assert event["payload"]["content"]["role"] == "agent"
        assert event["payload"]["content"]["message"] == "Your order has shipped."

    def test_normalize_input_message(self) -> None:
        """Input message should produce agent.input event with human role."""
        normalizer = AgentForceNormalizer()
        event = normalizer.normalize_message(
            {
                "Id": "msg-002",
                "AiAgentInteractionId": "int-001",
                "AiAgentInteractionMessageTypeId": "Input",
                "ContentText": "Check my order status",
            }
        )
        assert event["event_type"] == "agent.input"
        assert event["payload"]["content"]["role"] == "human"

    def test_normalize_interaction_with_json_attributes(self) -> None:
        """Interaction with JSON AttributeText should be parsed."""
        normalizer = AgentForceNormalizer()
        event = normalizer.normalize_interaction(
            {
                "Id": "int-001",
                "AiAgentSessionId": "sess-001",
                "AiAgentInteractionTypeId": "Turn",
                "TelemetryTraceId": "trace-abc",
                "TelemetryTraceSpanId": "span-001",
                "TopicApiName": "Order_Status",
                "AttributeText": '{"intent": "check_order", "confidence": 0.95}',
            }
        )
        assert event["event_type"] == "agent.interaction"
        assert event["payload"]["attributes"]["intent"] == "check_order"
        assert event["identity"]["trace_id"] == "trace-abc"


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


class TestAdapterLifecycle:
    """Verify adapter lifecycle without needing a live Salesforce connection."""

    def test_adapter_requires_credentials_or_connection(self) -> None:
        """connect() should raise SalesforceAuthError if neither is provided."""
        adapter = AgentForceAdapter(stratix=EventCollector(), org_id=_TEST_ORG_ID)
        with pytest.raises(SalesforceAuthError, match="credentials.*connection"):
            adapter.connect()

    def test_adapter_with_pre_built_connection(self) -> None:
        """Adapter should connect when given a SalesforceConnection directly."""
        creds = SalesforceCredentials(
            client_id="test-client-id",
            username="test@example.com",
            private_key="-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----",
            access_token="fake-token-for-test",
            token_expiry=time.time() + 3600,
        )
        conn = SalesforceConnection(credentials=creds)
        adapter = AgentForceAdapter(
            stratix=EventCollector(),
            connection=conn,
            org_id=_TEST_ORG_ID,
        )
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY
        assert adapter._connected is True
        adapter.disconnect()
        assert adapter._status == AdapterStatus.DISCONNECTED

    def test_adapter_import_without_connect_raises(self) -> None:
        """import_sessions() should raise if adapter not connected."""
        adapter = AgentForceAdapter(stratix=EventCollector(), org_id=_TEST_ORG_ID)
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.import_sessions()

    def test_adapter_health_check(self) -> None:
        """health_check() should return AdapterHealth with correct metadata."""
        creds = SalesforceCredentials(
            client_id="test-client-id",
            username="test@example.com",
            private_key="test-key",
            access_token="fake-token",
            token_expiry=time.time() + 3600,
        )
        conn = SalesforceConnection(credentials=creds)
        adapter = AgentForceAdapter(stratix=EventCollector(), connection=conn, org_id=_TEST_ORG_ID)
        adapter.connect()

        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "salesforce_agentforce"
        assert health.adapter_version == "0.1.0"
        assert health.circuit_open is False

    def test_adapter_info(self) -> None:
        """get_adapter_info() should expose correct capabilities."""
        adapter = AgentForceAdapter(org_id=_TEST_ORG_ID)
        info = adapter.get_adapter_info()
        assert info.name == "AgentForceAdapter"
        assert info.framework == "salesforce_agentforce"
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_TOOLS in info.capabilities

    def test_adapter_framework_and_version(self) -> None:
        """Adapter should expose FRAMEWORK and VERSION class attrs."""
        assert AgentForceAdapter.FRAMEWORK == "salesforce_agentforce"
        assert AgentForceAdapter.VERSION == "0.1.0"


# ---------------------------------------------------------------------------
# Credentials model
# ---------------------------------------------------------------------------


class TestSalesforceCredentials:
    """Verify SalesforceCredentials model behavior."""

    def test_is_expired_when_token_old(self) -> None:
        """Credentials should be expired when token_expiry is in the past."""
        creds = SalesforceCredentials(
            client_id="test",
            username="test@example.com",
            private_key="test-key",
            token_expiry=time.time() - 100,
        )
        assert creds.is_expired is True

    def test_is_not_expired_when_token_fresh(self) -> None:
        """Credentials should not be expired when token_expiry is in the future."""
        creds = SalesforceCredentials(
            client_id="test",
            username="test@example.com",
            private_key="test-key",
            token_expiry=time.time() + 3600,
        )
        assert creds.is_expired is False

    def test_repr_redacts_secrets(self) -> None:
        """repr() should redact private_key and access_token."""
        creds = SalesforceCredentials(
            client_id="my-client-id-12345",
            username="admin@example.com",
            private_key="super-secret-key",
            access_token="my-access-token",
        )
        r = repr(creds)
        assert "super-secret-key" not in r
        assert "my-access-token" not in r
        assert "REDACTED" in r


# ---------------------------------------------------------------------------
# Capture config integration
# ---------------------------------------------------------------------------


class TestCaptureConfigIntegration:
    """Verify CaptureConfig propagation to the adapter."""

    def test_capture_config_propagates(self) -> None:
        """CaptureConfig should be stored and accessible on the adapter."""
        config = CaptureConfig(
            l1_agent_io=True,
            l3_model_metadata=True,
            l5a_tool_calls=False,
        )
        adapter = AgentForceAdapter(
            stratix=EventCollector(),
            capture_config=config,
            org_id=_TEST_ORG_ID,
        )
        assert adapter._capture_config.l5a_tool_calls is False
        assert adapter._capture_config.l3_model_metadata is True

    def test_serialize_for_replay(self) -> None:
        """serialize_for_replay() should include capture config."""
        adapter = AgentForceAdapter(stratix=EventCollector(), org_id=_TEST_ORG_ID)
        replay = adapter.serialize_for_replay()
        assert replay.adapter_name == "AgentForceAdapter"
        assert replay.framework == "salesforce_agentforce"
        assert "capture_config" in replay.config
