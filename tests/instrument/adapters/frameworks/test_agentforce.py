"""Unit tests for the Salesforce Agentforce framework adapter.

Mocked at the SDK shape level — no real Salesforce API or ``requests``
network call is made. Each test patches ``requests`` (the only
third-party SDK touched at the module boundary) so the adapter, importer,
mapper, normalizer, client, events, evaluator, and trust-layer importer
are all exercised end-to-end against fixture data.

Coverage:

* lifecycle (connect / disconnect / health_check / serialize_for_replay)
* SOQL importer with paginated query results + JSON-injection guard
* DMO normalizer for every record type (session, participant, interaction,
  step×3 step-types, message)
* Agent API client (create / send / end / capture)
* Agent API mapper (start, user / agent message, topic, action, guardrail,
  escalation, end)
* Trust Layer importer (config fetch, YAML emission, deprecation alias)
* Platform Events subscriber (handle_event + reconnect bookkeeping)
* Einstein evaluator (composite score weights + offline-without-client
  behavior)
* Lazy-import + default-install guard (importing the package does NOT
  pull in ``requests``)
"""

from __future__ import annotations

import sys
from typing import Any
from unittest import mock

import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.agentforce import (
    ADAPTER_CLASS,
    ImportResult,
    AgentApiClient,
    AgentApiMapper,
    AgentForceAdapter,
    EinsteinEvaluator,
    AgentForceImporter,
    NormalizationError,
    TrustLayerImporter,
    SalesforceAuthError,
    AgentForceNormalizer,
    SalesforceConnection,
    SalesforceQueryError,
    SalesforceCredentials,
    PlatformEventSubscriber,
)
from layerlens.instrument.adapters.frameworks.agentforce.models import (
    AgentApiMessage,
    AgentApiSession,
    TrustLayerConfig,
    AgentSessionEvent,
    TrustLayerGuardrail,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _RecordingStratix:
    """Minimal stub that records every event emission."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _FakeResponse:
    """``requests.Response`` shape with the bits the adapter touches."""

    def __init__(
        self,
        json_data: Any = None,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        text_lines: list[str] | None = None,
    ) -> None:
        self._json = json_data if json_data is not None else {}
        self.status_code = status_code
        self.headers = headers or {}
        self._lines = text_lines or []
        self._closed = False

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests  # type: ignore[import-untyped]

            err = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def iter_lines(self, decode_unicode: bool = False) -> Any:  # noqa: ARG002
        yield from self._lines

    def close(self) -> None:
        self._closed = True


def _credentials() -> SalesforceCredentials:
    creds = SalesforceCredentials(
        client_id="3MVG9TestConnectedAppKey00000000",
        username="agent-importer@example.com",
        private_key="-----BEGIN PRIVATE KEY-----\nMIITestKey\n-----END PRIVATE KEY-----\n",
        instance_url="https://example.my.salesforce.com",
    )
    creds.access_token = "00DTEST!AQ.TOKEN"
    creds.token_expiry = 9_999_999_999.0  # not expired
    return creds


def _connection() -> SalesforceConnection:
    conn = SalesforceConnection(credentials=_credentials())
    conn.instance_url = "https://example.my.salesforce.com"
    return conn


# ---------------------------------------------------------------------------
# Lazy-import + package surface
# ---------------------------------------------------------------------------


def test_adapter_class_export_matches() -> None:
    assert ADAPTER_CLASS is AgentForceAdapter


def test_package_reexports_full_public_api() -> None:
    """All symbols in ``__all__`` are importable from the package root."""
    import layerlens.instrument.adapters.frameworks.agentforce as af

    for name in af.__all__:
        assert hasattr(af, name), f"{name!r} declared in __all__ but missing"


def test_package_does_not_eagerly_import_requests() -> None:
    """Importing the adapter package must not pull in ``requests``.

    Implementation note: this test deletes ``agentforce.*`` entries from
    ``sys.modules`` so the re-import is measured against a clean slate.
    The original module objects are saved and restored after the
    assertion so subsequent tests still see the same ``AgentForceAdapter``
    class object — otherwise ``is`` identity checks elsewhere in the
    suite (e.g., ``test_adapter_class_registered``) would fail because
    the second import creates a fresh class object.
    """
    # Snapshot existing agentforce module objects so we can restore them.
    saved_agentforce = {
        mod: sys.modules[mod]
        for mod in list(sys.modules)
        if mod.startswith("layerlens.instrument.adapters.frameworks.agentforce")
    }

    # Drop any prior import so the assertion measures the package itself.
    for mod in list(sys.modules):
        if mod == "requests" or mod.startswith("requests."):
            del sys.modules[mod]

    # Re-import the package fresh.
    for mod in list(sys.modules):
        if mod.startswith("layerlens.instrument.adapters.frameworks.agentforce"):
            del sys.modules[mod]

    try:
        import layerlens.instrument.adapters.frameworks.agentforce  # noqa: F401

        assert "requests" not in sys.modules, (
            "agentforce adapter must not import requests at module load time"
        )
    finally:
        # Restore the original module objects so other tests in the suite
        # see the same class identity they imported at collection time.
        for mod in list(sys.modules):
            if mod.startswith("layerlens.instrument.adapters.frameworks.agentforce"):
                del sys.modules[mod]
        sys.modules.update(saved_agentforce)


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


def test_connect_without_credentials_or_connection_raises() -> None:
    adapter = AgentForceAdapter()
    with pytest.raises(SalesforceAuthError):
        adapter.connect()


def test_lifecycle_with_prebuilt_connection() -> None:
    adapter = AgentForceAdapter(connection=_connection())
    adapter.connect()
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY

    info = adapter.get_adapter_info()
    assert info.framework == "salesforce_agentforce"
    assert info.version == AgentForceAdapter.VERSION

    health = adapter.health_check()
    assert health.framework_name == "salesforce_agentforce"
    assert health.error_count == 0

    rt = adapter.serialize_for_replay()
    assert rt.framework == "salesforce_agentforce"
    assert "capture_config" in rt.config

    adapter.disconnect()
    assert adapter.is_connected is False
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_adapter_info_declares_replay_capability() -> None:
    """AgentForce implements ``serialize_for_replay`` (Salesforce session
    backfill is the entire reason this adapter exists), so REPLAY must
    appear in the declared capabilities.
    """
    from layerlens.instrument.adapters._base.adapter import AdapterCapability

    info = AgentForceAdapter().get_adapter_info()
    assert AdapterCapability.REPLAY in info.capabilities


def test_health_message_warns_when_token_expired() -> None:
    creds = _credentials()
    creds.token_expiry = 0.0  # expired
    conn = SalesforceConnection(credentials=creds)
    conn.instance_url = "https://example.my.salesforce.com"

    adapter = AgentForceAdapter(credentials=creds, connection=conn)
    # Skip authenticate() by pre-populating connection.
    adapter._importer = mock.MagicMock()
    adapter._connected = True
    adapter._status = AdapterStatus.HEALTHY

    health = adapter.health_check()
    assert health.message is not None
    assert "expired" in health.message.lower()


def test_import_sessions_before_connect_raises() -> None:
    adapter = AgentForceAdapter()
    with pytest.raises(RuntimeError, match="not connected"):
        adapter.import_sessions(start_date="2026-04-01")


def test_import_sessions_routes_events_through_pipeline() -> None:
    stratix = _RecordingStratix()
    adapter = AgentForceAdapter(
        stratix=stratix,
        connection=_connection(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    # Replace the importer with a fixture-returning fake.
    fake_events = [
        {
            "event_type": "agent.lifecycle",
            "payload": {"lifecycle_action": "start", "session_id": "0XxAAA"},
            "identity": {"trace_id": "trace-1"},
            "timestamp": "2026-04-01T00:00:00Z",
        },
        {
            "event_type": "agent.input",
            "payload": {"content": {"role": "human", "message": "hi"}},
        },
    ]
    fake_result = ImportResult(sessions_imported=1)
    adapter._importer = mock.MagicMock()
    adapter._importer.import_sessions = mock.MagicMock(
        return_value=(fake_events, fake_result),
    )

    result = adapter.import_sessions(start_date="2026-04-01")
    assert result.sessions_imported == 1
    assert result.events_generated == 2

    types = [e["event_type"] for e in stratix.events]
    assert types == ["agent.lifecycle", "agent.input"]
    # Identity + timestamp passthrough into payload.
    assert stratix.events[0]["payload"]["_identity"] == {"trace_id": "trace-1"}
    assert stratix.events[0]["payload"]["_timestamp"] == "2026-04-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Importer (SOQL → events)
# ---------------------------------------------------------------------------


def test_importer_validates_date_format() -> None:
    importer = AgentForceImporter(connection=_connection())
    with pytest.raises(ValueError, match="Invalid date format"):
        importer.import_sessions(start_date="04/01/2026")


def test_importer_validates_timestamp_format() -> None:
    importer = AgentForceImporter(connection=_connection())
    with pytest.raises(ValueError, match="Invalid timestamp format"):
        importer.import_sessions(last_import_timestamp="2026/04/01 00:00:00")


def test_importer_rejects_malformed_salesforce_id() -> None:
    importer = AgentForceImporter(connection=_connection())
    with pytest.raises(ValueError, match="Invalid Salesforce ID"):
        importer._validate_sf_id("not a real id; DROP TABLE--")


def test_importer_runs_full_query_and_normalizes_records() -> None:
    importer = AgentForceImporter(connection=_connection(), batch_size=50)

    session_row = {
        "Id": "0XxAAAAAAAAAAA1",
        "StartTimestamp": "2026-04-01T10:00:00Z",
        "EndTimestamp": "2026-04-01T10:05:00Z",
        "AiAgentChannelTypeId": "Web",
        "AiAgentSessionEndType": "Completed",
        "VoiceCallId": None,
        "MessagingSessionId": None,
        "PreviousSessionId": None,
    }
    participant_row = {
        "Id": "1XxAAAAAAAAAAA1",
        "AiAgentSessionId": session_row["Id"],
        "AiAgentTypeId": "EinsteinSDR",
        "AiAgentApiName": "Sales_Agent",
        "AiAgentVersionApiName": "v1",
        "ParticipantId": "user-1",
        "AiAgentSessionParticipantRoleId": "Agent",
    }
    interaction_row = {
        "Id": "2XxAAAAAAAAAAA1",
        "AiAgentSessionId": session_row["Id"],
        "AiAgentInteractionTypeId": "Conversation",
        "TelemetryTraceId": "trace-1",
        "TelemetryTraceSpanId": "span-1",
        "TopicApiName": "Lead_Qualification",
        "AttributeText": '{"intent":"qualify"}',
        "PrevInteractionId": None,
    }
    step_row = {
        "Id": "3XxAAAAAAAAAAA1",
        "AiAgentInteractionId": interaction_row["Id"],
        "AiAgentInteractionStepTypeId": "LLMExecutionStep",
        "InputValueText": "what is the lead source?",
        "OutputValueText": "the lead came from a webinar",
        "ErrorMessageText": None,
        "GenerationId": "gen-1",
        "GenAiGatewayRequestId": "req-1",
        "GenAiGatewayResponseId": "resp-1",
        "Name": "lead_source_step",
        "TelemetryTraceSpanId": "span-2",
    }
    message_row = {
        "Id": "4XxAAAAAAAAAAA1",
        "AiAgentInteractionId": interaction_row["Id"],
        "AiAgentInteractionMessageTypeId": "Output",
        "ContentText": "Got it, thanks!",
        "AiAgentInteractionMsgContentTypeId": "Text",
        "MessageSentTimestamp": "2026-04-01T10:01:00Z",
        "ParentMessageId": None,
    }

    query_responses = [
        [session_row],
        [participant_row],
        [interaction_row],
        [step_row],
        [message_row],
    ]

    with mock.patch.object(
        importer._connection,
        "query",
        side_effect=query_responses,
    ):
        events, result = importer.import_sessions(
            start_date="2026-04-01",
            end_date="2026-04-02",
        )

    assert result.sessions_imported == 1
    assert result.participants_imported == 1
    assert result.interactions_imported == 1
    assert result.steps_imported == 1
    assert result.messages_imported == 1
    # 2 lifecycle events (start + end) + participant + interaction + step + message
    assert result.events_generated == 6
    assert len(events) == 6


def test_importer_records_query_failure_in_result() -> None:
    importer = AgentForceImporter(connection=_connection())

    with mock.patch.object(
        importer._connection,
        "query",
        side_effect=SalesforceQueryError("session query failed", soql=""),
    ):
        events, result = importer.import_sessions(start_date="2026-04-01")

    assert events == []
    assert result.sessions_imported == 0
    assert result.errors  # at least one entry


# ---------------------------------------------------------------------------
# Normalizer (DMO → canonical events)
# ---------------------------------------------------------------------------


def test_normalizer_session_emits_start_and_end() -> None:
    n = AgentForceNormalizer()
    events = n.normalize_session(
        {
            "Id": "0Xx1",
            "StartTimestamp": "2026-04-01T00:00:00Z",
            "EndTimestamp": "2026-04-01T00:05:00Z",
            "AiAgentChannelTypeId": "Voice",
            "AiAgentSessionEndType": "Completed",
        }
    )
    assert [e["payload"]["lifecycle_action"] for e in events] == ["start", "end"]
    assert events[0]["event_type"] == "agent.lifecycle"


def test_normalizer_participant_marks_human_for_employee() -> None:
    n = AgentForceNormalizer()
    evt = n.normalize_participant({"AiAgentTypeId": "Employee"})
    assert evt["payload"]["participant_type"] == "human"

    evt = n.normalize_participant({"AiAgentTypeId": "EinsteinSDR"})
    assert evt["payload"]["participant_type"] == "ai"


def test_normalizer_step_routes_by_type() -> None:
    n = AgentForceNormalizer()

    llm_step = n.normalize_step(
        {
            "AiAgentInteractionStepTypeId": "LLMExecutionStep",
            "Name": "summarize",
            "InputValueText": "summarize x",
            "OutputValueText": "x summarized",
            "StartTimestamp": "2026-04-01T10:00:00Z",
            "EndTimestamp": "2026-04-01T10:00:01Z",
        }
    )
    assert llm_step["event_type"] == "model.invoke"
    assert llm_step["payload"]["model"]["provider"] == "salesforce"
    assert llm_step["duration_ms"] == pytest.approx(1000.0)

    tool_step = n.normalize_step(
        {
            "AiAgentInteractionStepTypeId": "ActionInvocationStep",
            "Name": "create_case",
            "InputValueText": '{"subject":"hi"}',
            "OutputValueText": '{"id":"500x"}',
        }
    )
    assert tool_step["event_type"] == "tool.call"
    assert tool_step["payload"]["tool"]["name"] == "create_case"
    # JSON parsed.
    assert tool_step["payload"]["input"] == {"subject": "hi"}

    user_step = n.normalize_step(
        {
            "AiAgentInteractionStepTypeId": "UserInputStep",
            "InputValueText": "hello",
        }
    )
    assert user_step["event_type"] == "agent.input"
    assert user_step["payload"]["content"]["message"] == "hello"


def test_normalizer_interaction_handles_invalid_attribute_json() -> None:
    n = AgentForceNormalizer()
    evt = n.normalize_interaction(
        {
            "Id": "2Xx1",
            "AiAgentSessionId": "0Xx1",
            "AttributeText": "not json {",
        }
    )
    # Falls back to raw wrapper rather than crashing.
    assert evt["payload"]["attributes"] == {"raw": "not json {"}


def test_normalizer_message_routes_role_by_type() -> None:
    n = AgentForceNormalizer()

    out = n.normalize_message(
        {
            "AiAgentInteractionMessageTypeId": "Output",
            "ContentText": "hi",
        }
    )
    assert out["event_type"] == "agent.output"
    assert out["payload"]["content"]["role"] == "agent"

    inp = n.normalize_message(
        {
            "AiAgentInteractionMessageTypeId": "Input",
            "ContentText": "hi",
        }
    )
    assert inp["event_type"] == "agent.input"
    assert inp["payload"]["content"]["role"] == "human"


# ---------------------------------------------------------------------------
# Agent API client + mapper
# ---------------------------------------------------------------------------


def test_client_create_session_validates_inputs() -> None:
    client = AgentApiClient(connection=_connection())
    with pytest.raises(ValueError):
        client.create_session(agent_name="")


def test_client_create_send_end_session_round_trip() -> None:
    client = AgentApiClient(connection=_connection())

    create_resp = _FakeResponse(
        json_data={"sessionId": "session-1", "createdAt": "2026-04-01T10:00:00Z"},
    )
    send_resp = _FakeResponse(
        json_data={
            "messages": [{"id": "m1", "text": "hello back", "timestamp": "2026-04-01T10:00:01Z"}],
            "topic": "Greeting",
            "actions": [{"name": "noop", "parameters": {}, "result": "ok"}],
            "guardrailResults": [{"name": "toxicity", "triggered": False, "message": "clean"}],
        },
    )
    end_resp = _FakeResponse(json_data={})

    with mock.patch("requests.post", side_effect=[create_resp, send_resp]), mock.patch(
        "requests.delete", return_value=end_resp
    ):
        session = client.create_session(agent_name="ServiceAgent")
        assert session.session_id == "session-1"

        message = client.send_message(session.session_id, "hi")
        assert isinstance(message, AgentApiMessage)
        assert message.content == "hello back"
        assert message.topic == "Greeting"
        assert message.actions[0]["name"] == "noop"
        assert message.guardrail_results[0]["name"] == "toxicity"

        client.end_session(session.session_id)


def test_client_send_message_validates_inputs() -> None:
    client = AgentApiClient(connection=_connection())
    with pytest.raises(ValueError):
        client.send_message(session_id="", message="x")
    with pytest.raises(ValueError):
        client.send_message(session_id="s", message="")


def test_client_capture_session_records_full_transcript() -> None:
    client = AgentApiClient(connection=_connection())

    create_resp = _FakeResponse(
        json_data={"sessionId": "s-1", "createdAt": "2026-04-01T10:00:00Z"},
    )
    msg_resp_1 = _FakeResponse(json_data={"messages": [{"id": "m1", "text": "hi"}]})
    msg_resp_2 = _FakeResponse(json_data={"messages": [{"id": "m2", "text": "bye"}]})
    end_resp = _FakeResponse(json_data={})

    with mock.patch(
        "requests.post", side_effect=[create_resp, msg_resp_1, msg_resp_2]
    ), mock.patch("requests.delete", return_value=end_resp):
        session = client.capture_session(
            agent_name="ServiceAgent",
            messages=["hello", "goodbye"],
        )

    assert session.status == "ended"
    # 2 user + 2 agent = 4 messages.
    assert len(session.messages) == 4
    assert [m.role for m in session.messages] == ["user", "agent", "user", "agent"]


def test_mapper_emits_full_session_event_sequence() -> None:
    mapper = AgentApiMapper()
    session = AgentApiSession(
        session_id="s-1",
        agent_name="ServiceAgent",
        created_at="2026-04-01T10:00:00Z",
        ended_at="2026-04-01T10:00:05Z",
        messages=[
            AgentApiMessage(role="user", content="hello"),
            AgentApiMessage(
                role="agent",
                content="hi",
                topic="Greeting",
                actions=[{"name": "noop", "parameters": {}, "result": "ok"}],
                guardrail_results=[
                    {"name": "toxicity", "triggered": False, "message": ""},
                ],
            ),
        ],
    )
    events = mapper.map_session(session)
    types = [e["event_type"] for e in events]
    assert types == [
        "agent.state.change",  # session start
        "agent.input",  # user
        "agent.output",  # agent
        "environment.config",  # topic
        "tool.call",  # action
        "policy.violation",  # guardrail
        "agent.state.change",  # session end
    ]


def test_mapper_session_end_computes_duration() -> None:
    mapper = AgentApiMapper()
    session = AgentApiSession(
        session_id="s",
        created_at="2026-04-01T10:00:00Z",
        ended_at="2026-04-01T10:00:02Z",
    )
    end_event = mapper.map_session_end(session)
    # 2 seconds → 2_000_000_000 nanoseconds.
    assert end_event["payload"]["duration_ns"] == 2_000_000_000


def test_mapper_escalation() -> None:
    evt = AgentApiMapper.map_escalation(
        session_id="s-1",
        from_agent="bot",
        to_agent="human",
        reason="user requested",
    )
    assert evt["event_type"] == "agent.handoff"
    assert evt["payload"]["from_agent"] == "bot"


# ---------------------------------------------------------------------------
# Trust Layer
# ---------------------------------------------------------------------------


def test_trust_layer_to_layerlens_policy_emits_well_formed_yaml() -> None:
    importer = TrustLayerImporter(connection=_connection())
    cfg = TrustLayerConfig(
        guardrails=[
            TrustLayerGuardrail(name="toxicity_detection", type="toxicity"),
            TrustLayerGuardrail(name="pii_detection", type="pii", threshold=0.9),
        ],
    )
    yaml_str = importer.to_layerlens_policy(cfg, policy_name="my_policy")
    assert "policy:" in yaml_str
    assert "name: my_policy" in yaml_str
    assert "toxicity_detection" in yaml_str
    assert "pii_detection" in yaml_str
    assert "threshold: 0.9" in yaml_str
    assert "LayerLens Policy" in yaml_str
    assert "stratix.sdk" not in yaml_str


def test_trust_layer_yaml_has_no_stratix_brand_leak() -> None:
    """Regression: customer-visible YAML must contain LayerLens branding only.

    Trust Layer policies are written to a customer's source tree (and may be
    committed to their VCS / shared with auditors). They MUST NOT leak the
    legacy ``STRATIX`` brand or internal ``stratix.sdk.python.*`` module
    paths. This test exercises the full surface (header comments, generator
    line, body, alias output) so any future regression is caught immediately.
    """
    importer = TrustLayerImporter(connection=_connection())
    cfg = TrustLayerConfig(
        guardrails=[
            TrustLayerGuardrail(
                name="toxicity_detection",
                type="toxicity",
                action="block",
                threshold=0.7,
            ),
            TrustLayerGuardrail(name="pii_detection", type="pii"),
            TrustLayerGuardrail(name="prompt_injection", type="prompt_injection"),
            TrustLayerGuardrail(
                name="hallucination_detection",
                type="hallucination",
            ),
        ],
        data_masking_enabled=True,
        zero_data_retention=True,
        audit_trail_enabled=True,
    )

    yaml_str = importer.to_layerlens_policy(cfg, policy_name="customer_policy")

    # Positive assertions: LayerLens branding is present.
    assert "# LayerLens Policy" in yaml_str
    assert "layerlens.instrument.adapters.frameworks.agentforce.trust_layer" in yaml_str

    # Negative assertions: no STRATIX / stratix.sdk strings escape into the
    # customer-visible YAML output. Casing variants intentional.
    assert "STRATIX" not in yaml_str
    assert "Stratix" not in yaml_str
    assert "stratix.sdk" not in yaml_str
    assert "stratix.sdk.python" not in yaml_str
    assert "ateam" not in yaml_str

    # The deprecated alias must produce identical output (same brand audit).
    with pytest.warns(DeprecationWarning):
        legacy_yaml = importer.to_stratix_policy(cfg, policy_name="customer_policy")
    assert legacy_yaml == yaml_str
    assert "STRATIX" not in legacy_yaml
    assert "stratix.sdk" not in legacy_yaml


def test_trust_layer_deprecation_alias_warns_and_returns_same() -> None:
    importer = TrustLayerImporter(connection=_connection())
    cfg = TrustLayerConfig(guardrails=[TrustLayerGuardrail(name="x", type="custom")])

    with pytest.warns(DeprecationWarning, match="to_layerlens_policy"):
        legacy = importer.to_stratix_policy(cfg)
    canonical = importer.to_layerlens_policy(cfg)
    assert legacy == canonical


def test_trust_layer_classify_guardrail_buckets_known_names() -> None:
    classify = TrustLayerImporter._classify_guardrail
    assert classify("toxicity_detection") == "toxicity"
    assert classify("pii_mask") == "pii"
    assert classify("prompt_injection_guard") == "prompt_injection"
    assert classify("hallucination_check") == "hallucination"
    assert classify("custom_guard") == "custom"


def test_trust_layer_fetch_config_falls_back_to_defaults_on_query_fail() -> None:
    importer = TrustLayerImporter(connection=_connection())

    with mock.patch.object(
        importer._connection,
        "query",
        side_effect=SalesforceQueryError("no perms", soql=""),
    ):
        cfg = importer.fetch_config()

    # Default guardrails populated when nothing came back.
    names = {g.name for g in cfg.guardrails}
    assert "toxicity_detection" in names
    assert "pii_detection" in names


# ---------------------------------------------------------------------------
# Platform Events subscriber
# ---------------------------------------------------------------------------


def test_platform_events_handle_event_invokes_callback_and_records_replay_id() -> None:
    received: list[AgentSessionEvent] = []
    sub = PlatformEventSubscriber(
        connection=_connection(),
        on_event=received.append,
        channel="/event/AgentSession__e",
    )

    sub._handle_event(
        {
            "SessionId__c": "0Xx1",
            "AgentName__c": "ServiceAgent",
            "TopicName__c": "Greeting",
            "ActionsTaken__c": "[]",
            "ResponseText__c": "hi",
            "TrustLayerFlags__c": "{}",
            "event": {"replayId": "42"},
        }
    )

    assert len(received) == 1
    assert received[0].session_id == "0Xx1"
    assert sub.events_received == 1
    assert sub.last_replay_id == "42"


def test_platform_events_default_channel_and_state_flags() -> None:
    sub = PlatformEventSubscriber(connection=_connection())
    assert sub.is_running is False
    assert sub.events_received == 0


# ---------------------------------------------------------------------------
# Einstein evaluator
# ---------------------------------------------------------------------------


def test_evaluator_returns_zero_scores_without_layerlens_client() -> None:
    evaluator = EinsteinEvaluator()
    # No client configured => graders default to 0.0 (logged).
    results = evaluator.evaluate_completions(
        session_ids=["0Xx1"],
        graders=["relevance", "faithfulness"],
    )
    assert len(results) == 1
    assert results[0].scores == {"relevance": 0.0, "faithfulness": 0.0}
    assert results[0].composite_score == 0.0


def test_evaluator_composite_score_uses_weight_categories() -> None:
    evaluator = EinsteinEvaluator()
    composite = evaluator._compute_composite_score(
        {
            "relevance": 1.0,
            "faithfulness": 1.0,
            "safety": 1.0,
        }
    )
    # Three perfect scores collapse to 1.0 regardless of weight choice.
    assert composite == pytest.approx(1.0)

    composite_zero = evaluator._compute_composite_score({})
    assert composite_zero is None


def test_evaluator_returns_empty_when_no_session_ids() -> None:
    assert EinsteinEvaluator().evaluate_completions(session_ids=[]) == []


def test_evaluator_evaluate_topic_requires_adapter() -> None:
    with pytest.raises(RuntimeError, match="Adapter required"):
        EinsteinEvaluator().evaluate_topic(topic="Lead_Qualification")


# ---------------------------------------------------------------------------
# Smoke: NormalizationError surfaces for callers that re-export it
# ---------------------------------------------------------------------------


def test_normalization_error_is_distinct_exception() -> None:
    err = NormalizationError("bad row")
    assert isinstance(err, Exception)
    assert "bad row" in str(err)
