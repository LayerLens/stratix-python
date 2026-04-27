"""Runnable sample: drive the Salesforce Agentforce adapter end-to-end.

This sample is **fully mocked** — both the Salesforce REST surface and the
LayerLens telemetry sink are stubbed in-process. It demonstrates::

    1. Adapter construction with explicit ``CaptureConfig``.
    2. Three import-shaped flows:
       - SOQL session backfill (Data Cloud DMOs).
       - Agent API live capture (synchronous request / response).
       - Einstein Trust Layer policy export.
    3. Event routing through ``BaseAdapter`` → recording sink.
    4. Clean shutdown with summary.

Run::

    pip install 'layerlens[agentforce]'
    python -m samples.instrument.agentforce.main

If the optional ``SALESFORCE_*`` environment variables are present, the
sample additionally exercises a single ``connect()`` call against the live
Salesforce org via the JWT Bearer flow. Otherwise the sample stays in
mock-only mode and exits with code 0.

Required environment for the smoke run:

* (none — the sample exits cleanly without any env vars)

Optional environment for the live auth check:

* ``SALESFORCE_CLIENT_ID`` — Connected App consumer key.
* ``SALESFORCE_USERNAME`` — Salesforce user the JWT is issued for.
* ``SALESFORCE_PRIVATE_KEY`` — PEM-encoded private key (or
  ``env:VARNAME`` reference, or a filesystem path).
* ``SALESFORCE_INSTANCE_URL`` — your org's My Domain URL
  (e.g. ``https://example.my.salesforce.com``).
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest import mock

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.frameworks.agentforce import (
    AgentApiClient,
    AgentForceAdapter,
    EinsteinEvaluator,
    SalesforceAuthError,
    SalesforceConnection,
    SalesforceCredentials,
    TrustLayerImporter,
)
from layerlens.instrument.adapters.frameworks.agentforce.models import (
    TrustLayerConfig,
    TrustLayerGuardrail,
)


class _RecordingSink:
    """Stand-in for an HTTP / OTLP sink — records every event in-process."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append((args[0], args[1]))


def _have_salesforce_env() -> bool:
    return all(
        os.environ.get(name)
        for name in (
            "SALESFORCE_CLIENT_ID",
            "SALESFORCE_USERNAME",
            "SALESFORCE_PRIVATE_KEY",
        )
    )


def _mock_credentials() -> SalesforceCredentials:
    """Build credentials that do NOT require a real Salesforce org."""
    creds = SalesforceCredentials(
        client_id="3MVG9SampleConnectedAppKey0000000",
        username="sample-importer@example.com",
        private_key="-----BEGIN PRIVATE KEY-----\nMIISample\n-----END PRIVATE KEY-----\n",
        instance_url="https://example.my.salesforce.com",
    )
    creds.access_token = "00DSAMPLE!AQ.TOKEN"
    creds.token_expiry = 9_999_999_999.0  # not expired
    return creds


def _mock_connection() -> SalesforceConnection:
    conn = SalesforceConnection(credentials=_mock_credentials())
    conn.instance_url = "https://example.my.salesforce.com"
    return conn


# ---------------------------------------------------------------------------
# Flow 1 — SOQL session backfill (Data Cloud DMO import)
# ---------------------------------------------------------------------------


def _flow_session_backfill(sink: _RecordingSink) -> int:
    """Import a synthetic AgentForce session via the SOQL importer path."""
    adapter = AgentForceAdapter(
        stratix=sink,
        connection=_mock_connection(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    # Replace the connection.query with fixture rows to simulate the SOQL
    # responses the importer would receive from a real Salesforce org.
    session_row = {
        "Id": "0XxSAMPLE00001A",
        "StartTimestamp": "2026-04-25T10:00:00Z",
        "EndTimestamp": "2026-04-25T10:00:30Z",
        "AiAgentChannelTypeId": "Web",
        "AiAgentSessionEndType": "Completed",
        "VoiceCallId": None,
        "MessagingSessionId": None,
        "PreviousSessionId": None,
    }
    participant_row = {
        "Id": "1XxSAMPLE00001B",
        "AiAgentSessionId": session_row["Id"],
        "AiAgentTypeId": "EinsteinServiceAgent",
        "AiAgentApiName": "Service_Agent",
        "AiAgentVersionApiName": "v1",
        "ParticipantId": "user-1",
        "AiAgentSessionParticipantRoleId": "Agent",
    }
    interaction_row = {
        "Id": "2XxSAMPLE00001C",
        "AiAgentSessionId": session_row["Id"],
        "AiAgentInteractionTypeId": "Conversation",
        "TelemetryTraceId": "trace-1",
        "TelemetryTraceSpanId": "span-1",
        "TopicApiName": "Order_Status",
        "AttributeText": '{"intent":"check_order"}',
        "PrevInteractionId": None,
    }
    step_row = {
        "Id": "3XxSAMPLE00001D",
        "AiAgentInteractionId": interaction_row["Id"],
        "AiAgentInteractionStepTypeId": "ActionInvocationStep",
        "InputValueText": '{"order_id":"O-123"}',
        "OutputValueText": '{"status":"shipped"}',
        "ErrorMessageText": None,
        "GenerationId": None,
        "GenAiGatewayRequestId": None,
        "GenAiGatewayResponseId": None,
        "Name": "lookup_order",
        "TelemetryTraceSpanId": "span-2",
    }

    fixture_responses = [
        [session_row],
        [participant_row],
        [interaction_row],
        [step_row],
        [],  # no AIAgentInteractionMessage rows
    ]
    with mock.patch.object(
        adapter._importer._connection,  # type: ignore[union-attr]
        "query",
        side_effect=fixture_responses,
    ):
        result = adapter.import_sessions(start_date="2026-04-25")

    print(
        f"[backfill] imported {result.sessions_imported} session, "
        f"{result.events_generated} events emitted"
    )
    adapter.disconnect()
    return 0


# ---------------------------------------------------------------------------
# Flow 2 — Agent API live capture (request / response)
# ---------------------------------------------------------------------------


def _flow_live_capture() -> int:
    """Drive a live Agent API session through the mocked REST surface."""

    class _R:
        status_code = 200
        headers: dict[str, str] = {}

        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

        def raise_for_status(self) -> None:
            return None

    create_resp = _R({"sessionId": "session-1", "createdAt": "2026-04-25T10:00:00Z"})
    send_resp = _R(
        {
            "messages": [
                {"id": "m1", "text": "Your order shipped on 2026-04-24."},
            ],
            "topic": "Order_Status",
            "actions": [
                {"name": "lookup_order", "parameters": {"id": "O-123"}, "result": "shipped"},
            ],
            "guardrailResults": [
                {"name": "toxicity", "triggered": False, "message": "clean"},
            ],
        }
    )
    end_resp = _R({})

    client = AgentApiClient(connection=_mock_connection())
    with mock.patch("requests.post", side_effect=[create_resp, send_resp]), mock.patch(
        "requests.delete", return_value=end_resp
    ):
        session = client.create_session(agent_name="Service_Agent")
        message = client.send_message(session.session_id, "Where is my order?")
        client.end_session(session.session_id)

    print(f"[live] session={session.session_id} agent_response={message!r}")
    return 0


# ---------------------------------------------------------------------------
# Flow 3 — Einstein Trust Layer policy export
# ---------------------------------------------------------------------------


def _flow_trust_layer_export() -> int:
    """Convert a Trust Layer config into LayerLens YAML policy."""
    importer = TrustLayerImporter(connection=_mock_connection())
    cfg = TrustLayerConfig(
        guardrails=[
            TrustLayerGuardrail(name="toxicity_detection", type="toxicity"),
            TrustLayerGuardrail(name="pii_detection", type="pii", threshold=0.9),
        ],
        zero_data_retention=True,
        audit_trail_enabled=True,
    )
    yaml_str = importer.to_layerlens_policy(cfg, policy_name="sample_policy")
    first_lines = "\n".join(yaml_str.splitlines()[:6])
    print("[trust-layer] generated policy YAML (first 6 lines):")
    print(first_lines)
    return 0


# ---------------------------------------------------------------------------
# Flow 4 — Einstein evaluator (graceful offline fallback)
# ---------------------------------------------------------------------------


def _flow_evaluator_offline() -> int:
    """Show the offline behavior of the evaluator (no LayerLens client)."""
    evaluator = EinsteinEvaluator()
    results = evaluator.evaluate_completions(
        session_ids=["0XxSAMPLE00001A"],
        graders=["relevance", "faithfulness", "safety"],
    )
    for r in results:
        print(
            f"[evaluator] session={r.session_id} composite={r.composite_score} "
            f"scores={r.scores}"
        )
    return 0


# ---------------------------------------------------------------------------
# Optional: live JWT auth check (only if SALESFORCE_* env vars present)
# ---------------------------------------------------------------------------


def _flow_live_auth_check() -> int:
    creds = SalesforceCredentials(
        client_id=os.environ["SALESFORCE_CLIENT_ID"],
        username=os.environ["SALESFORCE_USERNAME"],
        private_key=os.environ["SALESFORCE_PRIVATE_KEY"],
        instance_url=os.environ.get(
            "SALESFORCE_INSTANCE_URL",
            "https://login.salesforce.com",
        ),
    )
    adapter = AgentForceAdapter(credentials=creds, capture_config=CaptureConfig.standard())
    try:
        adapter.connect()
        print("[live-auth] AgentForce adapter authenticated against Salesforce.")
    except SalesforceAuthError as exc:
        print(f"[live-auth] Salesforce auth failed: {exc}", file=sys.stderr)
        return 1
    finally:
        adapter.disconnect()
    return 0


def main() -> int:
    sink = _RecordingSink()

    rc = _flow_session_backfill(sink)
    if rc:
        return rc

    rc = _flow_live_capture()
    if rc:
        return rc

    rc = _flow_trust_layer_export()
    if rc:
        return rc

    rc = _flow_evaluator_offline()
    if rc:
        return rc

    print(f"[summary] sink recorded {len(sink.events)} events across the backfill flow")

    if _have_salesforce_env():
        rc = _flow_live_auth_check()
        if rc:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
