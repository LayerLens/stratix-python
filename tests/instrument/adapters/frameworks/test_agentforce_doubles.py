"""End-to-end doubles for the Agentforce adapter against a fake Salesforce REST API
(LAY-3582 / T8; rebuilt for the real STDM in LAY-3599).

Agentforce is credential-gated in CI (no Salesforce org), so these tests stand
in for live verification. Unlike ``test_agentforce.py`` (which mocks the
``_SalesforceConnection`` object), this module exercises the REAL connection
class — OAuth client-credentials token exchange, SOQL query URL construction,
Bearer auth headers, and ``nextRecordsUrl`` pagination — over an
``httpx.MockTransport`` serving realistic Session Tracing Data Model payloads
(``ssot__AiAgent*__dlm``). The records are shaped exactly like the rows a live
``describe`` returned (UUID business ids in ``ssot__Id__c``, the ``NOT_SET``
sentinel, no token fields).

Injection seam: ``_SalesforceConnection.authenticate`` constructs
``httpx.Client(timeout=30.0)`` directly with no transport parameter, so the
module-level ``httpx`` name inside ``agentforce.py`` is monkeypatched with a
shim whose ``Client(**kwargs)`` returns a real ``httpx.Client`` bound to the
MockTransport (no src changes; restored by monkeypatch).
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional

import httpx
import pytest

import layerlens.instrument.adapters.frameworks.agentforce as _mod
from layerlens.instrument.adapters.frameworks.agentforce import AgentforceAdapter

from .conftest import find_event, find_events, capture_framework_trace

_INSTANCE = "https://unit-test.my.salesforce.com"
_ACCESS_TOKEN = "00D8b000001JZg2!AQEAQFakeSessionTokenValue1234"

# ---------------------------------------------------------------------------
# Realistic Session Tracing Data Model records (real ssot__ shapes)
# ---------------------------------------------------------------------------

_SESSION_A_ID = "019ed365-eb7b-73f6-bb95-15e93f8ed2f0"
_SESSION_B_ID = "019ed359-610e-7e45-8b84-27b6e7012d85"
_INT_A_ID = "4c47dd00-390a-4173-9b22-bc476d894bff"
_INT_B_ID = "ff10205d-fd63-454a-ae2d-1f0b72883f1c"

_SESSION_A: Dict[str, Any] = {
    "attributes": {"type": "ssot__AiAgentSession__dlm"},
    "Id": "a5W8b000000TkQvEAK",  # SF surrogate (not the join key)
    "ssot__Id__c": _SESSION_A_ID,
    "ssot__AiAgentChannelType__c": "Messaging",
    "ssot__AiAgentSessionEndType__c": "Completed",
    "ssot__StartTimestamp__c": "2026-06-01T09:15:23.000+0000",
    "ssot__EndTimestamp__c": "2026-06-01T09:21:47.000+0000",
}

_SESSION_B: Dict[str, Any] = {
    "attributes": {"type": "ssot__AiAgentSession__dlm"},
    "Id": "a5W8b000000TkRwEAK",
    "ssot__Id__c": _SESSION_B_ID,
    "ssot__AiAgentChannelType__c": "Voice",
    "ssot__AiAgentSessionEndType__c": "Escalated",
    "ssot__StartTimestamp__c": "2026-06-02T14:02:11.000+0000",
    "ssot__EndTimestamp__c": "2026-06-02T14:09:55.000+0000",
}

# Participants — agent identity lives here (live-populated). One USER row per
# session, carrying the agent it spoke to.
_PARTICIPANTS: Dict[str, List[Dict[str, Any]]] = {
    _SESSION_A_ID: [
        {
            "attributes": {"type": "ssot__AiAgentSessionParticipant__dlm"},
            "ssot__Id__c": "e986e990-a5e2-44fb-b327-33562c4a0161",
            "ssot__AiAgentSessionId__c": _SESSION_A_ID,
            "ssot__AiAgentApiName__c": "Resort_Concierge",
            "ssot__AiAgentVersionApiName__c": "v3",
            "ssot__AiAgentType__c": "AgentforceServiceAgent",
            "ssot__AiAgentSessionParticipantRole__c": "USER",
            "ssot__ParticipantId__c": "005K0000008aBcDEFG",
        }
    ],
    _SESSION_B_ID: [
        {
            "attributes": {"type": "ssot__AiAgentSessionParticipant__dlm"},
            "ssot__Id__c": "f0c0ddd9-b30a-46b3-b998-b3aba6bdbf42",
            "ssot__AiAgentSessionId__c": _SESSION_B_ID,
            "ssot__AiAgentApiName__c": "Resort_Concierge",
            "ssot__AiAgentVersionApiName__c": "v3",
            "ssot__AiAgentType__c": "AgentforceServiceAgent",
            "ssot__AiAgentSessionParticipantRole__c": "USER",
            "ssot__ParticipantId__c": "005K0000008aXyZHIJ",
        }
    ],
}

# Interactions (TURNs) keyed by session UUID.
_INTERACTIONS: Dict[str, List[Dict[str, Any]]] = {
    _SESSION_A_ID: [
        {
            "attributes": {"type": "ssot__AiAgentInteraction__dlm"},
            "ssot__Id__c": _INT_A_ID,
            "ssot__AiAgentSessionId__c": _SESSION_A_ID,
            "ssot__AiAgentInteractionType__c": "TURN",
            "ssot__TopicApiName__c": "Reservations_FAQ",
            "ssot__TelemetryTraceId__c": "ad91cc668ac1db7b",
            "ssot__PrevInteractionId__c": "NOT_SET",
            "ssot__StartTimestamp__c": "2026-06-01T09:15:24.000+0000",
            "ssot__EndTimestamp__c": "2026-06-01T09:15:27.000+0000",
        }
    ],
    _SESSION_B_ID: [
        {
            "attributes": {"type": "ssot__AiAgentInteraction__dlm"},
            "ssot__Id__c": _INT_B_ID,
            "ssot__AiAgentSessionId__c": _SESSION_B_ID,
            "ssot__AiAgentInteractionType__c": "TURN",
            "ssot__TopicApiName__c": "Billing_Disputes",
            "ssot__TelemetryTraceId__c": "8992026495fe7676",
            "ssot__PrevInteractionId__c": "NOT_SET",
            "ssot__StartTimestamp__c": "2026-06-02T14:02:12.000+0000",
            "ssot__EndTimestamp__c": "2026-06-02T14:02:18.000+0000",
        }
    ],
}

# Steps keyed by interaction UUID. Note: NO token fields exist; an LLM step
# carries generation / gateway ids and the input/output value text.
_STEPS: Dict[str, List[Dict[str, Any]]] = {
    _INT_A_ID: [
        {
            "attributes": {"type": "ssot__AiAgentInteractionStep__dlm"},
            "ssot__Id__c": "stepA1-uuid",
            "ssot__AiAgentInteractionId__c": _INT_A_ID,
            "ssot__AiAgentInteractionStepType__c": "LLM_STEP",
            "SubType__c": None,
            "ssot__Name__c": "Draft reservation answer",
            "ssot__InputValueText__c": "Guest asks: can I get a late checkout on June 3rd?",
            "ssot__OutputValueText__c": "Late checkout until 2pm is available for your room type.",
            "ssot__GenerationId__c": "gen-A1-7c2",
            "ssot__GenAiGatewayRequestId__c": "req-A1-7c2",
            "ssot__GenAiGatewayResponseId__c": "resp-A1-7c2",
            "ssot__ErrorMessageText__c": None,
            "ssot__StartTimestamp__c": "2026-06-01T09:15:24.100+0000",
            "ssot__EndTimestamp__c": "2026-06-01T09:15:25.900+0000",
        },
        {
            "attributes": {"type": "ssot__AiAgentInteractionStep__dlm"},
            "ssot__Id__c": "stepA2-uuid",
            "ssot__AiAgentInteractionId__c": _INT_A_ID,
            "ssot__AiAgentInteractionStepType__c": "ACTION_STEP",
            "SubType__c": "Flow",
            "ssot__Name__c": "Get_Reservation_Status",
            "ssot__InputValueText__c": '{"reservationId": "RES-48213"}',
            "ssot__OutputValueText__c": '{"roomType": "Suite", "lateCheckoutEligible": true}',
            "ssot__GenerationId__c": "NOT_SET",
            "ssot__GenAiGatewayRequestId__c": "NOT_SET",
            "ssot__GenAiGatewayResponseId__c": "NOT_SET",
            "ssot__ErrorMessageText__c": None,
            "ssot__StartTimestamp__c": "2026-06-01T09:15:26.000+0000",
            "ssot__EndTimestamp__c": "2026-06-01T09:15:26.400+0000",
        },
    ],
    _INT_B_ID: [
        {
            "attributes": {"type": "ssot__AiAgentInteractionStep__dlm"},
            "ssot__Id__c": "stepB1-uuid",
            "ssot__AiAgentInteractionId__c": _INT_B_ID,
            "ssot__AiAgentInteractionStepType__c": "LLM_STEP",
            "SubType__c": None,
            "ssot__Name__c": "Detect billing dispute",
            "ssot__InputValueText__c": "Caller disputes a $250 minibar charge.",
            "ssot__OutputValueText__c": "This requires a human billing specialist.",
            "ssot__GenerationId__c": "gen-B1-9f4",
            "ssot__GenAiGatewayRequestId__c": "req-B1-9f4",
            "ssot__GenAiGatewayResponseId__c": "resp-B1-9f4",
            "ssot__ErrorMessageText__c": None,
            "ssot__StartTimestamp__c": "2026-06-02T14:02:12.100+0000",
            "ssot__EndTimestamp__c": "2026-06-02T14:02:14.000+0000",
        },
        {
            "attributes": {"type": "ssot__AiAgentInteractionStep__dlm"},
            # Escalation/handoff step type — not observed in the probe org
            # (which emitted LLM_STEP / ACTION_STEP / TOPIC_STEP only), but a
            # real Agentforce capability the classifier supports defensively.
            "ssot__Id__c": "stepB2-uuid",
            "ssot__AiAgentInteractionId__c": _INT_B_ID,
            "ssot__AiAgentInteractionStepType__c": "Escalation",
            "SubType__c": None,
            "ssot__Name__c": "Escalate to billing",
            "ssot__InputValueText__c": "Disputed minibar charge above auto-refund threshold",
            "ssot__OutputValueText__c": "NOT_SET",
            "ssot__GenerationId__c": "NOT_SET",
            "ssot__GenAiGatewayRequestId__c": "NOT_SET",
            "ssot__GenAiGatewayResponseId__c": "NOT_SET",
            "ssot__ErrorMessageText__c": None,
            "ssot__StartTimestamp__c": "2026-06-02T14:02:15.000+0000",
            "ssot__EndTimestamp__c": "2026-06-02T14:02:15.200+0000",
        },
    ],
}

# Messages keyed by interaction UUID — the human-readable turn.
_MESSAGES: Dict[str, List[Dict[str, Any]]] = {
    _INT_A_ID: [
        {
            "attributes": {"type": "ssot__AiAgentInteractionMessage__dlm"},
            "ssot__Id__c": "msgA-in",
            "ssot__AiAgentInteractionId__c": _INT_A_ID,
            "ssot__AiAgentInteractionMessageType__c": "Input",
            "ssot__ContentText__c": "Can I get a late checkout on June 3rd?",
            "ssot__MessageSentTimestamp__c": "2026-06-01T09:15:24.000+0000",
        },
        {
            "attributes": {"type": "ssot__AiAgentInteractionMessage__dlm"},
            "ssot__Id__c": "msgA-out",
            "ssot__AiAgentInteractionId__c": _INT_A_ID,
            "ssot__AiAgentInteractionMessageType__c": "Output",
            "ssot__ContentText__c": "Yes — late checkout until 2pm is available for your suite.",
            "ssot__MessageSentTimestamp__c": "2026-06-01T09:15:27.000+0000",
        },
    ],
    _INT_B_ID: [
        {
            "attributes": {"type": "ssot__AiAgentInteractionMessage__dlm"},
            "ssot__Id__c": "msgB-in",
            "ssot__AiAgentInteractionId__c": _INT_B_ID,
            "ssot__AiAgentInteractionMessageType__c": "Input",
            "ssot__ContentText__c": "I'm disputing a $250 minibar charge.",
            "ssot__MessageSentTimestamp__c": "2026-06-02T14:02:11.500+0000",
        },
        {
            "attributes": {"type": "ssot__AiAgentInteractionMessage__dlm"},
            "ssot__Id__c": "msgB-out",
            "ssot__AiAgentInteractionId__c": _INT_B_ID,
            "ssot__AiAgentInteractionMessageType__c": "Output",
            "ssot__ContentText__c": "I'm transferring you to a billing specialist.",
            "ssot__MessageSentTimestamp__c": "2026-06-02T14:02:18.000+0000",
        },
    ],
}


# ---------------------------------------------------------------------------
# Fake Salesforce REST API served over httpx.MockTransport
# ---------------------------------------------------------------------------


class FakeSalesforceAPI:
    """In-memory Salesforce REST double: OAuth token + /query with pagination."""

    def __init__(self, page_size: int = 1) -> None:
        self.requests: List[httpx.Request] = []
        self.clients: List[httpx.Client] = []
        self.token_status = 200
        self.page_size = page_size
        self.sessions: List[Dict[str, Any]] = [_SESSION_A, _SESSION_B]
        self._last_filtered: List[Dict[str, Any]] = []

    # -- httpx handler --

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/services/oauth2/token":
            return self._token(request)
        if path == "/services/data/v62.0/query/":
            return self._query(request)
        match = re.match(r"^/services/data/v62\.0/query/01g8b0000012Zqy-(\d+)$", path)
        if match:
            return self._sessions_page(request, offset=int(match.group(1)))
        return httpx.Response(404, json=[{"errorCode": "NOT_FOUND", "message": f"no route for {path}"}])

    def _token(self, request: httpx.Request) -> httpx.Response:
        if self.token_status != 200:
            return httpx.Response(
                self.token_status,
                json={"error": "invalid_client", "error_description": "client credentials are invalid"},
            )
        form = dict(httpx.QueryParams(request.content.decode()))
        assert form["grant_type"] == "client_credentials"
        assert form["client_id"]
        assert form["client_secret"]
        return httpx.Response(
            200,
            json={
                "access_token": _ACCESS_TOKEN,
                "signature": "c2lnbmF0dXJl",
                "scope": "api",
                "instance_url": _INSTANCE,
                "id": "https://login.salesforce.com/id/00D8b000001JZg2EAG/0058b00000GxTn9AAF",
                "token_type": "Bearer",
                "issued_at": "1780000000000",
            },
        )

    @staticmethod
    def _where_id(soql: str) -> Optional[str]:
        m = re.search(r"=\s*'([^']+)'", soql)
        return m.group(1) if m else None

    def _query(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {_ACCESS_TOKEN}"
        soql = request.url.params["q"]
        # Children first — their object names contain the parent base string.
        if "FROM ssot__AiAgentSessionParticipant__dlm" in soql:
            return self._records(_PARTICIPANTS.get(self._where_id(soql) or "", []))
        if "FROM ssot__AiAgentInteractionStep__dlm" in soql:
            return self._records(_STEPS.get(self._where_id(soql) or "", []))
        if "FROM ssot__AiAgentInteractionMessage__dlm" in soql:
            return self._records(_MESSAGES.get(self._where_id(soql) or "", []))
        if "FROM ssot__AiAgentInteraction__dlm" in soql:
            return self._records(_INTERACTIONS.get(self._where_id(soql) or "", []))
        if "FROM ssot__AiAgentSession__dlm" in soql:
            return self._sessions_page(request, offset=0, soql=soql)
        return httpx.Response(400, json=[{"errorCode": "MALFORMED_QUERY", "message": soql}])

    @staticmethod
    def _records(records: List[Dict[str, Any]]) -> httpx.Response:
        return httpx.Response(200, json={"totalSize": len(records), "done": True, "records": records})

    def _sessions_page(self, request: httpx.Request, offset: int, soql: str = "") -> httpx.Response:
        matching = self._filtered_sessions(soql) if soql else self._last_filtered
        self._last_filtered = matching
        page = matching[offset : offset + self.page_size]
        body: Dict[str, Any] = {"totalSize": len(matching), "records": page}
        next_offset = offset + self.page_size
        if next_offset < len(matching):
            body["done"] = False
            body["nextRecordsUrl"] = f"/services/data/v62.0/query/01g8b0000012Zqy-{next_offset}"
        else:
            body["done"] = True
        return httpx.Response(200, json=body)

    def _filtered_sessions(self, soql: str) -> List[Dict[str, Any]]:
        sessions = list(self.sessions)
        cursor = re.search(r"ssot__StartTimestamp__c > (\S+)", soql)
        if cursor:
            # ISO-8601 strings compare lexicographically.
            sessions = [s for s in sessions if s["ssot__StartTimestamp__c"] > cursor.group(1)]
        limit = re.search(r"LIMIT (\d+)", soql)
        if limit:
            sessions = sessions[: int(limit.group(1))]
        return sessions


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _setup(
    mock_client: Any,
    monkeypatch: Any,
    api: Optional[FakeSalesforceAPI] = None,
    connect: bool = True,
) -> tuple:
    """Patch the httpx seam, capture uploads, and (optionally) connect."""
    api = api or FakeSalesforceAPI()

    class _HttpxShim:
        @staticmethod
        def Client(**kwargs: Any) -> httpx.Client:
            client = httpx.Client(transport=httpx.MockTransport(api.handler), **kwargs)
            api.clients.append(client)
            return client

    monkeypatch.setattr(_mod, "httpx", _HttpxShim)
    uploaded = capture_framework_trace(mock_client)
    adapter = AgentforceAdapter(mock_client)
    if connect:
        adapter.connect(
            credentials={
                "client_id": "3MVG9pFakeConnectedAppId",
                "client_secret": "FAKE_CLIENT_SECRET",
                "instance_url": _INSTANCE,
            }
        )
    return adapter, api, uploaded


# ---------------------------------------------------------------------------
# Connect / OAuth
# ---------------------------------------------------------------------------


class TestConnect:
    def test_oauth_client_credentials_flow(self, mock_client, monkeypatch):
        adapter, api, _ = _setup(mock_client, monkeypatch)

        token_request = api.requests[0]
        assert token_request.url == f"{_INSTANCE}/services/oauth2/token"
        form = dict(httpx.QueryParams(token_request.content.decode()))
        assert form["grant_type"] == "client_credentials"
        assert form["client_id"] == "3MVG9pFakeConnectedAppId"

        info = adapter.adapter_info()
        assert info.connected is True
        assert info.metadata["instance_url"] == _INSTANCE
        adapter.disconnect()
        assert api.clients[0].is_closed

    def test_auth_failure_raises_and_closes_client(self, mock_client, monkeypatch):
        api = FakeSalesforceAPI()
        api.token_status = 400
        adapter, api, _ = _setup(mock_client, monkeypatch, api=api, connect=False)

        with pytest.raises(httpx.HTTPStatusError):
            adapter.connect(
                credentials={
                    "client_id": "3MVG9pFakeConnectedAppId",
                    "client_secret": "WRONG_SECRET",
                    "instance_url": _INSTANCE,
                }
            )
        assert adapter.adapter_info().connected is False
        assert api.clients[0].is_closed  # failed connect must not leak the client


# ---------------------------------------------------------------------------
# import_sessions end-to-end (real ssot__ STDM)
# ---------------------------------------------------------------------------


class TestImportSessions:
    def test_two_sessions_normalized_to_trace_events(self, mock_client, monkeypatch):
        adapter, api, uploaded = _setup(mock_client, monkeypatch)
        summary = adapter.import_sessions(limit=50)
        adapter.disconnect()

        assert summary["sessions_imported"] == 2
        assert summary["errors"] == 0
        assert summary["next_cursor"] == _SESSION_B["ssot__StartTimestamp__c"]

        events = uploaded["events"]
        assert summary["events_emitted"] == len(events)

        # Each session is a separate trace.
        assert len({e["trace_id"] for e in events}) == 2

        # No AgentConfiguration DMO exists and the STDM has no tokens, so the
        # rewrite must emit neither environment.config nor cost.record.
        assert len(find_events(events, "environment.config")) == 0
        assert len(find_events(events, "cost.record")) == 0

        # agent.lifecycle start event per session (the ateam import_service hook).
        lifecycles = find_events(events, "agent.lifecycle")
        assert len(lifecycles) == 2
        assert all(e["payload"]["lifecycle_action"] == "start" for e in lifecycles)
        assert {e["payload"]["session_id"] for e in lifecycles} == {_SESSION_A_ID, _SESSION_B_ID}

        # agent.input carries the session envelope + agent identity (participant).
        inputs = find_events(events, "agent.input")
        assert len(inputs) == 2
        first = inputs[0]["payload"]
        assert first["session_id"] == _SESSION_A_ID
        assert first["agent_name"] == "Resort_Concierge"
        assert first["agent_version"] == "v3"
        assert first["channel"] == "Messaging"
        assert first["start_time"] == _SESSION_A["ssot__StartTimestamp__c"]
        assert first["content"] == "Can I get a late checkout on June 3rd?"

        # Reasoning steps -> model.invoke carrying gen-id metadata, NO tokens.
        invokes = find_events(events, "model.invoke")
        assert len(invokes) == 2
        mi = invokes[0]["payload"]
        assert mi["generation_id"] == "gen-A1-7c2"
        assert mi["gateway_request_id"] == "req-A1-7c2"
        assert mi["gateway_response_id"] == "resp-A1-7c2"
        assert mi["messages"].startswith("Guest asks:")
        assert mi["output_message"].startswith("Late checkout")
        assert "tokens_prompt" not in mi and "tokens_total" not in mi

        # InvokeAction/Flow step -> tool.call.
        tool = find_event(events, "tool.call")
        assert tool["payload"]["tool_name"] == "Get_Reservation_Status"
        assert json.loads(tool["payload"]["input"]) == {"reservationId": "RES-48213"}
        assert "lateCheckoutEligible" in tool["payload"]["output"]

        # Escalation step -> agent.handoff.
        handoff = find_event(events, "agent.handoff")
        assert handoff["payload"]["step_name"] == "Escalate to billing"
        assert handoff["payload"]["reason"].startswith("Disputed minibar charge")

        # Messages -> agent.interaction turns (user + agent).
        interactions = find_events(events, "agent.interaction")
        roles = {e["payload"].get("role") for e in interactions}
        assert {"user", "agent"} <= roles

        # agent.output closes each session with the end type.
        outputs = find_events(events, "agent.output")
        assert len(outputs) == 2
        assert outputs[0]["payload"]["outcome"] == "Completed"
        assert outputs[1]["payload"]["outcome"] == "Escalated"

        # Migration contract: session id recoverable from every event.
        assert all(e["payload"].get("session_id") in {_SESSION_A_ID, _SESSION_B_ID} for e in events)

        # The fictional schema must be entirely gone from the wire.
        all_soql = " ".join(r.url.params.get("q", "") for r in api.requests if "/query" in r.url.path)
        assert "AIAgentSession__dlm" not in all_soql
        assert "AIAgentConfiguration__dlm" not in all_soql

        # Every Data Cloud query carried the OAuth Bearer token.
        query_requests = [r for r in api.requests if "/query" in r.url.path]
        assert query_requests
        for request in query_requests:
            assert request.headers["Authorization"] == f"Bearer {_ACCESS_TOKEN}"

    def test_session_query_paginates_via_next_records_url(self, mock_client, monkeypatch):
        adapter, api, uploaded = _setup(mock_client, monkeypatch, api=FakeSalesforceAPI(page_size=1))
        summary = adapter.import_sessions()
        adapter.disconnect()

        assert summary["sessions_imported"] == 2
        # The real connection followed nextRecordsUrl for page 2.
        assert any(re.match(r"^/services/data/v62\.0/query/01g8b0000012Zqy-\d+$", r.url.path) for r in api.requests)
        assert len(find_events(uploaded["events"], "agent.input")) == 2

    def test_since_cursor_incremental_import(self, mock_client, monkeypatch):
        adapter, api, uploaded = _setup(mock_client, monkeypatch)
        summary = adapter.import_sessions(since_cursor="2026-06-02T00:00:00Z")
        adapter.disconnect()

        # SOQL carried the cursor predicate on the real timestamp field.
        sessions_query = next(
            r.url.params["q"] for r in api.requests if "FROM ssot__AiAgentSession__dlm" in r.url.params.get("q", "")
        )
        assert "ssot__StartTimestamp__c > 2026-06-02T00:00:00Z" in sessions_query

        # Only session B (start past the cursor) was imported.
        assert summary["sessions_imported"] == 1
        assert summary["next_cursor"] == _SESSION_B["ssot__StartTimestamp__c"]
        inputs = find_events(uploaded["events"], "agent.input")
        assert len(inputs) == 1
        assert inputs[0]["payload"]["session_id"] == _SESSION_B_ID
