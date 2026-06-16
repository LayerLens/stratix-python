"""End-to-end doubles for the Agentforce adapter against a fake Salesforce REST API
(LAY-3582 / T8).

Agentforce is credential-gated (no Salesforce org), so these tests stand in
for live verification. Unlike ``test_agentforce.py`` (which mocks the
``_SalesforceConnection`` object), this module exercises the REAL connection
class — OAuth client-credentials token exchange, SOQL query URL construction,
Bearer auth headers, and ``nextRecordsUrl`` pagination — over an
``httpx.MockTransport`` serving realistic Salesforce Data Cloud payloads.

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
# Realistic Data Cloud records
# ---------------------------------------------------------------------------

_SESSION_A: Dict[str, Any] = {
    "attributes": {"type": "AIAgentSession__dlm"},
    "Id": "a5W8b000000TkQvEAK",
    "Name": "Session 2026-06-01 09:15",
    "StartTime": "2026-06-01T09:15:23Z",
    "EndTime": "2026-06-01T09:21:47Z",
    "Status": "Completed",
    "AgentId": "0XxK0000000NfXuKAK",
    "AgentName": "Resort Concierge",
    "ParticipantId": "005K0000008aBcDEFG",
    "ParticipantName": "Dana Okafor",
    "Channel": "Messaging",
    "Outcome": "Resolved",
}

_SESSION_B: Dict[str, Any] = {
    "attributes": {"type": "AIAgentSession__dlm"},
    "Id": "a5W8b000000TkRwEAK",
    "Name": "Session 2026-06-02 14:02",
    "StartTime": "2026-06-02T14:02:11Z",
    "EndTime": "2026-06-02T14:09:55Z",
    "Status": "Escalated",
    "AgentId": "0XxK0000000NfXuKAK",
    "AgentName": "Resort Concierge",
    "ParticipantId": "005K0000008aXyZHIJ",
    "ParticipantName": "Luis Romero",
    "Channel": "Voice",
    "Outcome": "Transferred",
}

_INTERACTIONS: Dict[str, List[Dict[str, Any]]] = {
    _SESSION_A["Id"]: [
        {
            "attributes": {"type": "AIAgentInteraction__dlm"},
            "Id": "a5X8b000000Gh1aEAC",
            "SessionId": _SESSION_A["Id"],
            "StepType": "Generative",
            "StepName": "Draft reservation answer",
            "Sequence": 1,
            "Input": "Guest asks: can I get a late checkout on June 3rd?",
            "Output": "Late checkout until 2pm is available for your room type.",
            "ModelName": "sfdc_ai__DefaultOpenAIGPT4Omni",
            "PromptTokens": 412,
            "CompletionTokens": 38,
            "ToolName": None,
            "ToolInput": None,
            "ToolOutput": None,
            "EscalationTarget": None,
            "ErrorMessage": None,
        },
        {
            "attributes": {"type": "AIAgentInteraction__dlm"},
            "Id": "a5X8b000000Gh1bEAC",
            "SessionId": _SESSION_A["Id"],
            "StepType": "Flow",
            "StepName": "Check reservation",
            "Sequence": 2,
            "Input": None,
            "Output": None,
            "ModelName": None,
            "PromptTokens": None,
            "CompletionTokens": None,
            "ToolName": "Get_Reservation_Status",
            "ToolInput": '{"reservationId": "RES-48213"}',
            "ToolOutput": '{"roomType": "Suite", "lateCheckoutEligible": true}',
            "EscalationTarget": None,
            "ErrorMessage": None,
        },
    ],
    _SESSION_B["Id"]: [
        {
            "attributes": {"type": "AIAgentInteraction__dlm"},
            "Id": "a5X8b000000Gh2aEAC",
            "SessionId": _SESSION_B["Id"],
            "StepType": "Generative",
            "StepName": "Detect billing dispute",
            "Sequence": 1,
            "Input": "Caller disputes a $250 minibar charge.",
            "Output": "This requires a human billing specialist.",
            "ModelName": "sfdc_ai__DefaultOpenAIGPT4Omni",
            "PromptTokens": 287,
            "CompletionTokens": 22,
            "ToolName": None,
            "ToolInput": None,
            "ToolOutput": None,
            "EscalationTarget": None,
            "ErrorMessage": None,
        },
        {
            "attributes": {"type": "AIAgentInteraction__dlm"},
            "Id": "a5X8b000000Gh2bEAC",
            "SessionId": _SESSION_B["Id"],
            "StepType": "Escalation",
            "StepName": "Escalate to billing",
            "Sequence": 2,
            "Input": "Disputed minibar charge above auto-refund threshold",
            "Output": None,
            "ModelName": None,
            "PromptTokens": None,
            "CompletionTokens": None,
            "ToolName": None,
            "ToolInput": None,
            "ToolOutput": None,
            "EscalationTarget": "Billing_Disputes_Queue",
            "ErrorMessage": None,
        },
    ],
}

_AGENT_CONFIG: Dict[str, Any] = {
    "attributes": {"type": "AIAgentConfiguration__dlm"},
    "Id": "a5Y8b000000CfG1EAK",
    "AgentId": "0XxK0000000NfXuKAK",
    "AgentName": "Resort Concierge",
    "Description": "Guest services agent for the resort messaging channel",
    "ModelName": "sfdc_ai__DefaultOpenAIGPT4Omni",
    "Instructions": "Help guests with reservations, amenities, and billing questions.",
    "TopicCount": 4,
    "ActionCount": 12,
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

    def _query(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {_ACCESS_TOKEN}"
        soql = request.url.params["q"]
        if "FROM AIAgentSession__dlm" in soql:
            return self._sessions_page(request, offset=0, soql=soql)
        if "FROM AIAgentInteraction__dlm" in soql:
            session_id = re.search(r"SessionId = '([^']+)'", soql).group(1)
            records = _INTERACTIONS.get(session_id, [])
            return httpx.Response(200, json={"totalSize": len(records), "done": True, "records": records})
        if "FROM AIAgentConfiguration__dlm" in soql:
            agent_id = re.search(r"AgentId = '([^']+)'", soql).group(1)
            records = [_AGENT_CONFIG] if agent_id == _AGENT_CONFIG["AgentId"] else []
            return httpx.Response(200, json={"totalSize": len(records), "done": True, "records": records})
        return httpx.Response(400, json=[{"errorCode": "MALFORMED_QUERY", "message": soql}])

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
        cursor = re.search(r"StartTime > (\S+)", soql)
        if cursor:
            # ISO-8601 strings compare lexicographically.
            sessions = [s for s in sessions if s["StartTime"] > cursor.group(1)]
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
# import_sessions end-to-end
# ---------------------------------------------------------------------------


class TestImportSessions:
    def test_two_sessions_normalized_to_trace_events(self, mock_client, monkeypatch):
        adapter, api, uploaded = _setup(mock_client, monkeypatch)
        summary = adapter.import_sessions(limit=50)
        adapter.disconnect()

        assert summary["sessions_imported"] == 2
        assert summary["errors"] == 0
        assert summary["next_cursor"] == _SESSION_B["StartTime"]

        events = uploaded["events"]
        assert summary["events_emitted"] == len(events)

        # Each session is a separate trace.
        assert len({e["trace_id"] for e in events}) == 2

        # environment.config from AIAgentConfiguration__dlm, once per session.
        configs = find_events(events, "environment.config")
        assert len(configs) == 2
        assert configs[0]["payload"]["agent_name"] == "Resort Concierge"
        assert configs[0]["payload"]["model"] == "sfdc_ai__DefaultOpenAIGPT4Omni"
        assert configs[0]["payload"]["topic_count"] == 4
        assert configs[0]["payload"]["action_count"] == 12

        # agent.input carries the session envelope.
        inputs = find_events(events, "agent.input")
        assert len(inputs) == 2
        first = inputs[0]["payload"]
        assert first["session_id"] == _SESSION_A["Id"]
        assert first["agent_name"] == "Resort Concierge"
        assert first["participant_name"] == "Dana Okafor"
        assert first["channel"] == "Messaging"
        assert first["start_time"] == _SESSION_A["StartTime"]

        # Generative steps -> model.invoke (+ cost.record with token rollups).
        invokes = find_events(events, "model.invoke")
        assert len(invokes) == 2
        mi = invokes[0]["payload"]
        assert mi["model"] == "sfdc_ai__DefaultOpenAIGPT4Omni"
        assert mi["tokens_prompt"] == 412
        assert mi["tokens_completion"] == 38
        assert mi["tokens_total"] == 450
        assert mi["messages"].startswith("Guest asks:")
        assert mi["output_message"].startswith("Late checkout")
        costs = find_events(events, "cost.record")
        assert len(costs) == 2
        assert costs[0]["payload"]["tokens_total"] == 450
        assert costs[0]["span_id"] == invokes[0]["span_id"]

        # Flow step -> tool.call.
        tool = find_event(events, "tool.call")
        assert tool["payload"]["tool_name"] == "Get_Reservation_Status"
        assert json.loads(tool["payload"]["input"]) == {"reservationId": "RES-48213"}
        assert "lateCheckoutEligible" in tool["payload"]["output"]

        # Escalation step -> agent.handoff.
        handoff = find_event(events, "agent.handoff")
        assert handoff["payload"]["escalation_target"] == "Billing_Disputes_Queue"
        assert handoff["payload"]["reason"].startswith("Disputed minibar charge")

        # agent.output closes each session with status/outcome.
        outputs = find_events(events, "agent.output")
        assert len(outputs) == 2
        assert outputs[0]["payload"]["status"] == "Completed"
        assert outputs[0]["payload"]["outcome"] == "Resolved"
        assert outputs[1]["payload"]["status"] == "Escalated"
        assert outputs[1]["payload"]["outcome"] == "Transferred"

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

        # SOQL carried the cursor predicate.
        sessions_query = next(
            r.url.params["q"] for r in api.requests if "FROM AIAgentSession__dlm" in r.url.params.get("q", "")
        )
        assert "StartTime > 2026-06-02T00:00:00Z" in sessions_query

        # Only session B (StartTime past the cursor) was imported.
        assert summary["sessions_imported"] == 1
        assert summary["next_cursor"] == _SESSION_B["StartTime"]
        inputs = find_events(uploaded["events"], "agent.input")
        assert len(inputs) == 1
        assert inputs[0]["payload"]["session_id"] == _SESSION_B["Id"]
