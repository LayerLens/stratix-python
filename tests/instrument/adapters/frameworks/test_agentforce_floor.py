"""Offline redaction + attestation + error-shape floor for the Salesforce
Agentforce adapter (``salesforce_agentforce`` / code name ``agentforce``).

Agentforce is a *blocked* adapter (no Salesforce org in CI) and a read-only
batch importer of the Session Tracing Data Model (STDM). This floor closes the
W2 census cells that the existing suites leave ``gap``/``partial`` — attestation
and the full-payload redaction sweep — so a regression fails in plain CI with no
credentials and no network:

* Redaction   — a real STDM session (LLM + tool/ACTION + escalation/handoff +
                topic/unknown steps + Input/Output messages) is imported through
                the adapter's real ``import_sessions`` parser with
                ``capture_content=False``. Every content-bearing STDM field
                carries a SENTINEL; the sweep over ``json.dumps(events)`` proves
                the SENTINEL never leaks, the per-family content keys are all
                stripped, and — non-vacuity — the structural metadata (agent
                name, gateway/generation ids, tool name, handoff origin) DOES
                survive. A ``capture_content=True`` control proves the same path
                carries the SENTINEL and content keys otherwise.
* Attestation — the REAL recorded STDM fixture
                (``tests/fixtures/recorded/agentforce/default.json``, captured
                live + scrubbed) is replayed through the real
                ``_SalesforceConnection`` over ``httpx.MockTransport``. Each
                imported session flushes a trace whose attestation chain
                reconstructs and ``verify_chain(...)`` returns valid; a tamper
                control breaks an interior link and proves the check is not
                vacuous.
* Error shape — (a) a real STDM error step (``ssot__ErrorMessageText__c``)
                surfaces as ``agent.error`` with the adapter's honest
                ``error_type == "step_error"``, ``status == "error"`` and the
                real Salesforce error text carried verbatim — NOT also a
                ``model.invoke``; (b) a real ``httpx.HTTPStatusError`` from the
                OAuth token exchange propagates out of ``connect()`` (the adapter
                never swallows a real Salesforce transport failure) and the
                transport client is closed, not leaked.
* Cost        — the STDM has no token fields, so the adapter emits **no
                ``cost.record``** (``na`` by design, honest — ateam does not
                fabricate cost). Asserted absent over the *real* recorded import
                where ``model.invoke`` events DO exist, so the zero is not
                vacuous. (Per the W2 cost adjudication: a ``cost_usd``-present
                assertion is intentionally NOT written here — there is no
                cost.record event to hang it on; this is a documented ``na``
                cell, not a source-bug hold.)

The only mock is the network boundary (``httpx.MockTransport`` for the recorded
Salesforce wire, and a directly-injected connection double for the controlled
redaction rows); every ``ssot__`` parse, classification and emit is the real
adapter code.
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import httpx
import pytest

import layerlens.instrument.adapters.frameworks.agentforce as _mod
from layerlens.models import CreateTracesResponse
from layerlens.attestation._verify import verify_chain
from layerlens.attestation._envelope import HashScope, AttestationEnvelope
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.agentforce import (
    AgentforceAdapter,
    _SalesforceCredentials,
)

from .conftest import find_event, find_events, capture_framework_trace
from ..._recorded import load_recorded, mock_transport

SENTINEL = "LL-SENTINEL-7f3a9c2e"

# Real STDM identifiers (from a live describe of the dev org — LAY-3599).
SESSION_UUID = "019ed365-eb7b-73f6-bb95-15e93f8ed2f0"
INTERACTION_UUID = "4c47dd00-390a-4173-9b22-bc476d894bff"
AGENT_API_NAME = "Agentforce_Employee_Agent"


# ---------------------------------------------------------------------------
# Controlled STDM rows — every *content* field carries the SENTINEL; every
# *metadata* field (names, topics, ids, channel) is deliberately SENTINEL-free
# so the sweep bites only real content leaks, not legitimate structural fields.
# ---------------------------------------------------------------------------


def _session_row() -> dict:
    return {
        "Id": "q0w000SurrogateAAA",
        "ssot__Id__c": SESSION_UUID,
        "ssot__AiAgentChannelType__c": "EmployeeAgent",  # metadata
        "ssot__AiAgentSessionEndType__c": "Completed",  # metadata
        "ssot__StartTimestamp__c": "2026-06-17T02:25:42.878+0000",
        "ssot__EndTimestamp__c": "2026-06-17T02:25:50.000+0000",
    }


def _participant_row() -> dict:
    return {
        "Id": "q0w000ParticipantA",
        "ssot__Id__c": "e986e990-a5e2-44fb-b327-33562c4a0161",
        "ssot__AiAgentSessionId__c": SESSION_UUID,
        "ssot__AiAgentApiName__c": AGENT_API_NAME,  # metadata (agent identity)
        "ssot__AiAgentVersionApiName__c": "v1",
        "ssot__AiAgentType__c": "AgentforceEmployeeAgent",
        "ssot__AiAgentSessionParticipantRole__c": "USER",
        "ssot__ParticipantId__c": "005fj00000Gz7jJAAR",
    }


def _interaction_row() -> dict:
    return {
        "Id": "q0t000SurrogateI",
        "ssot__Id__c": INTERACTION_UUID,
        "ssot__AiAgentSessionId__c": SESSION_UUID,
        "ssot__AiAgentInteractionType__c": "TURN",
        "ssot__TopicApiName__c": "GeneralFAQ_16jfj000001nP7x",  # metadata
        "ssot__TelemetryTraceId__c": "ad91cc668ac1db7b",
        "ssot__PrevInteractionId__c": "NOT_SET",
        "ssot__StartTimestamp__c": "2026-06-17T02:25:42.878+0000",
        "ssot__EndTimestamp__c": "2026-06-17T02:25:43.688+0000",
    }


def _step_row(step_type: str, name: str, *, inp: Optional[str], out: Optional[str], gen_ids: bool) -> dict:
    return {
        "Id": f"q0s000{name}",
        "ssot__Id__c": f"step-{name}",
        "ssot__AiAgentInteractionId__c": INTERACTION_UUID,
        "ssot__AiAgentInteractionStepType__c": step_type,
        "SubType__c": None,
        "ssot__Name__c": name,  # metadata (step/tool name, span name)
        "ssot__InputValueText__c": inp,
        "ssot__OutputValueText__c": out,
        "ssot__GenerationId__c": "gen-abc-123" if gen_ids else None,  # metadata
        "ssot__GenAiGatewayRequestId__c": "req-abc-123" if gen_ids else None,
        "ssot__GenAiGatewayResponseId__c": "resp-abc-123" if gen_ids else None,
        "ssot__ErrorMessageText__c": None,
        "ssot__StartTimestamp__c": "2026-06-17T02:25:42.900+0000",
        "ssot__EndTimestamp__c": "2026-06-17T02:25:43.600+0000",
    }


def _message_row(message_type: str, content: str) -> dict:
    return {
        "Id": f"q0r000{message_type}",
        "ssot__Id__c": f"msg-{message_type}",
        "ssot__AiAgentInteractionId__c": INTERACTION_UUID,
        "ssot__AiAgentSessionId__c": SESSION_UUID,
        "ssot__AiAgentInteractionMessageType__c": message_type,
        "ssot__ContentText__c": content,
        "ssot__MessageSentTimestamp__c": "2026-06-17T02:25:42.878+0000",
    }


def _make_mock_conn(
    sessions: List[dict],
    participants: List[dict],
    interactions: List[dict],
    steps: List[dict],
    messages: List[dict],
) -> Mock:
    """A double for ``_SalesforceConnection`` that routes SOQL on the real
    ``ssot__`` object names and honours the WHERE-clause id so per-interaction
    children resolve. Mirrors the proven router in ``test_agentforce.py``."""

    def _quoted_id(soql: str) -> Optional[str]:
        m = re.search(r"=\s*'([^']*)'", soql)
        return m.group(1) if m else None

    def _by(rows: List[dict], field: str, value: Optional[str]) -> List[dict]:
        return rows if value is None else [r for r in rows if r.get(field) == value]

    def _query(soql: str) -> List[dict]:
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


def _inject(adapter: AgentforceAdapter, conn: Mock) -> None:
    adapter._connection = conn
    adapter._connected = True
    adapter._credentials = _SalesforceCredentials(
        client_id="test",
        client_secret="test",
        instance_url="https://test.salesforce.com",
        access_token="fake-token",
    )
    adapter._metadata["instance_url"] = "https://test.salesforce.com"


def _sentinel_conn() -> Mock:
    """One session exercising every step family, SENTINEL in every content slot."""
    return _make_mock_conn(
        sessions=[_session_row()],
        participants=[_participant_row()],
        interactions=[_interaction_row()],
        steps=[
            _step_row("LLM_STEP", "generate_response", inp=f"model in {SENTINEL}", out=f"model out {SENTINEL}", gen_ids=True),
            _step_row("ACTION_STEP", "get_weather", inp=f"tool in {SENTINEL}", out=f"tool out {SENTINEL}", gen_ids=False),
            _step_row("Escalation", "escalate_to_human", inp=f"handoff reason {SENTINEL}", out=None, gen_ids=False),
            _step_row("TOPIC_STEP", "topic_route", inp=f"unk in {SENTINEL}", out=f"unk out {SENTINEL}", gen_ids=False),
        ],
        messages=[
            _message_row("Input", f"user asks {SENTINEL}"),
            _message_row("Output", f"agent replies {SENTINEL}"),
        ],
    )


# ---------------------------------------------------------------------------
# Redaction floor — full-payload SENTINEL sweep across every event family
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: capture_content=True carries the SENTINEL and every
        content key it rides on across all families."""
        uploaded = capture_framework_trace(mock_client)
        adapter = AgentforceAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        _inject(adapter, _sentinel_conn())
        adapter.import_sessions()
        adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"

        assert find_event(events, "agent.input")["payload"]["content"] == f"user asks {SENTINEL}"
        assert find_event(events, "agent.output")["payload"]["content"] == f"agent replies {SENTINEL}"
        mi = find_event(events, "model.invoke")["payload"]
        assert mi["messages"] == f"model in {SENTINEL}"
        assert mi["output_message"] == f"model out {SENTINEL}"
        tc = find_event(events, "tool.call")["payload"]
        assert tc["input"] == f"tool in {SENTINEL}"
        assert tc["output"] == f"tool out {SENTINEL}"
        assert find_event(events, "agent.handoff")["payload"]["reason"] == f"handoff reason {SENTINEL}"
        # The topic/unknown step interaction carries input/output content.
        unknown = next(
            e for e in find_events(events, "agent.interaction") if "input" in e["payload"] or "output" in e["payload"]
        )
        assert unknown["payload"]["input"] == f"unk in {SENTINEL}"

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False strips every content field — and the SENTINEL —
        from every stored event, while structural metadata survives."""
        uploaded = capture_framework_trace(mock_client)
        adapter = AgentforceAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        _inject(adapter, _sentinel_conn())
        adapter.import_sessions()
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "the import must still emit structural events without content"

        # 1) The strong sweep: the planted secret must not survive anywhere.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) Per-family content keys are all absent.
        for e in find_events(events, "agent.input"):
            assert "content" not in e["payload"], "agent.input leaked 'content'"
        for e in find_events(events, "agent.output"):
            assert "content" not in e["payload"], "agent.output leaked 'content'"
        mi = find_event(events, "model.invoke")["payload"]
        assert "messages" not in mi and "output_message" not in mi, "model.invoke leaked content"
        tc = find_event(events, "tool.call")["payload"]
        assert "input" not in tc and "output" not in tc, "tool.call leaked content"
        assert "reason" not in find_event(events, "agent.handoff")["payload"], "agent.handoff leaked 'reason'"
        for e in find_events(events, "agent.interaction"):
            assert "input" not in e["payload"], "agent.interaction leaked 'input'"
            assert "output" not in e["payload"], "agent.interaction leaked 'output'"
            assert "content" not in e["payload"], "agent.interaction leaked message 'content'"

        # 3) Non-vacuity: the structural metadata that is NOT content must remain,
        # proving the events were emitted (not simply dropped).
        assert find_event(events, "agent.input")["payload"]["agent_name"] == AGENT_API_NAME
        assert mi["generation_id"] == "gen-abc-123"
        assert mi["gateway_request_id"] == "req-abc-123"
        assert mi["gateway_response_id"] == "resp-abc-123"
        assert tc["tool_name"] == "get_weather"
        assert find_event(events, "agent.handoff")["payload"]["from_agent"] == AGENT_API_NAME


# ---------------------------------------------------------------------------
# Recorded-STDM replay seam (real _SalesforceConnection over MockTransport)
# ---------------------------------------------------------------------------
def _recorded_import(mock_client, monkeypatch):
    """Import the REAL recorded Salesforce STDM fixture through the real
    connection, capturing each flushed trace's full payload (events +
    attestation) so per-trace attestation can be verified independently."""
    monkeypatch.setattr(_mod, "_HAS_HTTPX", True)
    fixture = load_recorded("agentforce", "default")
    transport, _ = mock_transport(fixture)
    real_httpx = _mod.httpx

    class _Shim:
        def Client(self, **kwargs: Any) -> Any:
            kwargs.pop("transport", None)
            return real_httpx.Client(transport=transport, timeout=kwargs.get("timeout", 30.0))

        def __getattr__(self, name: str) -> Any:
            return getattr(real_httpx, name)

    monkeypatch.setattr(_mod, "httpx", _Shim())

    traces: List[Dict[str, Any]] = []

    def _capture(path: str) -> CreateTracesResponse:
        with open(path) as f:
            payload = json.load(f)[0]
        traces.append(payload)
        return CreateTracesResponse(trace_ids=[payload.get("trace_id") or "mock-trace-id"])

    mock_client.traces.upload.side_effect = _capture

    adapter = AgentforceAdapter(mock_client)
    adapter.connect(
        credentials={
            "client_id": "x",
            "client_secret": "y",
            "instance_url": "https://unit-test.my.salesforce.com",
        }
    )
    summary = adapter.import_sessions(limit=2)
    adapter.disconnect()
    return traces, summary


class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_stdm_import(self, mock_client, monkeypatch):
        traces, summary = _recorded_import(mock_client, monkeypatch)
        assert summary["sessions_imported"] == 2 and summary["errors"] == 0
        assert traces, "the real STDM import must flush at least one trace"

        tamperable: List[List[AttestationEnvelope]] = []
        for tr in traces:
            events = tr.get("events") or []
            attestation = tr.get("attestation") or {}
            raw = (attestation.get("chain") or {}).get("events") or []
            envelopes = [
                AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
                for e in raw
            ]
            assert envelopes, "no attestation envelopes captured for a real STDM trace"
            assert len(envelopes) == len(events), (
                f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
            )
            assert attestation.get("root_hash") is not None

            result = verify_chain(envelopes)
            assert result.valid, f"attestation chain invalid: {result.error}"
            if len(envelopes) >= 2:
                tamperable.append(envelopes)

        # Vacuity control: verify_chain must REJECT a broken interior link.
        assert tamperable, "expected at least one multi-event trace to break a link in"
        envelopes = tamperable[0]
        tampered = list(envelopes)
        tampered[1] = AttestationEnvelope(
            hash=tampered[1].hash,
            scope=tampered[1].scope,
            previous_hash="sha256:deadbeef-not-the-prior-hash",
        )
        broken = verify_chain(tampered)
        assert not broken.valid and broken.break_index == 1, "verify_chain failed to detect a broken link"


# ---------------------------------------------------------------------------
# Cost — honest absence (STDM has no tokens; the adapter fabricates no cost)
# ---------------------------------------------------------------------------
class TestCostHonestAbsence:
    def test_no_cost_record_over_real_stdm_import(self, mock_client, monkeypatch):
        traces, _ = _recorded_import(mock_client, monkeypatch)
        all_events = [e for tr in traces for e in (tr.get("events") or [])]
        # Non-vacuity: real model.invoke events exist in this import ...
        assert find_events(all_events, "model.invoke"), "expected model.invoke events in the real STDM import"
        # ... yet the STDM has no token fields, so NO cost.record is fabricated.
        assert find_events(all_events, "cost.record") == [], "STDM has no tokens — agentforce must not emit cost.record"


# ---------------------------------------------------------------------------
# Error shape — real STDM error step + real OAuth transport failure
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_step_error_surfaces_as_agent_error(self, mock_client):
        """A real STDM error step carries the Salesforce error text verbatim as
        agent.error (honest error_type/status), and is NOT also a model.invoke."""
        real_error = "Salesforce Einstein gateway returned HTTP 503: model overloaded"
        conn = _make_mock_conn(
            sessions=[_session_row()],
            participants=[_participant_row()],
            interactions=[_interaction_row()],
            steps=[
                {
                    **_step_row("LLM_STEP", "generate_response", inp="prompt", out=None, gen_ids=True),
                    "ssot__ErrorMessageText__c": real_error,
                }
            ],
            messages=[],
        )
        uploaded = capture_framework_trace(mock_client)
        adapter = AgentforceAdapter(mock_client, capture_config=CaptureConfig.full())
        _inject(adapter, conn)
        adapter.import_sessions()
        adapter.disconnect()

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        assert len(errors) == 1, f"expected exactly one agent.error, saw {[e['payload'] for e in errors]}"
        payload = errors[0]["payload"]
        assert payload["error_type"] == "step_error"
        assert payload["status"] == "error"
        assert payload["error_message"] == real_error  # verbatim, no mangling
        assert payload["session_id"] == SESSION_UUID
        assert payload["framework"] == "agentforce"
        # An errored LLM step is agent.error only — never also model.invoke.
        assert find_events(events, "model.invoke") == []

    def test_real_oauth_http_error_propagates_and_closes_client(self, mock_client, monkeypatch):
        """A real httpx.HTTPStatusError from the OAuth token exchange propagates
        out of connect() (never swallowed) and the transport client is closed."""
        monkeypatch.setattr(_mod, "_HAS_HTTPX", True)
        real_httpx = _mod.httpx
        clients: List[httpx.Client] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "invalid_client"})

        class _Shim:
            def Client(self, **kwargs: Any) -> Any:
                kwargs.pop("transport", None)
                client = real_httpx.Client(transport=httpx.MockTransport(_handler), timeout=kwargs.get("timeout", 30.0))
                clients.append(client)
                return client

            def __getattr__(self, name: str) -> Any:
                return getattr(real_httpx, name)

        monkeypatch.setattr(_mod, "httpx", _Shim())

        adapter = AgentforceAdapter(mock_client)
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            adapter.connect(
                credentials={
                    "client_id": "x",
                    "client_secret": "WRONG",
                    "instance_url": "https://unit-test.my.salesforce.com",
                }
            )
        assert excinfo.value.response.status_code == 400
        assert not adapter.is_connected
        assert clients and clients[0].is_closed, "a failed connect must close the transport client, not leak it"
        # A failed connect leaves the adapter unusable — imports are rejected.
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.import_sessions()
