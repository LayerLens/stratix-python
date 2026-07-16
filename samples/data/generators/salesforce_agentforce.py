"""ADP-W2 Family-B recorder for the ``salesforce_agentforce`` adapter (SEALED).

Salesforce Agentforce is credential-gated: importing sessions requires an
OAuth client-credentials grant against a provisioned Agentforce + Data Cloud
org (``SF_CLIENT_ID`` / ``SF_CLIENT_SECRET`` / ``SF_INSTANCE_URL``), and no such
org exists on any build machine. These fixtures are therefore recorded
**SEALED**, exactly like the Google Vertex / Azure OpenAI fixtures: the REAL
``AgentforceAdapter`` (its real ``_SalesforceConnection.authenticate()`` +
``.query()`` + the whole ``import_sessions`` STDM parser) runs against a REAL
``httpx`` client whose transport is an ``httpx.MockTransport`` serving recorded
Session-Tracing-Data-Model (``ssot__*__dlm``) rows. Only the Salesforce
*network* is sealed — every ``ssot__`` field parse, step classification
(LLM_STEP -> model.invoke, ACTION_STEP -> tool.call, escalation -> agent.handoff,
TOPIC_STEP -> agent.interaction), the agent-identity extraction from the
participant roster, and the attestation chain are genuine adapter output.

Agentforce is a **read-only batch importer** (one trace per Salesforce session),
not a live agent SDK, so there is no ``@trace`` wrapper: the adapter opens its
own per-session collector in ``_begin_run`` and flushes it in ``_end_run``. That
flush is observed through the shared ``_generate_fixtures`` capture seam
(``set_trace_observer`` + a no-op ``enqueue_upload``) so the sealed payload is
captured but never uploaded during generation. The samples upload the captured
fixture themselves at run time.

* ``generate_salesforce_agentforce_single`` -> ``salesforce_agentforce_order_status.jsonl``:
  a single Service Cloud customer-service session — a shopper asks where their
  order is; the Agentforce Service Agent routes the topic (TOPIC_STEP), reasons
  (LLM_STEP -> ``model.invoke`` carrying the real generation / gateway ids), calls
  a ``Get_Order_Status`` Apex action (ACTION_STEP -> ``tool.call``), and composes
  the answer (a second LLM_STEP). The STDM carries the agent's declared API name
  on the participant roster, so the Agent column renders that single agent; the
  flat steps render an honest span waterfall. NO ``cost.record`` — the STDM has
  no token fields and the adapter fabricates none.

* ``generate_salesforce_agentforce_multi`` -> ``salesforce_agentforce_billing_escalation.jsonl``:
  a MULTI-TOPIC session for the SAME agent — turn 1 resolves an order-status
  question, turn 2 is a billing dispute the agent cannot settle, so it emits an
  escalation step (-> ``agent.handoff`` with ``from_agent`` = the service agent).
  The STDM step schema carries NO target-agent field for an escalation, so
  ``to_agent`` is honestly ABSENT — this renders a single-agent handoff ORIGIN,
  NOT a fabricated two-node DAG. It is therefore NOT genuinely multi-agent; it is
  a multi-turn single-agent session with a handoff origin, mirroring how real
  Agentforce STDM records an escalation.

HONESTY: both fixtures are marked ``metadata.sealed = true`` with
``source = "synthetic-recorded"`` and ``captured_at = "pending-creds"``. The
``ssot__*`` rows are documented/synthetic (no real Salesforce org was queried),
the ``generation_id`` / gateway ids are synthetic placeholders, and — because the
STDM has no tokens — NO token or cost number is presented as a real billed call.
Provision ``SF_CLIENT_ID`` / ``SF_CLIENT_SECRET`` / ``SF_INSTANCE_URL`` (a real
Agentforce + Data Cloud org) and this recorder can be re-pointed at the live
STDM to replace the sealed rows with a genuine capture.
"""

from __future__ import annotations

import os
import re
import sys

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE

SINGLE_STEM = "salesforce_agentforce_order_status"
MULTI_STEM = "salesforce_agentforce_billing_escalation"

# Real STDM literal that stands in for null in these DMOs (adapter treats it as
# empty everywhere — exercising that path here keeps the fixture honest).
_NOT_SET = "NOT_SET"

_SEALED_META = {
    "sealed": True,
    "provider": "salesforce_agentforce",
    "source": "synthetic-recorded",
    "captured_at": "pending-creds",
    "reason": (
        "No Salesforce Agentforce + Data Cloud org exists (SF_CLIENT_ID / "
        "SF_CLIENT_SECRET / SF_INSTANCE_URL unset). Driven through the REAL "
        "AgentforceAdapter (real _SalesforceConnection.authenticate()/.query() + "
        "the full import_sessions STDM parser) against an httpx.MockTransport "
        "serving documented/synthetic ssot__*__dlm rows; only the Salesforce "
        "network is sealed. The STDM has no token fields, so NO cost/token number "
        "is presented as a real billed call. Re-point at a live org via "
        "SF_CLIENT_ID/SF_CLIENT_SECRET/SF_INSTANCE_URL to replace the sealed rows "
        "with a genuine capture."
    ),
}


# --------------------------------------------------------------------------
# STDM row builders — the exact ``ssot__*__dlm`` field shapes the real adapter
# parses (verified against a live describe in LAY-3599; mirrored in
# tests/fixtures/recorded/agentforce/default.json + test_agentforce_floor.py).
# Content is documented/synthetic Service-Cloud data — no real customer PII.
# --------------------------------------------------------------------------
def _session_row(session_uuid: str, *, channel: str, end_type: str, start: str, end: str) -> dict:
    return {
        "attributes": {"type": "ssot__AiAgentSession__dlm"},
        "ssot__Id__c": session_uuid,
        "ssot__AiAgentChannelType__c": channel,
        "ssot__AiAgentSessionEndType__c": end_type,
        "ssot__StartTimestamp__c": start,
        "ssot__EndTimestamp__c": end,
    }


def _participant_rows(session_uuid: str, *, agent_api: str, agent_version: str, agent_type: str) -> list:
    # The live roster carries one USER-role and one AGENT-role participant; the
    # adapter reads the agent identity off the first row carrying an api name.
    return [
        {
            "attributes": {"type": "ssot__AiAgentSessionParticipant__dlm"},
            "ssot__AiAgentSessionId__c": session_uuid,
            "ssot__AiAgentApiName__c": agent_api,
            "ssot__AiAgentVersionApiName__c": agent_version,
            "ssot__AiAgentType__c": agent_type,
            "ssot__AiAgentSessionParticipantRole__c": "USER",
            "ssot__ParticipantId__c": "005Hn00000Qa7kLIAR",
        },
        {
            "attributes": {"type": "ssot__AiAgentSessionParticipant__dlm"},
            "ssot__AiAgentSessionId__c": session_uuid,
            "ssot__AiAgentApiName__c": agent_api,
            "ssot__AiAgentVersionApiName__c": agent_version,
            "ssot__AiAgentType__c": agent_type,
            "ssot__AiAgentSessionParticipantRole__c": "AGENT",
            "ssot__ParticipantId__c": "0XxHn000001pQ9zKAE",
        },
    ]


def _interaction_row(interaction_uuid: str, session_uuid: str, *, topic: str, trace_id: str,
                     prev: str, start: str, end: str) -> dict:
    return {
        "attributes": {"type": "ssot__AiAgentInteraction__dlm"},
        "ssot__Id__c": interaction_uuid,
        "ssot__AiAgentSessionId__c": session_uuid,
        "ssot__AiAgentInteractionType__c": "TURN",
        "ssot__TopicApiName__c": topic,
        "ssot__TelemetryTraceId__c": trace_id,
        "ssot__PrevInteractionId__c": prev,
        "ssot__StartTimestamp__c": start,
        "ssot__EndTimestamp__c": end,
    }


def _step_row(interaction_uuid: str, step_uuid: str, *, step_type: str, name: str,
              inp: str = _NOT_SET, out: str = _NOT_SET, gen_id: str = _NOT_SET,
              gw_req: str = _NOT_SET, gw_resp: str = _NOT_SET, error: str = _NOT_SET,
              start: str, end: str) -> dict:
    return {
        "attributes": {"type": "ssot__AiAgentInteractionStep__dlm"},
        "ssot__Id__c": step_uuid,
        "ssot__AiAgentInteractionId__c": interaction_uuid,
        "ssot__AiAgentInteractionStepType__c": step_type,
        "SubType__c": None,
        "ssot__Name__c": name,
        "ssot__InputValueText__c": inp,
        "ssot__OutputValueText__c": out,
        "ssot__GenerationId__c": gen_id,
        "ssot__GenAiGatewayRequestId__c": gw_req,
        "ssot__GenAiGatewayResponseId__c": gw_resp,
        "ssot__ErrorMessageText__c": error,
        "ssot__StartTimestamp__c": start,
        "ssot__EndTimestamp__c": end,
    }


def _message_row(interaction_uuid: str, msg_uuid: str, *, msg_type: str, content: str, sent: str) -> dict:
    return {
        "attributes": {"type": "ssot__AiAgentInteractionMessage__dlm"},
        "ssot__Id__c": msg_uuid,
        "ssot__AiAgentInteractionId__c": interaction_uuid,
        "ssot__AiAgentInteractionMessageType__c": msg_type,
        "ssot__ContentText__c": content,
        "ssot__MessageSentTimestamp__c": sent,
    }


# --------------------------------------------------------------------------
# A sealed STDM "org" — routes the adapter's real SOQL GETs (and the OAuth POST)
# over an httpx.MockTransport, by object name + the WHERE-clause parent id. This
# is order-independent (unlike a strict sequential replay), so it exercises the
# real _SalesforceConnection without brittle response ordering.
# --------------------------------------------------------------------------
def _quoted_id(soql: str):
    m = re.search(r"=\s*'([^']*)'", soql)
    return m.group(1) if m else None


def _by(rows: list, field: str, value) -> list:
    return rows if value is None else [r for r in rows if r.get(field) == value]


def _make_transport(instance_url: str, *, sessions: list, participants: list,
                    interactions: list, steps: list, messages: list):
    import httpx

    def _query_records(soql: str) -> list:
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

    def handler(request: "httpx.Request") -> "httpx.Response":
        if request.url.path.endswith("/services/oauth2/token"):
            return httpx.Response(
                200,
                json={
                    "access_token": "[SEALED]",
                    "instance_url": instance_url,
                    "token_type": "Bearer",
                    "issued_at": "1782170151590",
                },
            )
        soql = request.url.params.get("q", "")
        records = _query_records(soql)
        return httpx.Response(200, json={"totalSize": len(records), "done": True, "records": records})

    return httpx.MockTransport(handler)


def _capture_import(client: Stratix, *, instance_url: str, rows: dict) -> dict:
    """Drive the REAL AgentforceAdapter over a sealed MockTransport and capture
    the single flushed trace via the shared observer seam."""
    import httpx

    import layerlens.instrument.adapters.frameworks.agentforce as _af

    transport = _make_transport(instance_url, **rows)
    real_httpx = _af.httpx

    class _Shim:
        def Client(self, **kwargs):
            kwargs.pop("transport", None)
            return real_httpx.Client(transport=transport, timeout=kwargs.get("timeout", 30.0))

        def __getattr__(self, name):
            return getattr(real_httpx, name)

    from layerlens.instrument.adapters.frameworks.agentforce import AgentforceAdapter

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig_enqueue = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    orig_httpx = _af.httpx
    orig_has = _af._HAS_HTTPX
    _af.httpx = _Shim()
    _af._HAS_HTTPX = True
    try:
        adapter = AgentforceAdapter(client, capture_config=_CAPTURE)
        adapter.connect(
            credentials={
                "client_id": "sealed",
                "client_secret": "sealed",
                "instance_url": instance_url,
            }
        )
        try:
            summary = adapter.import_sessions(limit=1)
        finally:
            adapter.disconnect()
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig_enqueue
        _af.httpx = orig_httpx
        _af._HAS_HTTPX = orig_has

    if summary.get("sessions_imported") != 1 or summary.get("errors"):
        raise RuntimeError(f"agentforce import did not cleanly import one session: {summary}")
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for salesforce_agentforce import")
    return payload


def _event_counts(payload: dict) -> dict:
    from collections import Counter

    return dict(Counter(e.get("event_type") for e in payload.get("events", [])))


# --------------------------------------------------------------------------
# Shared identity + timestamps for the sealed Service-Cloud org.
# --------------------------------------------------------------------------
_INSTANCE_URL = "https://sealed-agentforce.my.salesforce.com"
_AGENT_API = "Order_Support_Service_Agent"
_AGENT_VERSION = "v3"
_AGENT_TYPE = "AgentforceServiceAgent"
_CHANNEL = "MessagingForWeb"


# --------------------------------------------------------------------------
# Single: an order-status Service-Cloud session (one topic, one turn).
# --------------------------------------------------------------------------
def generate_salesforce_agentforce_single(client: Stratix) -> dict:
    """Record a sealed single-topic Agentforce order-status support session."""
    session_uuid = "019f0a11-3c4d-7a2b-9e10-2b7c1d4e5f60"
    interaction_uuid = "7a1c2d3e-4f50-4a6b-8c7d-9e0f1a2b3c4d"

    sessions = [
        _session_row(
            session_uuid,
            channel=_CHANNEL,
            end_type="Completed",
            start="2026-07-14T15:02:11.114+0000",
            end="2026-07-14T15:02:29.881+0000",
        )
    ]
    participants = _participant_rows(
        session_uuid, agent_api=_AGENT_API, agent_version=_AGENT_VERSION, agent_type=_AGENT_TYPE
    )
    interactions = [
        _interaction_row(
            interaction_uuid,
            session_uuid,
            topic="Order_Status_And_Tracking",
            trace_id="af14c9d0b73e2a11",
            prev=_NOT_SET,
            start="2026-07-14T15:02:14.220+0000",
            end="2026-07-14T15:02:29.640+0000",
        )
    ]
    steps = [
        _step_row(
            interaction_uuid, "b0000001-topic-4a6b-8c7d-000000000001",
            step_type="TOPIC_STEP", name="Order_Status_And_Tracking",
            start="2026-07-14T15:02:14.300+0000", end="2026-07-14T15:02:14.320+0000",
        ),
        _step_row(
            interaction_uuid, "b0000002-plan0-4a6b-8c7d-000000000002",
            step_type="LLM_STEP", name="Reasoning_And_Planning",
            inp=(
                "Customer message: \"Hi, I ordered a pair of trail running shoes last "
                "Tuesday and haven't gotten a shipping update. My order number is "
                "ORD-5582107. Where is it?\" Determine the customer's intent and the "
                "action needed to answer their order-status question."
            ),
            out=(
                "Intent: order status / delivery tracking for order ORD-5582107. Plan: "
                "call Get_Order_Status with the order number to retrieve fulfillment "
                "and shipment tracking, then summarize the delivery estimate for the "
                "customer."
            ),
            gen_id="4d1e9a77-2b60-4c31-9f80-a1b2c3d4e5f6",
            gw_req="9c2f10ab-77de-4a01-b3e9-1122334455aa",
            gw_resp="chatcmpl-SealedAF0rderStatus001",
            start="2026-07-14T15:02:14.400+0000", end="2026-07-14T15:02:15.980+0000",
        ),
        _step_row(
            interaction_uuid, "b0000003-actn0-4a6b-8c7d-000000000003",
            step_type="ACTION_STEP", name="Get_Order_Status",
            inp="{\"orderNumber\": \"ORD-5582107\"}",
            out=(
                "{\"orderNumber\": \"ORD-5582107\", \"status\": \"Shipped\", \"carrier\": "
                "\"UPS\", \"trackingNumber\": \"1Z999AA10123456784\", \"shippedOn\": "
                "\"2026-07-12\", \"estimatedDelivery\": \"2026-07-16\", \"items\": "
                "[{\"sku\": \"TRL-RUN-42\", \"name\": \"TrailBlazer Running Shoe\", "
                "\"qty\": 1}]}"
            ),
            start="2026-07-14T15:02:16.050+0000", end="2026-07-14T15:02:16.470+0000",
        ),
        _step_row(
            interaction_uuid, "b0000004-comp0-4a6b-8c7d-000000000004",
            step_type="LLM_STEP", name="Compose_Response",
            inp=(
                "Get_Order_Status result: order ORD-5582107 Shipped via UPS, tracking "
                "1Z999AA10123456784, estimated delivery 2026-07-16. Write a friendly, "
                "concise reply with the tracking number and delivery estimate."
            ),
            out=(
                "Good news — your order ORD-5582107 (TrailBlazer Running Shoe) shipped "
                "on July 12 via UPS and is on track to arrive by Tuesday, July 16. You "
                "can track it with UPS number 1Z999AA10123456784. Anything else I can "
                "help with?"
            ),
            gen_id="6f3b2c88-9d10-4e52-a0c1-b2d3e4f5a6b7",
            gw_req="a1b2c3d4-5566-4778-99aa-bbccddeeff00",
            gw_resp="chatcmpl-SealedAF0rderStatus002",
            start="2026-07-14T15:02:16.550+0000", end="2026-07-14T15:02:29.500+0000",
        ),
    ]
    messages = [
        _message_row(
            interaction_uuid, "c0000001-msg-in-000000000001",
            msg_type="Input",
            content=(
                "Hi, I ordered a pair of trail running shoes last Tuesday and haven't "
                "gotten a shipping update. My order number is ORD-5582107. Where is it?"
            ),
            sent="2026-07-14T15:02:14.220+0000",
        ),
        _message_row(
            interaction_uuid, "c0000002-msg-out-00000000002",
            msg_type="Output",
            content=(
                "Good news — your order ORD-5582107 (TrailBlazer Running Shoe) shipped "
                "on July 12 via UPS and is on track to arrive by Tuesday, July 16. You "
                "can track it with UPS number 1Z999AA10123456784. Anything else I can "
                "help with?"
            ),
            sent="2026-07-14T15:02:29.640+0000",
        ),
    ]

    payload = _capture_import(
        client,
        instance_url=_INSTANCE_URL,
        rows=dict(
            sessions=sessions, participants=participants, interactions=interactions,
            steps=steps, messages=messages,
        ),
    )
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "retail",
        "customer-service",
        "salesforce-agentforce",
        "service-cloud",
        "sealed-fixture",
    ]
    payload["metadata"] = dict(_SEALED_META)
    print(
        "  salesforce-agentforce single (service-cloud order status, sealed)  counts=%s"
        % _event_counts(payload)
    )
    print("  ->", _write([payload], "industry", SINGLE_STEM), "\n")
    return payload


# --------------------------------------------------------------------------
# Multi: a multi-topic session escalating a billing dispute (handoff ORIGIN).
# NOT genuinely multi-agent — the same service agent across two turns, with an
# honest single-agent handoff origin (to_agent absent; STDM has no target field).
# --------------------------------------------------------------------------
def generate_salesforce_agentforce_multi(client: Stratix) -> dict:
    """Record a sealed multi-topic Agentforce session with a billing escalation
    (agent.handoff ORIGIN — to_agent honestly absent, single-agent)."""
    session_uuid = "019f0a22-8b7c-7d6e-a1f2-3c4d5e6f7a80"
    turn1_uuid = "8b2d3e4f-5061-4b7c-9d8e-0f1a2b3c4d5e"
    turn2_uuid = "9c3e4f50-6172-4c8d-ae9f-1a2b3c4d5e6f"

    sessions = [
        _session_row(
            session_uuid,
            channel=_CHANNEL,
            end_type="EscalatedToHuman",
            start="2026-07-14T16:41:03.019+0000",
            end="2026-07-14T16:42:18.774+0000",
        )
    ]
    participants = _participant_rows(
        session_uuid, agent_api=_AGENT_API, agent_version=_AGENT_VERSION, agent_type=_AGENT_TYPE
    )
    interactions = [
        _interaction_row(
            turn1_uuid, session_uuid,
            topic="Order_Status_And_Tracking",
            trace_id="bc21d0e19a4f3b22",
            prev=_NOT_SET,
            start="2026-07-14T16:41:05.400+0000",
            end="2026-07-14T16:41:33.210+0000",
        ),
        _interaction_row(
            turn2_uuid, session_uuid,
            topic="Billing_Dispute_Resolution",
            trace_id="bc21d0e19a4f3b23",
            prev=turn1_uuid,
            start="2026-07-14T16:41:40.880+0000",
            end="2026-07-14T16:42:18.640+0000",
        ),
    ]
    steps = [
        # Turn 1 — order status resolved by the agent.
        _step_row(
            turn1_uuid, "d1000001-plan0-000000000001",
            step_type="LLM_STEP", name="Reasoning_And_Planning",
            inp=(
                "Customer message: \"I want to check on order ORD-5583991 and I also "
                "think I was charged twice.\" Handle the order-status part first."
            ),
            out="Intent: order status for ORD-5583991. Plan: call Get_Order_Status.",
            gen_id="1a2b3c4d-1111-4222-8333-444455556666",
            gw_req="55667788-99aa-4bbc-8cdd-eeff00112233",
            gw_resp="chatcmpl-SealedAFBilling001",
            start="2026-07-14T16:41:05.500+0000", end="2026-07-14T16:41:07.020+0000",
        ),
        _step_row(
            turn1_uuid, "d1000002-actn0-000000000002",
            step_type="ACTION_STEP", name="Get_Order_Status",
            inp="{\"orderNumber\": \"ORD-5583991\"}",
            out=(
                "{\"orderNumber\": \"ORD-5583991\", \"status\": \"Delivered\", "
                "\"deliveredOn\": \"2026-07-13\", \"total\": 84.98}"
            ),
            start="2026-07-14T16:41:07.100+0000", end="2026-07-14T16:41:07.560+0000",
        ),
        # Turn 2 — billing dispute the agent cannot settle -> escalation.
        _step_row(
            turn2_uuid, "d2000001-plan0-000000000003",
            step_type="LLM_STEP", name="Reasoning_And_Planning",
            inp=(
                "Customer message: \"I was charged $84.98 twice for order ORD-5583991. "
                "I want the duplicate charge refunded now.\" Assess whether this can be "
                "resolved automatically."
            ),
            out=(
                "A duplicate-charge refund exceeds the automated policy limit and needs "
                "a billing specialist to verify the payment ledger. Plan: look up the "
                "charges, then escalate to the billing queue."
            ),
            gen_id="2b3c4d5e-7777-4888-8999-aaaabbbbcccc",
            gw_req="66778899-aabb-4ccd-8dee-ff0011223344",
            gw_resp="chatcmpl-SealedAFBilling002",
            start="2026-07-14T16:41:41.000+0000", end="2026-07-14T16:41:43.180+0000",
        ),
        _step_row(
            turn2_uuid, "d2000002-actn0-000000000004",
            step_type="ACTION_STEP", name="Lookup_Payment_Charges",
            inp="{\"orderNumber\": \"ORD-5583991\"}",
            out=(
                "{\"orderNumber\": \"ORD-5583991\", \"charges\": [{\"id\": \"ch_9f1a\", "
                "\"amount\": 84.98, \"status\": \"captured\"}, {\"id\": \"ch_9f2b\", "
                "\"amount\": 84.98, \"status\": \"captured\"}], \"duplicateSuspected\": true}"
            ),
            start="2026-07-14T16:41:43.260+0000", end="2026-07-14T16:41:43.780+0000",
        ),
        _step_row(
            turn2_uuid, "d2000003-escl0-000000000005",
            step_type="Escalation",
            name="Escalate_To_Billing_Specialist",
            inp=(
                "Duplicate charge confirmed for order ORD-5583991 (ch_9f1a + ch_9f2b, "
                "$84.98 each). Refund of the duplicate exceeds the automated limit — "
                "escalating to the billing specialist queue for a manual refund."
            ),
            start="2026-07-14T16:42:15.900+0000", end="2026-07-14T16:42:16.120+0000",
        ),
    ]
    messages = [
        _message_row(
            turn1_uuid, "e1000001-msg-in-000000001",
            msg_type="Input",
            content="I want to check on order ORD-5583991 and I also think I was charged twice.",
            sent="2026-07-14T16:41:05.400+0000",
        ),
        _message_row(
            turn1_uuid, "e1000002-msg-out-00000002",
            msg_type="Output",
            content=(
                "Your order ORD-5583991 was delivered on July 13 (total $84.98). Now let "
                "me look into the duplicate charge you mentioned."
            ),
            sent="2026-07-14T16:41:33.210+0000",
        ),
        _message_row(
            turn2_uuid, "e2000001-msg-in-000000003",
            msg_type="Input",
            content=(
                "I was charged $84.98 twice for order ORD-5583991. I want the duplicate "
                "charge refunded now."
            ),
            sent="2026-07-14T16:41:40.880+0000",
        ),
        _message_row(
            turn2_uuid, "e2000002-msg-out-00000004",
            msg_type="Output",
            content=(
                "I can confirm there are two $84.98 charges for order ORD-5583991, so a "
                "duplicate refund is warranted. That refund needs a billing specialist "
                "to process, so I'm escalating your case to our billing team now — "
                "they'll follow up shortly. I'm sorry for the trouble."
            ),
            sent="2026-07-14T16:42:18.640+0000",
        ),
    ]

    payload = _capture_import(
        client,
        instance_url=_INSTANCE_URL,
        rows=dict(
            sessions=sessions, participants=participants, interactions=interactions,
            steps=steps, messages=messages,
        ),
    )
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "retail",
        "customer-service",
        "salesforce-agentforce",
        "service-cloud",
        "billing-escalation",
        "sealed-fixture",
    ]
    payload["metadata"] = dict(_SEALED_META)
    handoffs = [
        ((e.get("payload") or {}).get("from_agent"), (e.get("payload") or {}).get("to_agent"))
        for e in payload.get("events", [])
        if e.get("event_type") == "agent.handoff"
    ]
    print(
        "  salesforce-agentforce multi (multi-topic billing escalation, sealed)  "
        "counts=%s handoffs(from,to)=%s" % (_event_counts(payload), handoffs)
    )
    print("  ->", _write([payload], "industry", MULTI_STEM), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_salesforce_agentforce_single(_client)
    generate_salesforce_agentforce_multi(_client)
