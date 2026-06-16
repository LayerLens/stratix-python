"""Deterministic doubles for the Bedrock Agents framework adapter (LAY-3582 / T8).

Bedrock Agents is credential-gated (no AWS account / agent), so these tests
stand in for live verification. What ``botocore.stub.Stubber`` could and could
not cover here (residual-risk register input):

- STUBBABLE: the static members of the InvokeAgent response (``contentType``,
  ``sessionId``, ``memoryId``, and an empty ``completion``) plus full
  request-parameter validation through the real boto3 event system. That path
  is exercised below (``TestStubbedFlow``) and in ``test_bedrock_agents.py``.
- NOT STUBBABLE: the ``completion`` member is an event stream. botocore's
  Stubber validates stub responses against the service model but has no
  event-stream marshalling, so it cannot synthesize the chunk/trace events
  that flow through ``completion`` on the real wire. Moreover, the keys this
  adapter actually reads in ``_after_invoke`` — top-level ``outputText``,
  ``output.text``, and ``trace`` — are not members of the InvokeAgent output
  shape at all: ``Stubber.add_response`` rejects them with
  ``ParamValidationError`` (pinned by ``test_stubber_rejects_adapter_visible_keys``).
- RESIDUAL RISK: on the real wire, agent output and trace steps arrive as
  events *inside* the ``completion`` stream; whether the boto3 ``after-call``
  hook ever observes them as the top-level ``outputText``/``trace`` keys this
  adapter reads is unverifiable without live AWS access. The adapter's
  ``trace.steps[].type`` schema also does not correspond to the documented
  orchestrationTrace union members (``invocationInput`` / ``rationale`` /
  ``observation`` / ``modelInvocationOutput``), so live verification (or a
  recorded production fixture) is the only way to confirm end-to-end
  extraction against real InvokeAgent traffic.

Consequently the rich-payload tests drive ``_before_invoke``/``_after_invoke``
directly — on an adapter connected to a real ``bedrock-agent-runtime`` client —
with realistic InvokeAgent request params and response dicts in the adapter's
expected schema.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.stub import Stubber  # noqa: E402
from botocore.exceptions import ParamValidationError  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

# ---------------------------------------------------------------------------
# Realistic InvokeAgent request / response doubles
# ---------------------------------------------------------------------------

_REQUEST_PARAMS: Dict[str, Any] = {
    "agentId": "AGT4XK9QZB",
    "agentAliasId": "TSTALIASID",
    "sessionId": "sess-7c2f9d3a-4b1e-4f6c-9a8d-2e5b7c1f0a3d",
    "inputText": "I need to move my flight LL2041 to June 15th.",
    "enableTrace": True,
}


def _action_group_step() -> Dict[str, Any]:
    return {
        "type": "ACTION_GROUP",
        "actionGroupName": "FlightOperations",
        "function": "changeFlight",
        "verb": "POST",
        "apiPath": "/flights/{flightId}/change",
        "executionType": "LAMBDA",
        "actionGroupInput": {"flightId": "LL2041", "newDate": "2026-06-15"},
        "actionGroupInvocationOutput": {
            "output": '{"status": "CONFIRMED", "fareDifference": 42.50}',
            "invocationId": "5f1d9e2c-8a3b-4c7d-b6e1-0f9a8d7c6b5a",
            "responseState": "REPROMPT",
        },
    }


def _knowledge_base_step() -> Dict[str, Any]:
    return {
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseId": "KB9ZXQW123",
        "knowledgeBaseLookupInput": "flight change fee policy international",
        "knowledgeBaseLookupOutput": {
            "retrievedReferences": [
                {
                    "content": {"text": "Changes made more than 24h before departure incur a $40 fee."},
                    "location": {"type": "S3", "s3Location": {"uri": "s3://ll-policies/change-fees.pdf"}},
                    "score": 0.92,
                },
                {
                    "content": {"text": "Same-day changes are subject to availability."},
                    "location": {"type": "S3", "s3Location": {"uri": "s3://ll-policies/same-day.pdf"}},
                    "score": 0.81,
                },
            ]
        },
    }


def _model_invocation_step() -> Dict[str, Any]:
    return {
        "type": "MODEL_INVOCATION",
        "foundationModel": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "modelInvocationOutput": {"usage": {"inputTokens": 642, "outputTokens": 118}},
    }


def _collaborator_step() -> Dict[str, Any]:
    return {
        "type": "AGENT_COLLABORATOR",
        "supervisorAgentId": "AGT4XK9QZB",
        "collaboratorAgentId": "AGTREBOOK1",
        "collaboratorName": "RebookingSpecialist",
        "collaboratorDescription": "Handles rebooking and fare-difference workflows",
        "collaboratorInvocationType": "SUPERVISOR_ROUTER",
        "rationale": "The customer wants to change an existing booking; route to the rebooking specialist.",
        "invocationInput": {"task": "Rebook LL2041 to 2026-06-15"},
    }


def _full_response(session_id: str = _REQUEST_PARAMS["sessionId"]) -> Dict[str, Any]:
    return {
        "sessionId": session_id,
        "contentType": "text/plain",
        "outputText": "Your flight LL2041 has been moved to June 15th. A $40 change fee applies.",
        "trace": {
            "steps": [
                _action_group_step(),
                _knowledge_base_step(),
                _model_invocation_step(),
                _collaborator_step(),
            ]
        },
    }


def _make_boto_client() -> Any:
    return boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        aws_session_token="testing",
    )


def _setup(mock_client: Any, config: Optional[CaptureConfig] = None) -> tuple:
    uploaded = capture_framework_trace(mock_client)
    boto_client = _make_boto_client()
    adapter = BedrockAgentsAdapter(mock_client, capture_config=config)
    adapter.connect(target=boto_client)
    return adapter, uploaded, boto_client


# ---------------------------------------------------------------------------
# What the Stubber can / cannot represent
# ---------------------------------------------------------------------------


class TestStubberLimits:
    def test_stubber_rejects_adapter_visible_keys(self):
        """Pins WHY the rich tests drive hooks directly: the keys the adapter
        reads (`outputText`, `trace`) are not in the InvokeAgent output shape,
        so a service-model-valid stub can never carry them."""
        stubber = Stubber(_make_boto_client())
        with pytest.raises(ParamValidationError):
            stubber.add_response(
                "invoke_agent",
                {
                    "completion": {},
                    "contentType": "text/plain",
                    "sessionId": "sess-1",
                    "outputText": "hello",
                    "trace": {"steps": []},
                },
            )


class TestStubbedFlow:
    def test_stub_valid_response_yields_no_output_content(self, mock_client):
        """A fully service-model-valid stubbed InvokeAgent flows through the
        real event system but carries none of the adapter-readable output keys
        — agent.output is emitted without `output` (documents the gap)."""
        adapter, uploaded, boto_client = _setup(mock_client)
        stubber = Stubber(boto_client)
        stubber.activate()
        stubber.add_response(
            "invoke_agent",
            {
                "completion": {},
                "contentType": "text/plain",
                "sessionId": _REQUEST_PARAMS["sessionId"],
                "memoryId": "mem-1",
            },
        )

        boto_client.invoke_agent(
            agentId=_REQUEST_PARAMS["agentId"],
            agentAliasId=_REQUEST_PARAMS["agentAliasId"],
            sessionId=_REQUEST_PARAMS["sessionId"],
            inputText=_REQUEST_PARAMS["inputText"],
            enableTrace=True,
        )
        adapter.disconnect()
        stubber.assert_no_pending_responses()

        events = uploaded["events"]
        inp = find_event(events, "agent.input")
        assert inp["payload"]["input"] == _REQUEST_PARAMS["inputText"]
        assert inp["payload"]["enable_trace"] is True
        out = find_event(events, "agent.output")
        assert "output" not in out["payload"]  # no stubbable output key exists
        assert not find_events(events, "tool.call")
        assert not find_events(events, "model.invoke")


# ---------------------------------------------------------------------------
# Rich realistic payloads via direct hook drive
# ---------------------------------------------------------------------------


class TestDirectHookDrive:
    def _invoke(self, adapter: BedrockAgentsAdapter, parsed: Optional[Dict[str, Any]] = None) -> None:
        adapter._before_invoke(params=dict(_REQUEST_PARAMS))
        adapter._after_invoke(parsed=parsed if parsed is not None else _full_response())

    def test_agent_io_and_config(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client)
        self._invoke(adapter)
        adapter.disconnect()

        events = uploaded["events"]
        cfg = find_event(events, "environment.config")
        assert cfg["payload"]["agent_id"] == "AGT4XK9QZB"
        assert cfg["payload"]["agent_alias_id"] == "TSTALIASID"
        assert cfg["payload"]["enable_trace"] is True

        inp = find_event(events, "agent.input")
        assert inp["payload"]["agent_id"] == "AGT4XK9QZB"
        assert inp["payload"]["session_id"] == _REQUEST_PARAMS["sessionId"]
        assert inp["payload"]["input"] == _REQUEST_PARAMS["inputText"]

        out = find_event(events, "agent.output")
        assert out["payload"]["session_id"] == _REQUEST_PARAMS["sessionId"]
        assert out["payload"]["output"].startswith("Your flight LL2041 has been moved")
        assert out["payload"]["latency_ms"] is not None

    def test_action_group_schema_metadata(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client)
        self._invoke(adapter)
        adapter.disconnect()

        tool_calls = find_events(uploaded["events"], "tool.call")
        ag = next(tc for tc in tool_calls if tc["payload"]["tool_type"] == "action_group")
        assert ag["payload"]["tool_name"] == "FlightOperations"
        assert ag["payload"]["function"] == "changeFlight"
        assert ag["payload"]["verb"] == "POST"
        assert ag["payload"]["api_path"] == "/flights/{flightId}/change"
        assert ag["payload"]["execution_type"] == "LAMBDA"
        assert ag["payload"]["invocation_id"] == "5f1d9e2c-8a3b-4c7d-b6e1-0f9a8d7c6b5a"
        assert ag["payload"]["status"] == "REPROMPT"
        assert ag["payload"]["input"] == {"flightId": "LL2041", "newDate": "2026-06-15"}
        assert "CONFIRMED" in ag["payload"]["output"]

    def test_knowledge_base_retrieval_ranking(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client)
        self._invoke(adapter)
        adapter.disconnect()

        tool_calls = find_events(uploaded["events"], "tool.call")
        kb = next(tc for tc in tool_calls if tc["payload"]["tool_type"] == "knowledge_base_retrieval")
        assert kb["payload"]["tool_name"] == "KB9ZXQW123"
        assert kb["payload"]["num_results"] == 2
        assert kb["payload"]["retrieval_scores"] == [0.92, 0.81]
        assert kb["payload"]["retrieval_score_max"] == 0.92
        assert kb["payload"]["retrieval_score_min"] == 0.81
        assert kb["payload"]["retrieval_sources"] == [
            "s3://ll-policies/change-fees.pdf",
            "s3://ll-policies/same-day.pdf",
        ]

    def test_model_invocation_tokens_and_cost(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client)
        self._invoke(adapter)
        adapter.disconnect()

        events = uploaded["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert mi["payload"]["provider"] == "aws_bedrock"
        assert mi["payload"]["tokens_prompt"] == 642
        assert mi["payload"]["tokens_completion"] == 118
        assert mi["payload"]["tokens_total"] == 760

        cost = find_event(events, "cost.record")
        assert cost["payload"]["tokens_total"] == 760
        assert cost["payload"]["model"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert cost["span_id"] == mi["span_id"]

    def test_collaborator_handoff_metadata(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client)
        self._invoke(adapter)
        adapter.disconnect()

        handoff = find_event(uploaded["events"], "agent.handoff")
        assert handoff["payload"]["from_agent"] == "AGT4XK9QZB"
        assert handoff["payload"]["to_agent"] == "AGTREBOOK1"
        assert handoff["payload"]["reason"] == "supervisor_delegation"
        # camelCase collaborator metadata is snake_cased into the payload.
        assert handoff["payload"]["collaborator_name"] == "RebookingSpecialist"
        assert handoff["payload"]["collaborator_description"] == "Handles rebooking and fare-difference workflows"
        assert handoff["payload"]["collaborator_invocation_type"] == "SUPERVISOR_ROUTER"
        assert handoff["payload"]["rationale"].startswith("The customer wants to change")
        assert handoff["payload"]["input"] == {"task": "Rebook LL2041 to 2026-06-15"}

    def test_content_gating_strips_rationale_and_io(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client, config=CaptureConfig(capture_content=False))
        self._invoke(adapter)
        adapter.disconnect()

        events = uploaded["events"]
        assert "input" not in find_event(events, "agent.input")["payload"]
        assert "output" not in find_event(events, "agent.output")["payload"]
        handoff = find_event(events, "agent.handoff")
        assert "rationale" not in handoff["payload"]
        assert "input" not in handoff["payload"]
        # Structural metadata survives gating.
        assert handoff["payload"]["collaborator_name"] == "RebookingSpecialist"

    def test_trace_integrity(self, mock_client):
        adapter, uploaded, _ = _setup(mock_client)
        self._invoke(adapter)
        adapter.disconnect()

        events = uploaded["events"]
        assert len({e["trace_id"] for e in events}) == 1
        root = find_event(events, "agent.input")["span_id"]
        for tc in find_events(events, "tool.call"):
            assert tc["parent_span_id"] == root
        seq = [e["sequence_id"] for e in events]
        assert seq == sorted(seq)
