"""Offline error + attestation + redaction + cost floor for the Bedrock Agents adapter.

Closes the W2 census cells for ``bedrock_agents`` — the net-new one being
**attestation** (there was no ``verify_chain`` test for this adapter anywhere) —
while consolidating the real-error-shape, redaction-SENTINEL and cost floors into
the W1 floor pattern, so a regression fails in plain CI with no AWS credentials
and no network:

* Error-shape  — a REAL ``botocore`` SDK exception (``EventStreamError``, the
                 shape a real ``InvokeAgent`` completion stream raises mid-drain)
                 is fired the real way: injected into the recorded ``completion``
                 EventStream and surfaced while the customer iterates
                 ``response["completion"]``. It flows through the adapter's real
                 ``_CompletionProxy.__iter__`` error path to ``agent.error`` with
                 the honest ``error_type == "EventStreamError"`` (the real class
                 name) and the exception message verbatim — and the customer sees
                 the SAME exception re-raised (transparency). A second method
                 covers the in-stream ``failureTrace`` -> ``agent.error``
                 (``error_type == "agent_failure"``, ``status == "error"``).
* Attestation  — the recorded-real Nova completion stream flushes a trace whose
                 attestation chain reconstructs and ``verify_chain(...)`` returns
                 valid; a tamper control breaking link 1 proves it is not vacuous.
* Redaction    — a real InvokeAgent turn (SENTINEL input + a SENTINEL-bearing
                 action-group tool call + final chunk) with ``capture_content=False``
                 keeps structural metadata but strips every content field — and a
                 SENTINEL sweep over ``json.dumps(events)`` finds nothing — with a
                 ``capture_content=True`` vacuity control proving the same path
                 DOES carry the content otherwise.
* Cost         — the recorded Nova stream (real token usage 986/121/1107, a priced
                 ``amazon.nova-micro-v1:0`` model) yields a ``cost.record`` whose
                 ``cost_usd`` is present and > 0, span-linked to its ``model.invoke``.

The ONLY mock is the network boundary: the ``botocore.stub.Stubber`` for the
``InvokeAgent`` call plus the single-read ``FakeEventStream`` injected into
``parsed['completion']`` (the sanctioned bedrock replay seam in
``tests/instrument/_recorded.py``). Every ``bedrock_agents`` object — the real
boto3 event system, the adapter's completion proxy and its real orchestration
parser — is exercised.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import pytest

boto3 = pytest.importorskip("boto3")

from botocore.stub import Stubber  # noqa: E402
from botocore.exceptions import EventStreamError  # noqa: E402

import layerlens.instrument.adapters.frameworks.bedrock_agents as _mod  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import FakeEventStream, load_recorded, fake_completion_stream  # noqa: E402

SENTINEL = "LL-SENTINEL-b3dr0ck-7f3a9c2e"
_SESSION_ID = "floor-session"


# ---------------------------------------------------------------------------
# Offline replay seam — the recorded/constructed completion EventStream is the
# ONLY mock; the boto3 event system + adapter proxy + parser are all real.
# ---------------------------------------------------------------------------
def _make_boto_client() -> Any:
    return boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        aws_session_token="testing",
    )


def _stream_injector(stream: Any):
    """An after-call hook that populates ``parsed['completion']`` with our stream,
    registered BEFORE connect so the adapter's own after-call hook then wraps it
    in the real completion proxy."""

    def _inject(**kwargs: Any) -> None:
        parsed = kwargs.get("parsed", {})
        if isinstance(parsed, dict):
            parsed["completion"] = stream

    return _inject


def _connect(mock_client: Any, stream: Any, *, capture_config: Optional[CaptureConfig] = None):
    """Wire a real boto3 client + Stubber + injected ``stream`` and connect the
    adapter. Returns ``(adapter, uploaded, boto)``; caller drives ``invoke_agent``."""
    uploaded = capture_framework_trace(mock_client)
    boto = _make_boto_client()
    boto.meta.events.register(_mod._AFTER_HOOK, _stream_injector(stream))

    adapter = BedrockAgentsAdapter(mock_client, capture_config=capture_config)
    adapter.connect(target=boto)

    stubber = Stubber(boto)
    stubber.activate()
    stubber.add_response(
        "invoke_agent",
        {"completion": {}, "contentType": "text/plain", "sessionId": _SESSION_ID},
    )
    return adapter, uploaded, boto


def _invoke(boto: Any, *, input_text: str = "What is 2 + 2? Answer with just the number.") -> Any:
    return boto.invoke_agent(
        agentId="9WBVYFT6RB",
        agentAliasId="U2YUM9TNKB",
        sessionId=_SESSION_ID,
        inputText=input_text,
        enableTrace=True,
    )


def _drain(response: Any) -> List[Any]:
    return list(response["completion"])


def _sentinel_stream() -> FakeEventStream:
    """A real-shaped InvokeAgent completion turn whose every content field carries
    the SENTINEL: an action-group tool call (params + output) then the final chunk.

    Mirrors the real orchestrationTrace wire shape the adapter parses, so
    ``capture_content`` gating runs through the adapter's real ``_set_if_capturing``
    path — not a hand-built event dict."""
    action_group = {
        "trace": {
            "agentId": "a-floor",
            "agentAliasId": "al-floor",
            "sessionId": _SESSION_ID,
            "trace": {
                "orchestrationTrace": {
                    "invocationInput": {
                        "invocationType": "ACTION_GROUP",
                        "traceId": "trace-ag-floor",
                        "actionGroupInvocationInput": {
                            "actionGroupName": "BookingActions",
                            "function": "rebook",
                            "parameters": [{"name": "passenger", "type": "string", "value": SENTINEL}],
                        },
                    }
                }
            },
        }
    }
    observation = {
        "trace": {
            "agentId": "a-floor",
            "agentAliasId": "al-floor",
            "sessionId": _SESSION_ID,
            "trace": {
                "orchestrationTrace": {
                    "observation": {
                        "type": "ACTION_GROUP",
                        "traceId": "trace-ag-floor",
                        "actionGroupInvocationOutput": {"text": f"result for {SENTINEL}"},
                    }
                }
            },
        }
    }
    chunk = {"chunk": {"bytes": f"Answer mentioning {SENTINEL}".encode("utf-8")}}
    return FakeEventStream([action_group, observation, chunk])


def _failure_stream(reason: str) -> FakeEventStream:
    """A real-shaped ``failureTrace`` completion event (the agent itself failing)."""
    failure = {
        "trace": {
            "agentId": "a-floor",
            "agentAliasId": "al-floor",
            "sessionId": _SESSION_ID,
            "trace": {"failureTrace": {"failureReason": reason, "failureCode": 500, "traceId": "trace-fail"}},
        }
    }
    return FakeEventStream([failure])


# ---------------------------------------------------------------------------
# Real error-shape floor (real botocore exception, fired the real way)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_eventstream_error_surfaces_as_agent_error(self, mock_client):
        # A genuine botocore SDK exception — the shape a real InvokeAgent
        # completion stream raises when the model/stream errors mid-turn. The
        # FakeEventStream raises it once its events are exhausted, so it surfaces
        # while the customer drains response["completion"] — exactly the real path.
        err = EventStreamError(
            {"Error": {"Code": "ModelStreamErrorException", "Message": "the model stream errored mid-turn"}},
            "InvokeAgent",
        )
        assert type(err).__name__ == "EventStreamError"  # not a hand-rolled stand-in
        real_message = str(err)

        # One real chunk before the error, so there is a genuine mid-stream failure.
        pre_chunk = {"chunk": {"bytes": b"partial"}}
        stream = FakeEventStream([pre_chunk], error=err)
        adapter, uploaded, boto = _connect(mock_client, stream, capture_config=CaptureConfig.full())

        # Transparency: the customer sees the SAME exception re-raised.
        with pytest.raises(EventStreamError) as excinfo:
            _drain(_invoke(boto))
        assert str(excinfo.value) == real_message
        adapter.disconnect()

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        assert len(errors) == 1, f"expected exactly one agent.error, saw {[e['payload'] for e in errors]}"
        payload = errors[0]["payload"]

        # Honest classification: the REAL exception class name (bite: lost if the
        # adapter stops classifying by the real type or stops emitting on a
        # mid-stream failure).
        assert payload["error_type"] == "EventStreamError"
        # The REAL exception message flows through verbatim (bite: dropped/mangled
        # error text fails here).
        assert payload["error"] == real_message
        assert "ModelStreamErrorException" in payload["error"]
        assert payload["framework"] == "bedrock_agents"

        # The mid-stream failure suppresses the (misleading) terminal agent.output.
        assert not find_events(events, "agent.output"), "agent.error must be terminal on a failed stream"

    def test_in_stream_failure_trace_surfaces_as_agent_error(self, mock_client):
        # The other real error surface: a failureTrace event (the Bedrock agent
        # itself failing) — the adapter's honest in-stream classification.
        reason = "Agent could not complete the request: max iterations exceeded"
        adapter, uploaded, boto = _connect(mock_client, _failure_stream(reason), capture_config=CaptureConfig.full())
        _drain(_invoke(boto))
        adapter.disconnect()

        err = find_event(uploaded["events"], "agent.error")["payload"]
        # agent.error is reserved for real failures — honest type + status + code.
        assert err["error_type"] == "agent_failure"
        assert err["status"] == "error"
        assert err["error"] == reason  # verbatim failureReason
        assert err["error_code"] == 500
        assert err["framework"] == "bedrock_agents"


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real recorded InvokeAgent turn
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_recorded_stream(self, mock_client):
        fixture = load_recorded("bedrock_agents", "default")
        adapter, uploaded, boto = _connect(mock_client, fake_completion_stream(fixture))
        _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "the recorded InvokeAgent turn must flush a non-empty trace"

        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the bedrock_agents trace"
        assert len(envelopes) == len(events), (
            f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
        )
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result_ok = verify_chain(envelopes)
        assert result_ok.valid, f"attestation chain invalid: {result_ok.error}"

        # Vacuity control: verify_chain must REJECT a broken interior link, proving
        # the pass above is not trivially true.
        assert len(envelopes) >= 2
        tampered = list(envelopes)
        tampered[1] = AttestationEnvelope(
            hash=tampered[1].hash,
            scope=tampered[1].scope,
            previous_hash="sha256:deadbeef-not-the-prior-hash",
        )
        broken = verify_chain(tampered)
        assert not broken.valid and broken.break_index == 1, "verify_chain failed to detect a broken link"


# ---------------------------------------------------------------------------
# Cost floor — real recorded Nova token shape must carry a priced cost_usd
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_recorded_nova_cost_usd_present_and_span_linked(self, mock_client):
        fixture = load_recorded("bedrock_agents", "default")
        adapter, uploaded, boto = _connect(mock_client, fake_completion_stream(fixture))
        _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        mi = find_event(events, "model.invoke")
        cost = find_event(events, "cost.record")

        # Real token usage came from the recorded modelInvocationOutput.metadata.usage.
        assert mi["payload"]["model"] == "amazon.nova-micro-v1:0"
        assert cost["payload"]["model"] == "amazon.nova-micro-v1:0"
        assert cost["payload"]["tokens_total"] == 1107
        # The dollar figure (bite: a broken BEDROCK_PRICING lookup makes this None).
        assert cost["payload"]["cost_usd"] is not None
        assert cost["payload"]["cost_usd"] > 0
        # Cost is attributed to its own model call.
        assert cost["span_id"] == mi["span_id"]


# ---------------------------------------------------------------------------
# Redaction content-absence over a real SENTINEL-bearing InvokeAgent turn
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: capture_content=True carries the SENTINEL through every
        content field, so the gated assertions below are not vacuous."""
        adapter, uploaded, boto = _connect(
            mock_client, _sentinel_stream(), capture_config=CaptureConfig(capture_content=True)
        )
        _drain(_invoke(boto, input_text=f"Please rebook {SENTINEL}"))
        adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        assert find_event(events, "agent.input")["payload"]["input"] == f"Please rebook {SENTINEL}"
        assert SENTINEL in find_event(events, "agent.output")["payload"]["output"]

        ag = next(tc for tc in find_events(events, "tool.call") if tc["payload"].get("tool_type") == "action_group")
        assert SENTINEL in json.dumps(ag["payload"]["input"])  # action-group args
        assert SENTINEL in ag["payload"]["output"]  # action-group result

    def test_content_absent_when_not_capturing(self, mock_client):
        """The gate: capture_content=False strips agent.input/agent.output and
        tool.call args/output — the SENTINEL appears NOWHERE in any payload —
        while structural metadata survives."""
        adapter, uploaded, boto = _connect(
            mock_client, _sentinel_stream(), capture_config=CaptureConfig(capture_content=False)
        )
        _drain(_invoke(boto, input_text=f"Please rebook {SENTINEL}"))
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "the lifecycle must still emit structural events without content"

        # 1) SENTINEL sweep over the serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) The content keys must be absent from every payload that would carry them.
        inp = find_event(events, "agent.input")
        assert "input" not in inp["payload"], "agent.input leaked 'input' under capture_content=False"
        assert inp["payload"]["agent_id"] == "9WBVYFT6RB", "structural agent_id must survive"

        out = find_event(events, "agent.output")
        assert "output" not in out["payload"], "agent.output leaked 'output' under capture_content=False"

        ag = next(tc for tc in find_events(events, "tool.call") if tc["payload"].get("tool_type") == "action_group")
        assert "input" not in ag["payload"], "tool.call leaked action-group 'input'"
        assert "output" not in ag["payload"], "tool.call leaked action-group 'output'"
        assert ag["payload"]["tool_name"] == "BookingActions", "structural tool_name must survive"
        assert ag["payload"]["function"] == "rebook", "structural function must survive"
