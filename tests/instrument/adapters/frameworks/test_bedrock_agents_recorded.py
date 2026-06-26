"""Recorded-real-response replay for the bedrock_agents framework (LAY-3614).

bedrock_agents was one of the two adapters that shipped on a *fictional* schema
(it read top-level ``outputText``/``trace`` that don't exist; the real data is
inside the ``InvokeAgent`` ``completion`` EventStream). This replays a REAL
captured completion stream (Nova, agent 9WBVYFT6RB, "what is 2+2") through the
adapter's EventStream-proxy seam and asserts the emitted events — the exact
class of regression the fictional schema slipped past every non-live layer.

The recorded stream is injected into ``parsed['completion']`` via an after-call
hook registered *before* the adapter connects, then a ``Stubber`` returns an
empty completion the injector overwrites — the proven double wiring, but the
stream content is the real captured wire, not hand-built members.
"""

from __future__ import annotations

from typing import Any, List

import pytest

boto3 = pytest.importorskip("boto3")

from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter  # noqa: E402

from .conftest import find_event, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, fake_completion_stream  # noqa: E402

_AFTER_HOOK = "after-call.bedrock-agent-runtime.InvokeAgent"
_SESSION_ID = "record-corpus-session"


def _make_boto_client() -> Any:
    return boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        aws_session_token="testing",
    )


def _stream_injector(stream: Any):
    def _inject(**kwargs: Any) -> None:
        parsed = kwargs.get("parsed", {})
        if isinstance(parsed, dict):
            parsed["completion"] = stream

    return _inject


def _setup(mock_client: Any):
    from botocore.stub import Stubber

    fixture = load_recorded("bedrock_agents", "default")
    uploaded = capture_framework_trace(mock_client)
    boto = _make_boto_client()
    boto.meta.events.register(_AFTER_HOOK, _stream_injector(fake_completion_stream(fixture)))

    adapter = BedrockAgentsAdapter(mock_client)
    adapter.connect(target=boto)

    stubber = Stubber(boto)
    stubber.activate()
    stubber.add_response(
        "invoke_agent",
        {"completion": {}, "contentType": "text/plain", "sessionId": _SESSION_ID},
    )
    return adapter, uploaded, boto


def _invoke(boto: Any) -> Any:
    return boto.invoke_agent(
        agentId="9WBVYFT6RB",
        agentAliasId="U2YUM9TNKB",
        sessionId=_SESSION_ID,
        inputText="What is 2 + 2? Answer with just the number.",
        enableTrace=True,
    )


def _drain(response: Any) -> List[Any]:
    return list(response["completion"])


class TestBedrockAgentsRecorded:
    def test_model_invoke_from_real_completion_stream(self, mock_client):
        adapter, uploaded, boto = _setup(mock_client)
        _drain(_invoke(boto))
        adapter.disconnect()

        mi = find_event(uploaded["events"], "model.invoke")
        # Tokens come from the real modelInvocationOutput.metadata.usage — the
        # fictional top-level path would have produced nothing here.
        assert mi["payload"]["provider"] == "aws_bedrock"
        assert mi["payload"]["model"] == "amazon.nova-micro-v1:0"
        assert mi["payload"]["tokens_prompt"] == 986
        assert mi["payload"]["tokens_completion"] == 121
        assert mi["payload"]["tokens_total"] == 1107

    def test_priced_cost_record_for_nova(self, mock_client):
        adapter, uploaded, boto = _setup(mock_client)
        _drain(_invoke(boto))
        adapter.disconnect()

        events = uploaded["events"]
        mi = find_event(events, "model.invoke")
        cost = find_event(events, "cost.record")
        assert cost["payload"]["model"] == "amazon.nova-micro-v1:0"
        assert cost["payload"]["cost_usd"] is not None and cost["payload"]["cost_usd"] > 0
        assert cost["span_id"] == mi["span_id"]

    def test_agent_output_from_real_chunk(self, mock_client):
        adapter, uploaded, boto = _setup(mock_client)
        _drain(_invoke(boto))
        adapter.disconnect()

        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["output"].strip() == "4"

    def test_customer_stream_unbroken(self, mock_client):
        """The proxy must pass every real event through to the caller, in order."""
        adapter, _, boto = _setup(mock_client)
        seen = _drain(_invoke(boto))
        adapter.disconnect()
        # 4 trace events + 1 chunk in the captured stream.
        assert len(seen) == 5
        assert "chunk" in seen[-1]

    def test_provenance(self):
        prov = load_recorded("bedrock_agents", "default")["provenance"]
        assert prov["provider"] == "bedrock_agents"
        assert prov["model"] == "amazon.nova-micro-v1:0"
        assert prov["captured_at"] and prov["captured_at"] != "pending-creds"
