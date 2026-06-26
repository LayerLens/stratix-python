"""Recorded-real-response replay for the AWS Bedrock provider (LAY-3614).

Replays a REAL captured Amazon Nova response — both the Converse API
(``converse``) and the Converse-shaped ``invoke_model`` body — through a real
``bedrock-runtime`` boto3 client over ``botocore.stub.Stubber`` and asserts the
adapter's emitted events. The fixture is the provider's raw transport response
(recorded upstream of the parser); the assertions are the events (downstream).
Unlike the hand-built ``test_bedrock`` doubles — which fabricate token counts and
synthesize the body — these run against the actual Nova shape, including the
``cacheReadInputTokenCount`` / ``cacheWriteInputTokenCount`` usage members the
fabricated doubles never carried.

See ``tests/instrument/_recorded.py`` for the corpus design + its snapshot limit.
"""

from __future__ import annotations

import json

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.stub import ANY, Stubber  # noqa: E402

from layerlens.instrument import trace  # noqa: E402
from layerlens.instrument.adapters.providers.bedrock import BedrockProvider  # noqa: E402

from ...conftest import find_event  # noqa: E402
from ..._recorded import load_recorded, bedrock_stub_response  # noqa: E402

# The recorded fixtures captured Amazon Nova (Converse-shaped for both ops).
_MODEL_ID = "amazon.nova-micro-v1:0"


def _make_client():
    """Real bedrock-runtime client with static fake credentials (offline-safe)."""
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        aws_session_token="testing",
    )


def _connect(client):
    provider = BedrockProvider()
    provider.connect(client)
    stubber = Stubber(client)
    stubber.activate()
    return provider, stubber


def _nova_invoke_request_body() -> str:
    return json.dumps(
        {
            "schemaVersion": "messages-v1",
            "system": [{"text": "You are terse."}],
            "messages": [{"role": "user", "content": [{"text": "Reply with exactly: pong"}]}],
            "inferenceConfig": {"maxTokens": 16},
        }
    )


class TestBedrockRecorded:
    def test_converse_real_shape(self, mock_client, capture_trace):
        fixture = load_recorded("bedrock", "converse")
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_response(
            "converse",
            bedrock_stub_response(fixture),
            {"modelId": _MODEL_ID, "messages": ANY, "inferenceConfig": {"maxTokens": 16}},
        )

        @trace(mock_client)
        def agent():
            r = client.converse(
                modelId=_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": "Reply with exactly: pong"}]}],
                inferenceConfig={"maxTokens": 16},
            )
            return r["output"]["message"]["content"][0]["text"]

        # Passthrough: the caller still sees the real recorded Nova text.
        assert agent() == "pong"
        stubber.assert_no_pending_responses()

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "aws_bedrock.converse"
        assert mi["payload"]["model"] == _MODEL_ID
        assert mi["payload"]["latency_ms"] > 0
        # stopReason from the real Converse response.
        assert mi["payload"]["stop_reason"] == "end_turn"
        # Input message normalized from the content-block list.
        assert mi["payload"]["messages"] == [{"role": "user", "content": "Reply with exactly: pong"}]
        # Output text parsed from output.message.content[].text.
        assert mi["payload"]["output_message"] == {"role": "assistant", "content": "pong"}
        # Usage straight from the recorded Converse usage block (6/3/9).
        assert mi["payload"]["usage"] == {
            "prompt_tokens": 6,
            "completion_tokens": 3,
            "total_tokens": 9,
        }

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "aws_bedrock"
        assert cost["payload"]["model"] == _MODEL_ID
        assert cost["payload"]["total_tokens"] == 9
        # nova-micro: 6 * 0.000035/1k input + 3 * 0.00014/1k output.
        assert cost["payload"]["cost_usd"] == pytest.approx(6.3e-07)
        assert cost["span_id"] == mi["span_id"]

        provider.disconnect()

    def test_invoke_model_real_shape(self, mock_client, capture_trace):
        fixture = load_recorded("bedrock", "invoke_model")
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_response(
            "invoke_model",
            bedrock_stub_response(fixture),
            {
                "modelId": _MODEL_ID,
                "body": ANY,
                "accept": "application/json",
                "contentType": "application/json",
            },
        )

        @trace(mock_client)
        def agent():
            r = client.invoke_model(
                modelId=_MODEL_ID,
                body=_nova_invoke_request_body(),
                accept="application/json",
                contentType="application/json",
            )
            return json.loads(r["body"].read())

        body = agent()
        stubber.assert_no_pending_responses()
        # Passthrough: the re-materialized body still carries the real Nova text.
        assert body["output"]["message"]["content"][0]["text"] == "pong"
        assert body["usage"]["totalTokens"] == 9

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "aws_bedrock.invoke_model"
        assert mi["payload"]["model"] == _MODEL_ID
        # Nova invoke_model bodies classify as the amazon family.
        assert mi["payload"]["family"] == "amazon"
        assert mi["payload"]["latency_ms"] > 0
        # stopReason from the parsed Converse-shaped body.
        assert mi["payload"]["stop_reason"] == "end_turn"
        # Input messages reconstructed from the request body (system + user).
        assert mi["payload"]["messages"] == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Reply with exactly: pong"},
        ]
        # Output text parsed from the recorded body's output.message.
        assert mi["payload"]["output_message"] == {"role": "assistant", "content": "pong"}
        # Usage parsed from the recorded Converse-shaped usage block (6/3/9).
        assert mi["payload"]["usage"] == {
            "prompt_tokens": 6,
            "completion_tokens": 3,
            "total_tokens": 9,
        }
        # The request-body must not leak into captured parameters.
        assert "body" not in mi["payload"]["parameters"]
        assert mi["payload"]["parameters"]["modelId"] == _MODEL_ID

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "aws_bedrock"
        assert cost["payload"]["model"] == _MODEL_ID
        assert cost["payload"]["total_tokens"] == 9
        # nova-micro: 6 * 0.000035/1k input + 3 * 0.00014/1k output.
        assert cost["payload"]["cost_usd"] == pytest.approx(6.3e-07)
        assert cost["span_id"] == mi["span_id"]

        provider.disconnect()

    def test_provenance_is_stamped(self):
        for scenario in ("converse", "invoke_model"):
            prov = load_recorded("bedrock", scenario)["provenance"]
            assert prov["provider"] == "aws_bedrock"
            assert prov["scenario"] == scenario
            assert prov["model"] == _MODEL_ID
            # captured_at makes staleness visible (snapshot, not freshness).
            assert prov["captured_at"] and prov["captured_at"] != "pending-creds"
