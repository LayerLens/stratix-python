"""Deterministic doubles for the AWS Bedrock provider adapter (LAY-3582 / T8).

Bedrock is credential-gated (no AWS account / model access), so these tests
stand in for live verification. A real ``bedrock-runtime`` boto3 client
(region us-east-1, static fake credentials) is driven through
``botocore.stub.Stubber`` so requests pass real botocore parameter validation
and serialization with no network, while responses use realistic shapes:

- ``invoke_model``: the body for ``anthropic.*`` models on Bedrock is an
  Anthropic Messages API response — built here from the real ``anthropic``
  SDK types (``Message``/``TextBlock``/``Usage`` incl. the ``usage`` block)
  and serialized to JSON inside a ``botocore.response.StreamingBody``.
- ``converse``: the Converse API response with ``output.message``,
  ``usage`` (inputTokens/outputTokens/totalTokens), ``stopReason``, and
  ``metrics`` — all required members of the service model.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.stub import ANY, Stubber  # noqa: E402
from anthropic.types import Usage, Message, TextBlock  # noqa: E402
from botocore.response import StreamingBody  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402

from layerlens.instrument import trace  # noqa: E402
from layerlens.instrument.adapters.providers.bedrock import (  # noqa: E402
    BedrockProvider,
    instrument_bedrock,
    uninstrument_bedrock,
)

from ...conftest import find_event, find_events  # noqa: E402

_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> Any:
    """Real bedrock-runtime client with static fake credentials (offline-safe)."""
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        aws_session_token="testing",
    )


def _anthropic_body_bytes(
    text: str = "Hello from Bedrock!",
    input_tokens: int = 12,
    output_tokens: int = 8,
    stop_reason: str = "end_turn",
) -> bytes:
    """Anthropic-on-Bedrock invoke_model body, built from real anthropic SDK types."""
    message = Message(
        id="msg_bdrk_01A2B3C4D5E6F7G8H9J0K1L2",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet-20241022",
        content=[TextBlock(type="text", text=text)],
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )
    return message.model_dump_json().encode("utf-8")


def _invoke_model_stub_response(body: bytes) -> Dict[str, Any]:
    return {
        "body": StreamingBody(io.BytesIO(body), len(body)),
        "contentType": "application/json",
        "ResponseMetadata": {"RequestId": "11111111-2222-3333-4444-555555555555"},
    }


def _converse_stub_response() -> Dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Paris is the capital of France."}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 9, "outputTokens": 7, "totalTokens": 16},
        "metrics": {"latencyMs": 142},
        "ResponseMetadata": {"RequestId": "66666666-7777-8888-9999-000000000000"},
    }


def _anthropic_request_body() -> str:
    return json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "system": "You are terse.",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello"}]}],
        }
    )


def _connect(client: Any) -> tuple:
    provider = BedrockProvider()
    provider.connect(client)
    stubber = Stubber(client)
    stubber.activate()
    return provider, stubber


# ---------------------------------------------------------------------------
# invoke_model
# ---------------------------------------------------------------------------


class TestInvokeModel:
    def test_model_invoke_and_cost_record(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_response(
            "invoke_model",
            _invoke_model_stub_response(_anthropic_body_bytes()),
            {
                "modelId": _MODEL_ID,
                "body": ANY,
                "accept": "application/json",
                "contentType": "application/json",
            },
        )

        @trace(mock_client)
        def my_agent():
            r = client.invoke_model(
                modelId=_MODEL_ID,
                body=_anthropic_request_body(),
                accept="application/json",
                contentType="application/json",
            )
            # The adapter re-materializes the single-read StreamingBody, so the
            # caller must still be able to read the full body (passthrough).
            return json.loads(r["body"].read())

        body = my_agent()
        stubber.assert_no_pending_responses()

        # Passthrough: caller sees the original anthropic-on-bedrock body.
        assert body["content"][0]["text"] == "Hello from Bedrock!"
        assert body["usage"]["input_tokens"] == 12
        assert body["usage"]["output_tokens"] == 8

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "aws_bedrock.invoke_model"
        assert mi["payload"]["model"] == _MODEL_ID
        assert mi["payload"]["family"] == "anthropic"
        assert mi["payload"]["latency_ms"] > 0
        assert mi["payload"]["stop_reason"] == "end_turn"
        assert mi["payload"]["response_id"] == "11111111-2222-3333-4444-555555555555"
        # Input extraction: system prompt + user message (content-block list flattened).
        assert mi["payload"]["messages"] == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Say hello"},
        ]
        assert mi["payload"]["output_message"] == {"role": "assistant", "content": "Hello from Bedrock!"}
        # Usage extracted from the anthropic body's usage block.
        assert mi["payload"]["usage"]["prompt_tokens"] == 12
        assert mi["payload"]["usage"]["completion_tokens"] == 8
        assert mi["payload"]["usage"]["total_tokens"] == 20
        # Captured params are the request-shape allowlist.
        assert mi["payload"]["parameters"]["modelId"] == _MODEL_ID
        assert mi["payload"]["parameters"]["accept"] == "application/json"
        assert "body" not in mi["payload"]["parameters"]
        assert "otel_gen_ai" in mi["payload"]

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "aws_bedrock"
        assert cost["payload"]["model"] == _MODEL_ID
        assert cost["payload"]["total_tokens"] == 20
        # BEDROCK_PRICING: 12 * 0.003/1k input + 8 * 0.015/1k output.
        assert cost["payload"]["cost_usd"] == pytest.approx(0.000156)
        # cost.record shares the model.invoke span.
        assert cost["span_id"] == mi["span_id"]

        provider.disconnect()

    def test_error_emits_agent_error(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_client_error(
            "invoke_model",
            service_error_code="ValidationException",
            service_message="The provided model identifier is invalid.",
            http_status_code=400,
        )

        @trace(mock_client)
        def my_agent():
            try:
                client.invoke_model(modelId="anthropic.not-a-model", body=_anthropic_request_body())
            except ClientError:
                pass
            return "recovered"

        my_agent()
        events = capture_trace["events"]
        error = find_event(events, "agent.error")
        assert error["payload"]["name"] == "aws_bedrock.invoke_model"
        assert "model identifier is invalid" in error["payload"]["error"]
        assert "latency_ms" in error["payload"]
        # No model.invoke / cost.record for the failed call.
        assert not find_events(events, "model.invoke")
        assert not find_events(events, "cost.record")

        provider.disconnect()


# ---------------------------------------------------------------------------
# converse
# ---------------------------------------------------------------------------


class TestConverse:
    def test_model_invoke_and_cost_record(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_response(
            "converse",
            _converse_stub_response(),
            {
                "modelId": _MODEL_ID,
                "messages": ANY,
                "inferenceConfig": {"maxTokens": 100, "temperature": 0.2},
            },
        )

        @trace(mock_client)
        def my_agent():
            r = client.converse(
                modelId=_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": "What is the capital of France?"}]}],
                inferenceConfig={"maxTokens": 100, "temperature": 0.2},
            )
            return r["output"]["message"]["content"][0]["text"]

        result = my_agent()
        stubber.assert_no_pending_responses()
        assert result == "Paris is the capital of France."

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "aws_bedrock.converse"
        assert mi["payload"]["model"] == _MODEL_ID
        assert mi["payload"]["latency_ms"] > 0
        assert mi["payload"]["stop_reason"] == "end_turn"
        assert mi["payload"]["response_id"] == "66666666-7777-8888-9999-000000000000"
        assert mi["payload"]["messages"] == [{"role": "user", "content": "What is the capital of France?"}]
        assert mi["payload"]["output_message"] == {
            "role": "assistant",
            "content": "Paris is the capital of France.",
        }
        assert mi["payload"]["usage"] == {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16}
        assert mi["payload"]["parameters"]["inferenceConfig"] == {"maxTokens": 100, "temperature": 0.2}

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "aws_bedrock"
        assert cost["payload"]["total_tokens"] == 16
        # 9 * 0.003/1k + 7 * 0.015/1k from BEDROCK_PRICING.
        assert cost["payload"]["cost_usd"] == pytest.approx(0.000132)

        provider.disconnect()

    def test_error_emits_agent_error(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_client_error(
            "converse",
            service_error_code="ThrottlingException",
            service_message="Too many requests, please wait before trying again.",
            http_status_code=429,
        )

        @trace(mock_client)
        def my_agent():
            try:
                client.converse(
                    modelId=_MODEL_ID,
                    messages=[{"role": "user", "content": [{"text": "Hi"}]}],
                )
            except ClientError:
                pass
            return "recovered"

        my_agent()
        error = find_event(capture_trace["events"], "agent.error")
        assert error["payload"]["name"] == "aws_bedrock.converse"
        assert "Too many requests" in error["payload"]["error"]

        provider.disconnect()


# ---------------------------------------------------------------------------
# Passthrough / no-op outside trace
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_no_events_outside_trace(self, mock_client):
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_response("invoke_model", _invoke_model_stub_response(_anthropic_body_bytes()))

        response = client.invoke_model(modelId=_MODEL_ID, body=_anthropic_request_body())
        body = json.loads(response["body"].read())
        assert body["content"][0]["text"] == "Hello from Bedrock!"
        assert not mock_client.traces.upload.called

        provider.disconnect()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_disconnect_stops_emission(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client)
        provider.disconnect()
        stubber.add_response("invoke_model", _invoke_model_stub_response(_anthropic_body_bytes()))

        @trace(mock_client)
        def my_agent():
            r = client.invoke_model(modelId=_MODEL_ID, body=_anthropic_request_body())
            return json.loads(r["body"].read())["content"][0]["text"]

        assert my_agent() == "Hello from Bedrock!"
        assert not find_events(capture_trace["events"], "model.invoke")

    def test_adapter_info(self):
        provider = BedrockProvider()
        info = provider.adapter_info()
        assert info.name == "aws_bedrock"
        assert info.adapter_type == "provider"
        assert info.connected is False

        provider.connect(_make_client())
        assert provider.adapter_info().connected is True
        provider.disconnect()
        assert provider.adapter_info().connected is False

    def test_instrument_and_uninstrument(self):
        client = _make_client()
        provider = instrument_bedrock(client)
        assert isinstance(provider, BedrockProvider)
        uninstrument_bedrock()  # must not raise
