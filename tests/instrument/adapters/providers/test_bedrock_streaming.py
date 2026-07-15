"""Streaming-path tests for the AWS Bedrock provider adapter (G8 / W4).

Bedrock streams in production via ``invoke_model_with_response_stream`` and
``converse_stream``. These pin the adapter's streaming contract.

Contract (BUG-1 fixed — streaming now aggregates)
-------------------------------------------------
``botocore`` returns a single-read ``botocore.eventstream.EventStream`` for
these operations. ``_wrap_stream`` wraps that stream in a re-iterable proxy
(:class:`~layerlens.instrument.adapters.providers.bedrock._StreamTee`) that
tees every chunk back to the caller (single-read semantics preserved) while
accumulating them. On stream exhaustion it emits a FULL ``model.invoke``:

* aggregated ``output_message`` + input ``messages``;
* aggregated ``usage`` (+ flat token fields) and a priced ``cost.record`` on
  the same span;
* ``ttft_ms`` / ``streaming_duration_ms`` timing.

Per-family accumulators cover Anthropic Messages SSE bytes and Nova/Converse
``contentBlockDelta``/``metadata`` events for ``invoke_model_with_response_stream``
and the Converse event stream for ``converse_stream``. These tests assert that
aggregated contract AND that iterator passthrough is preserved (the caller
still receives every real-shaped chunk). Deeper per-family stream coverage
lives in ``test_bedrock_bugfix.py``.

Test mechanics mirror ``test_bedrock.py``: a real ``bedrock-runtime`` boto3
client (region us-east-1, static fake creds) is driven through
``botocore.stub.Stubber`` so the REQUEST passes real botocore parameter
validation + serialization with no network. The streaming RESPONSE member is an
``eventstream`` shape; the Stubber's response validator insists that member be a
``dict``, but the real runtime returns an iterable ``EventStream``. We therefore
disable response validation for the streaming stub only (``_validate_response``)
and return a real-shaped iterator of chunk events — exactly what the adapter and
the caller see at runtime. Request-param validation (the load-bearing part) is
left intact.
"""

from __future__ import annotations

import json
from typing import Any, List

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.stub import ANY, Stubber  # noqa: E402

from layerlens.instrument import trace  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.providers.bedrock import BedrockProvider  # noqa: E402

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


def _connect(client: Any) -> tuple:
    provider = BedrockProvider()
    provider.connect(client)
    stubber = Stubber(client)
    # The Bedrock streaming operations declare their `body` / `stream` member as
    # an `eventstream` structure. The real runtime returns an iterable
    # `EventStream`; botocore's response validator (correctly, for non-streaming
    # ops) insists the member be a `dict`, so it rejects the iterator we need to
    # hand the adapter. Disable *response* validation for these stubs only —
    # request-param validation/serialization (the load-bearing realism) stays on.
    stubber._validate_response = lambda output_shape, response: None  # noqa: SLF001
    stubber.activate()
    return provider, stubber


def _anthropic_stream_chunk_bytes(*, text: str) -> bytes:
    """One Anthropic-on-Bedrock SSE event, JSON-encoded as the runtime delivers
    inside a ``chunk.bytes`` blob for ``invoke_model_with_response_stream``."""
    return json.dumps(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }
    ).encode("utf-8")


def _invoke_model_stream_response() -> dict:
    """``invoke_model_with_response_stream`` output: ``body`` is an EventStream of
    ``{'chunk': {'bytes': <sse-json-bytes>}}`` events (real wire shape)."""
    body_events: List[dict] = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_bdrk_stream_01",
                            "role": "assistant",
                            "model": "claude-3-5-sonnet-20241022",
                            "usage": {"input_tokens": 12, "output_tokens": 0},
                        },
                    }
                ).encode("utf-8")
            }
        },
        {"chunk": {"bytes": _anthropic_stream_chunk_bytes(text="Hello ")}},
        {"chunk": {"bytes": _anthropic_stream_chunk_bytes(text="from Bedrock!")}},
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 8},
                    }
                ).encode("utf-8")
            }
        },
    ]
    return {
        "body": iter(body_events),
        "contentType": "application/json",
        "ResponseMetadata": {"RequestId": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"},
    }


def _converse_stream_response() -> dict:
    """``converse_stream`` output: ``stream`` is an EventStream of Converse events
    (messageStart / contentBlockDelta / contentBlockStop / messageStop / metadata)."""
    stream_events: List[dict] = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Paris is "}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "the capital of France."}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 9, "outputTokens": 7, "totalTokens": 16},
                "metrics": {"latencyMs": 142},
            }
        },
    ]
    return {
        "stream": iter(stream_events),
        "ResponseMetadata": {"RequestId": "11112222-3333-4444-5555-666677778888"},
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


# ---------------------------------------------------------------------------
# invoke_model_with_response_stream
# ---------------------------------------------------------------------------


class TestInvokeModelWithResponseStream:
    def test_emits_bare_streaming_model_invoke(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_response(
            "invoke_model_with_response_stream",
            _invoke_model_stream_response(),
            {"modelId": _MODEL_ID, "body": ANY},
        )

        collected: List[dict] = []

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            r = client.invoke_model_with_response_stream(modelId=_MODEL_ID, body=_anthropic_request_body())
            # Iterator passthrough: the adapter must NOT consume the stream; the
            # caller drains the real-shaped chunk events itself.
            for event in r["body"]:
                collected.append(event)
            return "done"

        assert my_agent() == "done"
        stubber.assert_no_pending_responses()

        # Passthrough: caller saw every real-shaped chunk event, in order.
        assert len(collected) == 4
        first_delta = json.loads(collected[1]["chunk"]["bytes"])
        assert first_delta["delta"]["text"] == "Hello "

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "aws_bedrock.invoke_model_with_response_stream"
        assert mi["payload"]["model"] == _MODEL_ID
        assert mi["payload"]["streaming"] is True
        assert mi["payload"]["method"] == "invoke_model_with_response_stream"
        assert mi["payload"]["latency_ms"] > 0

        # BUG-1 fixed: the stream is aggregated into a full model.invoke.
        assert mi["payload"]["output_message"] == {"role": "assistant", "content": "Hello from Bedrock!"}
        assert mi["payload"]["usage"]["prompt_tokens"] == 12
        assert mi["payload"]["usage"]["completion_tokens"] == 8
        assert mi["payload"]["prompt_tokens"] == 12
        assert mi["payload"]["completion_tokens"] == 8
        assert mi["payload"]["stop_reason"] == "end_turn"
        assert mi["payload"]["ttft_ms"] >= 0
        assert mi["payload"]["streaming_duration_ms"] >= 0
        # Input messages parsed from the request body too.
        assert mi["payload"]["messages"] == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Say hello"},
        ]

        cost = find_event(events, "cost.record")
        assert cost["payload"]["model"] == _MODEL_ID
        assert cost["payload"]["cost_usd"] is not None and cost["payload"]["cost_usd"] > 0
        assert cost["span_id"] == mi["span_id"]

        provider.disconnect()

    def test_error_emits_agent_error(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_client_error(
            "invoke_model_with_response_stream",
            service_error_code="ValidationException",
            service_message="The provided model identifier is invalid.",
            http_status_code=400,
        )

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            from botocore.exceptions import ClientError

            try:
                client.invoke_model_with_response_stream(
                    modelId="anthropic.not-a-model", body=_anthropic_request_body()
                )
            except ClientError:
                pass
            return "recovered"

        assert my_agent() == "recovered"
        events = capture_trace["events"]
        error = find_event(events, "agent.error")
        assert error["payload"]["name"] == "aws_bedrock.invoke_model_with_response_stream"
        assert "model identifier is invalid" in error["payload"]["error"]
        assert "latency_ms" in error["payload"]
        # No model.invoke for the failed streaming call.
        assert not find_events(events, "model.invoke")

        provider.disconnect()


# ---------------------------------------------------------------------------
# converse_stream
# ---------------------------------------------------------------------------


class TestConverseStream:
    def test_emits_bare_streaming_model_invoke(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_response(
            "converse_stream",
            _converse_stream_response(),
            {
                "modelId": _MODEL_ID,
                "messages": ANY,
                "inferenceConfig": {"maxTokens": 100, "temperature": 0.2},
            },
        )

        collected: List[dict] = []

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            r = client.converse_stream(
                modelId=_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": "What is the capital of France?"}]}],
                inferenceConfig={"maxTokens": 100, "temperature": 0.2},
            )
            for event in r["stream"]:
                collected.append(event)
            return "done"

        assert my_agent() == "done"
        stubber.assert_no_pending_responses()

        # Passthrough: caller drained the real Converse event sequence itself.
        assert collected[0] == {"messageStart": {"role": "assistant"}}
        assert collected[1]["contentBlockDelta"]["delta"]["text"] == "Paris is "
        assert collected[-1]["metadata"]["usage"]["totalTokens"] == 16

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "aws_bedrock.converse_stream"
        assert mi["payload"]["model"] == _MODEL_ID
        assert mi["payload"]["streaming"] is True
        assert mi["payload"]["method"] == "converse_stream"
        assert mi["payload"]["latency_ms"] > 0

        # BUG-1 fixed: the Converse event stream is aggregated.
        assert mi["payload"]["output_message"] == {
            "role": "assistant",
            "content": "Paris is the capital of France.",
        }
        assert mi["payload"]["usage"] == {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16}
        assert mi["payload"]["stop_reason"] == "end_turn"
        assert mi["payload"]["ttft_ms"] >= 0
        assert mi["payload"]["messages"] == [{"role": "user", "content": "What is the capital of France?"}]

        cost = find_event(events, "cost.record")
        assert cost["payload"]["cost_usd"] is not None and cost["payload"]["cost_usd"] > 0
        assert cost["span_id"] == mi["span_id"]

        provider.disconnect()

    def test_error_emits_agent_error(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_client_error(
            "converse_stream",
            service_error_code="ThrottlingException",
            service_message="Too many requests, please wait before trying again.",
            http_status_code=429,
        )

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            from botocore.exceptions import ClientError

            try:
                client.converse_stream(
                    modelId=_MODEL_ID,
                    messages=[{"role": "user", "content": [{"text": "Hi"}]}],
                )
            except ClientError:
                pass
            return "recovered"

        assert my_agent() == "recovered"
        error = find_event(capture_trace["events"], "agent.error")
        assert error["payload"]["name"] == "aws_bedrock.converse_stream"
        assert "Too many requests" in error["payload"]["error"]

        provider.disconnect()


# ---------------------------------------------------------------------------
# Passthrough outside a trace
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_no_events_outside_trace(self, mock_client):
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_response(
            "converse_stream",
            _converse_stream_response(),
            {"modelId": _MODEL_ID, "messages": ANY},
        )

        r = client.converse_stream(
            modelId=_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        )
        chunks = list(r["stream"])
        assert chunks[0] == {"messageStart": {"role": "assistant"}}
        assert not mock_client.traces.upload.called

        provider.disconnect()
