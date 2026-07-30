"""RED-first bugfix tests for the AWS Bedrock provider adapter (ADP W1).

These pin three real defects the deterministic doubles missed:

* **BUG-1** — streaming (``invoke_model_with_response_stream`` /
  ``converse_stream``) captured *nothing*: the adapter emitted a bare
  ``streaming=True`` ``model.invoke`` with no aggregated text, no usage, no
  cost, no TTFT. The fix wraps the returned ``EventStream`` in a re-iterable
  proxy that tees chunks to the caller while accumulating output/usage and, on
  completion, emits a full ``model.invoke`` + priced ``cost.record`` + TTFT.
* **BUG-2** — the non-Nova invoke_model parsers were wrong for four families:
  mistral (``outputs[0].text``), ai21 Jamba (``choices[0].message.content`` +
  ``usage.{prompt,completion}_tokens``), cohere Command-R (top-level ``text`` +
  ``meta.billed_units``) and amazon Titan completion tokens
  (``results[0].tokenCount``). Meta was already correct and stays untouched.
* **BUG-3** — Converse tool-use was dropped: ``toolUse`` never produced a
  ``tool.call`` event, ``toolResult`` content was discarded from the input
  messages, and a pure-tool assistant turn yielded ``output_message=None``.

Test mechanics mirror ``test_bedrock.py`` / ``test_bedrock_streaming.py``: a
real ``bedrock-runtime`` boto3 client (static fake creds) driven through
``botocore.stub.Stubber`` so the REQUEST passes real botocore validation with
no network, while the RESPONSE uses realistic per-family wire shapes.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.stub import ANY, Stubber  # noqa: E402
from botocore.response import StreamingBody  # noqa: E402

from layerlens.instrument import trace  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.providers.bedrock import BedrockProvider  # noqa: E402

from ...conftest import find_event  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> Any:
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        aws_session_token="testing",
    )


def _connect(client: Any, *, streaming: bool = False) -> tuple:
    provider = BedrockProvider()
    provider.connect(client)
    stubber = Stubber(client)
    if streaming:
        # Streaming ops declare their body/stream member as an `eventstream`;
        # the real runtime returns an iterable EventStream, which botocore's
        # response validator rejects. Disable RESPONSE validation for these
        # stubs only — request-param validation stays intact.
        stubber._validate_response = lambda output_shape, response: None  # noqa: SLF001
    stubber.activate()
    return provider, stubber


def _invoke_stub_response(body: bytes, *, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"RequestId": "11111111-2222-3333-4444-555555555555"}
    if headers is not None:
        metadata["HTTPHeaders"] = headers
    return {
        "body": StreamingBody(io.BytesIO(body), len(body)),
        "contentType": "application/json",
        "ResponseMetadata": metadata,
    }


# ===========================================================================
# BUG-1 — streaming aggregation
# ===========================================================================


def _anthropic_invoke_stream_body() -> Dict[str, Any]:
    events: List[dict] = [
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
        {
            "chunk": {
                "bytes": json.dumps(
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}}
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "from Bedrock!"},
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 8}}
                ).encode("utf-8")
            }
        },
    ]
    return {
        "body": iter(events),
        "contentType": "application/json",
        "ResponseMetadata": {"RequestId": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"},
    }


def _nova_invoke_stream_body() -> Dict[str, Any]:
    def _b(d: dict) -> dict:
        return {"chunk": {"bytes": json.dumps(d).encode("utf-8")}}

    events: List[dict] = [
        _b({"messageStart": {"role": "assistant"}}),
        _b({"contentBlockDelta": {"delta": {"text": "4"}, "contentBlockIndex": 0}}),
        _b({"contentBlockStop": {"contentBlockIndex": 0}}),
        _b({"messageStop": {"stopReason": "end_turn"}}),
        _b({"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}}),
    ]
    return {"body": iter(events), "contentType": "application/json", "ResponseMetadata": {"RequestId": "nova-stream"}}


def _converse_stream_body() -> Dict[str, Any]:
    events: List[dict] = [
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
    return {"stream": iter(events), "ResponseMetadata": {"RequestId": "conv-stream"}}


def _converse_stream_tooluse_body() -> Dict[str, Any]:
    """A ``converse_stream`` turn that calls a tool.

    Wire shape: a ``contentBlockStart`` with ``start.toolUse.{name,toolUseId}``,
    then ``contentBlockDelta`` events whose ``delta.toolUse.input`` fragments
    concatenate into the tool's input JSON, a ``contentBlockStop``, and a
    ``messageStop`` carrying ``stopReason='tool_use'``. The two input fragments
    join to ``{"city":"Paris","days":3}``.
    """
    events: List[dict] = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tu_stream_1", "name": "get_forecast"}},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"city":"Par'}}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": 'is","days":3}'}}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {"inputTokens": 20, "outputTokens": 15, "totalTokens": 35},
                "metrics": {"latencyMs": 120},
            }
        },
    ]
    return {"stream": iter(events), "ResponseMetadata": {"RequestId": "conv-stream-tool"}}


_ANTHROPIC_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"


def _anthropic_request_body() -> str:
    return json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello"}]}],
        }
    )


class TestStreamingAggregation:
    def test_invoke_model_stream_aggregates_anthropic(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client, streaming=True)
        stubber.add_response(
            "invoke_model_with_response_stream",
            _anthropic_invoke_stream_body(),
            {"modelId": _ANTHROPIC_MODEL, "body": ANY},
        )

        collected: List[dict] = []

        @trace(mock_client, capture_config=CaptureConfig.full())
        def run():
            r = client.invoke_model_with_response_stream(modelId=_ANTHROPIC_MODEL, body=_anthropic_request_body())
            for ev in r["body"]:
                collected.append(ev)
            return "done"

        assert run() == "done"
        stubber.assert_no_pending_responses()
        # Passthrough preserved: caller still drained every real chunk.
        assert len(collected) == 4

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "aws_bedrock.invoke_model_with_response_stream"
        assert mi["payload"]["streaming"] is True
        # Aggregated output text.
        assert mi["payload"]["output_message"] == {"role": "assistant", "content": "Hello from Bedrock!"}
        # Aggregated usage from the stream events.
        assert mi["payload"]["usage"]["prompt_tokens"] == 12
        assert mi["payload"]["usage"]["completion_tokens"] == 8
        assert mi["payload"]["prompt_tokens"] == 12
        assert mi["payload"]["completion_tokens"] == 8
        # TTFT surfaced.
        assert "ttft_ms" in mi["payload"] and mi["payload"]["ttft_ms"] >= 0

        cost = find_event(events, "cost.record")
        assert cost["payload"]["model"] == _ANTHROPIC_MODEL
        assert cost["payload"]["cost_usd"] is not None and cost["payload"]["cost_usd"] > 0
        assert cost["span_id"] == mi["span_id"]
        provider.disconnect()

    def test_invoke_model_stream_aggregates_nova(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client, streaming=True)
        model_id = "amazon.nova-micro-v1:0"
        stubber.add_response(
            "invoke_model_with_response_stream",
            _nova_invoke_stream_body(),
            {"modelId": model_id, "body": ANY},
        )

        @trace(mock_client, capture_config=CaptureConfig.full())
        def run():
            r = client.invoke_model_with_response_stream(modelId=model_id, body=_anthropic_request_body())
            return [ev for ev in r["body"]]

        assert len(run()) == 5
        stubber.assert_no_pending_responses()

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["output_message"] == {"role": "assistant", "content": "4"}
        assert mi["payload"]["usage"]["prompt_tokens"] == 10
        assert mi["payload"]["usage"]["completion_tokens"] == 5
        cost = find_event(events, "cost.record")
        assert cost["payload"]["cost_usd"] is not None and cost["payload"]["cost_usd"] > 0
        assert cost["span_id"] == mi["span_id"]
        provider.disconnect()

    def test_converse_stream_aggregates(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client, streaming=True)
        stubber.add_response(
            "converse_stream",
            _converse_stream_body(),
            {"modelId": _ANTHROPIC_MODEL, "messages": ANY},
        )

        @trace(mock_client, capture_config=CaptureConfig.full())
        def run():
            r = client.converse_stream(
                modelId=_ANTHROPIC_MODEL,
                messages=[{"role": "user", "content": [{"text": "What is the capital of France?"}]}],
            )
            return [ev for ev in r["stream"]]

        assert len(run()) == 6
        stubber.assert_no_pending_responses()

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "aws_bedrock.converse_stream"
        assert mi["payload"]["streaming"] is True
        assert mi["payload"]["output_message"] == {"role": "assistant", "content": "Paris is the capital of France."}
        assert mi["payload"]["usage"]["prompt_tokens"] == 9
        assert mi["payload"]["usage"]["completion_tokens"] == 7
        assert mi["payload"]["usage"]["total_tokens"] == 16
        assert "ttft_ms" in mi["payload"] and mi["payload"]["ttft_ms"] >= 0

        cost = find_event(events, "cost.record")
        assert cost["payload"]["cost_usd"] is not None and cost["payload"]["cost_usd"] > 0
        assert cost["span_id"] == mi["span_id"]
        provider.disconnect()


# ===========================================================================
# BUG-2 — non-Nova invoke_model parsers
# ===========================================================================


def _mistral_body() -> bytes:
    return json.dumps({"outputs": [{"text": "The answer is 4.", "stop_reason": "stop"}]}).encode("utf-8")


def _ai21_body() -> bytes:
    return json.dumps(
        {
            "id": "cmpl-jamba-1",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "The answer is 4."}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 6, "total_tokens": 17},
        }
    ).encode("utf-8")


def _cohere_r_body() -> bytes:
    return json.dumps(
        {
            "response_id": "cohere-r-1",
            "text": "The answer is 4.",
            "generation_id": "gen-1",
            "finish_reason": "COMPLETE",
            "meta": {"billed_units": {"input_tokens": 13, "output_tokens": 7}},
        }
    ).encode("utf-8")


def _titan_body() -> bytes:
    return json.dumps(
        {
            "inputTextTokenCount": 14,
            "results": [{"tokenCount": 9, "outputText": "The answer is 4.", "completionReason": "FINISH"}],
        }
    ).encode("utf-8")


class TestNonNovaParsers:
    @pytest.mark.parametrize(
        "model_id,body,headers,exp_prompt,exp_completion",
        [
            # mistral carries NO token counts in the body — Bedrock returns them
            # in the response headers, which the adapter must read.
            (
                "mistral.mistral-7b-instruct-v0:2",
                _mistral_body(),
                {"x-amzn-bedrock-input-token-count": "10", "x-amzn-bedrock-output-token-count": "5"},
                10,
                5,
            ),
            ("ai21.jamba-1-5-mini-v1:0", _ai21_body(), None, 11, 6),
            ("cohere.command-r-v1:0", _cohere_r_body(), None, 13, 7),
        ],
    )
    def test_invoke_family_parsed_priced(
        self, mock_client, capture_trace, model_id, body, headers, exp_prompt, exp_completion
    ):
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_response(
            "invoke_model",
            _invoke_stub_response(body, headers=headers),
            {"modelId": model_id, "body": ANY},
        )

        @trace(mock_client, capture_config=CaptureConfig.full())
        def run():
            r = client.invoke_model(modelId=model_id, body=json.dumps({"prompt": "2+2?"}))
            return json.loads(r["body"].read())

        run()
        stubber.assert_no_pending_responses()

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        # Output text must be captured (was blank for mistral/ai21/cohere-R).
        assert mi["payload"]["output_message"] is not None
        assert "The answer is 4." in mi["payload"]["output_message"]["content"]
        # Usage must be non-zero on BOTH sides.
        assert mi["payload"]["usage"]["prompt_tokens"] == exp_prompt
        assert mi["payload"]["usage"]["completion_tokens"] == exp_completion

        cost = find_event(events, "cost.record")
        assert cost["payload"]["model"] == model_id
        # These three families have BEDROCK_PRICING rows, so cost must be priced.
        assert cost["payload"]["cost_usd"] is not None and cost["payload"]["cost_usd"] > 0
        assert cost["span_id"] == mi["span_id"]
        provider.disconnect()

    def test_titan_completion_tokens_from_results(self, mock_client, capture_trace):
        # amazon-titan output already parsed; the bug was completion tokens read
        # from a non-existent top-level `tokenCount` (→ 0) instead of
        # `results[0].tokenCount`. Titan has no BEDROCK_PRICING row yet, so
        # cost_usd stays None (a genuinely-unpriced model) — the token counts,
        # not the dollar figure, are what this pins.
        client = _make_client()
        provider, stubber = _connect(client)
        model_id = "amazon.titan-text-express-v1"
        stubber.add_response(
            "invoke_model",
            _invoke_stub_response(_titan_body()),
            {"modelId": model_id, "body": ANY},
        )

        @trace(mock_client, capture_config=CaptureConfig.full())
        def run():
            r = client.invoke_model(modelId=model_id, body=json.dumps({"inputText": "2+2?"}))
            return json.loads(r["body"].read())

        run()
        stubber.assert_no_pending_responses()

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["output_message"]["content"] == "The answer is 4."
        assert mi["payload"]["usage"]["prompt_tokens"] == 14
        # BUG-2: was 0 (read top-level tokenCount); must be results[0].tokenCount.
        assert mi["payload"]["usage"]["completion_tokens"] == 9
        assert mi["payload"]["completion_tokens"] == 9

        cost = find_event(events, "cost.record")
        assert cost["payload"]["total_tokens"] == 23
        assert cost["payload"]["completion_tokens"] == 9
        provider.disconnect()


# ===========================================================================
# BUG-3 — Converse tool-use
# ===========================================================================


class TestConverseToolUse:
    def test_tool_use_emits_tool_call_and_captures_result(self, mock_client, capture_trace):
        client = _make_client()
        provider, stubber = _connect(client)
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tu_2",
                                "name": "get_forecast",
                                "input": {"city": "Paris", "days": 3},
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 30, "outputTokens": 12, "totalTokens": 42},
            "metrics": {"latencyMs": 100},
            "ResponseMetadata": {"RequestId": "toolturn"},
        }
        stubber.add_response("converse", response, {"modelId": _ANTHROPIC_MODEL, "messages": ANY})

        request_messages = [
            {"role": "user", "content": [{"text": "What's the weather in Paris?"}]},
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}}],
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "tu_1", "content": [{"text": "22C and sunny"}], "status": "success"}}
                ],
            },
        ]

        @trace(mock_client, capture_config=CaptureConfig.full())
        def run():
            return client.converse(modelId=_ANTHROPIC_MODEL, messages=request_messages)

        run()
        stubber.assert_no_pending_responses()

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        # A pure-tool assistant turn must NOT collapse to output_message=None.
        assert mi["payload"]["output_message"] is not None

        # A tool.call event carrying the model's requested tool name + input.
        tc = find_event(events, "tool.call")
        assert tc["payload"]["tool_name"] == "get_forecast"
        assert tc["payload"]["arguments"] == {"city": "Paris", "days": 3}

        # The toolResult content from the input must be captured, not dropped.
        rendered = json.dumps(mi["payload"]["messages"])
        assert "22C and sunny" in rendered
        # And the prior toolUse in the input is captured too.
        assert "get_weather" in rendered

        cost = find_event(events, "cost.record")
        assert cost["payload"]["cost_usd"] is not None and cost["payload"]["cost_usd"] > 0
        provider.disconnect()

    def test_converse_stream_tool_use_emits_tool_call(self, mock_client, capture_trace):
        # BUG-3 (streaming path): the same "Converse tool-use dropped" defect
        # was still live on converse_stream — a toolUse block streamed as
        # contentBlockStart + delta.toolUse.input fragments never produced a
        # tool.call event and collapsed a pure-tool turn to output_message=None.
        client = _make_client()
        provider, stubber = _connect(client, streaming=True)
        stubber.add_response(
            "converse_stream",
            _converse_stream_tooluse_body(),
            {"modelId": _ANTHROPIC_MODEL, "messages": ANY},
        )

        @trace(mock_client, capture_config=CaptureConfig.full())
        def run():
            r = client.converse_stream(
                modelId=_ANTHROPIC_MODEL,
                messages=[{"role": "user", "content": [{"text": "What's the weather in Paris?"}]}],
            )
            return [ev for ev in r["stream"]]

        assert len(run()) == 7
        stubber.assert_no_pending_responses()

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "aws_bedrock.converse_stream"
        assert mi["payload"]["streaming"] is True
        # A pure-tool streaming turn must NOT collapse to empty output.
        assert mi["payload"]["output_message"] is not None
        assert mi["payload"]["output_message"]["content"]

        # A tool.call event carrying the streamed tool name + concatenated-input
        # arguments (parsed from the JSON fragments) + tool_use_id.
        tc = find_event(events, "tool.call")
        assert tc["payload"]["tool_name"] == "get_forecast"
        assert tc["payload"]["arguments"] == {"city": "Paris", "days": 3}
        assert tc["payload"]["tool_use_id"] == "tu_stream_1"
        # Parented under the streaming model.invoke span (parity with sync).
        assert tc["parent_span_id"] == mi["span_id"]

        cost = find_event(events, "cost.record")
        assert cost["payload"]["cost_usd"] is not None and cost["payload"]["cost_usd"] > 0
        assert cost["span_id"] == mi["span_id"]
        provider.disconnect()
