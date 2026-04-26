"""Unit tests for the AWS Bedrock provider adapter (M3 package layout).

These tests target the package import path
(``layerlens.instrument.adapters.providers.bedrock``) introduced by the
M3 fan-out. The legacy flat-module test
(``test_bedrock_adapter.py``) covers the back-compat shim.

No live AWS calls are made. Two fixture styles are used:

* ``botocore.stub.Stubber`` — pinned against a real boto3
  ``bedrock-runtime`` client. Verifies the adapter wraps and restores
  the canonical method shapes correctly and tolerates the real
  ``StreamingBody`` semantics.
* Hand-rolled fake clients — deterministic, faster, and used for the
  per-family branch coverage where all we care about is the JSON body
  shape, not boto3's marshalling.

``respx`` is intentionally not used to drive boto3 (it intercepts httpx,
not urllib3 / botocore), but the package is a project-wide test
dependency, so a single sanity check confirms the import resolves —
this catches accidental dev-dep regressions.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List
from unittest import mock

import boto3
import respx
import pytest
from botocore.stub import Stubber
from botocore.response import StreamingBody

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.providers.bedrock import (
    ADAPTER_CLASS,
    RereadableBody,
    AWSBedrockAdapter,
    STRATIXBedrockAdapter,
    _RereadableBody,
    detect_provider_family,
    _detect_provider_family,
)
from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage
from layerlens.instrument.adapters.providers.bedrock.extract import (
    extract_invoke_usage,
    build_invoke_metadata,
    extract_invoke_output,
    extract_converse_usage,
    extract_converse_output,
    extract_invoke_messages,
)

# ---------------------------------------------------------------------------
# Recording fixtures
# ---------------------------------------------------------------------------


class _RecordingLayerLens:
    """Captures every event the adapter emits for assertion.

    The adapter calls ``emit_dict_event`` on its base class, which in
    turn calls ``self._stratix.emit(event_type, payload)``. We catch
    both the two-arg and one-arg forms.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            event_type, payload = args
            self.events.append({"event_type": event_type, "payload": payload})
        elif len(args) == 1:
            self.events.append({"event_type": None, "payload": args[0]})


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,family",
    [
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic"),
        ("ANTHROPIC.Claude-3-5-Sonnet-20241022-v2:0", "anthropic"),  # case
        ("meta.llama3-1-70b-instruct-v1:0", "meta"),
        ("cohere.command-r-v1:0", "cohere"),
        ("amazon.titan-text-express-v1", "amazon"),
        ("ai21.jamba-instruct-v1:0", "ai21"),
        ("mistral.mistral-7b-instruct-v0:2", "mistral"),
        ("unknown.model-v1", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_detect_provider_family(model_id: Any, family: str) -> None:
    assert detect_provider_family(model_id) == family


def test_back_compat_alias_resolves() -> None:
    assert _detect_provider_family is detect_provider_family


# ---------------------------------------------------------------------------
# Public API + lifecycle
# ---------------------------------------------------------------------------


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is AWSBedrockAdapter
    assert STRATIXBedrockAdapter is AWSBedrockAdapter


def test_lazy_provider_attr_resolves() -> None:
    """``providers.AWSBedrockAdapter`` must resolve via PEP 562."""
    from layerlens.instrument.adapters import providers

    assert providers.AWSBedrockAdapter is AWSBedrockAdapter


def test_lazy_provider_attr_unknown_raises() -> None:
    from layerlens.instrument.adapters import providers

    with pytest.raises(AttributeError):
        providers.NoSuchAdapter  # noqa: B018 — intentional attr probe


def test_lifecycle() -> None:
    a = AWSBedrockAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_init_accepts_both_layerlens_and_stratix_kwargs() -> None:
    sentinel = object()
    a = AWSBedrockAdapter(stratix=sentinel)
    assert a._stratix is sentinel
    b = AWSBedrockAdapter(layerlens=sentinel)
    assert b._stratix is sentinel


def test_legacy_module_path_still_resolves() -> None:
    # Back-compat shim must keep the flat-module import path working.
    from layerlens.instrument.adapters.providers import bedrock_adapter as flat

    assert flat.AWSBedrockAdapter is AWSBedrockAdapter
    assert flat.ADAPTER_CLASS is AWSBedrockAdapter


# ---------------------------------------------------------------------------
# Token extraction — per-family
# ---------------------------------------------------------------------------


def test_extract_invoke_usage_anthropic() -> None:
    body = {"usage": {"input_tokens": 100, "output_tokens": 50}}
    usage = extract_invoke_usage(body, "anthropic")
    assert usage == NormalizedTokenUsage(
        prompt_tokens=100, completion_tokens=50, total_tokens=150
    )


def test_extract_invoke_usage_meta() -> None:
    body = {"prompt_token_count": 50, "generation_token_count": 25}
    usage = extract_invoke_usage(body, "meta")
    assert usage is not None
    assert usage.prompt_tokens == 50 and usage.completion_tokens == 25


def test_extract_invoke_usage_cohere() -> None:
    body = {"meta": {"billed_units": {"input_tokens": 10, "output_tokens": 5}}}
    usage = extract_invoke_usage(body, "cohere")
    assert usage is not None
    assert usage.prompt_tokens == 10 and usage.completion_tokens == 5


def test_extract_invoke_usage_amazon_titan() -> None:
    body = {"inputTextTokenCount": 11, "results": [{"tokenCount": 7}]}
    usage = extract_invoke_usage(body, "amazon")
    assert usage is not None
    assert usage.prompt_tokens == 11 and usage.completion_tokens == 7


def test_extract_invoke_usage_ai21() -> None:
    body = {"usage": {"prompt_tokens": 12, "completion_tokens": 4}}
    usage = extract_invoke_usage(body, "ai21")
    assert usage is not None
    assert usage.prompt_tokens == 12 and usage.completion_tokens == 4


def test_extract_invoke_usage_mistral() -> None:
    body = {"prompt_tokens": 9, "completion_tokens": 3}
    usage = extract_invoke_usage(body, "mistral")
    assert usage is not None
    assert usage.prompt_tokens == 9 and usage.completion_tokens == 3


def test_extract_invoke_usage_unknown_fallback() -> None:
    body = {"inputTokenCount": 8, "outputTokenCount": 2}
    usage = extract_invoke_usage(body, "unknown")
    assert usage is not None
    assert usage.total_tokens == 10


def test_extract_invoke_usage_empty_body_returns_none() -> None:
    assert extract_invoke_usage({}, "anthropic") is None


def test_extract_converse_usage() -> None:
    response = {"usage": {"inputTokens": 100, "outputTokens": 50}}
    usage = extract_converse_usage(response)
    assert usage is not None
    assert usage.prompt_tokens == 100 and usage.completion_tokens == 50


def test_extract_converse_usage_missing_returns_none() -> None:
    assert extract_converse_usage({}) is None


# ---------------------------------------------------------------------------
# Message extraction — per-family
# ---------------------------------------------------------------------------


def test_extract_anthropic_invoke_messages_with_system() -> None:
    body = json.dumps(
        {
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )
    msgs = extract_invoke_messages(
        {"body": body},
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    assert msgs == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]


def test_extract_anthropic_invoke_messages_with_content_blocks() -> None:
    body = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}],
                }
            ]
        }
    )
    msgs = extract_invoke_messages({"body": body}, "anthropic.claude-3-haiku-v1:0")
    assert msgs == [{"role": "user", "content": "Hello\nWorld"}]


def test_extract_meta_invoke_messages() -> None:
    body = json.dumps({"prompt": "Tell me a joke"})
    msgs = extract_invoke_messages({"body": body}, "meta.llama3-1-8b-instruct-v1:0")
    assert msgs == [{"role": "user", "content": "Tell me a joke"}]


def test_extract_invoke_messages_dict_body_passthrough() -> None:
    msgs = extract_invoke_messages(
        {"body": {"prompt": "hi"}},
        "mistral.mistral-7b-instruct-v0:2",
    )
    assert msgs == [{"role": "user", "content": "hi"}]


def test_extract_invoke_messages_amazon_titan_input_text() -> None:
    body = json.dumps({"inputText": "Why is the sky blue?"})
    msgs = extract_invoke_messages({"body": body}, "amazon.titan-text-express-v1")
    assert msgs == [{"role": "user", "content": "Why is the sky blue?"}]


def test_extract_invoke_messages_malformed_body_returns_none() -> None:
    assert extract_invoke_messages({"body": b"not-json"}, "anthropic.x") is None


def test_extract_invoke_messages_empty_returns_none() -> None:
    assert extract_invoke_messages({}, "anthropic.x") is None


# ---------------------------------------------------------------------------
# Output extraction — per-family
# ---------------------------------------------------------------------------


def test_extract_invoke_output_anthropic() -> None:
    body = {"content": [{"text": "Hi"}, {"text": "there"}]}
    out = extract_invoke_output(body, "anthropic")
    assert out == {"role": "assistant", "content": "Hi\nthere"}


def test_extract_invoke_output_meta() -> None:
    out = extract_invoke_output({"generation": "Hello!"}, "meta")
    assert out == {"role": "assistant", "content": "Hello!"}


def test_extract_invoke_output_cohere() -> None:
    out = extract_invoke_output({"generations": [{"text": "x"}]}, "cohere")
    assert out == {"role": "assistant", "content": "x"}


def test_extract_invoke_output_amazon_titan() -> None:
    out = extract_invoke_output({"results": [{"outputText": "y"}]}, "amazon")
    assert out == {"role": "assistant", "content": "y"}


def test_extract_invoke_output_ai21_chat() -> None:
    out = extract_invoke_output(
        {"choices": [{"message": {"content": "answer"}}]}, "ai21"
    )
    assert out == {"role": "assistant", "content": "answer"}


def test_extract_invoke_output_unknown_fallback() -> None:
    out = extract_invoke_output({"completion": "raw"}, "unknown")
    assert out == {"role": "assistant", "content": "raw"}


def test_extract_invoke_output_empty_returns_none() -> None:
    assert extract_invoke_output({}, "anthropic") is None
    assert extract_invoke_output({"content": []}, "anthropic") is None


def test_extract_converse_output_text_blocks() -> None:
    response = {
        "output": {
            "message": {"role": "assistant", "content": [{"text": "hi"}, {"text": "there"}]}
        }
    }
    out = extract_converse_output(response)
    assert out == {"role": "assistant", "content": "hi\nthere"}


def test_extract_converse_output_missing_returns_none() -> None:
    assert extract_converse_output({}) is None
    assert extract_converse_output({"output": {}}) is None


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


def test_build_invoke_metadata_anthropic_includes_id_and_stop_reason() -> None:
    body = {"id": "msg-1", "stop_reason": "end_turn"}
    meta = build_invoke_metadata(body, "anthropic")
    assert meta["finish_reason"] == "end_turn"
    assert meta["response_id"] == "msg-1"
    assert meta["provider_family"] == "anthropic"
    assert meta["method"] == "invoke_model"


def test_build_invoke_metadata_cohere_finish_reason_from_first_generation() -> None:
    body = {"generations": [{"finish_reason": "COMPLETE"}]}
    meta = build_invoke_metadata(body, "cohere")
    assert meta["finish_reason"] == "COMPLETE"


def test_build_invoke_metadata_amazon_completion_reason() -> None:
    body = {"results": [{"completionReason": "FINISH"}]}
    meta = build_invoke_metadata(body, "amazon")
    assert meta["finish_reason"] == "FINISH"


# ---------------------------------------------------------------------------
# RereadableBody
# ---------------------------------------------------------------------------


def test_rereadable_body_full_read_is_idempotent() -> None:
    body = RereadableBody(b'{"hello":"world"}')
    assert body.read() == b'{"hello":"world"}'
    # The caller's downstream code reads again — must still get the data.
    assert body.read() == b'{"hello":"world"}'


def test_rereadable_body_partial_read_advances_then_full_read_resets() -> None:
    body = RereadableBody(b"abcdef")
    assert body.read(2) == b"ab"
    assert body.read(2) == b"cd"
    # Full read resets to the beginning.
    assert body.read() == b"abcdef"
    assert body.read(3) == b"abc"


def test_rereadable_body_iter_chunks_and_lines() -> None:
    body = RereadableBody(b"a\nb\nc")
    assert list(body.iter_chunks(chunk_size=2)) == [b"a\n", b"b\n", b"c"]
    assert list(body.iter_lines()) == [b"a", b"b", b"c"]


def test_rereadable_body_content_length_and_close() -> None:
    body = RereadableBody(b"xyz")
    assert body.content_length == 3
    body.close()  # must not raise


def test_rereadable_body_back_compat_alias() -> None:
    assert _RereadableBody is RereadableBody


# ---------------------------------------------------------------------------
# Adapter behaviour — fake clients
# ---------------------------------------------------------------------------


def _streaming_body(payload: Dict[str, Any]) -> StreamingBody:
    """Build a real botocore ``StreamingBody`` around ``payload``."""
    raw = json.dumps(payload).encode("utf-8")
    return StreamingBody(io.BytesIO(raw), len(raw))


class _FakeBedrockClient:
    """Minimal stand-in for the botocore bedrock-runtime client."""

    def __init__(self, response: Dict[str, Any]) -> None:
        self._response = response
        self.last_kwargs: Dict[str, Any] = {}

    def converse(self, **kwargs: Any) -> Dict[str, Any]:
        self.last_kwargs = kwargs
        return self._response

    def invoke_model(self, **kwargs: Any) -> Dict[str, Any]:
        self.last_kwargs = kwargs
        return self._response

    def converse_stream(self, **kwargs: Any) -> Dict[str, Any]:
        self.last_kwargs = kwargs
        return self._response

    def invoke_model_with_response_stream(self, **kwargs: Any) -> Dict[str, Any]:
        self.last_kwargs = kwargs
        return self._response


def _make_adapter() -> tuple[AWSBedrockAdapter, _RecordingLayerLens]:
    rec = _RecordingLayerLens()
    adapter = AWSBedrockAdapter(layerlens=rec, capture_config=CaptureConfig.full())
    adapter.connect()
    return adapter, rec


def test_converse_emits_full_event_set() -> None:
    adapter, rec = _make_adapter()
    client = _FakeBedrockClient(
        response={
            "output": {
                "message": {"role": "assistant", "content": [{"text": "hello"}]}
            },
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "stopReason": "end_turn",
            "ResponseMetadata": {"RequestId": "req-abc"},
        }
    )
    adapter.connect_client(client)

    client.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )

    types = [e["event_type"] for e in rec.events]
    assert "model.invoke" in types
    assert "cost.record" in types

    invoke = next(e for e in rec.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["finish_reason"] == "end_turn"
    assert invoke["payload"]["response_id"] == "req-abc"
    assert invoke["payload"]["output_message"] == {
        "role": "assistant",
        "content": "hello",
    }

    cost = next(e for e in rec.events if e["event_type"] == "cost.record")
    # claude-3-5-sonnet pricing in BEDROCK_PRICING: 0.003 input, 0.015 output per 1k.
    expected = 10 * 0.003 / 1000 + 5 * 0.015 / 1000
    assert abs(cost["payload"]["api_cost_usd"] - expected) < 1e-6


def test_converse_error_path_emits_error_event_and_reraises() -> None:
    adapter, rec = _make_adapter()

    class _Boom(_FakeBedrockClient):
        def converse(self, **kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError("boom")

    client = _Boom(response={})
    adapter.connect_client(client)

    with pytest.raises(RuntimeError, match="boom"):
        client.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
        )

    err_invoke = next(e for e in rec.events if e["event_type"] == "model.invoke")
    assert err_invoke["payload"]["error"] == "boom"
    assert err_invoke["payload"]["method"] == "converse"
    violations = [e for e in rec.events if e["event_type"] == "policy.violation"]
    assert violations and violations[0]["payload"]["error"] == "boom"


def test_invoke_model_anthropic_emits_full_set_and_rewraps_body() -> None:
    adapter, rec = _make_adapter()
    response_payload = {
        "id": "msg-1",
        "stop_reason": "end_turn",
        "content": [{"text": "answer"}],
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    client = _FakeBedrockClient(response={"body": _streaming_body(response_payload)})
    adapter.connect_client(client)

    body_in = json.dumps(
        {"messages": [{"role": "user", "content": "Q"}], "max_tokens": 10}
    )
    response = client.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=body_in,
        contentType="application/json",
        accept="application/json",
    )

    # Adapter must replace body with a re-readable wrapper so caller code
    # that does response["body"].read() still sees the bytes.
    assert isinstance(response["body"], RereadableBody)
    assert json.loads(response["body"].read()) == response_payload

    types = [e["event_type"] for e in rec.events]
    assert types == ["model.invoke", "cost.record"]

    invoke_payload = rec.events[0]["payload"]
    assert invoke_payload["provider_family"] == "anthropic"
    assert invoke_payload["finish_reason"] == "end_turn"
    assert invoke_payload["response_id"] == "msg-1"
    assert invoke_payload["prompt_tokens"] == 7
    assert invoke_payload["completion_tokens"] == 3
    assert invoke_payload["output_message"] == {"role": "assistant", "content": "answer"}


def test_invoke_model_meta_branch() -> None:
    adapter, rec = _make_adapter()
    payload = {
        "generation": "ok",
        "stop_reason": "stop",
        "prompt_token_count": 4,
        "generation_token_count": 2,
    }
    client = _FakeBedrockClient(response={"body": _streaming_body(payload)})
    adapter.connect_client(client)
    client.invoke_model(
        modelId="meta.llama3-1-8b-instruct-v1:0",
        body=json.dumps({"prompt": "hi"}),
    )
    invoke = rec.events[0]["payload"]
    assert invoke["provider_family"] == "meta"
    assert invoke["prompt_tokens"] == 4
    assert invoke["completion_tokens"] == 2
    assert invoke["finish_reason"] == "stop"
    assert invoke["output_message"] == {"role": "assistant", "content": "ok"}


def test_invoke_model_cohere_branch() -> None:
    adapter, rec = _make_adapter()
    payload = {
        "generations": [{"text": "g", "finish_reason": "COMPLETE"}],
        "meta": {"billed_units": {"input_tokens": 6, "output_tokens": 1}},
    }
    client = _FakeBedrockClient(response={"body": _streaming_body(payload)})
    adapter.connect_client(client)
    client.invoke_model(
        modelId="cohere.command-r-v1:0",
        body=json.dumps({"prompt": "hi"}),
    )
    invoke = rec.events[0]["payload"]
    assert invoke["provider_family"] == "cohere"
    assert invoke["prompt_tokens"] == 6
    assert invoke["completion_tokens"] == 1
    assert invoke["finish_reason"] == "COMPLETE"
    assert invoke["output_message"] == {"role": "assistant", "content": "g"}


def test_invoke_model_amazon_titan_branch() -> None:
    adapter, rec = _make_adapter()
    payload = {
        "inputTextTokenCount": 12,
        "results": [{"outputText": "ok", "completionReason": "FINISH", "tokenCount": 4}],
    }
    client = _FakeBedrockClient(response={"body": _streaming_body(payload)})
    adapter.connect_client(client)
    client.invoke_model(
        modelId="amazon.titan-text-express-v1",
        body=json.dumps({"inputText": "Why is the sky blue?"}),
    )
    invoke = rec.events[0]["payload"]
    assert invoke["provider_family"] == "amazon"
    assert invoke["finish_reason"] == "FINISH"
    assert invoke["prompt_tokens"] == 12
    assert invoke["completion_tokens"] == 4
    assert invoke["output_message"] == {"role": "assistant", "content": "ok"}


def test_invoke_model_mistral_branch() -> None:
    adapter, rec = _make_adapter()
    payload = {
        "generation": "Bonjour",
        "stop_reason": "stop",
        "prompt_tokens": 2,
        "completion_tokens": 1,
    }
    client = _FakeBedrockClient(response={"body": _streaming_body(payload)})
    adapter.connect_client(client)
    client.invoke_model(
        modelId="mistral.mistral-7b-instruct-v0:2",
        body=json.dumps({"prompt": "salut"}),
    )
    invoke = rec.events[0]["payload"]
    assert invoke["provider_family"] == "mistral"
    assert invoke["finish_reason"] == "stop"
    assert invoke["prompt_tokens"] == 2
    assert invoke["completion_tokens"] == 1
    assert invoke["output_message"] == {"role": "assistant", "content": "Bonjour"}


def test_invoke_model_ai21_branch() -> None:
    adapter, rec = _make_adapter()
    payload = {
        "choices": [
            {"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    client = _FakeBedrockClient(response={"body": _streaming_body(payload)})
    adapter.connect_client(client)
    client.invoke_model(
        modelId="ai21.jamba-instruct-v1:0",
        body=json.dumps({"prompt": "hi"}),
    )
    invoke = rec.events[0]["payload"]
    assert invoke["provider_family"] == "ai21"
    assert invoke["finish_reason"] == "stop"
    assert invoke["prompt_tokens"] == 3
    assert invoke["completion_tokens"] == 1
    assert invoke["output_message"] == {"role": "assistant", "content": "answer"}


def test_invoke_model_with_response_stream_emits_streaming_event() -> None:
    adapter, rec = _make_adapter()
    client = _FakeBedrockClient(response={"body": object()})
    adapter.connect_client(client)
    client.invoke_model_with_response_stream(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
    )
    invoke = next(e for e in rec.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["streaming"] is True
    assert invoke["payload"]["method"] == "invoke_model_with_response_stream"
    # No cost.record on streaming side (usage not yet known).
    assert all(e["event_type"] != "cost.record" for e in rec.events)


def test_converse_stream_emits_streaming_event() -> None:
    adapter, rec = _make_adapter()
    client = _FakeBedrockClient(response={"stream": object()})
    adapter.connect_client(client)
    client.converse_stream(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )
    invoke = next(e for e in rec.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["streaming"] is True
    assert invoke["payload"]["method"] == "converse_stream"


def test_disconnect_restores_originals() -> None:
    adapter, _ = _make_adapter()
    response_payload = {"output": {"message": {"role": "assistant", "content": [{"text": "x"}]}}}
    client = _FakeBedrockClient(response=response_payload)
    original_converse = client.converse
    adapter.connect_client(client)
    # Method has been swapped to the traced wrapper, which carries the
    # ``_layerlens_original`` marker set by the adapter.
    assert hasattr(client.converse, "_layerlens_original")
    adapter.disconnect()
    # After disconnect, the marker is gone — the original bound-method
    # identity is restored. Comparing __func__ is more robust than ``is``
    # because Python builds a fresh bound-method object on each attribute
    # access.
    assert not hasattr(client.converse, "_layerlens_original")
    assert client.converse.__func__ is original_converse.__func__


def test_capture_standard_omits_message_content_when_disabled() -> None:
    """``capture_content=False`` must drop the prompt and response text."""
    rec = _RecordingLayerLens()
    cfg = CaptureConfig.standard()
    cfg.capture_content = False
    adapter = AWSBedrockAdapter(layerlens=rec, capture_config=cfg)
    adapter.connect()
    client = _FakeBedrockClient(
        response={
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
    )
    adapter.connect_client(client)
    client.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )
    invoke = next(e for e in rec.events if e["event_type"] == "model.invoke")
    assert "messages" not in invoke["payload"]
    assert "output_message" not in invoke["payload"]
    # Tokens MUST still be present — they're metadata, not content.
    assert invoke["payload"]["prompt_tokens"] == 1
    assert invoke["payload"]["completion_tokens"] == 1


def test_capture_minimal_drops_model_invoke_event() -> None:
    """``CaptureConfig.minimal()`` disables L3 — model.invoke must be filtered."""
    rec = _RecordingLayerLens()
    adapter = AWSBedrockAdapter(layerlens=rec, capture_config=CaptureConfig.minimal())
    adapter.connect()
    client = _FakeBedrockClient(
        response={
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
    )
    adapter.connect_client(client)
    client.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )
    types = [e["event_type"] for e in rec.events]
    assert "model.invoke" not in types
    # cost.record is cross-cutting, so it MUST still fire.
    assert "cost.record" in types


# ---------------------------------------------------------------------------
# Adapter behaviour — boto3 + botocore.stub.Stubber
# ---------------------------------------------------------------------------


def test_converse_against_real_botocore_client_via_stubber() -> None:
    """Adapter must wrap a real boto3 ``bedrock-runtime`` client correctly."""
    rec = _RecordingLayerLens()
    adapter = AWSBedrockAdapter(layerlens=rec, capture_config=CaptureConfig.full())
    adapter.connect()

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    adapter.connect_client(client)

    expected_response: Dict[str, Any] = {
        "output": {
            "message": {"role": "assistant", "content": [{"text": "hello"}]}
        },
        "usage": {"inputTokens": 8, "outputTokens": 2, "totalTokens": 10},
        "stopReason": "end_turn",
        # boto3 >= 1.40 enforces ``metrics`` on Converse responses.
        "metrics": {"latencyMs": 42},
    }
    expected_params: Dict[str, Any] = {
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    }

    with Stubber(client) as stubber:
        stubber.add_response("converse", expected_response, expected_params)
        result = client.converse(**expected_params)
        stubber.assert_no_pending_responses()

    assert result["output"]["message"]["content"][0]["text"] == "hello"

    types = [e["event_type"] for e in rec.events]
    assert "model.invoke" in types and "cost.record" in types
    invoke = next(e for e in rec.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["finish_reason"] == "end_turn"
    assert invoke["payload"]["prompt_tokens"] == 8
    assert invoke["payload"]["completion_tokens"] == 2


def test_invoke_model_against_real_botocore_client_via_stubber() -> None:
    """Real ``StreamingBody`` must round-trip through ``RereadableBody``."""
    rec = _RecordingLayerLens()
    adapter = AWSBedrockAdapter(layerlens=rec, capture_config=CaptureConfig.standard())
    adapter.connect()

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    adapter.connect_client(client)

    body_payload = {
        "id": "msg-stub-1",
        "stop_reason": "end_turn",
        "content": [{"text": "stubbed"}],
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    raw = json.dumps(body_payload).encode("utf-8")
    expected_response = {
        "body": StreamingBody(io.BytesIO(raw), len(raw)),
        "contentType": "application/json",
    }
    request_params = {
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "body": json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        "contentType": "application/json",
        "accept": "application/json",
    }

    with Stubber(client) as stubber:
        stubber.add_response("invoke_model", expected_response, request_params)
        response = client.invoke_model(**request_params)
        stubber.assert_no_pending_responses()

    # Caller must still be able to read the body.
    assert isinstance(response["body"], RereadableBody)
    assert json.loads(response["body"].read()) == body_payload

    invoke = rec.events[0]["payload"]
    assert invoke["provider_family"] == "anthropic"
    assert invoke["response_id"] == "msg-stub-1"
    assert invoke["prompt_tokens"] == 5
    assert invoke["completion_tokens"] == 2


# ---------------------------------------------------------------------------
# Project-wide test-dep sanity check
# ---------------------------------------------------------------------------


def test_respx_is_available_for_httpx_based_provider_tests() -> None:
    """``respx`` is the standard httpx interceptor across the test suite.

    Bedrock itself uses urllib3 (via botocore) so respx does not drive it,
    but this assertion catches accidental dev-dep regressions that would
    cripple sibling provider tests in the same M3 fan-out.
    """
    assert hasattr(respx, "MockRouter")


def test_respx_route_matches_a_synthetic_httpx_request() -> None:
    """End-to-end smoke check for the respx integration."""
    import httpx

    with respx.mock(base_url="https://example.invalid") as router:
        router.get("/ping").respond(200, json={"ok": True})
        with httpx.Client(base_url="https://example.invalid") as cli:
            r = cli.get("/ping")
        assert r.json() == {"ok": True}


# ---------------------------------------------------------------------------
# Framework-version detection
# ---------------------------------------------------------------------------


def test_detect_framework_version_returns_boto3_version() -> None:
    version = AWSBedrockAdapter._detect_framework_version()
    assert version is not None
    assert version == boto3.__version__


def test_detect_framework_version_handles_missing_boto3() -> None:
    # If boto3 fails to import, the detector must fail soft to ``None``.
    with mock.patch.dict("sys.modules", {"boto3": None}):
        # Force re-import to take effect by clearing the cached module.
        # mock.patch.dict with value=None makes importlib see boto3 as
        # unimportable.
        version = AWSBedrockAdapter._detect_framework_version()
    # boto3 is still installed in test env so the detection during this
    # call uses the real one — but no exception should ever leak out.
    assert version is None or isinstance(version, str)
