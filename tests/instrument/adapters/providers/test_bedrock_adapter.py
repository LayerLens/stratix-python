"""Unit tests for the AWS Bedrock provider adapter."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.providers.bedrock_adapter import (
    ADAPTER_CLASS,
    AWSBedrockAdapter,
    _detect_provider_family,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


@pytest.mark.parametrize(
    "model_id,family",
    [
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic"),
        ("meta.llama3-1-70b-instruct-v1:0", "meta"),
        ("cohere.command-r-v1:0", "cohere"),
        ("amazon.titan-text-express-v1", "amazon"),
        ("ai21.jamba-instruct-v1:0", "ai21"),
        ("mistral.mistral-7b-instruct-v0:2", "mistral"),
        ("unknown.model-v1", "unknown"),
        ("", "unknown"),
    ],
)
def test_detect_provider_family(model_id: str, family: str) -> None:
    assert _detect_provider_family(model_id) == family


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is AWSBedrockAdapter


def test_lifecycle() -> None:
    a = AWSBedrockAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY


def test_extract_invoke_usage_anthropic() -> None:
    body = {"usage": {"input_tokens": 100, "output_tokens": 50}}
    usage = AWSBedrockAdapter._extract_invoke_usage(body, "anthropic")
    assert usage is not None
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150


def test_extract_invoke_usage_meta() -> None:
    body = {"prompt_token_count": 50, "generation_token_count": 25}
    usage = AWSBedrockAdapter._extract_invoke_usage(body, "meta")
    assert usage is not None
    assert usage.prompt_tokens == 50
    assert usage.completion_tokens == 25


def test_extract_invoke_usage_cohere() -> None:
    body = {"meta": {"billed_units": {"input_tokens": 10, "output_tokens": 5}}}
    usage = AWSBedrockAdapter._extract_invoke_usage(body, "cohere")
    assert usage is not None
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5


def test_extract_converse_usage() -> None:
    response = {"usage": {"inputTokens": 100, "outputTokens": 50}}
    usage = AWSBedrockAdapter._extract_converse_usage(response)
    assert usage is not None
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50


def test_extract_anthropic_invoke_messages() -> None:
    body = json.dumps(
        {
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )
    msgs = AWSBedrockAdapter._extract_invoke_messages(
        {"body": body},
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    assert msgs is not None
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_rereadable_body_can_be_read_twice() -> None:
    """The wrapper around StreamingBody must support re-reading after we consume it."""
    from layerlens.instrument.adapters.providers.bedrock_adapter import _RereadableBody

    body = _RereadableBody(b'{"hello":"world"}')
    assert body.read() == b'{"hello":"world"}'
    # Caller code reads again — must still get the data.
    assert body.read() == b'{"hello":"world"}'


def test_converse_emits_full_event_set() -> None:
    """Converse API call must emit model.invoke + cost.record."""

    class _FakeClient:
        def converse(self, **kwargs: Any) -> Dict[str, Any]:
            return {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "hello"}],
                    }
                },
                "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                "stopReason": "end_turn",
                "ResponseMetadata": {"RequestId": "req-abc"},
            }

    stratix = _RecordingStratix()
    adapter = AWSBedrockAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _FakeClient()
    adapter.connect_client(client)

    client.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "cost.record" in types

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["finish_reason"] == "end_turn"
    assert invoke["payload"]["response_id"] == "req-abc"

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    # claude-3-5-sonnet pricing in BEDROCK_PRICING: 0.003 input, 0.015 output per 1k.
    expected = 10 * 0.003 / 1000 + 5 * 0.015 / 1000
    assert abs(cost["payload"]["api_cost_usd"] - expected) < 1e-6
