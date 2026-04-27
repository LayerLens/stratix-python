"""Unit tests for the Cohere provider adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.providers.cohere_adapter import (
    ADAPTER_CLASS,
    CohereAdapter,
)


class _RecordingStratix:
    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this stratix pass the
    # BaseAdapter fail-fast check. Tests asserting cross-tenant
    # isolation override this default.
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


def _make_v1_response(
    text: str = "hello",
    input_tokens: int = 10,
    output_tokens: int = 5,
    response_id: str = "gen-abc",
    finish_reason: str = "COMPLETE",
    tool_calls: List[Any] = None,
) -> Any:
    """Build a v1 Cohere chat response."""
    billed = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    meta = SimpleNamespace(billed_units=billed)
    return SimpleNamespace(
        text=text,
        generation_id=response_id,
        meta=meta,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
    )


def _make_v2_response(
    text: str = "hello",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> Any:
    """Build a v2 Cohere chat response."""
    text_block = SimpleNamespace(type="text", text=text)
    message = SimpleNamespace(content=[text_block], tool_calls=None)
    billed = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    meta = SimpleNamespace(billed_units=billed)
    return SimpleNamespace(
        id="msg-xyz",
        message=message,
        meta=meta,
        finish_reason="COMPLETE",
    )


def _make_client(*, returns_v1: Any = None, returns_v2: Any = None) -> Any:
    def chat(**kwargs: Any) -> Any:
        return returns_v1

    def v2_chat(**kwargs: Any) -> Any:
        return returns_v2

    def embed(**kwargs: Any) -> Any:
        return SimpleNamespace(
            embeddings=[[0.1, 0.2]],
            meta=SimpleNamespace(billed_units=SimpleNamespace(input_tokens=4)),
        )

    v2 = SimpleNamespace(chat=v2_chat)
    return SimpleNamespace(chat=chat, v2=v2, embed=embed)


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is CohereAdapter


def test_lifecycle() -> None:
    a = CohereAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_v1_chat_emits_invoke_and_cost() -> None:
    stratix = _RecordingStratix()
    adapter = CohereAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    client = _make_client(returns_v1=_make_v1_response())
    adapter.connect_client(client)

    client.chat(model="command-r-plus", message="hi")

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "cost.record" in types

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["provider"] == "cohere"
    assert invoke["payload"]["model"] == "command-r-plus"
    assert invoke["payload"]["prompt_tokens"] == 10
    assert invoke["payload"]["completion_tokens"] == 5
    assert invoke["payload"]["parameters"]["api_version"] == "v1"
    assert invoke["payload"]["finish_reason"] == "COMPLETE"


def test_v1_chat_input_message_captured() -> None:
    stratix = _RecordingStratix()
    adapter = CohereAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_client(returns_v1=_make_v1_response())
    adapter.connect_client(client)

    client.chat(
        model="command-r",
        message="hello world",
        preamble="You are a helpful assistant.",
    )

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    msgs = invoke["payload"].get("messages")
    assert msgs is not None
    # Preamble inserted as system message at position 0.
    assert msgs[0]["role"] == "system"
    assert "helpful" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "hello" in msgs[1]["content"]


def test_v1_output_text_captured() -> None:
    stratix = _RecordingStratix()
    adapter = CohereAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_client(returns_v1=_make_v1_response(text="answer-text"))
    adapter.connect_client(client)
    client.chat(model="command-r", message="x")
    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    out = invoke["payload"].get("output_message")
    assert out is not None
    assert out["content"] == "answer-text"


def test_v1_tool_calls_emitted() -> None:
    stratix = _RecordingStratix()
    adapter = CohereAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    tc = SimpleNamespace(name="lookup", parameters={"q": "weather"})
    client = _make_client(returns_v1=_make_v1_response(tool_calls=[tc]))
    adapter.connect_client(client)
    client.chat(model="command-r", message="x")

    tool_events = [e for e in stratix.events if e["event_type"] == "tool.call"]
    assert len(tool_events) == 1
    assert tool_events[0]["payload"]["tool_name"] == "lookup"


def test_v2_chat_emits_invoke() -> None:
    stratix = _RecordingStratix()
    adapter = CohereAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    client = _make_client(returns_v2=_make_v2_response())
    adapter.connect_client(client)

    client.v2.chat(model="command-r-plus", messages=[{"role": "user", "content": "hi"}])

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["parameters"]["api_version"] == "v2"
    assert invoke["payload"]["output_message"]["content"] == "hello"


def test_provider_error_emits_policy_violation() -> None:
    stratix = _RecordingStratix()
    adapter = CohereAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    def bad_chat(**kwargs: Any) -> Any:
        raise RuntimeError("rate limited")

    client = SimpleNamespace(chat=bad_chat, v2=None, embed=None)
    adapter.connect_client(client)

    with pytest.raises(RuntimeError):
        client.chat(model="command-r", message="x")

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "policy.violation" in types


def test_disconnect_restores_originals() -> None:
    adapter = CohereAdapter(org_id="test-org")
    adapter.connect()

    client = _make_client(returns_v1=_make_v1_response())
    original_chat = client.chat
    adapter.connect_client(client)
    assert client.chat is not original_chat
    adapter.disconnect()
    assert client.chat is original_chat


def test_known_model_priced() -> None:
    """``command-r-plus`` is in the canonical PRICING table."""
    stratix = _RecordingStratix()
    adapter = CohereAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_client(
        returns_v1=_make_v1_response(input_tokens=1000, output_tokens=500),
    )
    adapter.connect_client(client)
    client.chat(model="command-r-plus", message="x")

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    # command-r-plus: 0.003 input + 0.015 output per 1k.
    expected = 1000 * 0.003 / 1000 + 500 * 0.015 / 1000
    assert cost["payload"]["api_cost_usd"] == pytest.approx(expected, rel=1e-4)


def test_embed_emits_events() -> None:
    stratix = _RecordingStratix()
    adapter = CohereAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_client()
    adapter.connect_client(client)

    client.embed(model="embed-english-v3.0", texts=["hi"])

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["request_type"] == "embedding"
