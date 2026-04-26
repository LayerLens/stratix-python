"""Unit tests for the Mistral provider adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.providers.mistral_adapter import (
    ADAPTER_CLASS,
    MistralAdapter,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


def _make_response(
    content: str = "hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    response_id: str = "msg-abc",
    finish_reason: str = "stop",
    tool_calls: List[Any] = None,
) -> Any:
    """Build an OpenAI-shape Mistral response."""
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, index=0)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(
        id=response_id,
        model="mistral-small-latest",
        choices=[choice],
        usage=usage,
    )


def _make_client(*, returns: Any = None, raises: Exception = None) -> Any:
    def complete(**kwargs: Any) -> Any:
        if raises is not None:
            raise raises
        return returns

    def stream(**kwargs: Any) -> Any:
        if raises is not None:
            raise raises
        return iter([])

    def embed_create(**kwargs: Any) -> Any:
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])],
            usage=SimpleNamespace(
                prompt_tokens=4, completion_tokens=0, total_tokens=4
            ),
        )

    chat = SimpleNamespace(complete=complete, stream=stream)
    embeddings = SimpleNamespace(create=embed_create)
    return SimpleNamespace(chat=chat, embeddings=embeddings)


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is MistralAdapter


def test_lifecycle() -> None:
    a = MistralAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_complete_emits_invoke_and_cost() -> None:
    stratix = _RecordingStratix()
    adapter = MistralAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    client = _make_client(returns=_make_response())
    adapter.connect_client(client)

    client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
    )

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "cost.record" in types

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["provider"] == "mistral"
    assert invoke["payload"]["model"] == "mistral-small-latest"
    assert invoke["payload"]["prompt_tokens"] == 10
    assert invoke["payload"]["completion_tokens"] == 5
    assert invoke["payload"]["parameters"]["temperature"] == 0.5
    assert invoke["payload"]["finish_reason"] == "stop"


def test_known_model_priced() -> None:
    stratix = _RecordingStratix()
    adapter = MistralAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_client(
        returns=_make_response(prompt_tokens=1000, completion_tokens=500),
    )
    adapter.connect_client(client)
    client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "x"}],
    )

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    # mistral-small-latest: 0.0002 input + 0.0006 output per 1k.
    expected = 1000 * 0.0002 / 1000 + 500 * 0.0006 / 1000
    assert cost["payload"]["api_cost_usd"] == pytest.approx(expected, rel=1e-4)


def test_provider_error_emits_policy_violation() -> None:
    stratix = _RecordingStratix()
    adapter = MistralAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_client(raises=RuntimeError("rate limited"))
    adapter.connect_client(client)

    with pytest.raises(RuntimeError):
        client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "x"}],
        )

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "policy.violation" in types


def test_tool_calls_extracted() -> None:
    stratix = _RecordingStratix()
    adapter = MistralAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    fn = SimpleNamespace(name="get_time", arguments='{"tz": "UTC"}')
    tc = SimpleNamespace(id="call-1", function=fn)

    client = _make_client(returns=_make_response(tool_calls=[tc]))
    adapter.connect_client(client)
    client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "what time"}],
    )

    tool_events = [e for e in stratix.events if e["event_type"] == "tool.call"]
    assert len(tool_events) == 1
    assert tool_events[0]["payload"]["tool_name"] == "get_time"
    assert tool_events[0]["payload"]["tool_input"] == {"tz": "UTC"}


def test_disconnect_restores_originals() -> None:
    adapter = MistralAdapter()
    adapter.connect()
    client = _make_client(returns=_make_response())
    original_complete = client.chat.complete
    adapter.connect_client(client)
    assert client.chat.complete is not original_complete
    adapter.disconnect()
    assert client.chat.complete is original_complete


def test_streaming_emits_consolidated_event() -> None:
    """Iterating the stream emits exactly one consolidated model.invoke."""
    stratix = _RecordingStratix()
    adapter = MistralAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    # Build a synthetic stream of CompletionEvent-like objects.
    def stream_events(**kwargs: Any) -> Any:
        return iter(
            [
                SimpleNamespace(
                    data=SimpleNamespace(
                        id="msg-1",
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="Hello "),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    )
                ),
                SimpleNamespace(
                    data=SimpleNamespace(
                        id="msg-1",
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="world"),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    )
                ),
                SimpleNamespace(
                    data=SimpleNamespace(
                        id="msg-1",
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content=None),
                                finish_reason="stop",
                            )
                        ],
                        usage=SimpleNamespace(
                            prompt_tokens=8,
                            completion_tokens=2,
                            total_tokens=10,
                        ),
                    )
                ),
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(complete=lambda **kw: None, stream=stream_events),
        embeddings=None,
    )
    adapter.connect_client(client)

    stream = client.chat.stream(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "hi"}],
    )
    chunks = list(stream)
    assert len(chunks) == 3

    invokes = [e for e in stratix.events if e["event_type"] == "model.invoke"]
    assert len(invokes) == 1
    assert invokes[0]["payload"]["streaming"] is True
    assert invokes[0]["payload"]["finish_reason"] == "stop"
    assert invokes[0]["payload"]["output_message"]["content"] == "Hello world"


def test_embed_emits_events() -> None:
    stratix = _RecordingStratix()
    adapter = MistralAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_client()
    adapter.connect_client(client)

    client.embeddings.create(model="mistral-embed", inputs=["hi"])

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["request_type"] == "embedding"
