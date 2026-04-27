"""Unit tests for the Ollama provider adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.providers.ollama_adapter import (
    ADAPTER_CLASS,
    OllamaAdapter,
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


def _make_chat_response() -> Dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": "hello"},
        "prompt_eval_count": 10,
        "eval_count": 5,
        "prompt_eval_duration": 1_000_000_000,  # 1s
        "eval_duration": 2_000_000_000,  # 2s
        "done_reason": "stop",
    }


def _make_client(*, chat_response: Any = None) -> Any:
    client = SimpleNamespace()
    client.chat = lambda **kw: chat_response or _make_chat_response()
    client.generate = lambda **kw: {
        "response": "hi",
        "prompt_eval_count": 5,
        "eval_count": 2,
    }
    client.embeddings = lambda **kw: {"embedding": [0.1, 0.2]}
    return client


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is OllamaAdapter


def test_connect_uses_env_endpoint(monkeypatch: Any) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://my-ollama:11434")
    a = OllamaAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    assert a._endpoint == "http://my-ollama:11434"


def test_connect_default_endpoint(monkeypatch: Any) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    a = OllamaAdapter(org_id="test-org")
    a.connect()
    assert a._endpoint == "http://localhost:11434"


def test_chat_emits_zero_api_cost() -> None:
    """Local inference => api_cost_usd is exactly 0.0."""
    stratix = _RecordingStratix()
    adapter = OllamaAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_client()
    adapter.connect_client(client)

    client.chat(model="llama3.1", messages=[{"role": "user", "content": "hi"}])

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["api_cost_usd"] == 0.0


def test_infra_cost_calculated_when_configured() -> None:
    stratix = _RecordingStratix()
    adapter = OllamaAdapter(
        stratix=stratix,
        capture_config=CaptureConfig.full(),
        cost_per_second=0.01,
    )
    adapter.connect()
    client = _make_client()
    adapter.connect_client(client)

    client.chat(model="llama3.1", messages=[{"role": "user", "content": "hi"}])

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    # 1s + 2s = 3s @ $0.01/s = $0.03
    assert cost["payload"]["infra_cost_usd"] == 0.03


def test_generate_method_works() -> None:
    stratix = _RecordingStratix()
    adapter = OllamaAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_client()
    adapter.connect_client(client)

    client.generate(model="llama3.1", prompt="Why is the sky blue?")

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["method"] == "generate"
    # Generate response captured as output_message.
    assert invoke["payload"].get("output_message") is not None


def test_disconnect_restores_originals() -> None:
    adapter = OllamaAdapter(org_id="test-org")
    adapter.connect()
    client = _make_client()
    original_chat = client.chat
    adapter.connect_client(client)
    assert client.chat is not original_chat
    adapter.disconnect()
    assert client.chat is original_chat
