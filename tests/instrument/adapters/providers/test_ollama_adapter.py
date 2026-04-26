"""HTTP-fixture tests for the Ollama LLM provider adapter.

These tests instrument a real :class:`ollama.Client` and intercept the
underlying HTTP traffic with :mod:`respx`, asserting that the adapter
emits the canonical telemetry events (``model.invoke``,
``cost.record``, ``policy.violation``) and that all Ollama-specific
invariants hold:

* ``api_cost_usd`` is always exactly ``0.0`` (local inference).
* ``infra_cost_usd`` appears only when ``cost_per_second`` is set.
* The endpoint and method are recorded in event metadata.
* ``connect_client`` works on the real ``ollama.Client`` instance.
* ``disconnect`` restores all originally-bound methods.

Hitting the real httpx layer (rather than a hand-rolled
``SimpleNamespace`` mock) gives us confidence the adapter integrates
with the actual Ollama Python SDK as shipped on PyPI.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
import respx
import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.providers.ollama_adapter import (
    ADAPTER_CLASS,
    OllamaAdapter,
)

# Ollama's Python SDK normalises ``http://localhost:11434`` to
# ``http://127.0.0.1:11434`` before httpx sees the URL, so respx must
# match against the IP form.
_OLLAMA_BASE = "http://127.0.0.1:11434"


class _RecordingStratix:
    """Minimal capture sink that records every emitted event."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


def _by_type(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    return [e for e in events if e["event_type"] == event_type]


def _chat_response_body(
    *,
    content: str = "hello",
    prompt_eval_count: int = 10,
    eval_count: int = 5,
    prompt_eval_duration_ns: int = 1_000_000_000,
    eval_duration_ns: int = 2_000_000_000,
    done_reason: str = "stop",
) -> Dict[str, Any]:
    return {
        "model": "llama3.1",
        "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "total_duration": prompt_eval_duration_ns + eval_duration_ns,
        "load_duration": 0,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": prompt_eval_duration_ns,
        "eval_count": eval_count,
        "eval_duration": eval_duration_ns,
    }


def _generate_response_body(
    *,
    response: str = "why the sky is blue",
    prompt_eval_count: int = 5,
    eval_count: int = 3,
) -> Dict[str, Any]:
    return {
        "model": "llama3.1",
        "created_at": "2026-01-01T00:00:00Z",
        "response": response,
        "done": True,
        "done_reason": "stop",
        "total_duration": 1_500_000_000,
        "load_duration": 0,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": 500_000_000,
        "eval_count": eval_count,
        "eval_duration": 1_000_000_000,
    }


# ---------------------------------------------------------------------------
# Module / registry plumbing
# ---------------------------------------------------------------------------


def test_adapter_class_export() -> None:
    """The lazy registry contract: ``ADAPTER_CLASS`` is the class itself."""
    assert ADAPTER_CLASS is OllamaAdapter


def test_lazy_export_from_providers_package() -> None:
    """``from ...providers import OllamaAdapter`` triggers lazy import."""
    from layerlens.instrument.adapters.providers import OllamaAdapter as Lazy

    assert Lazy is OllamaAdapter


# ---------------------------------------------------------------------------
# Endpoint detection
# ---------------------------------------------------------------------------


def test_connect_default_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``OLLAMA_HOST`` set, the adapter records the local default."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    adapter = OllamaAdapter()
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY
    assert adapter._endpoint == "http://localhost:11434"


def test_connect_uses_env_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OLLAMA_HOST`` overrides the default and is stamped on every event."""
    monkeypatch.setenv("OLLAMA_HOST", "http://my-ollama:11434")
    adapter = OllamaAdapter()
    adapter.connect()
    assert adapter._endpoint == "http://my-ollama:11434"


# ---------------------------------------------------------------------------
# Chat path — HTTP-fixture round-trip
# ---------------------------------------------------------------------------


@respx.mock(base_url=_OLLAMA_BASE)
def test_chat_emits_zero_api_cost(respx_mock: respx.MockRouter) -> None:
    """Local inference => ``api_cost_usd`` is exactly 0.0."""
    respx_mock.post("/api/chat").mock(
        return_value=httpx.Response(200, json=_chat_response_body())
    )

    from ollama import Client

    sink = _RecordingStratix()
    adapter = OllamaAdapter(stratix=sink, capture_config=CaptureConfig.full())
    adapter.connect()
    client = Client()
    adapter.connect_client(client)

    response = client.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.message.content == "hello"

    cost_events = _by_type(sink.events, "cost.record")
    assert len(cost_events) == 1
    payload = cost_events[0]["payload"]
    assert payload["api_cost_usd"] == 0.0
    assert payload["provider"] == "ollama"
    assert payload["prompt_tokens"] == 10
    assert payload["completion_tokens"] == 5
    assert payload["total_tokens"] == 15
    # Without cost_per_second, no infra cost should be emitted.
    assert "infra_cost_usd" not in payload


@respx.mock(base_url=_OLLAMA_BASE)
def test_chat_records_method_and_endpoint(respx_mock: respx.MockRouter) -> None:
    """The ``model.invoke`` event records method and endpoint metadata."""
    respx_mock.post("/api/chat").mock(
        return_value=httpx.Response(200, json=_chat_response_body())
    )

    from ollama import Client

    sink = _RecordingStratix()
    adapter = OllamaAdapter(stratix=sink, capture_config=CaptureConfig.full())
    adapter.connect()
    client = Client()
    adapter.connect_client(client)

    client.chat(model="llama3.1", messages=[{"role": "user", "content": "hi"}])

    invokes = _by_type(sink.events, "model.invoke")
    assert len(invokes) == 1
    payload = invokes[0]["payload"]
    assert payload["method"] == "chat"
    assert payload["endpoint"] == "http://localhost:11434"
    assert payload["finish_reason"] == "stop"
    assert payload["model"] == "llama3.1"
    # Output captured under capture_content=True.
    assert payload["output_message"]["content"] == "hello"
    # Latency is a non-negative float.
    assert isinstance(payload["latency_ms"], float)
    assert payload["latency_ms"] >= 0.0


@respx.mock(base_url=_OLLAMA_BASE)
def test_chat_with_cost_per_second_emits_infra_cost(
    respx_mock: respx.MockRouter,
) -> None:
    """``cost_per_second`` => ``infra_cost_usd = total_seconds * rate``."""
    respx_mock.post("/api/chat").mock(
        return_value=httpx.Response(
            200,
            # 1s prompt eval + 2s eval = 3s @ $0.01/s => $0.03
            json=_chat_response_body(),
        )
    )

    from ollama import Client

    sink = _RecordingStratix()
    adapter = OllamaAdapter(
        stratix=sink,
        capture_config=CaptureConfig.full(),
        cost_per_second=0.01,
    )
    adapter.connect()
    client = Client()
    adapter.connect_client(client)

    client.chat(model="llama3.1", messages=[{"role": "user", "content": "hi"}])

    cost_events = _by_type(sink.events, "cost.record")
    assert cost_events[0]["payload"]["infra_cost_usd"] == pytest.approx(0.03)
    # api_cost is still 0 — infra cost is additive, not a substitute.
    assert cost_events[0]["payload"]["api_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# Generate path
# ---------------------------------------------------------------------------


@respx.mock(base_url=_OLLAMA_BASE)
def test_generate_captures_prompt_as_input_message(
    respx_mock: respx.MockRouter,
) -> None:
    """``generate`` synthesises a single user-role message from the prompt."""
    respx_mock.post("/api/generate").mock(
        return_value=httpx.Response(200, json=_generate_response_body())
    )

    from ollama import Client

    sink = _RecordingStratix()
    adapter = OllamaAdapter(stratix=sink, capture_config=CaptureConfig.full())
    adapter.connect()
    client = Client()
    adapter.connect_client(client)

    client.generate(model="llama3.1", prompt="Why is the sky blue?")

    invokes = _by_type(sink.events, "model.invoke")
    payload = invokes[0]["payload"]
    assert payload["method"] == "generate"
    assert payload["messages"] == [
        {"role": "user", "content": "Why is the sky blue?"}
    ]
    assert payload["output_message"]["content"] == "why the sky is blue"
    assert payload["prompt_tokens"] == 5
    assert payload["completion_tokens"] == 3


# ---------------------------------------------------------------------------
# Embeddings path
# ---------------------------------------------------------------------------


@respx.mock(base_url=_OLLAMA_BASE)
def test_embeddings_emits_model_invoke_with_method(
    respx_mock: respx.MockRouter,
) -> None:
    """Embeddings are L3 ``model.invoke`` events with ``method=embeddings``."""
    respx_mock.post("/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})
    )

    from ollama import Client

    sink = _RecordingStratix()
    adapter = OllamaAdapter(stratix=sink, capture_config=CaptureConfig.full())
    adapter.connect()
    client = Client()
    adapter.connect_client(client)

    response = client.embeddings(model="nomic-embed-text", prompt="hello")
    assert len(response.embedding) == 3

    invokes = _by_type(sink.events, "model.invoke")
    assert any(e["payload"]["method"] == "embeddings" for e in invokes)
    # Embeddings response carries no token counts — the cost payload
    # falls back to zeros and api_cost_usd is still 0.0.
    cost = _by_type(sink.events, "cost.record")[0]["payload"]
    assert cost["api_cost_usd"] == 0.0
    assert cost["prompt_tokens"] == 0
    assert cost["completion_tokens"] == 0


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


@respx.mock(base_url=_OLLAMA_BASE)
def test_chat_error_emits_policy_violation_and_reraises(
    respx_mock: respx.MockRouter,
) -> None:
    """An HTTP-500 from Ollama raises and emits both error events."""
    respx_mock.post("/api/chat").mock(
        return_value=httpx.Response(500, json={"error": "model not found"})
    )

    from ollama import Client, ResponseError

    sink = _RecordingStratix()
    adapter = OllamaAdapter(stratix=sink, capture_config=CaptureConfig.full())
    adapter.connect()
    client = Client()
    adapter.connect_client(client)

    with pytest.raises(ResponseError):
        client.chat(model="nope", messages=[{"role": "user", "content": "hi"}])

    invokes = _by_type(sink.events, "model.invoke")
    violations = _by_type(sink.events, "policy.violation")
    assert any(
        "model not found" in str(e["payload"].get("error", "")) for e in invokes
    )
    assert any(v["payload"]["provider"] == "ollama" for v in violations)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_disconnect_restores_originals() -> None:
    """``disconnect`` removes the wrappers; later attribute access returns
    the original class-level bound method, not the traced wrapper.

    We can't use ``is`` against the pre-connect bound method (Python
    re-binds the descriptor on every attribute access), so we identify
    the wrapper by its sentinel ``_layerlens_original`` attribute.
    """
    from ollama import Client

    adapter = OllamaAdapter()
    adapter.connect()
    client = Client()

    adapter.connect_client(client)
    # Wrappers have the sentinel attribute.
    assert hasattr(client.chat, "_layerlens_original")
    assert hasattr(client.generate, "_layerlens_original")
    assert hasattr(client.embeddings, "_layerlens_original")

    adapter.disconnect()
    # After disconnect, the instance attributes are gone and access
    # falls through to the class-level bound method, which has no
    # sentinel marker.
    assert not hasattr(client.chat, "_layerlens_original")
    assert not hasattr(client.generate, "_layerlens_original")
    assert not hasattr(client.embeddings, "_layerlens_original")
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_dict_response_path_extract_usage() -> None:
    """The dict-response branch of ``_extract_usage`` mirrors the SDK form."""
    usage = OllamaAdapter._extract_usage(
        {"prompt_eval_count": 7, "eval_count": 11}
    )
    assert usage is not None
    assert usage.prompt_tokens == 7
    assert usage.completion_tokens == 11
    assert usage.total_tokens == 18


def test_extract_usage_handles_none_response() -> None:
    assert OllamaAdapter._extract_usage(None) is None
