"""Unit tests for the EmbeddingAdapter — focused on error-aware emission.

The adapter wraps OpenAI / Cohere / SentenceTransformer client methods
and forwards them through to the underlying provider. When a wrapped
call raises (rate limit, network failure, dimension mismatch), the
adapter MUST emit a discrete ``model.error`` event before re-raising —
this is the cross-pollination #2 contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


# --- OpenAI wrapper ---------------------------------------------------------


def test_openai_wrapper_emits_model_error_on_raise() -> None:
    stratix = _RecordingStratix()
    adapter = EmbeddingAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    class _OpenAIClient:
        class _Embeddings:
            def create(self, **kwargs: Any) -> Any:
                raise RuntimeError("rate limited")

        def __init__(self) -> None:
            self.embeddings = _OpenAIClient._Embeddings()

    client = _OpenAIClient()
    adapter.wrap_openai(client)

    with pytest.raises(RuntimeError, match="rate limited"):
        client.embeddings.create(model="text-embedding-3-small", input=["hi"])

    error_events = [e for e in stratix.events if e["event_type"] == "model.error"]
    assert len(error_events) == 1
    payload = error_events[0]["payload"]
    assert payload["framework"] == "embedding"
    assert payload["provider"] == "openai"
    assert payload["model"] == "text-embedding-3-small"
    assert payload["phase"] == "embedding.create"


def test_openai_wrapper_does_not_emit_error_on_success() -> None:
    stratix = _RecordingStratix()
    adapter = EmbeddingAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    class _OpenAIClient:
        class _Embeddings:
            def create(self, **kwargs: Any) -> Any:
                return SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.1] * 1536)],
                    usage=SimpleNamespace(total_tokens=4),
                )

        def __init__(self) -> None:
            self.embeddings = _OpenAIClient._Embeddings()

    client = _OpenAIClient()
    adapter.wrap_openai(client)
    client.embeddings.create(model="text-embedding-3-small", input=["hi"])

    assert all(e["event_type"] != "model.error" for e in stratix.events)
    assert any(e["event_type"] == "embedding.create" for e in stratix.events)


# --- Cohere wrapper ---------------------------------------------------------


def test_cohere_wrapper_emits_model_error_on_raise() -> None:
    stratix = _RecordingStratix()
    adapter = EmbeddingAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    class _CohereClient:
        def embed(self, **kwargs: Any) -> Any:
            raise ConnectionError("no network")

    client = _CohereClient()
    adapter.wrap_cohere(client)

    with pytest.raises(ConnectionError):
        client.embed(model="embed-english-v3.0", texts=["hi"])

    error_events = [e for e in stratix.events if e["event_type"] == "model.error"]
    assert len(error_events) == 1
    payload = error_events[0]["payload"]
    assert payload["provider"] == "cohere"
    assert payload["exception_type"] == "ConnectionError"


# --- SentenceTransformer wrapper -------------------------------------------


def test_sentence_transformer_wrapper_emits_model_error_on_raise() -> None:
    stratix = _RecordingStratix()
    adapter = EmbeddingAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    class _STModel:
        def encode(self, sentences: Any, **kwargs: Any) -> Any:
            raise ValueError("dimension mismatch")

    model = _STModel()
    adapter.wrap_sentence_transformer(model)

    with pytest.raises(ValueError):
        model.encode(["hi", "there"])

    error_events = [e for e in stratix.events if e["event_type"] == "model.error"]
    assert len(error_events) == 1
    payload = error_events[0]["payload"]
    assert payload["provider"] == "sentence_transformers"


def test_adapter_health_remains_healthy_after_emitted_error() -> None:
    """Emitting an error event must NOT trip the circuit breaker on its own."""
    stratix = _RecordingStratix()
    adapter = EmbeddingAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    class _OpenAIClient:
        class _Embeddings:
            def create(self, **kwargs: Any) -> Any:
                raise RuntimeError("err")

        def __init__(self) -> None:
            self.embeddings = _OpenAIClient._Embeddings()

    client = _OpenAIClient()
    adapter.wrap_openai(client)
    with pytest.raises(RuntimeError):
        client.embeddings.create(model="text-embedding-3-small", input=["hi"])

    # One emit of model.error must not push the adapter into ERROR state.
    assert adapter.status == AdapterStatus.HEALTHY
