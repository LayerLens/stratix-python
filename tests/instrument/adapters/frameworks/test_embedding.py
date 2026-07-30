"""Tests for the embedding-provider adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from layerlens.instrument import trace_context
from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter

from .conftest import find_event, find_events, capture_framework_trace


def _openai_result(dimensions: int = 3, total_tokens: int = 7, n: int = 1):
    data = [SimpleNamespace(embedding=[0.0] * dimensions) for _ in range(n)]
    usage = SimpleNamespace(total_tokens=total_tokens)
    return SimpleNamespace(data=data, usage=usage)


def _cohere_result(dimensions: int = 4, n: int = 2):
    return SimpleNamespace(embeddings=[[0.0] * dimensions for _ in range(n)])


class TestAdapterInfo:
    def test_name(self, mock_client):
        a = EmbeddingAdapter(mock_client)
        info = a.adapter_info()
        assert info.name == "embedding"
        assert info.adapter_type == "framework"


class TestOpenAIWrapping:
    def test_emits_embedding_create_inside_trace(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = EmbeddingAdapter(mock_client)

        fake_create = Mock(return_value=_openai_result(dimensions=1536, total_tokens=12, n=3))
        openai_client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))

        adapter.wrap_openai(openai_client)
        with trace_context(mock_client):
            openai_client.embeddings.create(model="text-embedding-3-small", input=["a", "b", "c"])

        events = uploaded["events"]
        evt = find_event(events, "embedding.create")
        assert evt["payload"]["provider"] == "openai"
        assert evt["payload"]["model"] == "text-embedding-3-small"
        assert evt["payload"]["batch_size"] == 3
        assert evt["payload"]["dimensions"] == 1536
        assert evt["payload"]["total_tokens"] == 12
        assert evt["payload"]["latency_ms"] >= 0

    def test_passthrough_outside_trace(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = EmbeddingAdapter(mock_client)

        fake_create = Mock(return_value=_openai_result())
        openai_client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
        adapter.wrap_openai(openai_client)

        # No trace_context — call should pass through silently
        result = openai_client.embeddings.create(model="x", input=["a"])
        assert result is fake_create.return_value
        # No events were captured outside an active trace context
        assert uploaded.get("events", []) == []

    def test_disconnect_restores_original(self, mock_client):
        adapter = EmbeddingAdapter(mock_client)
        fake_create = Mock(return_value=_openai_result())
        openai_client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))

        adapter.wrap_openai(openai_client)
        wrapped = openai_client.embeddings.create
        assert wrapped is not fake_create
        adapter.disconnect()
        assert openai_client.embeddings.create is fake_create

    def test_idempotent_wrap(self, mock_client):
        adapter = EmbeddingAdapter(mock_client)
        fake_create = Mock(return_value=_openai_result())
        openai_client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
        adapter.wrap_openai(openai_client)
        wrapped = openai_client.embeddings.create
        adapter.wrap_openai(openai_client)  # second wrap is a no-op
        assert openai_client.embeddings.create is wrapped


class TestCohereWrapping:
    def test_emits_embedding_create(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = EmbeddingAdapter(mock_client)

        cohere_client = SimpleNamespace(embed=Mock(return_value=_cohere_result(dimensions=1024, n=2)))
        adapter.wrap_cohere(cohere_client)
        with trace_context(mock_client):
            cohere_client.embed(model="embed-english-v3.0", texts=["a", "b"])

        evt = find_event(uploaded["events"], "embedding.create")
        assert evt["payload"]["provider"] == "cohere"
        assert evt["payload"]["dimensions"] == 1024
        assert evt["payload"]["batch_size"] == 2


class TestSentenceTransformerWrapping:
    def test_emits_with_shape_dimensions(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = EmbeddingAdapter(mock_client)

        # Fake "tensor": shape attribute + len()
        fake_result = SimpleNamespace(shape=(4, 768))
        st_model = SimpleNamespace(encode=Mock(return_value=fake_result))
        adapter.wrap_sentence_transformer(st_model)

        with trace_context(mock_client):
            st_model.encode(["s1", "s2", "s3", "s4"])

        evt = find_event(uploaded["events"], "embedding.create")
        assert evt["payload"]["provider"] == "sentence_transformers"
        assert evt["payload"]["dimensions"] == 768
        assert evt["payload"]["batch_size"] == 4

    def test_emits_with_list_dimensions(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = EmbeddingAdapter(mock_client)

        st_model = SimpleNamespace(encode=Mock(return_value=[[0.0] * 384 for _ in range(2)]))
        adapter.wrap_sentence_transformer(st_model)
        with trace_context(mock_client):
            st_model.encode(["s1", "s2"])

        evt = find_event(uploaded["events"], "embedding.create")
        assert evt["payload"]["dimensions"] == 384


class TestAutoWrap:
    def test_connect_with_openai_target_wraps_it(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = EmbeddingAdapter(mock_client)

        fake_create = Mock(return_value=_openai_result())
        openai_client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))

        adapter.connect(target=openai_client)
        with trace_context(mock_client):
            openai_client.embeddings.create(model="x", input=["a"])

        assert find_events(uploaded["events"], "embedding.create")
