"""Tests for FEA-1910 Embedding & Vector Store Adapters."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from layerlens.instrument.adapters._base.adapter import AdapterStatus
from layerlens.instrument.adapters.frameworks.embedding.embedding_adapter import EmbeddingAdapter
from layerlens.instrument.adapters.frameworks.embedding.vector_store_adapter import VectorStoreAdapter


class TestEmbeddingAdapter(unittest.TestCase):
    """Tests for the EmbeddingAdapter (ADP-060)."""

    def setUp(self):
        self.adapter = EmbeddingAdapter()
        self.adapter.connect()

    def tearDown(self):
        self.adapter.disconnect()

    def test_connect_sets_healthy(self):
        adapter = EmbeddingAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY
        adapter.disconnect()

    def test_disconnect_sets_disconnected(self):
        self.adapter.disconnect()
        assert not self.adapter.is_connected
        assert self.adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check(self):
        health = self.adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "embedding"

    def test_get_adapter_info(self):
        info = self.adapter.get_adapter_info()
        assert info.name == "EmbeddingAdapter"
        assert info.framework == "embedding"
        assert len(info.capabilities) > 0

    def test_serialize_for_replay(self):
        trace = self.adapter.serialize_for_replay()
        assert trace.adapter_name == "EmbeddingAdapter"
        assert trace.framework == "embedding"

    def test_wrap_openai_intercepts_create(self):
        """Verify OpenAI embeddings.create wrapper captures events."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 384)]
        mock_response.usage.total_tokens = 10
        mock_client.embeddings.create.return_value = mock_response

        wrapped = self.adapter.wrap_openai(mock_client)
        result = wrapped.embeddings.create(model="text-embedding-3-small", input=["hello"])

        assert result == mock_response
        assert len(self.adapter._trace_events) == 1
        evt = self.adapter._trace_events[0]
        assert evt["event_type"] == "embedding.create"
        payload = evt["payload"]
        assert payload["provider"] == "openai"
        assert payload["dimensions"] == 384
        assert payload["total_tokens"] == 10

    def test_wrap_cohere_intercepts_embed(self):
        """Verify Cohere embed wrapper captures events."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_response

        wrapped = self.adapter.wrap_cohere(mock_client)
        result = wrapped.embed(texts=["hello"], model="embed-english-v3.0")

        assert result == mock_response
        assert len(self.adapter._trace_events) == 1
        payload = self.adapter._trace_events[0]["payload"]
        assert payload["provider"] == "cohere"
        assert payload["dimensions"] == 1024

    def test_wrap_sentence_transformer(self):
        """Verify SentenceTransformer encode wrapper captures events."""

        mock_model = MagicMock()

        class FakeResult:
            shape = (2, 768)

        mock_model.encode.return_value = FakeResult()

        wrapped = self.adapter.wrap_sentence_transformer(mock_model)
        wrapped.encode(["hello", "world"])

        assert len(self.adapter._trace_events) == 1
        payload = self.adapter._trace_events[0]["payload"]
        assert payload["provider"] == "sentence_transformers"
        assert payload["dimensions"] == 768
        assert payload["batch_size"] == 2


class TestVectorStoreAdapter(unittest.TestCase):
    """Tests for the VectorStoreAdapter (ADP-061)."""

    def setUp(self):
        self.adapter = VectorStoreAdapter()
        self.adapter.connect()

    def tearDown(self):
        self.adapter.disconnect()

    def test_connect_disconnect(self):
        assert self.adapter.is_connected
        self.adapter.disconnect()
        assert not self.adapter.is_connected

    def test_health_check(self):
        health = self.adapter.health_check()
        assert health.framework_name == "vector_store"

    def test_get_adapter_info(self):
        info = self.adapter.get_adapter_info()
        assert info.name == "VectorStoreAdapter"

    def test_wrap_pinecone_query(self):
        """Verify Pinecone query wrapper captures events."""
        mock_index = MagicMock()
        match1 = MagicMock(score=0.95)
        match2 = MagicMock(score=0.87)
        mock_index.query.return_value = MagicMock(matches=[match1, match2])

        wrapped = self.adapter.wrap_pinecone(mock_index)
        wrapped.query(vector=[0.1] * 384, top_k=5)

        assert len(self.adapter._trace_events) == 1
        payload = self.adapter._trace_events[0]["payload"]
        assert payload["provider"] == "pinecone"
        assert payload["match_count"] == 2
        assert payload["top_k"] == 5
        assert payload["score_min"] == 0.87
        assert payload["score_max"] == 0.95

    def test_wrap_chroma_query(self):
        """Verify Chroma query wrapper captures events."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1", "id2", "id3"]],
            "distances": [[0.1, 0.2, 0.3]],
        }

        wrapped = self.adapter.wrap_chroma(mock_collection)
        wrapped.query(query_texts=["hello"], n_results=3)

        assert len(self.adapter._trace_events) == 1
        payload = self.adapter._trace_events[0]["payload"]
        assert payload["provider"] == "chroma"
        assert payload["result_count"] == 3
        assert payload["distance_min"] == 0.1

    def test_wrap_weaviate_near_vector(self):
        """Verify Weaviate near_vector wrapper captures events."""
        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_query.near_vector.return_value = MagicMock(objects=[1, 2])
        mock_collection.query = mock_query

        wrapped = self.adapter.wrap_weaviate(mock_collection)
        wrapped.query.near_vector(near_vector=[0.1] * 384, limit=5)

        assert len(self.adapter._trace_events) == 1
        payload = self.adapter._trace_events[0]["payload"]
        assert payload["provider"] == "weaviate"
        assert payload["query_type"] == "near_vector"
        assert payload["result_count"] == 2


if __name__ == "__main__":
    unittest.main()
