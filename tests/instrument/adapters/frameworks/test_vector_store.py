"""Tests for the vector-store adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from layerlens.instrument import trace_context
from layerlens.instrument.adapters.frameworks.vector_store import VectorStoreAdapter

from .conftest import find_event, capture_framework_trace


class TestAdapterInfo:
    def test_name(self, mock_client):
        a = VectorStoreAdapter(mock_client)
        info = a.adapter_info()
        assert info.name == "vector_store"
        assert info.adapter_type == "framework"


class TestPinecone:
    def _matches(self, scores):
        return [SimpleNamespace(score=s, id=f"id{i}") for i, s in enumerate(scores)]

    def test_emits_retrieval_query_with_score_summary(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = VectorStoreAdapter(mock_client)

        result = SimpleNamespace(matches=self._matches([0.95, 0.82, 0.75]))
        index = SimpleNamespace(query=Mock(return_value=result))
        adapter.wrap_pinecone(index)

        with trace_context(mock_client):
            index.query(vector=[0.1] * 8, top_k=5, filter={"x": 1}, namespace="ns")

        evt = find_event(uploaded["events"], "retrieval.query")
        assert evt["payload"]["provider"] == "pinecone"
        assert evt["payload"]["top_k"] == 5
        assert evt["payload"]["has_filter"] is True
        assert evt["payload"]["namespace"] == "ns"
        assert evt["payload"]["match_count"] == 3
        assert evt["payload"]["score_min"] == 0.75
        assert evt["payload"]["score_max"] == 0.95
        assert evt["payload"]["score_mean"] == 0.84

    def test_pass_through_outside_trace(self, mock_client):
        adapter = VectorStoreAdapter(mock_client)
        index = SimpleNamespace(query=Mock(return_value=SimpleNamespace(matches=[])))
        adapter.wrap_pinecone(index)
        # No active trace — should not raise, should not emit
        index.query(vector=[0.1], top_k=3)

    def test_empty_matches_omits_score_keys(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = VectorStoreAdapter(mock_client)

        result = SimpleNamespace(matches=[])
        index = SimpleNamespace(query=Mock(return_value=result))
        adapter.wrap_pinecone(index)
        with trace_context(mock_client):
            index.query(vector=[0.1], top_k=10)

        evt = find_event(uploaded["events"], "retrieval.query")
        assert "score_min" not in evt["payload"]
        assert "score_max" not in evt["payload"]
        assert evt["payload"]["match_count"] == 0

    def test_disconnect_restores_original(self, mock_client):
        adapter = VectorStoreAdapter(mock_client)
        original = Mock(return_value=SimpleNamespace(matches=[]))
        index = SimpleNamespace(query=original)
        adapter.wrap_pinecone(index)
        assert index.query is not original
        adapter.disconnect()
        assert index.query is original


class TestChroma:
    def test_emits_with_distance_summary(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = VectorStoreAdapter(mock_client)

        chroma_result = {
            "ids": [["a", "b", "c"]],
            "distances": [[0.10, 0.42, 0.99]],
            "documents": [["doc-a", "doc-b", "doc-c"]],
        }
        collection = SimpleNamespace(query=Mock(return_value=chroma_result))
        adapter.wrap_chroma(collection)

        with trace_context(mock_client):
            collection.query(query_texts=["q"], n_results=3, where={"x": 1})

        evt = find_event(uploaded["events"], "retrieval.query")
        assert evt["payload"]["provider"] == "chroma"
        assert evt["payload"]["n_results"] == 3
        assert evt["payload"]["has_filter"] is True
        assert evt["payload"]["result_count"] == 3
        assert evt["payload"]["distance_min"] == 0.1
        assert evt["payload"]["distance_max"] == 0.99


class TestWeaviate:
    def test_emits_near_vector(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = VectorStoreAdapter(mock_client)

        objects = [SimpleNamespace(uuid=f"u{i}") for i in range(5)]
        result = SimpleNamespace(objects=objects)
        near_vector = Mock(return_value=result)
        # Weaviate collection has a query object with near_vector / near_text methods.
        collection = SimpleNamespace(query=SimpleNamespace(near_vector=near_vector))
        adapter.wrap_weaviate(collection)

        with trace_context(mock_client):
            collection.query.near_vector(vector=[0.1] * 8, limit=5)

        evt = find_event(uploaded["events"], "retrieval.query")
        assert evt["payload"]["provider"] == "weaviate"
        assert evt["payload"]["query_type"] == "near_vector"
        assert evt["payload"]["result_count"] == 5
        assert evt["payload"]["limit"] == 5

    def test_emits_near_text(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = VectorStoreAdapter(mock_client)

        near_text = Mock(return_value=SimpleNamespace(objects=[]))
        # Provide a near_vector too so adapter sees the Weaviate shape;
        # we only call near_text.
        collection = SimpleNamespace(query=SimpleNamespace(near_vector=Mock(), near_text=near_text))
        adapter.wrap_weaviate(collection)

        with trace_context(mock_client):
            collection.query.near_text(query="hello", limit=2)

        evt = find_event(uploaded["events"], "retrieval.query")
        assert evt["payload"]["query_type"] == "near_text"
        assert evt["payload"]["limit"] == 2


class TestDisconnect:
    def test_disconnect_clears_all_originals(self, mock_client):
        adapter = VectorStoreAdapter(mock_client)
        # Wrap one of each
        index = SimpleNamespace(query=Mock())
        adapter.wrap_pinecone(index)
        adapter.disconnect()
        # After disconnect, originals should be empty
        assert adapter._originals == {}
