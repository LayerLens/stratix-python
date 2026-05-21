"""End-to-end: embedding + vector_store adapters and benchmark importer.

- EmbeddingAdapter is wired against a (mocked) real OpenAI client shape,
  and we verify embedding.create events carry the right metadata.
- VectorStoreAdapter is exercised against a real ephemeral Chroma
  in-process collection — actual vector storage, no mocks.
- BenchmarkImporter reads real files written to tmp_path (CSV, JSONL,
  JSON-with-wrapper) and verifies the import shape.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import pytest

from layerlens.benchmarks import BenchmarkImporter
from layerlens.instrument import trace_context
from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter
from layerlens.instrument.adapters.frameworks.vector_store import VectorStoreAdapter

from .conftest import events_of, first_event

# ----------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------


class TestEmbeddingE2E:
    def _fake_openai_embed_response(self, dimensions=1536, tokens=42, n=2):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.01] * dimensions) for _ in range(n)],
            usage=SimpleNamespace(total_tokens=tokens),
            model="text-embedding-3-small",
        )

    def test_openai_embedding_event(self, client_and_uploads):
        client, uploads = client_and_uploads
        adapter = EmbeddingAdapter(client)

        fake_create = Mock(return_value=self._fake_openai_embed_response())
        fake_openai = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
        adapter.wrap_openai(fake_openai)

        with trace_context(client):
            result = fake_openai.embeddings.create(model="text-embedding-3-small", input=["a", "b"])
            assert len(result.data) == 2

        evt = first_event(uploads, "embedding.create")
        p = evt["payload"]
        assert p["provider"] == "openai"
        assert p["model"] == "text-embedding-3-small"
        assert p["batch_size"] == 2
        assert p["dimensions"] == 1536
        assert p["total_tokens"] == 42
        assert "latency_ms" in p


# ----------------------------------------------------------------------
# Vector store — real Chroma collection
# ----------------------------------------------------------------------


class TestVectorStoreE2E:
    """Use a real in-process Chroma collection. No network, no mocks."""

    def setup_method(self, method):
        chromadb = pytest.importorskip("chromadb")
        # Unique collection name per test — Chroma's rust backend keeps state
        # across EphemeralClient instances within a process, so naming a
        # collection "e2e_test" twice raises InternalError.
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self.chroma_client.create_collection(name=f"e2e_{method.__name__}")
        self.collection.add(
            ids=["d1", "d2", "d3"],
            documents=["the cat sat on the mat", "the dog barked loudly", "fish swim in water"],
            metadatas=[{"category": "animal"}, {"category": "animal"}, {"category": "animal"}],
        )

    def test_chroma_query_emits_retrieval_event(self, client_and_uploads):
        client, uploads = client_and_uploads
        adapter = VectorStoreAdapter(client)
        adapter.wrap_chroma(self.collection)

        with trace_context(client):
            result = self.collection.query(query_texts=["cat"], n_results=2)

        assert "ids" in result
        assert len(result["ids"][0]) == 2

        evt = first_event(uploads, "retrieval.query")
        p = evt["payload"]
        assert p["provider"] == "chroma"
        assert p["n_results"] == 2
        assert p["result_count"] == 2
        assert "distance_min" in p
        assert "distance_max" in p
        assert "distance_mean" in p

    def test_chroma_query_with_filter(self, client_and_uploads):
        client, uploads = client_and_uploads
        adapter = VectorStoreAdapter(client)
        adapter.wrap_chroma(self.collection)

        with trace_context(client):
            self.collection.query(
                query_texts=["animal"],
                n_results=3,
                where={"category": "animal"},
            )

        evt = first_event(uploads, "retrieval.query")
        assert evt["payload"]["has_filter"] is True

    def test_disconnect_stops_event_emission(self, client_and_uploads):
        """After disconnect, queries should NOT emit retrieval.query events
        (bound-method identity comparisons aren't stable in Python, so we
        check behaviour instead)."""
        client, uploads = client_and_uploads
        adapter = VectorStoreAdapter(client)
        adapter.wrap_chroma(self.collection)

        with trace_context(client):
            self.collection.query(query_texts=["cat"], n_results=1)
        assert events_of(uploads, "retrieval.query")  # emitted while wrapped
        uploads.clear()

        adapter.disconnect()
        with trace_context(client):
            self.collection.query(query_texts=["cat"], n_results=1)
        assert events_of(uploads, "retrieval.query") == []  # silent after disconnect


# ----------------------------------------------------------------------
# Benchmark importer — real files on disk
# ----------------------------------------------------------------------


class TestBenchmarkImporterE2E:
    def test_csv_with_schema_mapping(self, tmp_path: Path):
        path = tmp_path / "qa.csv"
        path.write_text(
            "question,answer,difficulty\n"
            "What is 2+2?,4,easy\n"
            "What's the capital of France?,Paris,easy\n"
            "Prove FLT,...,hard\n"
        )
        importer = BenchmarkImporter(imported_by="e2e")
        result = importer.import_csv(
            str(path),
            schema_mapping={"question": "prompt", "answer": "expected_output"},
            tags=["smoke"],
        )

        assert result.success
        assert result.records_imported == 3
        # Schema mapping applied
        assert all("prompt" in r for r in result.records)
        assert all("expected_output" in r for r in result.records)
        # Non-mapped column stays as-is
        assert all("difficulty" in r for r in result.records)
        # Metadata has tags + mapping recorded
        assert "smoke" in result.metadata.tags
        assert result.metadata.imported_by == "e2e"

    def test_jsonl_roundtrip_with_record_count(self, tmp_path: Path):
        path = tmp_path / "qa.jsonl"
        records_in = [{"prompt": f"q{i}", "expected_output": f"a{i}"} for i in range(25)]
        path.write_text("\n".join(json.dumps(r) for r in records_in))

        importer = BenchmarkImporter()
        result = importer.import_json(str(path))

        assert result.records_imported == 25
        assert result.records[0]["prompt"] == "q0"
        assert result.records[-1]["expected_output"] == "a24"

    def test_json_wrapper_object(self, tmp_path: Path):
        path = tmp_path / "wrapped.json"
        path.write_text(json.dumps({"records": [{"x": 1}, {"x": 2}], "version": "1.0", "source": "test"}))
        importer = BenchmarkImporter()
        result = importer.import_json(str(path))
        assert result.records == [{"x": 1}, {"x": 2}]

    def test_to_dict_is_json_serialisable(self, tmp_path: Path):
        path = tmp_path / "tiny.csv"
        path.write_text("a,b\n1,2\n")
        importer = BenchmarkImporter()
        result = importer.import_csv(str(path))
        text = json.dumps(result.to_dict(), default=str)
        roundtrip = json.loads(text)
        assert roundtrip["records_imported"] == 1
        assert roundtrip["success"] is True

    def test_imported_property_tracks_benchmarks(self, tmp_path: Path):
        importer = BenchmarkImporter()
        a = tmp_path / "a.csv"
        a.write_text("x\n1\n")
        b = tmp_path / "b.csv"
        b.write_text("y\n2\n")
        importer.import_csv(str(a))
        importer.import_csv(str(b))
        assert len(importer.imported) == 2
