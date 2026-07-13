"""Recorded-real-response *equivalent* for the vector-store adapter (LAY-3614, G5).

Honest deviation from the http/boto3 replay pattern, stated up front: chromadb's
``EphemeralClient`` is a fully **in-process** engine — a real Chroma collection,
real HNSW index, real distance math — with **no network transport to record**.
There is therefore no upstream HTTP/RPC body to capture and replay; the
"recorded real response" for this adapter is the collection's own deterministic
query result. So this test runs a genuine, credential-free, offline Chroma query
(explicit vectors, explicit query) and asserts the adapter's emitted
``retrieval.query`` event carries the REAL match count and REAL distances the
Chroma engine computed — exercising the real ``collection.query`` return shape
(``{ids: [[...]], distances: [[...]], ...}``) through the real adapter parser,
which the hand-built unit doubles never combine with a real Chroma engine.

This is the honest G5 entry for vector_store: a real run, not a fabricated
fixture (which would just mirror our own assumed shape).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

chromadb = pytest.importorskip("chromadb")

from layerlens.instrument import trace_context  # noqa: E402
from layerlens.instrument.adapters.frameworks.vector_store import VectorStoreAdapter  # noqa: E402

from .conftest import find_event, capture_framework_trace  # noqa: E402


def _collection() -> Any:
    """A real, offline Chroma collection with three explicit 2-D vectors and an
    L2 index — no embedding function, no network, no creds."""
    client = chromadb.EphemeralClient()
    # Unique name: chromadb shares process-global state across EphemeralClient
    # instances, so a fixed name could collide on a re-run in the same process.
    col = client.create_collection(name=f"g5_recorded_{uuid.uuid4().hex[:8]}", metadata={"hnsw:space": "l2"})
    col.add(
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )
    return col


class TestVectorStoreRecorded:
    def test_chroma_query_over_real_engine(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        col = _collection()

        adapter = VectorStoreAdapter(mock_client)
        adapter.connect(target=col)  # auto-detects Chroma and wraps collection.query
        try:
            with trace_context(mock_client):
                result = col.query(query_embeddings=[[1.0, 0.0]], n_results=2)
        finally:
            adapter.disconnect()

        # The REAL Chroma engine ranked the two nearest vectors: exact hit "a"
        # (L2=0) then "c" (L2=1); "b" (L2=2) is excluded by n_results=2.
        assert result["ids"][0] == ["a", "c"]
        real_distances = result["distances"][0]
        assert real_distances[0] == pytest.approx(0.0)
        assert real_distances[1] == pytest.approx(1.0)

        evt = find_event(uploaded["events"], "retrieval.query")
        assert evt["payload"]["provider"] == "chroma"
        assert evt["payload"]["n_results"] == 2
        # match count + distance summary are the REAL values the engine returned.
        assert evt["payload"]["result_count"] == 2
        assert evt["payload"]["distance_min"] == pytest.approx(0.0)
        assert evt["payload"]["distance_max"] == pytest.approx(1.0)
        assert evt["payload"]["distance_mean"] == pytest.approx(0.5)
        assert evt["payload"]["latency_ms"] >= 0
