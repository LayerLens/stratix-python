"""Sample: vector-store retrieval instrumented with layerlens (offline).

Runs fully offline against an in-process Chroma ``EphemeralClient`` with
explicit vectors (no embedding-function download, no network, no creds). The
adapter wraps ``collection.query`` so the query emits a real
``retrieval.query`` event with the provider, result count, and distance
summary. Single-agent / cross-cutting: it instruments the retrieval surface,
not an agent framework.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.vector_store import VectorStoreAdapter


def main() -> None:
    try:
        import chromadb  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install chromadb")
        return

    cc = chromadb.EphemeralClient()
    coll = cc.get_or_create_collection("sample")
    # Explicit vectors keep the workload offline (no ONNX embedding download).
    coll.add(
        ids=["a", "b", "c"],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        documents=["alpha ocean", "beta river", "gamma lake"],
    )

    adapter = VectorStoreAdapter(None)
    adapter.connect(coll)
    try:
        with capture_events("vector_store_query"):
            result = coll.query(query_embeddings=[[0.15, 0.25, 0.35]], n_results=2)
            print("matched ids:", result.get("ids"))
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
