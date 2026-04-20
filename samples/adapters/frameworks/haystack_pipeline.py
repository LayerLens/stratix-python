"""Sample: Haystack pipeline — tiny QA over an in-memory doc store."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.haystack import HaystackAdapter


def main() -> None:
    try:
        from haystack import Document, Pipeline  # type: ignore[import-not-found]
        from haystack.document_stores.in_memory import InMemoryDocumentStore  # type: ignore[import-not-found]
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install 'layerlens[haystack]' haystack-ai")
        return

    store = InMemoryDocumentStore()
    store.write_documents([Document(content="Grass is green due to chlorophyll.")])

    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))

    HaystackAdapter(None).connect(pipeline)
    with capture_events("haystack_pipeline"):
        result = pipeline.run({"retriever": {"query": "Why is grass green?", "top_k": 1}})
        print("docs:", [d.content for d in result["retriever"]["documents"]])


if __name__ == "__main__":
    main()
