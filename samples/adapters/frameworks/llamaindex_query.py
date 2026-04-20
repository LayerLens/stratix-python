"""Sample: LlamaIndex RAG query over an in-memory document."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter


def main() -> None:
    try:
        from llama_index.core import Document, VectorStoreIndex  # type: ignore[import-not-found]
        from llama_index.embeddings.openai import OpenAIEmbedding  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        print("Install: pip install 'layerlens[llamaindex]' llama-index llama-index-embeddings-openai")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to build the OpenAI embedding index.")
        return

    docs = [Document(text="Grass is green because of chlorophyll.")]
    index = VectorStoreIndex.from_documents(docs)

    LlamaIndexAdapter(None).connect(index)
    with capture_events("llamaindex_query"):
        engine = index.as_query_engine()
        resp = engine.query("Why is grass green?")
        print("reply:", resp)


if __name__ == "__main__":
    main()
