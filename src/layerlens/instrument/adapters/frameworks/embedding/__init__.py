"""
LayerLens Embedding & Vector Store Adapters (FEA-1910)

Provides adapters for tracing embedding operations and vector store queries
across popular providers and databases.
"""

from __future__ import annotations

from layerlens.instrument.adapters.frameworks.embedding.embedding_adapter import (
    ADAPTER_CLASS,
    EmbeddingAdapter,
)
from layerlens.instrument.adapters.frameworks.embedding.vector_store_adapter import VectorStoreAdapter

__all__ = [
    "ADAPTER_CLASS",
    "EmbeddingAdapter",
    "VectorStoreAdapter",
]
