"""
STRATIX Vector Store Adapter (ADP-061)

Traces retrieval operations across popular vector databases:
Pinecone, Weaviate, and Chroma. Captures query parameters,
result relevance scores, and retrieval latency.

Emits ``retrieval.query`` events with filter parameters, top-k results,
and score distributions.
"""

from __future__ import annotations

import time
import logging
from typing import Any

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters._base.truncation import DEFAULT_POLICY
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

logger = logging.getLogger(__name__)


class VectorStoreAdapter(BaseAdapter):
    """
    LayerLens adapter for vector store databases.

    Wraps query/search methods on Pinecone, Weaviate, and Chroma clients
    to emit ``retrieval.query`` events capturing filter params, top-k
    results, score distributions, and latency.

    Usage::

        from layerlens.instrument.adapters.frameworks.embedding import VectorStoreAdapter

        adapter = VectorStoreAdapter()
        adapter.connect()

        # Wrap a Pinecone index
        index = adapter.wrap_pinecone(pinecone_index)
        results = index.query(vector=[0.1, 0.2, ...], top_k=10)
    """

    FRAMEWORK = "vector_store"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/embedding/``). Pinecone/Weaviate/Chroma
    # client wrappers operate on dict / list responses; no Pydantic
    # interaction.
    requires_pydantic = PydanticCompat.V1_OR_V2

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)
        # Per-adapter wiring of the field-specific truncation policy
        # (cross-pollination audit §2.4). Vector-store query results
        # can include large content payloads (matched documents); the
        # policy caps those without dropping the structural fields
        # (scores, IDs, metadata) needed for retrieval analytics.
        self._truncation_policy = DEFAULT_POLICY
        self._originals: dict[str, Any] = {}
        self._clients: list[Any] = []

    # -- Lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        self._restore_originals()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        self._close_sinks()

    def health_check(self) -> AdapterHealth:
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            adapter_version=self.VERSION,
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="VectorStoreAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
            ],
            author="STRATIX Team",
            description="Traces vector retrieval operations across Pinecone, Weaviate, and Chroma",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="VectorStoreAdapter",
            framework=self.FRAMEWORK,
            trace_id="",
            events=list(self._trace_events),
        )

    # -- Provider wrappers -------------------------------------------------

    def wrap_pinecone(self, index: Any) -> Any:
        """Wrap a Pinecone Index's query method."""
        if hasattr(index, "query"):
            original = index.query
            self._originals["pinecone.query"] = (index, original)
            index.query = self._make_pinecone_wrapper(original)
            self._clients.append(index)
        return index

    def wrap_weaviate(self, collection: Any) -> Any:
        """Wrap a Weaviate collection's query methods."""
        if hasattr(collection, "query"):
            query_obj = collection.query
            if hasattr(query_obj, "near_vector"):
                original = query_obj.near_vector
                self._originals["weaviate.near_vector"] = (query_obj, original)
                query_obj.near_vector = self._make_weaviate_wrapper(original, "near_vector")
            if hasattr(query_obj, "near_text"):
                original = query_obj.near_text
                self._originals["weaviate.near_text"] = (query_obj, original)
                query_obj.near_text = self._make_weaviate_wrapper(original, "near_text")
            self._clients.append(collection)
        return collection

    def wrap_chroma(self, collection: Any) -> Any:
        """Wrap a Chroma Collection's query method."""
        if hasattr(collection, "query"):
            original = collection.query
            self._originals["chroma.query"] = (collection, original)
            collection.query = self._make_chroma_wrapper(original)
            self._clients.append(collection)
        return collection

    # -- Internal wrappers -------------------------------------------------

    def _make_pinecone_wrapper(self, original: Any) -> Any:
        adapter = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            top_k = kwargs.get("top_k", 10)
            has_filter = "filter" in kwargs and kwargs["filter"] is not None
            namespace = kwargs.get("namespace", "")

            start = time.monotonic()
            result = original(*args, **kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000

            # Extract score distribution from matches
            scores: list[float] = []
            match_count = 0
            if hasattr(result, "matches"):
                match_count = len(result.matches)
                scores = [m.score for m in result.matches if hasattr(m, "score")]

            adapter.emit_dict_event(
                "retrieval.query",
                {
                    "provider": "pinecone",
                    "top_k": top_k,
                    "has_filter": has_filter,
                    "namespace": namespace,
                    "match_count": match_count,
                    "score_min": round(min(scores), 4) if scores else None,
                    "score_max": round(max(scores), 4) if scores else None,
                    "score_mean": round(sum(scores) / len(scores), 4) if scores else None,
                    "latency_ms": round(elapsed_ms, 2),
                },
            )
            return result

        return wrapper

    def _make_weaviate_wrapper(self, original: Any, method_name: str) -> Any:
        adapter = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            limit = kwargs.get("limit", 10)

            start = time.monotonic()
            result = original(*args, **kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000

            result_count = 0
            if hasattr(result, "objects"):
                result_count = len(result.objects)

            adapter.emit_dict_event(
                "retrieval.query",
                {
                    "provider": "weaviate",
                    "query_type": method_name,
                    "limit": limit,
                    "result_count": result_count,
                    "latency_ms": round(elapsed_ms, 2),
                },
            )
            return result

        return wrapper

    def _make_chroma_wrapper(self, original: Any) -> Any:
        adapter = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            n_results = kwargs.get("n_results", 10)
            has_where = "where" in kwargs and kwargs["where"] is not None

            start = time.monotonic()
            result = original(*args, **kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000

            result_count = 0
            distances: list[float] = []
            if isinstance(result, dict):
                ids = result.get("ids", [[]])
                result_count = len(ids[0]) if ids and ids[0] else 0
                dist_list = result.get("distances", [[]])
                if dist_list and dist_list[0]:
                    distances = dist_list[0]

            adapter.emit_dict_event(
                "retrieval.query",
                {
                    "provider": "chroma",
                    "n_results": n_results,
                    "has_filter": has_where,
                    "result_count": result_count,
                    "distance_min": round(min(distances), 4) if distances else None,
                    "distance_max": round(max(distances), 4) if distances else None,
                    "latency_ms": round(elapsed_ms, 2),
                },
            )
            return result

        return wrapper

    # -- Cleanup -----------------------------------------------------------

    def _restore_originals(self) -> None:
        for key, (obj, original) in self._originals.items():
            try:
                if key == "pinecone.query" or key == "chroma.query":
                    obj.query = original
                elif key.startswith("weaviate."):
                    method = key.split(".", 1)[1]
                    setattr(obj, method, original)
            except Exception:
                logger.debug("Could not restore %s", key)
        self._originals.clear()
