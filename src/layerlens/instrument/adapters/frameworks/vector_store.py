"""Vector-store adapter.

Wraps ``query`` / ``search`` methods on common vector databases to emit
``retrieval.query`` events with provider, top-k, filter presence,
result count, score/distance distribution, and latency.

Supported stores:

- Pinecone — ``index.query``
- Weaviate — ``collection.query.near_vector`` and ``near_text``
- Chroma — ``collection.query``

Usage::

    adapter = VectorStoreAdapter(client)
    adapter.connect()
    adapter.wrap_pinecone(pinecone_index)
    # ... use index.query(...) inside a @trace ...
    adapter.disconnect()
"""

from __future__ import annotations

import time
import inspect
import logging
from typing import Any, Dict, List, Tuple, Optional

from ..._context import _current_collector
from ._base_framework import FrameworkAdapter

log = logging.getLogger(__name__)


class VectorStoreAdapter(FrameworkAdapter):
    """Trace retrieval calls across Pinecone, Weaviate, and Chroma."""

    name = "vector_store"

    def __init__(self, client: Any, capture_config: Any = None) -> None:
        super().__init__(client, capture_config)
        # key -> (target_object, original_callable, attr_name)
        self._originals: Dict[str, Tuple[Any, Any, str]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        if target is not None:
            self._auto_wrap(target)

    def _on_disconnect(self) -> None:
        for key, (obj, original, attr) in self._originals.items():
            try:
                setattr(obj, attr, original)
            except Exception:
                log.debug("layerlens.vector_store: could not restore %s", key, exc_info=True)
        self._originals.clear()

    def _auto_wrap(self, target: Any) -> None:
        # The old heuristic ("has .query and no top-level .near_vector ->
        # Pinecone") routed BOTH Chroma and Weaviate to Pinecone: a Chroma
        # Collection and a Weaviate Collection both expose ``.query`` and neither
        # exposes a *top-level* ``.near_vector`` (Weaviate's lives on the query
        # sub-object). So a real Chroma run emitted provider='pinecone' /
        # match_count=0 and Weaviate's wrapper was never reached (LAY-3616).
        query_attr = getattr(target, "query", None)
        if query_attr is None:
            return
        # Weaviate: ``collection.query`` is an object exposing near_vector / near_text.
        if hasattr(query_attr, "near_vector") or hasattr(query_attr, "near_text"):
            self.wrap_weaviate(target)
            return
        # Chroma: a full-CRUD Collection whose ``.query`` returns dicts. Tell it
        # apart from a Pinecone Index (whose ``.query`` returns objects with
        # ``.matches``) by the chromadb module or its collection-management
        # surface (add/get/count) — Pinecone indexes have none of those.
        module_root = (type(target).__module__ or "").split(".", 1)[0]
        is_chroma = module_root == "chromadb" or (
            hasattr(target, "add") and hasattr(target, "get") and hasattr(target, "count")
        )
        if is_chroma:
            self.wrap_chroma(target)
            return
        # Pinecone Index: ``index.query`` returns an object with ``.matches``.
        self.wrap_pinecone(target)

    # ------------------------------------------------------------------
    # Public wrappers
    # ------------------------------------------------------------------

    def wrap_pinecone(self, index: Any) -> Any:
        """Wrap ``index.query`` for a Pinecone Index."""
        if not hasattr(index, "query"):
            return index
        key = f"pinecone.query.{id(index)}"
        if key in self._originals:
            return index
        original = index.query
        self._originals[key] = (index, original, "query")
        index.query = self._make_pinecone_wrapper(original)
        return index

    def wrap_chroma(self, collection: Any) -> Any:
        """Wrap ``collection.query`` for a Chroma Collection."""
        if not hasattr(collection, "query"):
            return collection
        key = f"chroma.query.{id(collection)}"
        if key in self._originals:
            return collection
        original = collection.query
        self._originals[key] = (collection, original, "query")
        collection.query = self._make_chroma_wrapper(original)
        return collection

    def wrap_weaviate(self, collection: Any) -> Any:
        """Wrap ``collection.query.near_vector`` and ``.near_text``."""
        query_obj = getattr(collection, "query", None)
        if query_obj is None:
            return collection
        for method_name in ("near_vector", "near_text"):
            if not hasattr(query_obj, method_name):
                continue
            key = f"weaviate.{method_name}.{id(query_obj)}"
            if key in self._originals:
                continue
            original = getattr(query_obj, method_name)
            self._originals[key] = (query_obj, original, method_name)
            setattr(
                query_obj,
                method_name,
                self._make_weaviate_wrapper(original, method_name),
            )
        return collection

    # ------------------------------------------------------------------
    # Wrappers
    # ------------------------------------------------------------------

    def _make_pinecone_wrapper(self, original: Any) -> Any:
        adapter = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return original(*args, **kwargs)
            top_k = kwargs.get("top_k", 10)
            has_filter = bool(kwargs.get("filter"))
            namespace = kwargs.get("namespace", "")
            start = time.monotonic()
            result = original(*args, **kwargs)
            latency_ms = (time.monotonic() - start) * 1000

            matches = getattr(result, "matches", None) or []
            scores = _collect_scores(matches)

            adapter._emit(
                "retrieval.query",
                adapter._payload(
                    provider="pinecone",
                    top_k=top_k,
                    has_filter=has_filter,
                    namespace=namespace,
                    match_count=len(matches),
                    latency_ms=round(latency_ms, 2),
                    **_score_summary(scores, key_prefix="score"),
                ),
            )
            return result

        return wrapper

    def _make_chroma_wrapper(self, original: Any) -> Any:
        adapter = self

        def _emit(result: Any, n_results: Any, has_filter: bool, latency_ms: float) -> None:
            result_count, distances = _chroma_result_stats(result)
            adapter._emit(
                "retrieval.query",
                adapter._payload(
                    provider="chroma",
                    n_results=n_results,
                    has_filter=has_filter,
                    result_count=result_count,
                    latency_ms=round(latency_ms, 2),
                    **_score_summary(distances, key_prefix="distance"),
                ),
            )

        # chromadb's async client (AsyncHttpClient) exposes ``query`` as a
        # coroutine; a sync wrapper would hand the caller an un-awaited coroutine
        # and record result_count=0. Install a matching async wrapper instead.
        if inspect.iscoroutinefunction(original):

            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                if _current_collector.get() is None:
                    return await original(*args, **kwargs)
                n_results = kwargs.get("n_results", 10)
                has_filter = bool(kwargs.get("where"))
                start = time.monotonic()
                result = await original(*args, **kwargs)
                _emit(result, n_results, has_filter, (time.monotonic() - start) * 1000)
                return result

            return awrapper

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return original(*args, **kwargs)
            n_results = kwargs.get("n_results", 10)
            has_filter = bool(kwargs.get("where"))
            start = time.monotonic()
            result = original(*args, **kwargs)
            _emit(result, n_results, has_filter, (time.monotonic() - start) * 1000)
            return result

        return wrapper

    def _make_weaviate_wrapper(self, original: Any, method_name: str) -> Any:
        adapter = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return original(*args, **kwargs)
            limit = kwargs.get("limit", 10)
            start = time.monotonic()
            result = original(*args, **kwargs)
            latency_ms = (time.monotonic() - start) * 1000

            objects = getattr(result, "objects", None) or []
            adapter._emit(
                "retrieval.query",
                adapter._payload(
                    provider="weaviate",
                    query_type=method_name,
                    limit=limit,
                    result_count=len(objects),
                    latency_ms=round(latency_ms, 2),
                ),
            )
            return result

        return wrapper


def _collect_scores(matches: Any) -> List[float]:
    out: List[float] = []
    for m in matches:
        score = getattr(m, "score", None)
        if isinstance(score, (int, float)):
            out.append(float(score))
    return out


def _chroma_result_stats(result: Any) -> Tuple[int, List[float]]:
    """Chroma returns ``{ids: [[...]], distances: [[...]], ...}``."""
    if not isinstance(result, dict):
        return 0, []
    ids = result.get("ids") or [[]]
    result_count = len(ids[0]) if ids and ids[0] else 0
    dist_list = result.get("distances") or [[]]
    distances: List[float] = []
    if dist_list and dist_list[0]:
        for d in dist_list[0]:
            if isinstance(d, (int, float)):
                distances.append(float(d))
    return result_count, distances


def _score_summary(values: List[float], *, key_prefix: str) -> Dict[str, Optional[float]]:
    """Return min/max/mean rounded to 4 dp, or empty dict if no values."""
    if not values:
        return {}
    return {
        f"{key_prefix}_min": round(min(values), 4),
        f"{key_prefix}_max": round(max(values), 4),
        f"{key_prefix}_mean": round(sum(values) / len(values), 4),
    }
