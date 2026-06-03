"""Embedding-provider adapter.

Wraps ``embed`` / ``embeddings.create`` / ``encode`` methods on common
embedding clients to emit ``embedding.create`` events with provider,
model, batch size, vector dimensions, token usage, and latency.

Supported providers:

- OpenAI — ``client.embeddings.create``
- Cohere — ``client.embed``
- HuggingFace sentence-transformers — ``model.encode``

Usage::

    adapter = EmbeddingAdapter(client)
    adapter.connect()
    adapter.wrap_openai(openai_client)
    # ... use openai_client.embeddings.create(...) inside a @trace ...
    adapter.disconnect()
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, Tuple, Optional

from ..._context import _current_collector
from ._base_framework import FrameworkAdapter

log = logging.getLogger(__name__)


class EmbeddingAdapter(FrameworkAdapter):
    """Trace embedding calls across OpenAI, Cohere, and sentence-transformers."""

    name = "embedding"

    def __init__(self, client: Any, capture_config: Any = None) -> None:
        super().__init__(client, capture_config)
        # key -> (target_object, original_callable)
        self._originals: Dict[str, Tuple[Any, Any]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        # No required dependency at connect time; users wrap clients explicitly.
        if target is not None:
            self._auto_wrap(target)

    def _on_disconnect(self) -> None:
        for key, (obj, original) in self._originals.items():
            try:
                if key == "openai.embeddings.create":
                    obj.embeddings.create = original
                elif key == "cohere.embed":
                    obj.embed = original
                elif key == "sentence_transformers.encode":
                    obj.encode = original
            except Exception:
                log.debug("layerlens.embedding: could not restore %s", key, exc_info=True)
        self._originals.clear()

    def _auto_wrap(self, target: Any) -> None:
        """Best-effort detection — useful for ``adapter.connect(target=...)``."""
        if hasattr(target, "embeddings") and hasattr(target.embeddings, "create"):
            self.wrap_openai(target)
        elif hasattr(target, "embed"):
            self.wrap_cohere(target)
        elif hasattr(target, "encode"):
            self.wrap_sentence_transformer(target)

    # ------------------------------------------------------------------
    # Public wrappers
    # ------------------------------------------------------------------

    def wrap_openai(self, client: Any) -> Any:
        """Wrap ``client.embeddings.create``."""
        if not (hasattr(client, "embeddings") and hasattr(client.embeddings, "create")):
            return client
        if "openai.embeddings.create" in self._originals:
            return client
        original = client.embeddings.create
        self._originals["openai.embeddings.create"] = (client, original)
        client.embeddings.create = self._make_openai_wrapper(original)
        return client

    def wrap_cohere(self, client: Any) -> Any:
        """Wrap ``client.embed``."""
        if not hasattr(client, "embed"):
            return client
        if "cohere.embed" in self._originals:
            return client
        original = client.embed
        self._originals["cohere.embed"] = (client, original)
        client.embed = self._make_cohere_wrapper(original)
        return client

    def wrap_sentence_transformer(self, model: Any) -> Any:
        """Wrap ``SentenceTransformer.encode``."""
        if not hasattr(model, "encode"):
            return model
        if "sentence_transformers.encode" in self._originals:
            return model
        original = model.encode
        self._originals["sentence_transformers.encode"] = (model, original)
        model.encode = self._make_st_wrapper(original)
        return model

    # ------------------------------------------------------------------
    # Wrappers
    # ------------------------------------------------------------------

    def _make_openai_wrapper(self, original: Any) -> Any:
        adapter = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return original(*args, **kwargs)
            model = kwargs.get("model", "unknown")
            input_data = kwargs.get("input", args[0] if args else [])
            batch_size = len(input_data) if isinstance(input_data, list) else 1
            start = time.monotonic()
            result = original(*args, **kwargs)
            latency_ms = (time.monotonic() - start) * 1000

            dimensions = _extract_dimensions_openai(result)
            tokens = _extract_total_tokens(result)

            adapter._emit(
                "embedding.create",
                adapter._payload(
                    provider="openai",
                    model=model,
                    batch_size=batch_size,
                    dimensions=dimensions,
                    total_tokens=tokens,
                    latency_ms=round(latency_ms, 2),
                ),
            )
            return result

        return wrapper

    def _make_cohere_wrapper(self, original: Any) -> Any:
        adapter = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return original(*args, **kwargs)
            model = kwargs.get("model", "embed-english-v3.0")
            texts = kwargs.get("texts", args[0] if args else [])
            batch_size = len(texts) if isinstance(texts, list) else 1
            start = time.monotonic()
            result = original(*args, **kwargs)
            latency_ms = (time.monotonic() - start) * 1000

            dimensions = _extract_dimensions_cohere(result)

            adapter._emit(
                "embedding.create",
                adapter._payload(
                    provider="cohere",
                    model=model,
                    batch_size=batch_size,
                    dimensions=dimensions,
                    latency_ms=round(latency_ms, 2),
                ),
            )
            return result

        return wrapper

    def _make_st_wrapper(self, original: Any) -> Any:
        adapter = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return original(*args, **kwargs)
            sentences = args[0] if args else kwargs.get("sentences", [])
            batch_size = len(sentences) if isinstance(sentences, list) else 1
            start = time.monotonic()
            result = original(*args, **kwargs)
            latency_ms = (time.monotonic() - start) * 1000

            dimensions = _extract_dimensions_st(result)

            adapter._emit(
                "embedding.create",
                adapter._payload(
                    provider="sentence_transformers",
                    model="local",
                    batch_size=batch_size,
                    dimensions=dimensions,
                    latency_ms=round(latency_ms, 2),
                ),
            )
            return result

        return wrapper


def _extract_dimensions_openai(result: Any) -> Optional[int]:
    try:
        data = result.data
        if data:
            first = data[0]
            embedding = getattr(first, "embedding", None) or (
                first.get("embedding") if isinstance(first, dict) else None
            )
            if embedding is not None:
                return len(embedding)
    except (AttributeError, IndexError, TypeError):
        pass
    return None


def _extract_dimensions_cohere(result: Any) -> Optional[int]:
    try:
        embeddings = getattr(result, "embeddings", None) or (
            result.get("embeddings") if isinstance(result, dict) else None
        )
        if embeddings:
            return len(embeddings[0])
    except (AttributeError, IndexError, TypeError):
        pass
    return None


def _extract_dimensions_st(result: Any) -> Optional[int]:
    shape = getattr(result, "shape", None)
    if shape is not None and len(shape) > 1:
        return int(shape[1])
    # Fallback: list of lists
    if isinstance(result, list) and result and isinstance(result[0], (list, tuple)):
        return len(result[0])
    return None


def _extract_total_tokens(result: Any) -> Optional[int]:
    try:
        usage = getattr(result, "usage", None)
        if usage is None:
            return None
        total = getattr(usage, "total_tokens", None)
        if isinstance(total, int):
            return total
    except AttributeError:
        pass
    return None
