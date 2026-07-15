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
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                adapter._emit_embedding_error(
                    exc, (time.monotonic() - start) * 1000, provider="openai", model=model
                )
                raise
            latency_ms = (time.monotonic() - start) * 1000

            dimensions = _extract_dimensions_openai(result)
            prompt_tokens, total_tokens = _extract_openai_usage(result)

            payload = adapter._payload(
                provider="openai",
                model=model,
                batch_size=batch_size,
                dimensions=dimensions,
                total_tokens=total_tokens,
                latency_ms=round(latency_ms, 2),
            )
            # Price the call: embeddings bill every token at the input rate, so
            # the embedding.create carries its own cost_usd and a paired
            # cost.record is emitted for the platform cost rollup.
            cost_record = adapter._embedding_cost_record("openai", model, prompt_tokens, total_tokens)
            if cost_record is not None and cost_record.get("cost_usd") is not None:
                payload["cost_usd"] = cost_record["cost_usd"]
            adapter._emit("embedding.create", payload)
            if cost_record is not None:
                adapter._emit("cost.record", cost_record)
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
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                adapter._emit_embedding_error(
                    exc, (time.monotonic() - start) * 1000, provider="cohere", model=model
                )
                raise
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
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                model_id = _st_model_id(getattr(original, "__self__", None))
                adapter._emit_embedding_error(
                    exc, (time.monotonic() - start) * 1000,
                    provider="sentence_transformers", model=model_id,
                )
                raise
            latency_ms = (time.monotonic() - start) * 1000

            dimensions = _extract_dimensions_st(result)

            payload = adapter._payload(
                provider="sentence_transformers",
                batch_size=batch_size,
                dimensions=dimensions,
                latency_ms=round(latency_ms, 2),
            )
            # The real loaded model id, read from the bound instance — not the
            # hardcoded "local" placeholder (S20d). Omitted honestly if the
            # instance exposes no name.
            model_id = _st_model_id(getattr(original, "__self__", None))
            if model_id:
                payload["model"] = model_id
            adapter._emit("embedding.create", payload)
            return result

        return wrapper

    # ------------------------------------------------------------------
    # Error + cost helpers
    # ------------------------------------------------------------------

    def _emit_embedding_error(
        self, error: BaseException, latency_ms: float, *, provider: str, model: Optional[str] = None
    ) -> None:
        """Emit an ``agent.error`` for a failed embedding/vector call.

        The wrapped SDK exception still propagates verbatim (the caller re-raises);
        this records the failure so it is not silently lost. The free-text error is
        secret-scrubbed at the collector chokepoint; ``error_type``/``status`` are
        the surviving category the schema lock requires.
        """
        payload = self._payload(
            provider=provider,
            error=str(error),
            error_type=type(error).__name__,
            status="error",
            latency_ms=round(latency_ms, 2),
        )
        if model:
            payload["model"] = model
        self._emit("agent.error", payload)

    def _embedding_cost_record(
        self, provider: str, model: Optional[str], prompt_tokens: Optional[int], total_tokens: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """Build a priced ``cost.record`` payload for an embedding call, or ``None``.

        Embeddings bill every token at the model's input rate, so the billable
        count is the prompt/total token usage. Returns ``None`` when there is
        nothing to price (no model id, or no token usage — e.g. local encoders).
        ``_price_cost_record`` fills ``cost_usd`` from PRICING when the model
        resolves to a rate (unpriced/local models stay tokens-only).
        """
        if not model:
            return None
        billable = prompt_tokens if prompt_tokens is not None else total_tokens
        if billable is None:
            return None
        total = total_tokens if total_tokens is not None else billable
        payload = self._payload(
            provider=provider,
            model=model,
            prompt_tokens=int(billable),
            total_tokens=int(total),
        )
        self._price_cost_record(payload)
        return payload


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


def _st_model_id(model: Any) -> Optional[str]:
    """The honest loaded model id of a SentenceTransformer instance, or None.

    Prefers the model card's declared id/name, then the underlying transformer
    config's ``_name_or_path`` (the load name/path). Best-effort and never
    raises; returns None when the instance exposes no name (S20d).
    """
    if model is None:
        return None
    mcd = getattr(model, "model_card_data", None)
    for attr in ("model_id", "model_name", "base_model"):
        val = getattr(mcd, attr, None)
        if val:
            return str(val)
    try:
        cfg = model._first_module().auto_model.config  # type: ignore[attr-defined]
        name = getattr(cfg, "_name_or_path", None)
        if name:
            return str(name)
    except Exception:  # noqa: BLE001 — best-effort id read, never fatal
        pass
    return None


def _extract_openai_usage(result: Any) -> Tuple[Optional[int], Optional[int]]:
    """``(prompt_tokens, total_tokens)`` from an OpenAI embeddings response usage.

    For embeddings the two are equal (all tokens are input tokens); either may be
    absent depending on the response shape, so both are returned independently.
    """
    usage = getattr(result, "usage", None)
    if usage is None:
        return None, None
    prompt = getattr(usage, "prompt_tokens", None)
    total = getattr(usage, "total_tokens", None)
    prompt = prompt if isinstance(prompt, int) else None
    total = total if isinstance(total, int) else None
    return prompt, total
