"""
STRATIX Embedding Provider Adapter (ADP-060)

Wraps embedding API calls to capture dimension tracking, batch handling,
and per-item latency. Supports OpenAI, Cohere, and HuggingFace embedding
providers.

Emits ``embedding.create`` events with dimension, token, and latency metadata.
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
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

logger = logging.getLogger(__name__)


class EmbeddingAdapter(BaseAdapter):
    """
    LayerLens adapter for embedding providers.

    Wraps embedding client ``embed()`` / ``create()`` calls to emit
    ``embedding.create`` events with dimension tracking, batch handling,
    and per-item latency.

    Supported providers:
    - OpenAI (``openai.embeddings.create``)
    - Cohere (``cohere.Client.embed``)
    - HuggingFace (``sentence_transformers.SentenceTransformer.encode``)

    Usage::

        from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter

        adapter = EmbeddingAdapter()
        adapter.connect()

        # Wrap an OpenAI client
        client = adapter.wrap_openai(openai_client)
        result = client.embeddings.create(model="text-embedding-3-small", input=["hello"])
    """

    FRAMEWORK = "embedding"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/embedding/``). The pyproject extra is
    # empty (deps come from the underlying embedding store). Adapter
    # wraps client methods structurally and emits dict events.
    requires_pydantic = PydanticCompat.V1_OR_V2

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)
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
            name="EmbeddingAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            capabilities=[
                AdapterCapability.TRACE_MODELS,
            ],
            author="STRATIX Team",
            description="Traces embedding operations across OpenAI, Cohere, and HuggingFace providers",  # noqa: E501
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="EmbeddingAdapter",
            framework=self.FRAMEWORK,
            trace_id="",
            events=list(self._trace_events),
        )

    # -- Provider wrappers -------------------------------------------------

    def wrap_openai(self, client: Any) -> Any:
        """Wrap an OpenAI client's embeddings.create method."""
        if hasattr(client, "embeddings"):
            original = client.embeddings.create
            self._originals["openai.embeddings.create"] = (client, original)
            client.embeddings.create = self._make_openai_wrapper(original)
            self._clients.append(client)
        return client

    def wrap_cohere(self, client: Any) -> Any:
        """Wrap a Cohere client's embed method."""
        if hasattr(client, "embed"):
            original = client.embed
            self._originals["cohere.embed"] = (client, original)
            client.embed = self._make_cohere_wrapper(original)
            self._clients.append(client)
        return client

    def wrap_sentence_transformer(self, model: Any) -> Any:
        """Wrap a SentenceTransformer's encode method."""
        if hasattr(model, "encode"):
            original = model.encode
            self._originals["st.encode"] = (model, original)
            model.encode = self._make_st_wrapper(original)
            self._clients.append(model)
        return model

    # -- Internal wrappers -------------------------------------------------

    def _make_openai_wrapper(self, original: Any) -> Any:
        adapter = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "unknown")
            input_data = kwargs.get("input", args[0] if args else [])
            batch_size = len(input_data) if isinstance(input_data, list) else 1

            start = time.monotonic()
            result = original(*args, **kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000

            dimensions = None
            if hasattr(result, "data") and result.data:
                first = result.data[0]
                if hasattr(first, "embedding"):
                    dimensions = len(first.embedding)

            tokens = 0
            if hasattr(result, "usage") and hasattr(result.usage, "total_tokens"):
                tokens = result.usage.total_tokens

            adapter.emit_dict_event(
                "embedding.create",
                {
                    "provider": "openai",
                    "model": model,
                    "batch_size": batch_size,
                    "dimensions": dimensions,
                    "total_tokens": tokens,
                    "latency_ms": round(elapsed_ms, 2),
                },
            )
            return result

        return wrapper

    def _make_cohere_wrapper(self, original: Any) -> Any:
        adapter = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "embed-english-v3.0")
            texts = kwargs.get("texts", args[0] if args else [])
            batch_size = len(texts) if isinstance(texts, list) else 1

            start = time.monotonic()
            result = original(*args, **kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000

            dimensions = None
            if hasattr(result, "embeddings") and result.embeddings:
                dimensions = len(result.embeddings[0])

            adapter.emit_dict_event(
                "embedding.create",
                {
                    "provider": "cohere",
                    "model": model,
                    "batch_size": batch_size,
                    "dimensions": dimensions,
                    "latency_ms": round(elapsed_ms, 2),
                },
            )
            return result

        return wrapper

    def _make_st_wrapper(self, original: Any) -> Any:
        adapter = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sentences = args[0] if args else kwargs.get("sentences", [])
            batch_size = len(sentences) if isinstance(sentences, list) else 1

            start = time.monotonic()
            result = original(*args, **kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000

            dimensions = None
            if hasattr(result, "shape") and len(result.shape) > 1:
                dimensions = result.shape[1]

            adapter.emit_dict_event(
                "embedding.create",
                {
                    "provider": "sentence_transformers",
                    "model": "local",
                    "batch_size": batch_size,
                    "dimensions": dimensions,
                    "latency_ms": round(elapsed_ms, 2),
                },
            )
            return result

        return wrapper

    # -- Cleanup -----------------------------------------------------------

    def _restore_originals(self) -> None:
        for key, (obj, original) in self._originals.items():
            try:
                if key == "openai.embeddings.create":
                    obj.embeddings.create = original
                elif key == "cohere.embed":
                    obj.embed = original
                elif key == "st.encode":
                    obj.encode = original
            except Exception:
                logger.debug("Could not restore %s", key)
        self._originals.clear()


ADAPTER_CLASS = EmbeddingAdapter
