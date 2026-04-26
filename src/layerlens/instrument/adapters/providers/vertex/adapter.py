"""Google Vertex AI LLM provider adapter (M3 multi-vendor port).

Wraps ``GenerativeModel.generate_content`` from either
``google.generativeai`` (legacy generative-language SDK) or
``vertexai.generative_models`` / ``vertexai.preview.generative_models``
(Vertex AI SDK). Intercepts sync, async, and streaming calls and emits
``model.invoke``, ``cost.record``, ``tool.call``, and
``policy.violation`` events.

Unlike the older ``google_vertex_adapter.py`` which is Gemini-only,
this adapter supports the three model families that share the Vertex
``generate_content`` surface:

* **Gemini** — Google's native models (e.g. ``gemini-2.5-pro``).
* **Anthropic on Vertex** — Claude via the Model Garden
  (``publishers/anthropic/models/claude-opus-4-6``).
* **Llama on Vertex** — Meta via the Model Garden
  (``publishers/meta/models/llama-3.3-70b-instruct-maas``).

Authentication is delegated to the standard Google credential chain;
see :mod:`layerlens.instrument.adapters.providers.vertex.auth` for the
two supported sources (Service-Account JSON via
``GOOGLE_APPLICATION_CREDENTIALS`` or Application Default Credentials).

Ported from
``ateam/stratix/sdk/python/adapters/llm_providers/google_vertex_adapter.py``
(~348 LOC). ``stratix.*`` → ``layerlens.*``; ``_stratix_original``
attribute is mirrored as ``_layerlens_original`` with a
backwards-compatible ``_stratix_original`` alias for callers that still
detect the wrapper through the legacy attribute name.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Iterator, Optional

from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.providers.vertex.auth import (
    detect_location,
    detect_project_id,
    detect_credential_source,
)
from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage
from layerlens.instrument.adapters.providers._base.provider import LLMProviderAdapter
from layerlens.instrument.adapters.providers.vertex.pricing import (
    VERTEX_PRICING,
    normalize_vertex_model,
)
from layerlens.instrument.adapters.providers.vertex.messages import (
    extract_usage,
    extract_output_text,
    extract_function_calls,
    normalize_vertex_contents,
)

logger = logging.getLogger(__name__)


# Provider tag emitted on ``model.invoke`` / ``cost.record`` events.
_PROVIDER = "google_vertex"


class VertexAdapter(LLMProviderAdapter):
    """LayerLens adapter for the Google Vertex AI generative SDK.

    Wraps ``GenerativeModel.generate_content`` for sync and streaming
    invocations against any of the three Vertex-hosted model families
    (Gemini, Anthropic, Llama). Extracts token usage from
    ``usage_metadata`` and function calls from
    ``candidates[0].content.parts[].function_call``.

    Usage::

        from vertexai.generative_models import GenerativeModel
        from layerlens.instrument.adapters.providers.vertex import VertexAdapter

        adapter = VertexAdapter()
        adapter.connect()

        model = GenerativeModel("gemini-2.5-pro")
        adapter.connect_client(model)

        response = model.generate_content("Why is the sky blue?")
    """

    FRAMEWORK = _PROVIDER
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Optional[CaptureConfig] = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)
        self._project_id: Optional[str] = detect_project_id()
        self._location: Optional[str] = detect_location()
        self._credential_source: str = detect_credential_source()

    # --- Lifecycle ----------------------------------------------------

    def connect_client(self, client: Any) -> Any:
        """Wrap a ``GenerativeModel`` instance with tracing.

        Args:
            client: A ``google.generativeai.GenerativeModel`` or
                ``vertexai.generative_models.GenerativeModel`` instance
                (any object exposing ``generate_content``).

        Returns:
            The same client, with ``generate_content`` replaced by a
            traced wrapper. The original is captured on
            ``self._originals["generate_content"]`` and restored by
            :meth:`disconnect`.
        """
        self._client = client

        if hasattr(client, "generate_content"):
            original = client.generate_content
            self._originals["generate_content"] = original
            client.generate_content = self._wrap_generate_content(original)

        return client

    def _restore_originals(self) -> None:
        if self._client is None:
            return
        if "generate_content" in self._originals:
            try:
                self._client.generate_content = self._originals["generate_content"]
            except Exception:
                logger.warning("Could not restore generate_content")

    @staticmethod
    def _detect_framework_version() -> Optional[str]:
        """Return the active Google SDK version, preferring vertexai."""
        try:
            import vertexai  # type: ignore[import-not-found,import-untyped,unused-ignore]

            version = getattr(vertexai, "__version__", None)
            if version is not None:
                return str(version)
        except ImportError:
            pass
        try:
            import google.generativeai as genai  # type: ignore[import-not-found,import-untyped,unused-ignore]

            version = getattr(genai, "__version__", None)
            return str(version) if version is not None else None
        except ImportError:
            return None

    # --- Wrappers -----------------------------------------------------

    def _wrap_generate_content(self, original: Any) -> Any:
        adapter = self

        def traced_generate(*args: Any, **kwargs: Any) -> Any:
            model_name = adapter._resolve_model_name()
            is_stream = bool(kwargs.get("stream", False))
            start_ns = time.time_ns()

            params = adapter._extract_generation_params(kwargs)
            input_messages = normalize_vertex_contents(
                args[0] if args else kwargs.get("contents"),
            )

            base_metadata = adapter._base_metadata(model_name)

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider=_PROVIDER,
                        model=model_name,
                        parameters=params,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        input_messages=input_messages,
                        metadata=base_metadata if base_metadata else None,
                    )
                    adapter._emit_provider_error(
                        _PROVIDER, str(exc), model=model_name
                    )
                except Exception:
                    logger.warning("Error emitting Vertex error event", exc_info=True)
                raise

            if is_stream:
                return adapter._wrap_stream(
                    response,
                    model_name,
                    params,
                    start_ns,
                    input_messages,
                    base_metadata,
                )

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = extract_usage(response)
                output_message = extract_output_text(response)

                metadata: Dict[str, Any] = dict(base_metadata)
                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    fr = getattr(candidates[0], "finish_reason", None)
                    if fr is not None:
                        fr_name = getattr(fr, "name", None)
                        metadata["finish_reason"] = (
                            fr_name if fr_name is not None else str(fr)
                        )

                adapter._emit_model_invoke(
                    provider=_PROVIDER,
                    model=model_name,
                    parameters=params,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    input_messages=input_messages,
                    output_message=output_message,
                    metadata=metadata if metadata else None,
                )
                adapter._emit_cost_record(
                    model=model_name,
                    usage=usage,
                    provider=_PROVIDER,
                    pricing_table=VERTEX_PRICING,
                )

                tool_calls = extract_function_calls(response)
                if tool_calls:
                    adapter._emit_tool_calls(tool_calls, parent_model=model_name)
            except Exception:
                logger.warning("Error emitting Vertex trace events", exc_info=True)

            return response

        # Mirror the legacy attribute name from ateam alongside the new
        # one so detection code that grew up against ``stratix.*`` still
        # works against the new ``layerlens.*`` package.
        traced_generate._layerlens_original = original  # type: ignore[attr-defined]
        traced_generate._stratix_original = original  # type: ignore[attr-defined]
        return traced_generate

    def _wrap_stream(
        self,
        stream: Any,
        model_name: Optional[str],
        params: Dict[str, Any],
        start_ns: int,
        input_messages: Optional[List[Dict[str, str]]] = None,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        adapter = self
        final_usage: Optional[NormalizedTokenUsage] = None
        stream_finish_reason: Optional[str] = None
        base_meta: Dict[str, Any] = dict(base_metadata or {})

        class TracedStream:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __iter__(self) -> Iterator[Any]:
                return self

            def __next__(self) -> Any:
                nonlocal final_usage, stream_finish_reason
                try:
                    chunk = next(self._inner)
                except StopIteration:
                    try:
                        elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                        stream_meta: Dict[str, Any] = dict(base_meta)
                        stream_meta["streaming"] = True
                        if stream_finish_reason is not None:
                            stream_meta["finish_reason"] = stream_finish_reason
                        adapter._emit_model_invoke(
                            provider=_PROVIDER,
                            model=model_name,
                            parameters=params,
                            usage=final_usage,
                            latency_ms=elapsed_ms,
                            metadata=stream_meta,
                            input_messages=input_messages,
                        )
                        if final_usage:
                            adapter._emit_cost_record(
                                model=model_name,
                                usage=final_usage,
                                provider=_PROVIDER,
                                pricing_table=VERTEX_PRICING,
                            )
                    except Exception:
                        logger.warning(
                            "Error emitting Vertex stream events", exc_info=True
                        )
                    raise

                try:
                    chunk_usage = extract_usage(chunk)
                    if chunk_usage:
                        final_usage = chunk_usage
                    chunk_candidates = getattr(chunk, "candidates", None) or []
                    if chunk_candidates:
                        fr = getattr(chunk_candidates[0], "finish_reason", None)
                        if fr is not None:
                            fr_name = getattr(fr, "name", None)
                            stream_finish_reason = (
                                fr_name if fr_name is not None else str(fr)
                            )
                except Exception:
                    logger.debug("Error extracting Vertex stream usage", exc_info=True)
                return chunk

            def __enter__(self) -> "TracedStream":
                return self

            def __exit__(self, *args: Any) -> Any:
                if hasattr(self._inner, "__exit__"):
                    return self._inner.__exit__(*args)
                if hasattr(self._inner, "close"):
                    self._inner.close()
                return None

            def close(self) -> None:
                if hasattr(self._inner, "close"):
                    self._inner.close()

        return TracedStream(stream)

    # --- Helpers ------------------------------------------------------

    def _resolve_model_name(self) -> Optional[str]:
        """Return the bare model identifier from the wrapped client.

        Strips ``models/`` and ``publishers/<vendor>/models/`` prefixes
        so the value is consistent across Gemini, Anthropic-on-Vertex,
        and Llama-on-Vertex pricing-table lookups.
        """
        raw = getattr(self._client, "model_name", None) or getattr(
            self._client, "_model_name", None
        )
        if raw is None:
            return None
        return normalize_vertex_model(str(raw))

    @staticmethod
    def _extract_generation_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Pull the standard generation knobs out of the call kwargs."""
        params: Dict[str, Any] = {}
        gen_config = kwargs.get("generation_config")
        if gen_config is None:
            return params
        if isinstance(gen_config, dict):
            for key in ("temperature", "max_output_tokens", "top_p", "top_k"):
                if key in gen_config:
                    params[key] = gen_config[key]
            return params
        # Object form (vertexai.generative_models.GenerationConfig).
        for key in ("temperature", "max_output_tokens", "top_p", "top_k"):
            value = getattr(gen_config, key, None)
            if value is not None:
                params[key] = value
        return params

    def _base_metadata(self, model_name: Optional[str]) -> Dict[str, Any]:
        """Return the per-invocation metadata block.

        Includes the credential source and (when discoverable) the
        Google Cloud project / location. These flow into the
        ``model.invoke`` event so trace consumers can correlate
        invocations with billing and IAM dashboards.
        """
        meta: Dict[str, Any] = {
            "credential_source": self._credential_source,
        }
        if self._project_id:
            meta["gcp_project"] = self._project_id
        if self._location:
            meta["gcp_location"] = self._location
        if model_name:
            meta["vendor"] = _detect_vendor(model_name)
        return meta


def _detect_vendor(model_name: str) -> str:
    """Classify a Vertex model into its publishing vendor."""
    lower = model_name.lower()
    if lower.startswith("claude") or "anthropic" in lower:
        return "anthropic"
    if lower.startswith("llama") or lower.startswith("meta"):
        return "meta"
    return "google"


# Registry lazy-loading convention.
ADAPTER_CLASS = VertexAdapter
