"""
Azure OpenAI LLM Provider Adapter (ADP-055)

Same wrapping as OpenAI (same SDK) with additional capture of
deployment_name, azure_endpoint, api_version, and region.
Uses Azure-specific pricing table.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from layerlens.instrument.adapters._capture import CaptureConfig
from layerlens.instrument.adapters.llm_providers.base_provider import LLMProviderAdapter
from layerlens.instrument.adapters.llm_providers.openai_adapter import OpenAIAdapter
from layerlens.instrument.adapters.llm_providers.pricing import AZURE_PRICING
from layerlens.instrument.adapters.llm_providers.token_usage import NormalizedTokenUsage

logger = logging.getLogger(__name__)

_CAPTURE_PARAMS = frozenset({
    "model", "temperature", "max_tokens", "top_p",
    "frequency_penalty", "presence_penalty", "response_format",
    "tool_choice",
})


class AzureOpenAIAdapter(LLMProviderAdapter):
    """
    STRATIX adapter for Azure OpenAI Service.

    Uses the same openai SDK but captures Azure-specific metadata
    (deployment, endpoint, region) and uses Azure pricing.
    """

    FRAMEWORK = "azure_openai"
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)
        self._azure_metadata: dict[str, Any] = {}

    @staticmethod
    def _sanitize_endpoint(url: Any) -> str | None:
        """Strip query parameters from Azure endpoint URL to prevent token leakage."""
        if url is None:
            return None
        url_str = str(url)
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url_str)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

    def connect_client(self, client: Any) -> Any:
        """Wrap Azure OpenAI client methods with tracing."""
        self._client = client

        # Capture Azure-specific metadata (sanitize endpoint to prevent token leakage)
        raw_endpoint = getattr(client, "_base_url", None) or getattr(client, "base_url", None)
        self._azure_metadata = {
            "azure_endpoint": self._sanitize_endpoint(raw_endpoint),
            "api_version": getattr(client, "_api_version", None),
        }
        # Some Azure clients have _custom_query with api-version
        custom_query = getattr(client, "_custom_query", None)
        if custom_query and isinstance(custom_query, dict):
            self._azure_metadata["api_version"] = custom_query.get("api-version", self._azure_metadata.get("api_version"))

        # Wrap chat.completions.create
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            original_create = client.chat.completions.create
            self._originals["chat.completions.create"] = original_create
            client.chat.completions.create = self._wrap_chat_create(original_create)

        # Wrap embeddings.create
        if hasattr(client, "embeddings"):
            original_embed = client.embeddings.create
            self._originals["embeddings.create"] = original_embed
            client.embeddings.create = self._wrap_embeddings_create(original_embed)

        return client

    def _restore_originals(self) -> None:
        if self._client is None:
            return
        if "chat.completions.create" in self._originals:
            try:
                self._client.chat.completions.create = self._originals["chat.completions.create"]
            except Exception:
                logger.warning("Could not restore chat.completions.create")
        if "embeddings.create" in self._originals:
            try:
                self._client.embeddings.create = self._originals["embeddings.create"]
            except Exception:
                logger.warning("Could not restore embeddings.create")

    @staticmethod
    def _detect_framework_version() -> str | None:
        try:
            import openai
            return getattr(openai, "__version__", None)
        except ImportError:
            return None

    def _wrap_chat_create(self, original: Any) -> Any:
        adapter = self

        def traced_create(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            params = {k: kwargs[k] for k in _CAPTURE_PARAMS if k in kwargs}
            start_ns = time.time_ns()

            # Extract input messages for content capture
            input_messages = adapter._normalize_messages(kwargs.get("messages"))

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="azure_openai",
                        model=model,
                        parameters=params,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata=adapter._azure_metadata,
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("azure_openai", str(exc), model=model)
                except Exception:
                    logger.warning("Error emitting Azure error event", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = OpenAIAdapter._extract_usage_from_obj(
                    getattr(response, "usage", None)
                ) if getattr(response, "usage", None) else None

                output_message = OpenAIAdapter._extract_output_message(response)

                # Extract response metadata and merge with Azure metadata
                merged_metadata = dict(adapter._azure_metadata)
                choices = getattr(response, "choices", None) or []
                if choices:
                    fr = getattr(choices[0], "finish_reason", None)
                    if fr is not None:
                        merged_metadata["finish_reason"] = fr
                resp_id = getattr(response, "id", None)
                if resp_id is not None:
                    merged_metadata["response_id"] = resp_id
                resp_model = getattr(response, "model", None)
                if resp_model is not None:
                    merged_metadata["response_model"] = resp_model
                sys_fp = getattr(response, "system_fingerprint", None)
                if sys_fp is not None:
                    merged_metadata["system_fingerprint"] = sys_fp

                adapter._emit_model_invoke(
                    provider="azure_openai",
                    model=model,
                    parameters=params,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    metadata=merged_metadata,
                    input_messages=input_messages,
                    output_message=output_message,
                )
                adapter._emit_cost_record(
                    model=model, usage=usage, provider="azure_openai",
                    pricing_table=AZURE_PRICING,
                )

                tool_calls = OpenAIAdapter._extract_tool_calls(response)
                if tool_calls:
                    adapter._emit_tool_calls(tool_calls, parent_model=model)
            except Exception:
                logger.warning("Error emitting Azure trace events", exc_info=True)

            return response

        traced_create._stratix_original = original
        return traced_create

    def _wrap_embeddings_create(self, original: Any) -> Any:
        adapter = self

        def traced_embed(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            start_ns = time.time_ns()

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="azure_openai",
                        model=model,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={**adapter._azure_metadata, "request_type": "embedding"},
                    )
                except Exception:
                    logger.warning("Error emitting Azure embedding error", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = OpenAIAdapter._extract_usage_from_obj(
                    getattr(response, "usage", None)
                ) if getattr(response, "usage", None) else None

                adapter._emit_model_invoke(
                    provider="azure_openai",
                    model=model,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    metadata={**adapter._azure_metadata, "request_type": "embedding"},
                )
                adapter._emit_cost_record(
                    model=model, usage=usage, provider="azure_openai",
                    pricing_table=AZURE_PRICING,
                )
            except Exception:
                logger.warning("Error emitting Azure embedding events", exc_info=True)

            return response

        traced_embed._stratix_original = original
        return traced_embed


# Registry lazy-loading convention
ADAPTER_CLASS = AzureOpenAIAdapter
