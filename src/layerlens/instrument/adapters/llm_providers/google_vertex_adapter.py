"""
Google Vertex AI LLM Provider Adapter (ADP-056)

Wraps GenerativeModel.generate_content to intercept sync, async, and
streaming calls. Parses function calls from response candidates.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from layerlens.instrument.adapters._capture import CaptureConfig
from layerlens.instrument.adapters.llm_providers.base_provider import LLMProviderAdapter
from layerlens.instrument.adapters.llm_providers.token_usage import NormalizedTokenUsage

logger = logging.getLogger(__name__)


class GoogleVertexAdapter(LLMProviderAdapter):
    """
    STRATIX adapter for the Google Vertex AI (Gemini) SDK.

    Wraps GenerativeModel.generate_content for sync and streaming.
    Extracts tokens from usage_metadata and function calls from candidates.
    """

    FRAMEWORK = "google_vertex"
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)

    def connect_client(self, client: Any) -> Any:
        """
        Wrap a GenerativeModel instance with tracing.

        Args:
            client: A google.generativeai.GenerativeModel or
                    vertexai.generative_models.GenerativeModel instance
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
    def _detect_framework_version() -> str | None:
        try:
            import google.generativeai as genai
            return getattr(genai, "__version__", None)
        except ImportError:
            pass
        try:
            import vertexai
            return getattr(vertexai, "__version__", None)
        except ImportError:
            return None

    def _wrap_generate_content(self, original: Any) -> Any:
        adapter = self

        def traced_generate(*args: Any, **kwargs: Any) -> Any:
            model_name = getattr(adapter._client, "model_name", None) or getattr(adapter._client, "_model_name", None)
            # Strip "models/" prefix for consistent pricing lookups
            if model_name and model_name.startswith("models/"):
                model_name = model_name[len("models/"):]
            is_stream = kwargs.get("stream", False)
            start_ns = time.time_ns()

            # Extract generation config params
            params: dict[str, Any] = {}
            gen_config = kwargs.get("generation_config")
            if gen_config:
                if hasattr(gen_config, "temperature"):
                    params["temperature"] = gen_config.temperature
                elif isinstance(gen_config, dict):
                    params = {k: gen_config[k] for k in ("temperature", "max_output_tokens", "top_p", "top_k") if k in gen_config}

            # Extract input messages for content capture
            input_messages = adapter._normalize_vertex_contents(
                args[0] if args else kwargs.get("contents"),
            )

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="google_vertex",
                        model=model_name,
                        parameters=params,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("google_vertex", str(exc), model=model_name)
                except Exception:
                    logger.warning("Error emitting Vertex error event", exc_info=True)
                raise

            if is_stream:
                return adapter._wrap_stream(response, model_name, params, start_ns, input_messages)

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = adapter._extract_usage(response)
                output_message = adapter._extract_output_text(response)

                # Extract response metadata
                metadata: dict[str, Any] = {}
                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    fr = getattr(candidates[0], "finish_reason", None)
                    if fr is not None:
                        # finish_reason is an enum in Vertex; capture its name
                        fr_name = getattr(fr, "name", None)
                        metadata["finish_reason"] = fr_name if fr_name is not None else str(fr)

                adapter._emit_model_invoke(
                    provider="google_vertex",
                    model=model_name,
                    parameters=params,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    input_messages=input_messages,
                    output_message=output_message,
                    metadata=metadata if metadata else None,
                )
                adapter._emit_cost_record(model=model_name, usage=usage, provider="google_vertex")

                tool_calls = adapter._extract_function_calls(response)
                if tool_calls:
                    adapter._emit_tool_calls(tool_calls, parent_model=model_name)
            except Exception:
                logger.warning("Error emitting Vertex trace events", exc_info=True)

            return response

        traced_generate._stratix_original = original
        return traced_generate

    def _wrap_stream(
        self,
        stream: Any,
        model_name: str | None,
        params: dict[str, Any],
        start_ns: int,
        input_messages: list[dict[str, str]] | None = None,
    ) -> Any:
        adapter = self
        final_usage: NormalizedTokenUsage | None = None
        stream_finish_reason: str | None = None

        class TracedStream:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __iter__(self):
                return self

            def __next__(self):
                nonlocal final_usage, stream_finish_reason
                try:
                    chunk = next(self._inner)
                except StopIteration:
                    try:
                        elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                        stream_meta: dict[str, Any] = {"streaming": True}
                        if stream_finish_reason is not None:
                            stream_meta["finish_reason"] = stream_finish_reason
                        adapter._emit_model_invoke(
                            provider="google_vertex",
                            model=model_name,
                            parameters=params,
                            usage=final_usage,
                            latency_ms=elapsed_ms,
                            metadata=stream_meta,
                            input_messages=input_messages,
                        )
                        if final_usage:
                            adapter._emit_cost_record(
                                model=model_name, usage=final_usage, provider="google_vertex",
                            )
                    except Exception:
                        logger.warning("Error emitting Vertex stream events", exc_info=True)
                    raise

                # Extract usage and finish_reason from each chunk (last one wins)
                try:
                    chunk_usage = adapter._extract_usage(chunk)
                    if chunk_usage:
                        final_usage = chunk_usage
                    chunk_candidates = getattr(chunk, "candidates", None) or []
                    if chunk_candidates:
                        fr = getattr(chunk_candidates[0], "finish_reason", None)
                        if fr is not None:
                            fr_name = getattr(fr, "name", None)
                            stream_finish_reason = fr_name if fr_name is not None else str(fr)
                except Exception:
                    logger.debug("Error extracting Vertex stream usage", exc_info=True)
                return chunk

        return TracedStream(stream)

    @staticmethod
    def _extract_usage(response: Any) -> NormalizedTokenUsage | None:
        """Extract token usage from Vertex response's usage_metadata."""
        metadata = getattr(response, "usage_metadata", None)
        if not metadata:
            return None
        prompt = getattr(metadata, "prompt_token_count", 0) or 0
        completion = getattr(metadata, "candidates_token_count", 0) or 0
        total = getattr(metadata, "total_token_count", 0) or (prompt + completion)
        reasoning = getattr(metadata, "thoughts_token_count", None)

        return NormalizedTokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            reasoning_tokens=reasoning,
        )

    @staticmethod
    def _normalize_vertex_contents(contents: Any) -> list[dict[str, str]] | None:
        """Normalize Vertex AI contents to [{role, content}]."""
        if contents is None:
            return None
        try:
            messages: list[dict[str, str]] = []
            # String input (single prompt)
            if isinstance(contents, str):
                messages.append({"role": "user", "content": contents[:10_000]})
                return messages
            # List of strings or Content objects
            if isinstance(contents, list):
                for item in contents:
                    if isinstance(item, str):
                        messages.append({"role": "user", "content": item[:10_000]})
                    elif hasattr(item, "role") and hasattr(item, "parts"):
                        role = str(getattr(item, "role", "user"))
                        parts_text = []
                        for part in getattr(item, "parts", []):
                            text = getattr(part, "text", None)
                            if text:
                                parts_text.append(str(text))
                        if parts_text:
                            messages.append({"role": role, "content": "\n".join(parts_text)[:10_000]})
                    elif isinstance(item, dict):
                        role = str(item.get("role", "user"))
                        parts = item.get("parts", [])
                        parts_text = []
                        for p in parts:
                            if isinstance(p, str):
                                parts_text.append(p)
                            elif isinstance(p, dict) and "text" in p:
                                parts_text.append(str(p["text"]))
                        if parts_text:
                            messages.append({"role": role, "content": "\n".join(parts_text)[:10_000]})
            return messages if messages else None
        except Exception:
            logger.debug("Error normalizing Vertex contents", exc_info=True)
            return None

    @staticmethod
    def _extract_output_text(response: Any) -> dict[str, str] | None:
        """Extract output text from Vertex response."""
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                return None
            content = getattr(candidates[0], "content", None)
            if not content:
                return None
            parts = getattr(content, "parts", None) or []
            texts = []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    texts.append(str(text))
            if texts:
                return {"role": "model", "content": "\n".join(texts)[:10_000]}
        except Exception:
            logger.debug("Error extracting Vertex output text", exc_info=True)
        return None

    @staticmethod
    def _extract_function_calls(response: Any) -> list[dict[str, Any]]:
        """Extract function calls from Vertex response candidates."""
        tool_calls = []
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                return tool_calls
            content = getattr(candidates[0], "content", None)
            if not content:
                return tool_calls
            parts = getattr(content, "parts", None) or []
            for part in parts:
                fn_call = getattr(part, "function_call", None)
                if fn_call:
                    tool_calls.append({
                        "name": getattr(fn_call, "name", "unknown"),
                        "arguments": dict(getattr(fn_call, "args", {}) or {}),
                    })
        except Exception:
            logger.debug("Error extracting Vertex function calls", exc_info=True)
        return tool_calls


# Registry lazy-loading convention
ADAPTER_CLASS = GoogleVertexAdapter
