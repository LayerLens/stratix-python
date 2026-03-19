"""
AWS Bedrock LLM Provider Adapter (ADP-057)

Wraps invoke_model, invoke_model_with_response_stream, converse,
and converse_stream. Parses modelId to detect provider family
for token extraction.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from layerlens.instrument.adapters._capture import CaptureConfig
from layerlens.instrument.adapters.llm_providers.base_provider import LLMProviderAdapter
from layerlens.instrument.adapters.llm_providers.pricing import BEDROCK_PRICING
from layerlens.instrument.adapters.llm_providers.token_usage import NormalizedTokenUsage

logger = logging.getLogger(__name__)


def _detect_provider_family(model_id: str) -> str:
    """Detect the provider family from a Bedrock modelId."""
    if not model_id:
        return "unknown"
    lower = model_id.lower()
    if lower.startswith("anthropic."):
        return "anthropic"
    if lower.startswith("meta."):
        return "meta"
    if lower.startswith("cohere."):
        return "cohere"
    if lower.startswith("amazon."):
        return "amazon"
    if lower.startswith("ai21."):
        return "ai21"
    if lower.startswith("mistral."):
        return "mistral"
    return "unknown"


class AWSBedrockAdapter(LLMProviderAdapter):
    """
    STRATIX adapter for AWS Bedrock (bedrock-runtime).

    Wraps invoke_model, invoke_model_with_response_stream,
    converse, and converse_stream. Parses modelId for
    provider-specific token extraction.
    """

    FRAMEWORK = "aws_bedrock"
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)

    def connect_client(self, client: Any) -> Any:
        """Wrap a Bedrock runtime client with tracing."""
        self._client = client

        # Wrap invoke_model
        if hasattr(client, "invoke_model"):
            original = client.invoke_model
            self._originals["invoke_model"] = original
            client.invoke_model = self._wrap_invoke_model(original)

        # Wrap converse
        if hasattr(client, "converse"):
            original = client.converse
            self._originals["converse"] = original
            client.converse = self._wrap_converse(original)

        # Wrap invoke_model_with_response_stream
        if hasattr(client, "invoke_model_with_response_stream"):
            original = client.invoke_model_with_response_stream
            self._originals["invoke_model_with_response_stream"] = original
            client.invoke_model_with_response_stream = self._wrap_invoke_stream(original)

        # Wrap converse_stream
        if hasattr(client, "converse_stream"):
            original = client.converse_stream
            self._originals["converse_stream"] = original
            client.converse_stream = self._wrap_converse_stream(original)

        return client

    def _restore_originals(self) -> None:
        if self._client is None:
            return
        for method_name, original in self._originals.items():
            try:
                setattr(self._client, method_name, original)
            except Exception:
                logger.warning("Could not restore %s", method_name)

    @staticmethod
    def _detect_framework_version() -> str | None:
        try:
            import boto3
            return getattr(boto3, "__version__", None)
        except ImportError:
            return None

    def _wrap_invoke_model(self, original: Any) -> Any:
        adapter = self

        def traced_invoke(*args: Any, **kwargs: Any) -> Any:
            model_id = kwargs.get("modelId", "")
            start_ns = time.time_ns()

            # Extract input messages from body
            input_messages = adapter._extract_invoke_messages(kwargs, model_id)

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="aws_bedrock",
                        model=model_id,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={"method": "invoke_model"},
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("aws_bedrock", str(exc), model=model_id)
                except Exception:
                    logger.warning("Error emitting Bedrock error event", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                # Parse response body
                body = response.get("body")
                if body and hasattr(body, "read"):
                    body_bytes = body.read()
                    body_data = json.loads(body_bytes)
                    # Replace body with a new StreamingBody-like that can still be read
                    response["body"] = _RereadableBody(body_bytes)
                else:
                    body_data = {}

                family = _detect_provider_family(model_id)
                usage = adapter._extract_invoke_usage(body_data, family)

                output_message = adapter._extract_invoke_output(body_data, family)

                # Extract response metadata from invoke_model body
                invoke_metadata: dict[str, Any] = {"method": "invoke_model", "provider_family": family}
                if family == "anthropic":
                    sr = body_data.get("stop_reason")
                    if sr is not None:
                        invoke_metadata["finish_reason"] = sr
                    rid = body_data.get("id")
                    if rid is not None:
                        invoke_metadata["response_id"] = rid
                elif family in ("meta", "mistral"):
                    sr = body_data.get("stop_reason")
                    if sr is not None:
                        invoke_metadata["finish_reason"] = sr
                elif family == "cohere":
                    # Cohere uses finish_reason directly in generations
                    gens = body_data.get("generations", [])
                    if gens and isinstance(gens, list):
                        sr = gens[0].get("finish_reason")
                        if sr is not None:
                            invoke_metadata["finish_reason"] = sr
                else:
                    # Generic fallback
                    sr = body_data.get("stop_reason") or body_data.get("finish_reason")
                    if sr is not None:
                        invoke_metadata["finish_reason"] = sr

                adapter._emit_model_invoke(
                    provider="aws_bedrock",
                    model=model_id,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    metadata=invoke_metadata,
                    input_messages=input_messages,
                    output_message=output_message,
                )
                adapter._emit_cost_record(
                    model=model_id, usage=usage, provider="aws_bedrock",
                    pricing_table=BEDROCK_PRICING,
                )
            except Exception:
                logger.warning("Error emitting Bedrock invoke events", exc_info=True)

            return response

        traced_invoke._stratix_original = original
        return traced_invoke

    def _wrap_converse(self, original: Any) -> Any:
        adapter = self

        def traced_converse(*args: Any, **kwargs: Any) -> Any:
            model_id = kwargs.get("modelId", "")
            start_ns = time.time_ns()

            # Extract input messages from Converse API
            input_messages = adapter._normalize_messages(kwargs.get("messages"))

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="aws_bedrock",
                        model=model_id,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={"method": "converse"},
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("aws_bedrock", str(exc), model=model_id)
                except Exception:
                    logger.warning("Error emitting Bedrock converse error", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = adapter._extract_converse_usage(response)
                output_message = adapter._extract_converse_output(response)

                # Extract response metadata from converse response
                converse_metadata: dict[str, Any] = {"method": "converse"}
                stop_reason = response.get("stopReason")
                if stop_reason is not None:
                    converse_metadata["finish_reason"] = stop_reason
                resp_meta = response.get("ResponseMetadata", {})
                request_id = resp_meta.get("RequestId")
                if request_id is not None:
                    converse_metadata["response_id"] = request_id

                adapter._emit_model_invoke(
                    provider="aws_bedrock",
                    model=model_id,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    metadata=converse_metadata,
                    input_messages=input_messages,
                    output_message=output_message,
                )
                adapter._emit_cost_record(
                    model=model_id, usage=usage, provider="aws_bedrock",
                    pricing_table=BEDROCK_PRICING,
                )
            except Exception:
                logger.warning("Error emitting Bedrock converse events", exc_info=True)

            return response

        traced_converse._stratix_original = original
        return traced_converse

    def _wrap_invoke_stream(self, original: Any) -> Any:
        """Wrap invoke_model_with_response_stream.

        Note: output_message is intentionally not extracted here because the
        response is a stream — content is not available until the caller
        fully consumes the iterator, which happens after this wrapper returns.
        """
        adapter = self

        def traced_invoke_stream(*args: Any, **kwargs: Any) -> Any:
            model_id = kwargs.get("modelId", "")
            start_ns = time.time_ns()

            # Extract input messages from body
            input_messages = adapter._extract_invoke_messages(kwargs, model_id)

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="aws_bedrock",
                        model=model_id,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={"method": "invoke_model_with_response_stream"},
                        input_messages=input_messages,
                    )
                except Exception:
                    logger.warning("Error emitting Bedrock stream error", exc_info=True)
                raise

            # Emit basic event — stream content processing deferred
            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                adapter._emit_model_invoke(
                    provider="aws_bedrock",
                    model=model_id,
                    latency_ms=elapsed_ms,
                    metadata={"method": "invoke_model_with_response_stream", "streaming": True},
                    input_messages=input_messages,
                )
            except Exception:
                logger.warning("Error emitting Bedrock stream events", exc_info=True)

            return response

        traced_invoke_stream._stratix_original = original
        return traced_invoke_stream

    def _wrap_converse_stream(self, original: Any) -> Any:
        """Wrap converse_stream.

        Note: output_message is intentionally not extracted here because the
        response is a stream — content is not available until the caller
        fully consumes the iterator, which happens after this wrapper returns.
        """
        adapter = self

        def traced_converse_stream(*args: Any, **kwargs: Any) -> Any:
            model_id = kwargs.get("modelId", "")
            start_ns = time.time_ns()

            # Extract input messages from Converse API
            input_messages = adapter._normalize_messages(kwargs.get("messages"))

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="aws_bedrock",
                        model=model_id,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={"method": "converse_stream"},
                        input_messages=input_messages,
                    )
                except Exception:
                    logger.warning("Error emitting Bedrock converse_stream error", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                adapter._emit_model_invoke(
                    provider="aws_bedrock",
                    model=model_id,
                    latency_ms=elapsed_ms,
                    metadata={"method": "converse_stream", "streaming": True},
                    input_messages=input_messages,
                )
            except Exception:
                logger.warning("Error emitting Bedrock converse_stream events", exc_info=True)

            return response

        traced_converse_stream._stratix_original = original
        return traced_converse_stream

    # --- Message extraction ---

    @staticmethod
    def _extract_invoke_messages(
        kwargs: dict[str, Any],
        model_id: str,
    ) -> list[dict[str, str]] | None:
        """Extract messages from invoke_model body (JSON string) based on provider family."""
        try:
            body = kwargs.get("body")
            if not body:
                return None
            if isinstance(body, (str, bytes)):
                body_data = json.loads(body)
            elif isinstance(body, dict):
                body_data = body
            else:
                return None

            family = _detect_provider_family(model_id)
            messages: list[dict[str, str]] = []

            if family == "anthropic":
                # Anthropic Messages API format
                system = body_data.get("system", "")
                if system:
                    messages.append({"role": "system", "content": str(system)[:10_000]})
                for msg in body_data.get("messages", []):
                    if isinstance(msg, dict) and "role" in msg:
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            parts = [str(p.get("text", "")) for p in content if isinstance(p, dict) and "text" in p]
                            content = "\n".join(parts)
                        messages.append({"role": str(msg["role"]), "content": str(content)[:10_000]})
            elif family in ("meta", "mistral"):
                prompt = body_data.get("prompt", "")
                if prompt:
                    messages.append({"role": "user", "content": str(prompt)[:10_000]})
            else:
                # Try generic prompt field
                prompt = body_data.get("prompt") or body_data.get("inputText", "")
                if prompt:
                    messages.append({"role": "user", "content": str(prompt)[:10_000]})

            return messages if messages else None
        except Exception:
            logger.debug("Error extracting Bedrock invoke messages", exc_info=True)
            return None

    # --- Output extraction ---

    @staticmethod
    def _extract_invoke_output(
        body_data: dict[str, Any],
        family: str,
    ) -> dict[str, str] | None:
        """Extract output message from invoke_model response body."""
        try:
            if not body_data:
                return None

            content = ""
            if family == "anthropic":
                # Anthropic Messages API: content[0].text
                content_blocks = body_data.get("content", [])
                if content_blocks and isinstance(content_blocks, list):
                    parts = []
                    for block in content_blocks:
                        if isinstance(block, dict) and "text" in block:
                            parts.append(str(block["text"]))
                    content = "\n".join(parts)
            elif family in ("meta", "mistral"):
                content = str(body_data.get("generation", ""))
            elif family == "cohere":
                generations = body_data.get("generations", [])
                if generations and isinstance(generations, list):
                    content = str(generations[0].get("text", ""))
            elif family == "amazon":
                results = body_data.get("results", [])
                if results and isinstance(results, list):
                    content = str(results[0].get("outputText", ""))
            else:
                # Try common fields
                content = str(
                    body_data.get("generation", "")
                    or body_data.get("completion", "")
                    or body_data.get("outputText", "")
                )

            if content:
                return {"role": "assistant", "content": content[:10_000]}
            return None
        except Exception:
            logger.debug("Error extracting Bedrock invoke output", exc_info=True)
            return None

    @staticmethod
    def _extract_converse_output(response: dict[str, Any]) -> dict[str, str] | None:
        """Extract output message from Converse API response."""
        try:
            output = response.get("output", {})
            message = output.get("message", {})
            if not message:
                return None
            content_blocks = message.get("content", [])
            if not content_blocks:
                return None
            parts = []
            for block in content_blocks:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
            if parts:
                return {"role": "assistant", "content": "\n".join(parts)[:10_000]}
            return None
        except Exception:
            logger.debug("Error extracting Bedrock converse output", exc_info=True)
            return None

    # --- Token extraction ---

    @staticmethod
    def _extract_invoke_usage(
        body_data: dict[str, Any],
        family: str,
    ) -> NormalizedTokenUsage | None:
        """Extract tokens from invoke_model response body based on provider family."""
        if not body_data:
            return None

        if family == "anthropic":
            usage = body_data.get("usage", {})
            return NormalizedTokenUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            )

        if family == "meta":
            prompt = body_data.get("prompt_token_count", 0)
            completion = body_data.get("generation_token_count", 0)
            return NormalizedTokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            )

        if family == "cohere":
            meta = body_data.get("meta", {})
            tokens = meta.get("billed_units", {})
            prompt = tokens.get("input_tokens", 0)
            completion = tokens.get("output_tokens", 0)
            return NormalizedTokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            )

        # Fallback: try common field names
        prompt = body_data.get("inputTokenCount", 0) or body_data.get("prompt_tokens", 0)
        completion = body_data.get("outputTokenCount", 0) or body_data.get("completion_tokens", 0)
        if prompt or completion:
            return NormalizedTokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            )

        return None

    @staticmethod
    def _extract_converse_usage(response: dict[str, Any]) -> NormalizedTokenUsage | None:
        """Extract tokens from Converse API response."""
        usage = response.get("usage", {})
        if not usage:
            return None
        prompt = usage.get("inputTokens", 0)
        completion = usage.get("outputTokens", 0)
        return NormalizedTokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )


class _RereadableBody:
    """Allows Bedrock response body to be re-read after we consume it for tracing.

    Implements the subset of the botocore StreamingBody interface that
    callers typically use after invoke_model.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, amt: int | None = None) -> bytes:
        if amt is None:
            # Full read: always return all data and reset position
            # so the body can be re-read (matching StreamingBody semantics
            # after we've already consumed the original)
            self._pos = 0
            return self._data
        result = self._data[self._pos:self._pos + amt]
        self._pos += amt
        return result

    def iter_chunks(self, chunk_size: int = 1024):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]

    def iter_lines(self):
        for line in self._data.split(b"\n"):
            if line:
                yield line

    def close(self) -> None:
        pass

    @property
    def content_length(self) -> int:
        return len(self._data)


# Registry lazy-loading convention
ADAPTER_CLASS = AWSBedrockAdapter
