"""OpenAI LLM Provider Adapter.

Wraps the OpenAI Python SDK client to intercept chat completions,
embeddings, and streaming calls. Emits ``model.invoke``,
``cost.record``, ``tool.call``, and ``policy.violation`` events.

Ported from ``ateam/stratix/sdk/python/adapters/llm_providers/openai_adapter.py``.
"""

from __future__ import annotations

import json
import time
import logging
from typing import Any, Dict, List, Iterator, Optional

from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage
from layerlens.instrument.adapters.providers._base.provider import LLMProviderAdapter

logger = logging.getLogger(__name__)

# Parameters to capture from request kwargs.
_CAPTURE_PARAMS = frozenset(
    {
        "model",
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "response_format",
        "tool_choice",
    }
)


class OpenAIAdapter(LLMProviderAdapter):
    """LayerLens adapter for the OpenAI Python SDK.

    Wraps ``client.chat.completions.create`` and
    ``client.embeddings.create`` to emit ``model.invoke``,
    ``cost.record``, and ``tool.call`` events.

    Usage::

        from openai import OpenAI
        from layerlens.instrument.adapters.providers.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter()
        adapter.connect()

        client = OpenAI()
        adapter.connect_client(client)

        # Now every client.chat.completions.create() call is instrumented.
    """

    FRAMEWORK = "openai"
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Optional[CaptureConfig] = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)

    def connect_client(self, client: Any) -> Any:
        """Wrap OpenAI client methods with tracing."""
        self._client = client

        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            original_create = client.chat.completions.create
            self._originals["chat.completions.create"] = original_create
            client.chat.completions.create = self._wrap_chat_create(original_create)

        if hasattr(client, "embeddings"):
            original_embed = client.embeddings.create
            self._originals["embeddings.create"] = original_embed
            client.embeddings.create = self._wrap_embeddings_create(original_embed)

        return client

    def _restore_originals(self) -> None:
        """Restore original methods on the client."""
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
    def _detect_framework_version() -> Optional[str]:
        try:
            import openai

            version = getattr(openai, "__version__", None)
            return str(version) if version is not None else None
        except ImportError:
            return None

    # --- Wrapping methods ---

    def _wrap_chat_create(self, original: Any) -> Any:
        adapter = self

        def traced_create(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            params = {k: kwargs[k] for k in _CAPTURE_PARAMS if k in kwargs}
            is_stream = kwargs.get("stream", False)
            start_ns = time.time_ns()

            input_messages = adapter._normalize_messages(kwargs.get("messages"))

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="openai",
                        model=model,
                        parameters=params,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("openai", str(exc), model=model)
                except Exception:
                    logger.warning("Error emitting OpenAI error event", exc_info=True)
                raise

            if is_stream:
                return adapter._wrap_stream(response, model, params, start_ns, input_messages)

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = adapter._extract_usage(response)
                output_message = adapter._extract_output_message(response)

                metadata: Dict[str, Any] = {}
                choices = getattr(response, "choices", None) or []
                if choices:
                    fr = getattr(choices[0], "finish_reason", None)
                    if fr is not None:
                        metadata["finish_reason"] = fr
                resp_id = getattr(response, "id", None)
                if resp_id is not None:
                    metadata["response_id"] = resp_id
                resp_model = getattr(response, "model", None)
                if resp_model is not None:
                    metadata["response_model"] = resp_model
                sys_fp = getattr(response, "system_fingerprint", None)
                if sys_fp is not None:
                    metadata["system_fingerprint"] = sys_fp
                svc_tier = getattr(response, "service_tier", None)
                if svc_tier is not None:
                    metadata["service_tier"] = svc_tier
                seed = kwargs.get("seed")
                if seed is not None:
                    metadata["seed"] = seed

                adapter._emit_model_invoke(
                    provider="openai",
                    model=model,
                    parameters=params,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    input_messages=input_messages,
                    output_message=output_message,
                    metadata=metadata if metadata else None,
                )
                adapter._emit_cost_record(model=model, usage=usage, provider="openai")

                tool_calls = adapter._extract_tool_calls(response)
                if tool_calls:
                    adapter._emit_tool_calls(tool_calls, parent_model=model)
            except Exception:
                logger.warning("Error emitting OpenAI trace events", exc_info=True)

            return response

        traced_create._layerlens_original = original  # type: ignore[attr-defined]
        return traced_create

    def _wrap_stream(
        self,
        stream: Any,
        model: Optional[str],
        params: Dict[str, Any],
        start_ns: int,
        input_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Any:
        """Wrap a streaming response to accumulate chunks and emit on completion."""
        adapter = self
        accumulated_content: List[str] = []
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
        final_usage: Optional[NormalizedTokenUsage] = None
        stream_finish_reason: Optional[str] = None
        stream_response_id: Optional[str] = None
        stream_response_model: Optional[str] = None
        stream_system_fingerprint: Optional[str] = None

        class TracedStream:
            """Wrapper that intercepts stream iteration."""

            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __iter__(self) -> Iterator[Any]:
                return self

            def __next__(self) -> Any:
                try:
                    chunk = next(self._inner)
                except StopIteration:
                    try:
                        elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                        output_msg: Optional[Dict[str, str]] = None
                        if accumulated_content:
                            output_msg = {
                                "role": "assistant",
                                "content": "".join(accumulated_content)[:10_000],
                            }
                        stream_meta: Dict[str, Any] = {"streaming": True}
                        if stream_finish_reason is not None:
                            stream_meta["finish_reason"] = stream_finish_reason
                        if stream_response_id is not None:
                            stream_meta["response_id"] = stream_response_id
                        if stream_response_model is not None:
                            stream_meta["response_model"] = stream_response_model
                        if stream_system_fingerprint is not None:
                            stream_meta["system_fingerprint"] = stream_system_fingerprint
                        adapter._emit_model_invoke(
                            provider="openai",
                            model=model,
                            parameters=params,
                            usage=final_usage,
                            latency_ms=elapsed_ms,
                            metadata=stream_meta,
                            input_messages=input_messages,
                            output_message=output_msg,
                        )
                        if final_usage:
                            adapter._emit_cost_record(
                                model=model,
                                usage=final_usage,
                                provider="openai",
                            )
                        if accumulated_tool_calls:
                            tcs = [
                                {
                                    "name": tc.get("name", ""),
                                    "arguments": tc.get("arguments", ""),
                                    "id": tc.get("id"),
                                }
                                for tc in accumulated_tool_calls.values()
                            ]
                            adapter._emit_tool_calls(tcs, parent_model=model)
                    except Exception:
                        logger.warning("Error emitting OpenAI stream events", exc_info=True)
                    raise

                try:
                    self._process_chunk(chunk)
                except Exception:
                    logger.debug("Error processing OpenAI stream chunk", exc_info=True)
                return chunk

            def _process_chunk(self, chunk: Any) -> None:
                nonlocal final_usage, stream_finish_reason, stream_response_id
                nonlocal stream_response_model, stream_system_fingerprint
                chunk_id = getattr(chunk, "id", None)
                if chunk_id is not None:
                    stream_response_id = chunk_id
                chunk_model = getattr(chunk, "model", None)
                if chunk_model is not None:
                    stream_response_model = chunk_model
                chunk_fp = getattr(chunk, "system_fingerprint", None)
                if chunk_fp is not None:
                    stream_system_fingerprint = chunk_fp
                choices = getattr(chunk, "choices", None) or []
                for choice in choices:
                    fr = getattr(choice, "finish_reason", None)
                    if fr is not None:
                        stream_finish_reason = fr
                    delta = getattr(choice, "delta", None)
                    if delta:
                        content = getattr(delta, "content", None)
                        if content:
                            accumulated_content.append(content)
                        tc_deltas = getattr(delta, "tool_calls", None) or []
                        for tc_delta in tc_deltas:
                            idx = getattr(tc_delta, "index", 0)
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {
                                    "id": getattr(tc_delta, "id", None),
                                    "name": "",
                                    "arguments": "",
                                }
                            fn = getattr(tc_delta, "function", None)
                            if fn:
                                name = getattr(fn, "name", None)
                                if name:
                                    accumulated_tool_calls[idx]["name"] = name
                                args = getattr(fn, "arguments", None)
                                if args:
                                    accumulated_tool_calls[idx]["arguments"] += args
                            tc_id = getattr(tc_delta, "id", None)
                            if tc_id:
                                accumulated_tool_calls[idx]["id"] = tc_id

                usage = getattr(chunk, "usage", None)
                if usage:
                    final_usage = adapter._extract_usage_from_obj(usage)

            def __enter__(self) -> Any:
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
                        provider="openai",
                        model=model,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={"request_type": "embedding"},
                    )
                except Exception:
                    logger.warning("Error emitting OpenAI embedding error", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = adapter._extract_usage(response)
                adapter._emit_model_invoke(
                    provider="openai",
                    model=model,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    metadata={"request_type": "embedding"},
                )
                adapter._emit_cost_record(model=model, usage=usage, provider="openai")
            except Exception:
                logger.warning("Error emitting OpenAI embedding events", exc_info=True)

            return response

        traced_embed._layerlens_original = original  # type: ignore[attr-defined]
        return traced_embed

    # --- Token extraction ---

    def _extract_usage(self, response: Any) -> Optional[NormalizedTokenUsage]:
        """Extract token usage from a synchronous OpenAI response."""
        usage = getattr(response, "usage", None)
        if not usage:
            return None
        return self._extract_usage_from_obj(usage)

    @staticmethod
    def _extract_usage_from_obj(usage: Any) -> NormalizedTokenUsage:
        """Extract :class:`NormalizedTokenUsage` from an OpenAI Usage object."""
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", 0) or (prompt + completion)

        cached: Optional[int] = None
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            cached = getattr(details, "cached_tokens", None)

        reasoning: Optional[int] = None
        comp_details = getattr(usage, "completion_tokens_details", None)
        if comp_details:
            reasoning = getattr(comp_details, "reasoning_tokens", None)

        return NormalizedTokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cached_tokens=cached,
            reasoning_tokens=reasoning,
        )

    @staticmethod
    def _extract_output_message(response: Any) -> Optional[Dict[str, str]]:
        """Extract the assistant output message from an OpenAI response."""
        try:
            choices = getattr(response, "choices", None) or []
            if not choices:
                return None
            message = getattr(choices[0], "message", None)
            if not message:
                return None
            content = getattr(message, "content", None)
            if content:
                return {"role": "assistant", "content": str(content)[:10_000]}
        except Exception:
            logger.debug("Error extracting OpenAI output message", exc_info=True)
        return None

    @staticmethod
    def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from an OpenAI response."""
        tool_calls: List[Dict[str, Any]] = []
        try:
            choices = getattr(response, "choices", None) or []
            if not choices:
                return tool_calls
            message = getattr(choices[0], "message", None)
            if not message:
                return tool_calls
            tcs = getattr(message, "tool_calls", None) or []
            for tc in tcs:
                fn = getattr(tc, "function", None)
                if fn:
                    args_str = getattr(fn, "arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args = args_str
                    tool_calls.append(
                        {
                            "name": getattr(fn, "name", "unknown"),
                            "arguments": args,
                            "id": getattr(tc, "id", None),
                        }
                    )
        except Exception:
            logger.debug("Error extracting OpenAI tool calls", exc_info=True)
        return tool_calls


# Registry lazy-loading convention.
ADAPTER_CLASS = OpenAIAdapter
