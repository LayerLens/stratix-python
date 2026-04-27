"""Anthropic LLM Provider Adapter.

Wraps the Anthropic Python SDK client to intercept message completions
and streaming calls. Emits ``model.invoke``, ``cost.record``,
``tool.call``, and ``policy.violation`` events.

Ported from ``ateam/stratix/sdk/python/adapters/llm_providers/anthropic_adapter.py``.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Iterator, Optional

from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage
from layerlens.instrument.adapters.providers._base.provider import LLMProviderAdapter

logger = logging.getLogger(__name__)

_CAPTURE_PARAMS = frozenset(
    {
        "model",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "tool_choice",
    }
)


class AnthropicAdapter(LLMProviderAdapter):
    """LayerLens adapter for the Anthropic Python SDK.

    Wraps ``client.messages.create`` and ``client.messages.stream`` to
    emit ``model.invoke``, ``cost.record``, and ``tool.call`` events.

    Usage::

        from anthropic import Anthropic
        from layerlens.instrument.adapters.providers.anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter()
        adapter.connect()

        client = Anthropic()
        adapter.connect_client(client)
    """

    FRAMEWORK = "anthropic"
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Optional[CaptureConfig] = None,
        *,
        org_id: Optional[str] = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config, org_id=org_id)

    def connect_client(self, client: Any) -> Any:
        """Wrap Anthropic client methods with tracing."""
        self._client = client

        if hasattr(client, "messages"):
            original_create = client.messages.create
            self._originals["messages.create"] = original_create
            client.messages.create = self._wrap_messages_create(original_create)

            if hasattr(client.messages, "stream"):
                original_stream = client.messages.stream
                self._originals["messages.stream"] = original_stream
                client.messages.stream = self._wrap_messages_stream(original_stream)

        return client

    def _restore_originals(self) -> None:
        if self._client is None:
            return
        if "messages.create" in self._originals:
            try:
                self._client.messages.create = self._originals["messages.create"]
            except Exception:
                logger.warning("Could not restore messages.create")
        if "messages.stream" in self._originals:
            try:
                self._client.messages.stream = self._originals["messages.stream"]
            except Exception:
                logger.warning("Could not restore messages.stream")

    @staticmethod
    def _detect_framework_version() -> Optional[str]:
        try:
            import anthropic

            version = getattr(anthropic, "__version__", None)
            return str(version) if version is not None else None
        except ImportError:
            return None

    # --- Wrapping methods ---

    def _wrap_messages_create(self, original: Any) -> Any:
        adapter = self

        def traced_create(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            params = {k: kwargs[k] for k in _CAPTURE_PARAMS if k in kwargs}
            if "system" in kwargs:
                params["has_system"] = True
            tools = kwargs.get("tools")
            if tools:
                params["tools_count"] = len(tools)
            is_stream = kwargs.get("stream", False)
            start_ns = time.time_ns()

            input_messages = adapter._normalize_messages(
                kwargs.get("messages"),
                system=kwargs.get("system"),
            )

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="anthropic",
                        model=model,
                        parameters=params,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("anthropic", str(exc), model=model)
                except Exception:
                    logger.warning("Error emitting Anthropic error event", exc_info=True)
                raise

            if is_stream:
                return adapter._wrap_stream_response(response, model, params, start_ns, input_messages)

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = adapter._extract_usage(response)
                output_message = adapter._extract_output_message(response)

                metadata: Dict[str, Any] = {}
                stop_reason = getattr(response, "stop_reason", None)
                if stop_reason is not None:
                    metadata["finish_reason"] = stop_reason
                resp_id = getattr(response, "id", None)
                if resp_id is not None:
                    metadata["response_id"] = resp_id
                resp_model = getattr(response, "model", None)
                if resp_model is not None:
                    metadata["response_model"] = resp_model
                resp_usage = getattr(response, "usage", None)
                if resp_usage is not None:
                    cache_create = getattr(resp_usage, "cache_creation_input_tokens", None)
                    if cache_create is not None:
                        metadata["cache_creation_input_tokens"] = cache_create
                    cache_read = getattr(resp_usage, "cache_read_input_tokens", None)
                    if cache_read is not None:
                        metadata["cache_read_input_tokens"] = cache_read

                adapter._emit_model_invoke(
                    provider="anthropic",
                    model=model,
                    parameters=params,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    input_messages=input_messages,
                    output_message=output_message,
                    metadata=metadata if metadata else None,
                )
                adapter._emit_cost_record(model=model, usage=usage, provider="anthropic")

                tool_calls = adapter._extract_tool_use(response)
                if tool_calls:
                    adapter._emit_tool_calls(tool_calls, parent_model=model)
            except Exception:
                logger.warning("Error emitting Anthropic trace events", exc_info=True)

            return response

        traced_create._layerlens_original = original  # type: ignore[attr-defined]
        return traced_create

    def _wrap_messages_stream(self, original: Any) -> Any:
        """Wrap the ``messages.stream`` context manager."""
        adapter = self

        def traced_stream(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            params = {k: kwargs[k] for k in _CAPTURE_PARAMS if k in kwargs}
            if "system" in kwargs:
                params["has_system"] = True
            tools = kwargs.get("tools")
            if tools:
                params["tools_count"] = len(tools)
            start_ns = time.time_ns()

            input_messages = adapter._normalize_messages(
                kwargs.get("messages"),
                system=kwargs.get("system"),
            )

            try:
                stream_ctx = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="anthropic",
                        model=model,
                        parameters=params,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("anthropic", str(exc), model=model)
                except Exception:
                    logger.warning("Error emitting Anthropic stream error", exc_info=True)
                raise

            return _TracedStreamManager(adapter, stream_ctx, model, params, start_ns, input_messages)

        traced_stream._layerlens_original = original  # type: ignore[attr-defined]
        return traced_stream

    def _wrap_stream_response(
        self,
        stream: Any,
        model: Optional[str],
        params: Dict[str, Any],
        start_ns: int,
        input_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Any:
        """Wrap a streaming response (from ``stream=True``) iterator."""
        adapter = self
        accumulated_tool_calls: List[Dict[str, Any]] = []
        accumulated_content: List[str] = []
        final_usage: Optional[NormalizedTokenUsage] = None
        stream_finish_reason: Optional[str] = None
        stream_response_id: Optional[str] = None
        stream_response_model: Optional[str] = None

        class TracedStream:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __iter__(self) -> Iterator[Any]:
                return self

            def __next__(self) -> Any:
                try:
                    event = next(self._inner)
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
                        adapter._emit_model_invoke(
                            provider="anthropic",
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
                                provider="anthropic",
                            )
                        if accumulated_tool_calls:
                            adapter._emit_tool_calls(accumulated_tool_calls, parent_model=model)
                    except Exception:
                        logger.warning("Error emitting Anthropic stream events", exc_info=True)
                    raise

                try:
                    _process_stream_event(event)
                except Exception:
                    logger.debug("Error processing Anthropic stream event", exc_info=True)
                return event

            def __enter__(self) -> Any:
                return self

            def __exit__(self, *args: Any) -> Any:
                if hasattr(self._inner, "__exit__"):
                    return self._inner.__exit__(*args)
                return None

            def close(self) -> None:
                if hasattr(self._inner, "close"):
                    self._inner.close()

        def _process_stream_event(event: Any) -> None:
            nonlocal final_usage, stream_finish_reason, stream_response_id, stream_response_model
            event_type = getattr(event, "type", None)
            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta and getattr(delta, "type", None) == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        accumulated_content.append(text)
            if event_type == "message_delta":
                stop_reason = getattr(event, "delta", None)
                if stop_reason is not None:
                    sr = getattr(stop_reason, "stop_reason", None)
                    if sr is not None:
                        stream_finish_reason = sr
                usage_data = getattr(event, "usage", None)
                if usage_data:
                    output = getattr(usage_data, "output_tokens", 0) or 0
                    prior_prompt = final_usage.prompt_tokens if final_usage else 0
                    final_usage = NormalizedTokenUsage(
                        prompt_tokens=prior_prompt,
                        completion_tokens=output,
                        total_tokens=prior_prompt + output,
                    )
            elif event_type == "message_start":
                msg = getattr(event, "message", None)
                if msg:
                    msg_id = getattr(msg, "id", None)
                    if msg_id is not None:
                        stream_response_id = msg_id
                    msg_model = getattr(msg, "model", None)
                    if msg_model is not None:
                        stream_response_model = msg_model
                    usage_data = getattr(msg, "usage", None)
                    if usage_data:
                        final_usage = adapter._extract_usage_from_obj(usage_data)
            elif event_type == "content_block_start":
                block = getattr(event, "content_block", None)
                if block and getattr(block, "type", None) == "tool_use":
                    accumulated_tool_calls.append(
                        {
                            "name": getattr(block, "name", "unknown"),
                            "input": {},
                            "id": getattr(block, "id", None),
                            "_json_parts": [],
                        }
                    )
            elif event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta and getattr(delta, "type", None) == "input_json_delta":
                    json_str = getattr(delta, "partial_json", "")
                    if accumulated_tool_calls and json_str:
                        accumulated_tool_calls[-1]["_json_parts"].append(json_str)
            elif event_type == "content_block_stop":
                if accumulated_tool_calls and accumulated_tool_calls[-1].get("_json_parts"):
                    import json as _json

                    try:
                        full_json = "".join(accumulated_tool_calls[-1].pop("_json_parts"))
                        accumulated_tool_calls[-1]["input"] = _json.loads(full_json)
                    except Exception:
                        accumulated_tool_calls[-1].pop("_json_parts", None)

        return TracedStream(stream)

    # --- Token extraction ---

    def _extract_usage(self, response: Any) -> Optional[NormalizedTokenUsage]:
        usage = getattr(response, "usage", None)
        if not usage:
            return None
        return self._extract_usage_from_obj(usage)

    @staticmethod
    def _extract_usage_from_obj(usage: Any) -> NormalizedTokenUsage:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

        cached = getattr(usage, "cache_read_input_tokens", None)
        reasoning = getattr(usage, "thinking_tokens", None)

        return NormalizedTokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cached_tokens=cached,
            reasoning_tokens=reasoning,
        )

    @staticmethod
    def _extract_output_message(response: Any) -> Optional[Dict[str, str]]:
        """Extract the assistant output message from an Anthropic response."""
        try:
            content = getattr(response, "content", None) or []
            parts: List[str] = []
            for block in content:
                if getattr(block, "type", None) == "text":
                    parts.append(getattr(block, "text", ""))
            if parts:
                return {"role": "assistant", "content": "\n".join(parts)[:10_000]}
        except Exception:
            logger.debug("Error extracting Anthropic output message", exc_info=True)
        return None

    @staticmethod
    def _extract_tool_use(response: Any) -> List[Dict[str, Any]]:
        """Extract ``tool_use`` blocks from an Anthropic response."""
        tool_calls: List[Dict[str, Any]] = []
        try:
            content = getattr(response, "content", None) or []
            for block in content:
                if getattr(block, "type", None) == "tool_use":
                    tool_calls.append(
                        {
                            "name": getattr(block, "name", "unknown"),
                            "input": getattr(block, "input", {}),
                            "id": getattr(block, "id", None),
                        }
                    )
        except Exception:
            logger.debug("Error extracting Anthropic tool_use blocks", exc_info=True)
        return tool_calls


class _TracedStreamManager:
    """Wraps the Anthropic ``messages.stream()`` context manager."""

    def __init__(
        self,
        adapter: AnthropicAdapter,
        inner: Any,
        model: Optional[str],
        params: Dict[str, Any],
        start_ns: int,
        input_messages: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self._adapter = adapter
        self._inner = inner
        self._model = model
        self._params = params
        self._start_ns = start_ns
        self._input_messages = input_messages

    def __enter__(self) -> Any:
        stream = self._inner.__enter__()
        return self._adapter._wrap_stream_response(
            stream,
            self._model,
            self._params,
            self._start_ns,
            self._input_messages,
        )

    def __exit__(self, *args: Any) -> Any:
        return self._inner.__exit__(*args)


# Registry lazy-loading convention.
ADAPTER_CLASS = AnthropicAdapter
