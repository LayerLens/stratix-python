"""Mistral AI LLM Provider Adapter.

Wraps the Mistral Python SDK (``mistralai`` >= 1.x) to intercept
``client.chat.complete`` and ``client.chat.stream`` calls. Emits
``model.invoke``, ``cost.record``, ``tool.call``, and
``policy.violation`` events.

Fresh-built (Mistral did not have an adapter in ``ateam`` source as of
2026-04-25). Follows the OpenAI / Anthropic adapter contract.
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
        "temperature",
        "max_tokens",
        "top_p",
        "random_seed",
        "response_format",
        "tool_choice",
        "safe_prompt",
    }
)


class MistralAdapter(LLMProviderAdapter):
    """LayerLens adapter for the Mistral AI Python SDK.

    Usage::

        from mistralai import Mistral
        from layerlens.instrument.adapters.providers.mistral_adapter import MistralAdapter

        adapter = MistralAdapter()
        adapter.connect()

        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        adapter.connect_client(client)

        client.chat.complete(
            model="mistral-small",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    FRAMEWORK = "mistral"
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Optional[CaptureConfig] = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)

    def connect_client(self, client: Any) -> Any:
        """Wrap ``client.chat.complete`` and ``client.chat.stream``."""
        self._client = client

        chat = getattr(client, "chat", None)
        if chat is None:
            return client

        if hasattr(chat, "complete") and callable(chat.complete):
            original_complete = chat.complete
            self._originals["chat.complete"] = original_complete
            chat.complete = self._wrap_complete(original_complete)

        if hasattr(chat, "stream") and callable(chat.stream):
            original_stream = chat.stream
            self._originals["chat.stream"] = original_stream
            chat.stream = self._wrap_stream_method(original_stream)

        # Embedding endpoint is at ``client.embeddings.create``.
        embeddings = getattr(client, "embeddings", None)
        if embeddings is not None and hasattr(embeddings, "create"):
            original_embed = embeddings.create
            self._originals["embeddings.create"] = original_embed
            embeddings.create = self._wrap_embed(original_embed)

        return client

    def _restore_originals(self) -> None:
        if self._client is None:
            return
        chat = getattr(self._client, "chat", None)
        if chat is not None:
            if "chat.complete" in self._originals:
                try:
                    chat.complete = self._originals["chat.complete"]
                except Exception:
                    logger.warning("Could not restore chat.complete")
            if "chat.stream" in self._originals:
                try:
                    chat.stream = self._originals["chat.stream"]
                except Exception:
                    logger.warning("Could not restore chat.stream")
        embeddings = getattr(self._client, "embeddings", None)
        if embeddings is not None and "embeddings.create" in self._originals:
            try:
                embeddings.create = self._originals["embeddings.create"]
            except Exception:
                logger.warning("Could not restore embeddings.create")

    @staticmethod
    def _detect_framework_version() -> Optional[str]:
        try:
            import mistralai  # type: ignore[import-not-found,import-untyped,unused-ignore]

            version = getattr(mistralai, "__version__", None)
            return str(version) if version is not None else None
        except ImportError:
            return None

    # --- Wrapping ---

    def _wrap_complete(self, original: Any) -> Any:
        adapter = self

        def traced_complete(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            params = {k: kwargs[k] for k in _CAPTURE_PARAMS if k in kwargs}
            start_ns = time.time_ns()

            input_messages = adapter._normalize_messages(kwargs.get("messages"))

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="mistral",
                        model=model,
                        parameters=params,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("mistral", str(exc), model=model)
                except Exception:
                    logger.warning("Error emitting Mistral error event", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = adapter._extract_usage(response)
                output_message = adapter._extract_output_message(response)

                metadata: Dict[str, Any] = {}
                resp_id = getattr(response, "id", None)
                if resp_id is not None:
                    metadata["response_id"] = resp_id
                resp_model = getattr(response, "model", None)
                if resp_model is not None:
                    metadata["response_model"] = resp_model
                choices = getattr(response, "choices", None) or []
                if choices:
                    finish = getattr(choices[0], "finish_reason", None)
                    if finish is not None:
                        metadata["finish_reason"] = str(finish)

                adapter._emit_model_invoke(
                    provider="mistral",
                    model=model,
                    parameters=params,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    input_messages=input_messages,
                    output_message=output_message,
                    metadata=metadata if metadata else None,
                )
                adapter._emit_cost_record(
                    model=model,
                    usage=usage,
                    provider="mistral",
                )

                tool_calls = adapter._extract_tool_calls(response)
                if tool_calls:
                    adapter._emit_tool_calls(tool_calls, parent_model=model)
            except Exception:
                logger.warning("Error emitting Mistral trace events", exc_info=True)

            return response

        traced_complete._layerlens_original = original  # type: ignore[attr-defined]
        return traced_complete

    def _wrap_stream_method(self, original: Any) -> Any:
        """Wrap ``client.chat.stream`` to emit one consolidated event on completion."""
        adapter = self

        def traced_stream(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            params = {k: kwargs[k] for k in _CAPTURE_PARAMS if k in kwargs}
            start_ns = time.time_ns()
            input_messages = adapter._normalize_messages(kwargs.get("messages"))

            try:
                stream = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="mistral",
                        model=model,
                        parameters=params,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("mistral", str(exc), model=model)
                except Exception:
                    logger.warning("Error emitting Mistral stream error", exc_info=True)
                raise

            return _MistralTracedStream(
                adapter, stream, model, params, start_ns, input_messages
            )

        traced_stream._layerlens_original = original  # type: ignore[attr-defined]
        return traced_stream

    def _wrap_embed(self, original: Any) -> Any:
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
                        provider="mistral",
                        model=model,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={"request_type": "embedding"},
                    )
                except Exception:
                    logger.warning("Error emitting Mistral embed error", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = adapter._extract_usage(response)
                adapter._emit_model_invoke(
                    provider="mistral",
                    model=model,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    metadata={"request_type": "embedding"},
                )
                adapter._emit_cost_record(
                    model=model,
                    usage=usage,
                    provider="mistral",
                )
            except Exception:
                logger.warning("Error emitting Mistral embed events", exc_info=True)

            return response

        traced_embed._layerlens_original = original  # type: ignore[attr-defined]
        return traced_embed

    # --- Token + content extraction ---

    @staticmethod
    def _extract_usage(response: Any) -> Optional[NormalizedTokenUsage]:
        """Extract usage from a Mistral response (``response.usage``)."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", 0) or (prompt + completion)
        return NormalizedTokenUsage(
            prompt_tokens=int(prompt),
            completion_tokens=int(completion),
            total_tokens=int(total),
        )

    @staticmethod
    def _extract_output_message(response: Any) -> Optional[Dict[str, str]]:
        try:
            choices = getattr(response, "choices", None) or []
            if not choices:
                return None
            message = getattr(choices[0], "message", None)
            if message is None:
                return None
            content = getattr(message, "content", None)
            if content:
                return {"role": "assistant", "content": str(content)[:10_000]}
        except Exception:
            logger.debug("Error extracting Mistral output message", exc_info=True)
        return None

    @staticmethod
    def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
        """Extract tool_calls from a Mistral response (OpenAI-compatible shape)."""
        import json as _json

        calls: List[Dict[str, Any]] = []
        try:
            choices = getattr(response, "choices", None) or []
            if not choices:
                return calls
            message = getattr(choices[0], "message", None)
            if message is None:
                return calls
            tool_calls = getattr(message, "tool_calls", None) or []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                args_str = getattr(fn, "arguments", "{}")
                try:
                    args = _json.loads(args_str) if isinstance(args_str, str) else args_str
                except (ValueError, TypeError):
                    args = args_str
                calls.append(
                    {
                        "name": getattr(fn, "name", "unknown"),
                        "arguments": args,
                        "id": getattr(tc, "id", None),
                    }
                )
        except Exception:
            logger.debug("Error extracting Mistral tool calls", exc_info=True)
        return calls


class _MistralTracedStream:
    """Wrap a Mistral chat stream to emit one consolidated ``model.invoke``.

    The Mistral SDK's stream returns a generator of ``CompletionEvent``
    objects with ``data.choices[0].delta.content`` text fragments. We
    accumulate content and tool-call deltas, then emit on iterator
    exhaustion (``StopIteration``).
    """

    def __init__(
        self,
        adapter: MistralAdapter,
        inner: Any,
        model: Optional[str],
        params: Dict[str, Any],
        start_ns: int,
        input_messages: Optional[List[Dict[str, str]]],
    ) -> None:
        self._adapter = adapter
        self._inner = iter(inner)
        self._model = model
        self._params = params
        self._start_ns = start_ns
        self._input_messages = input_messages
        self._content: List[str] = []
        self._final_usage: Optional[NormalizedTokenUsage] = None
        self._finish_reason: Optional[str] = None
        self._response_id: Optional[str] = None

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            event = next(self._inner)
        except StopIteration:
            self._emit_consolidated()
            raise
        try:
            self._absorb_event(event)
        except Exception:
            logger.debug("Error absorbing Mistral stream event", exc_info=True)
        return event

    def _absorb_event(self, event: Any) -> None:
        data = getattr(event, "data", event)
        resp_id = getattr(data, "id", None)
        if resp_id is not None:
            self._response_id = str(resp_id)
        choices = getattr(data, "choices", None) or []
        for choice in choices:
            delta = getattr(choice, "delta", None)
            if delta is not None:
                content = getattr(delta, "content", None)
                if content:
                    self._content.append(str(content))
            finish = getattr(choice, "finish_reason", None)
            if finish is not None:
                self._finish_reason = str(finish)
        usage = getattr(data, "usage", None)
        if usage is not None:
            self._final_usage = MistralAdapter._extract_usage(data)

    def _emit_consolidated(self) -> None:
        try:
            elapsed_ms = (time.time_ns() - self._start_ns) / 1_000_000
            output_message: Optional[Dict[str, str]] = None
            if self._content:
                output_message = {
                    "role": "assistant",
                    "content": "".join(self._content)[:10_000],
                }
            metadata: Dict[str, Any] = {"streaming": True}
            if self._finish_reason:
                metadata["finish_reason"] = self._finish_reason
            if self._response_id:
                metadata["response_id"] = self._response_id
            self._adapter._emit_model_invoke(
                provider="mistral",
                model=self._model,
                parameters=self._params,
                usage=self._final_usage,
                latency_ms=elapsed_ms,
                input_messages=self._input_messages,
                output_message=output_message,
                metadata=metadata,
            )
            if self._final_usage:
                self._adapter._emit_cost_record(
                    model=self._model,
                    usage=self._final_usage,
                    provider="mistral",
                )
        except Exception:
            logger.warning("Error emitting Mistral stream events", exc_info=True)


# Registry lazy-loading convention.
ADAPTER_CLASS = MistralAdapter
