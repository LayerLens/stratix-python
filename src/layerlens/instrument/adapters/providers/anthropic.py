from __future__ import annotations

import time
import logging
from typing import Any, Dict, List

from ..._context import _current_collector
from ._emit_helpers import emit_llm_error, emit_llm_events
from ._base_provider import MonkeyPatchProvider

log: logging.Logger = logging.getLogger(__name__)

_CAPTURE_PARAMS = frozenset(
    {
        "model",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "system",
        "tool_choice",
        "tools",
        "stream",
        "thinking",
    }
)


class AnthropicProvider(MonkeyPatchProvider):
    """Anthropic adapter with streaming, thinking-tokens, and cache-token capture."""

    name = "anthropic"
    capture_params = _CAPTURE_PARAMS

    @staticmethod
    def extract_output(response: Any) -> Any:
        try:
            content = response.content
        except AttributeError:
            return None
        if not content:
            return None
        blocks: List[Dict[str, Any]] = []
        for block in content:
            b_type = getattr(block, "type", None)
            if b_type == "text":
                blocks.append({"type": "text", "text": getattr(block, "text", None)})
            elif b_type == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": getattr(block, "id", None),
                        "tool_name": getattr(block, "name", None),
                        "input": getattr(block, "input", None),
                    }
                )
            elif b_type == "thinking":
                blocks.append({"type": "thinking", "thinking": getattr(block, "thinking", None)})
            else:
                blocks.append({"type": b_type})
        if len(blocks) == 1 and blocks[0].get("type") == "text":
            return {"type": "text", "text": blocks[0]["text"]}
        return {"type": "message", "blocks": blocks}

    @staticmethod
    def extract_meta(response: Any) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            cache_read = _opt_int(getattr(usage, "cache_read_input_tokens", None))
            cache_creation = _opt_int(getattr(usage, "cache_creation_input_tokens", None))
            input_tokens = _opt_int(getattr(usage, "input_tokens", 0)) or 0
            output_tokens = _opt_int(getattr(usage, "output_tokens", 0)) or 0
            # Anthropic's input_tokens excludes cached reads, so we add them for a full picture.
            prompt_tokens = input_tokens + (cache_read or 0)
            thinking_tokens = _count_thinking_tokens(response)
            usage_payload: Dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            if cache_read is not None:
                usage_payload["cached_tokens"] = cache_read
                usage_payload["cache_read_input_tokens"] = cache_read
            if cache_creation is not None:
                usage_payload["cache_creation_input_tokens"] = cache_creation
            if thinking_tokens:
                usage_payload["thinking_tokens"] = thinking_tokens
                usage_payload["reasoning_tokens"] = thinking_tokens
            meta["usage"] = usage_payload
        for attr, key in (
            ("model", "response_model"),
            ("id", "response_id"),
            ("stop_reason", "stop_reason"),
            ("stop_sequence", "stop_sequence"),
            ("role", "role"),
        ):
            val = getattr(response, attr, None)
            if isinstance(val, (str, int, float, bool)):
                meta[key] = val
        return meta

    @staticmethod
    def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            content = response.content
        except AttributeError:
            return out
        for block in content or []:
            if getattr(block, "type", None) == "tool_use":
                out.append(
                    {
                        "id": getattr(block, "id", None),
                        "type": "tool_use",
                        "tool_name": getattr(block, "name", None),
                        "arguments": getattr(block, "input", None),
                    }
                )
        return out

    @staticmethod
    def aggregate_stream(chunks: list[Any]) -> Any:
        if not chunks:
            return None
        return _StreamedMessage.from_events(chunks)

    @staticmethod
    def derive_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        # Only record presence of system prompt (not content) for privacy.
        if "system" in kwargs and kwargs["system"] is not None:
            extra["has_system"] = True
        tools = kwargs.get("tools")
        if tools:
            try:
                extra["tools_count"] = len(tools)
            except TypeError:
                pass
        return extra

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target
        if hasattr(target, "messages"):
            messages = target.messages
            orig = messages.create
            self._originals["messages.create"] = orig
            messages.create = self._wrap_sync("anthropic.messages.create", orig)
            if hasattr(messages, "acreate"):
                async_orig = messages.acreate
                self._originals["messages.acreate"] = async_orig
                messages.acreate = self._wrap_async("anthropic.messages.create", async_orig)
            if hasattr(messages, "stream"):
                stream_orig = messages.stream
                self._originals["messages.stream"] = stream_orig
                messages.stream = self._wrap_messages_stream(stream_orig)
        return target

    def _wrap_messages_stream(self, original: Any) -> Any:
        """Wrap ``messages.stream(...)`` — a context manager that yields events.

        The underlying Anthropic SDK returns a ``MessageStreamManager`` whose
        ``__enter__`` yields an iterable of stream events. We return a proxy that
        accumulates events, then on ``__exit__`` aggregates them and emits the
        usual ``model.invoke`` / ``tool.call`` / ``cost.record`` events.
        """
        event_name = "anthropic.messages.stream"
        provider_self = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return original(*args, **kwargs)
            start = time.time()
            try:
                inner_manager = original(*args, **kwargs)
            except Exception as exc:
                emit_llm_error(event_name, exc, (time.time() - start) * 1000)
                raise
            return _TracedMessageStream(inner_manager, provider_self, event_name, kwargs, start)

        return wrapped


def _opt_int(val: Any) -> Any:
    """Best-effort ``int`` coercion. Returns ``None`` for non-numeric inputs."""
    if val is None:
        return None
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _count_thinking_tokens(response: Any) -> int:
    """Rough thinking-token tally: sum of thinking-block character counts / 4.

    Anthropic does not surface a dedicated thinking_tokens field today, so this
    best-effort estimate matches ateam's heuristic. When Anthropic adds the
    field, callers will see ``usage.thinking_tokens`` populated from the API.
    """
    api_reported = _opt_int(getattr(getattr(response, "usage", None), "thinking_tokens", None))
    if api_reported is not None:
        return api_reported
    try:
        content = response.content or []
    except AttributeError:
        return 0
    if not isinstance(content, (list, tuple)):
        return 0
    total_chars = 0
    for block in content:
        if getattr(block, "type", None) == "thinking":
            text = getattr(block, "thinking", "") or ""
            if isinstance(text, str):
                total_chars += len(text)
    return total_chars // 4


class _StreamedMessage:
    """Minimal response shim assembled from Anthropic streaming events."""

    class _Block:
        __slots__ = ("type", "text", "thinking", "id", "name", "input")

        def __init__(self, block_type: str):
            self.type = block_type
            self.text = "" if block_type == "text" else None
            self.thinking = "" if block_type == "thinking" else None
            self.id = None
            self.name = None
            self.input = None

    class _Usage:
        __slots__ = (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "thinking_tokens",
        )

        def __init__(self) -> None:
            self.input_tokens = 0
            self.output_tokens = 0
            self.cache_read_input_tokens: int | None = None
            self.cache_creation_input_tokens: int | None = None
            self.thinking_tokens: int | None = None

    def __init__(self) -> None:
        self.id: str | None = None
        self.model: str | None = None
        self.role: str = "assistant"
        self.stop_reason: str | None = None
        self.stop_sequence: str | None = None
        self.content: list[_StreamedMessage._Block] = []
        self.usage = _StreamedMessage._Usage()

    @classmethod
    def from_events(cls, events: list[Any]) -> "_StreamedMessage":
        msg = cls()
        current_block: _StreamedMessage._Block | None = None
        tool_args_buffer: dict[int, str] = {}
        for event in events:
            etype = getattr(event, "type", None)
            if etype == "message_start":
                message = getattr(event, "message", None)
                if message is not None:
                    msg.id = getattr(message, "id", None)
                    msg.model = getattr(message, "model", None)
                    msg.role = getattr(message, "role", "assistant") or "assistant"
                    u = getattr(message, "usage", None)
                    if u is not None:
                        msg.usage.input_tokens = getattr(u, "input_tokens", 0) or 0
                        msg.usage.cache_read_input_tokens = getattr(u, "cache_read_input_tokens", None)
                        msg.usage.cache_creation_input_tokens = getattr(u, "cache_creation_input_tokens", None)
            elif etype == "content_block_start":
                block = getattr(event, "content_block", None)
                block_type = getattr(block, "type", "text") if block is not None else "text"
                current_block = cls._Block(block_type)
                if block is not None and block_type == "tool_use":
                    current_block.id = getattr(block, "id", None)
                    current_block.name = getattr(block, "name", None)
                msg.content.append(current_block)
            elif etype == "content_block_delta":
                if current_block is None:
                    continue
                delta = getattr(event, "delta", None)
                dtype = getattr(delta, "type", None) if delta is not None else None
                if dtype == "text_delta":
                    current_block.text = (current_block.text or "") + (getattr(delta, "text", "") or "")
                elif dtype == "thinking_delta":
                    current_block.thinking = (current_block.thinking or "") + (getattr(delta, "thinking", "") or "")
                elif dtype == "input_json_delta":
                    idx = getattr(event, "index", 0) or 0
                    tool_args_buffer[idx] = tool_args_buffer.get(idx, "") + (getattr(delta, "partial_json", "") or "")
            elif etype == "message_delta":
                delta = getattr(event, "delta", None)
                if delta is not None:
                    msg.stop_reason = getattr(delta, "stop_reason", None) or msg.stop_reason
                    msg.stop_sequence = getattr(delta, "stop_sequence", None) or msg.stop_sequence
                u = getattr(event, "usage", None)
                if u is not None:
                    out_tok = getattr(u, "output_tokens", None)
                    if out_tok is not None:
                        msg.usage.output_tokens = out_tok
        # Fold tool-use JSON fragments back onto their blocks.
        tool_blocks = [b for b in msg.content if b.type == "tool_use"]
        for idx, block in enumerate(tool_blocks):
            raw = tool_args_buffer.get(idx)
            if raw:
                try:
                    import json

                    block.input = json.loads(raw)
                except (ValueError, TypeError):
                    block.input = raw
        return msg


class _TracedMessageStream:
    """Proxy context manager around ``client.messages.stream(...)``.

    Forwards attribute access and enter/exit to the inner SDK manager while
    tapping every yielded event so we can aggregate on close.
    """

    def __init__(
        self,
        inner: Any,
        provider: AnthropicProvider,
        event_name: str,
        kwargs: Dict[str, Any],
        start: float,
    ) -> None:
        self._inner = inner
        self._provider = provider
        self._event_name = event_name
        self._kwargs = kwargs
        self._start = start
        self._events: List[Any] = []
        self._stream: Any = None
        self._error: Exception | None = None

    def __enter__(self) -> "_TracedMessageStream":
        self._stream = self._inner.__enter__()
        return self

    async def __aenter__(self) -> "_TracedMessageStream":
        self._stream = await self._inner.__aenter__()
        return self

    def __iter__(self) -> Any:
        for event in self._stream:
            self._events.append(event)
            yield event

    async def __aiter__(self) -> Any:
        async for event in self._stream:
            self._events.append(event)
            yield event

    def __getattr__(self, item: str) -> Any:
        return getattr(self._stream if self._stream is not None else self._inner, item)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        result = self._inner.__exit__(exc_type, exc, tb)
        self._emit(exc)
        return result

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        result = await self._inner.__aexit__(exc_type, exc, tb)
        self._emit(exc)
        return result

    def _emit(self, exc: Exception | None) -> None:
        latency_ms = (time.time() - self._start) * 1000
        if exc is not None:
            emit_llm_error(self._event_name, exc, latency_ms)
            return
        try:
            response = AnthropicProvider.aggregate_stream(self._events)
            if response is None:
                return
            emit_llm_events(
                self._event_name,
                self._kwargs,
                response,
                AnthropicProvider.extract_output,
                AnthropicProvider.extract_meta,
                self._provider.capture_params,
                latency_ms,
                pricing_table=self._provider.pricing_table,
                extract_tool_calls=AnthropicProvider.extract_tool_calls,
                extra_params=AnthropicProvider.derive_params(self._kwargs),
            )
        except Exception:
            log.debug("Error emitting Anthropic stream events", exc_info=True)


# --- Convenience API ---


def instrument_anthropic(client: Any) -> AnthropicProvider:
    from .._registry import get, register

    existing = get("anthropic")
    if existing is not None:
        existing.disconnect()
    provider = AnthropicProvider()
    provider.connect(client)
    register("anthropic", provider)
    return provider


def uninstrument_anthropic() -> None:
    from .._registry import unregister

    unregister("anthropic")
