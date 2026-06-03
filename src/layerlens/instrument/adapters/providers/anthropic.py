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
        "stop_sequences",
        "stream",
    }
)
# ``system``, ``tools``, ``tool_choice``, ``thinking``, ``metadata``, and
# ``messages`` are intentionally NOT in _CAPTURE_PARAMS: their raw values
# may contain prompt content or PII. Safe summaries are emitted by
# :meth:`AnthropicProvider.derive_params` instead (LAY-3334 AC).


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

        # Content block type counts + tool-use names per LAY-3334 AC.
        try:
            content = response.content or []
        except AttributeError:
            content = []
        if isinstance(content, (list, tuple)) and content:
            block_counts: Dict[str, int] = {}
            tool_use_names: list[str] = []
            for block in content:
                b_type = getattr(block, "type", None)
                if isinstance(b_type, str):
                    block_counts[b_type] = block_counts.get(b_type, 0) + 1
                    if b_type == "tool_use":
                        name = getattr(block, "name", None)
                        if isinstance(name, str):
                            tool_use_names.append(name)
            if block_counts:
                meta["content_block_counts"] = block_counts
            if tool_use_names:
                meta["tool_use_names"] = tool_use_names
            meta["has_thinking"] = block_counts.get("thinking", 0) > 0

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
        """Build a privacy-safe summary of request kwargs per LAY-3334.

        Raw ``system``, ``messages``, ``tools``, ``tool_choice``, ``metadata``,
        and ``thinking`` payloads are NEVER returned — only counts, lengths,
        and explicitly-safe fields (e.g. ``metadata.user_id`` for cost
        attribution, ``thinking.budget_tokens``).
        """
        extra: Dict[str, Any] = {}

        # System prompt: presence + length only, never content.
        system = kwargs.get("system")
        if system is not None:
            extra["has_system"] = True
            if isinstance(system, str):
                extra["system_length"] = len(system)
            elif isinstance(system, list):
                # Anthropic accepts a list of system blocks. Sum string lengths.
                try:
                    extra["system_length"] = sum(len(b.get("text", "")) if isinstance(b, dict) else 0 for b in system)
                except TypeError:
                    pass

        # Messages: count + per-role distribution. Content never copied.
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            extra["messages_count"] = len(messages)
            role_counts: Dict[str, int] = {}
            for m in messages:
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                if isinstance(role, str):
                    role_counts[role] = role_counts.get(role, 0) + 1
            if role_counts:
                extra["message_roles"] = role_counts

        # Tools: count + names. Schemas and descriptions dropped.
        tools = kwargs.get("tools")
        if tools:
            try:
                extra["tools_count"] = len(tools)
                names = [t.get("name") if isinstance(t, dict) else getattr(t, "name", None) for t in tools]
                extra["tool_names"] = [n for n in names if isinstance(n, str)]
            except TypeError:
                pass

        # Tool choice: type (auto/any/tool/none) + name when type=tool.
        tool_choice = kwargs.get("tool_choice")
        if tool_choice is not None:
            if isinstance(tool_choice, str):
                # E.g. ``tool_choice="auto"``.
                extra["tool_choice_type"] = tool_choice
            elif isinstance(tool_choice, dict):
                t_type = tool_choice.get("type")
                if isinstance(t_type, str):
                    extra["tool_choice_type"] = t_type
                t_name = tool_choice.get("name")
                if isinstance(t_name, str):
                    extra["tool_choice_name"] = t_name

        # Metadata: only ``user_id`` is captured (cost attribution per LAY-3334).
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            user_id = metadata.get("user_id")
            if isinstance(user_id, str):
                extra["metadata_user_id"] = user_id

        # Thinking: only ``budget_tokens`` is captured.
        thinking = kwargs.get("thinking")
        if isinstance(thinking, dict):
            budget = thinking.get("budget_tokens")
            if isinstance(budget, int):
                extra["thinking_budget_tokens"] = budget
            t_type = thinking.get("type")
            if isinstance(t_type, str):
                extra["thinking_type"] = t_type

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
        # ``message_stop`` is the SDK's signal that the stream is fully drained.
        # We track receipt of it so downstream consumers / tests can distinguish
        # "iteration ended cleanly" from "iteration ended due to early exit".
        self.stopped: bool = False

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
                        # Defensive: pick up ``thinking_tokens`` if Anthropic
                        # ever surfaces it on a streaming event. Falls back
                        # below to a char-count estimate over thinking blocks.
                        api_thinking = getattr(u, "thinking_tokens", None)
                        if api_thinking is not None:
                            msg.usage.thinking_tokens = api_thinking
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
                    api_thinking = getattr(u, "thinking_tokens", None)
                    if api_thinking is not None:
                        msg.usage.thinking_tokens = api_thinking
            elif etype == "message_stop":
                # Per LAY-3328 / LAY-3332 ACs the message_stop SSE event is an
                # explicit lifecycle signal that the stream finished cleanly.
                # The SDK doesn't carry additional payload on it, but we mark
                # ``stopped`` so consumers can distinguish a complete stream
                # from a torn-down iterator.
                msg.stopped = True
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
        # First content-delta timestamp — drives TTFT.
        self._first_delta_at: float | None = None

    def __enter__(self) -> "_TracedMessageStream":
        self._stream = self._inner.__enter__()
        return self

    async def __aenter__(self) -> "_TracedMessageStream":
        self._stream = await self._inner.__aenter__()
        return self

    def __iter__(self) -> Any:
        for event in self._stream:
            self._events.append(event)
            self._mark_first_delta(event)
            yield event

    async def __aiter__(self) -> Any:
        async for event in self._stream:
            self._events.append(event)
            self._mark_first_delta(event)
            yield event

    def _mark_first_delta(self, event: Any) -> None:
        # Per LAY-3329 / LAY-3332, TTFT measures time-to-first-content. Anthropic
        # emits ``message_start`` and ``content_block_start`` before any content
        # is generated, so anchor on the first ``content_block_delta`` instead.
        if self._first_delta_at is not None:
            return
        if getattr(event, "type", None) == "content_block_delta":
            self._first_delta_at = time.time()

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
            # LAY-3332: surface partial state alongside the error so observers
            # see what was received before the failure.
            partial_meta: Dict[str, Any] | None = None
            try:
                partial_response = AnthropicProvider.aggregate_stream(self._events)
                if partial_response is not None:
                    partial_meta = AnthropicProvider.extract_meta(partial_response) or None
            except Exception:  # noqa: BLE001 — best-effort partial meta
                partial_meta = None
            emit_llm_error(
                self._event_name,
                exc,
                latency_ms,
                partial_meta=partial_meta,
                partial_chunks=len(self._events),
            )
            return
        ttft_ms = (self._first_delta_at - self._start) * 1000 if self._first_delta_at else None
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
                ttft_ms=ttft_ms,
                streaming_duration_ms=latency_ms,
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
