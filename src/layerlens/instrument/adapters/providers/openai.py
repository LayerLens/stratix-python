from __future__ import annotations

import json
from typing import Any, Dict

from ._base_provider import MonkeyPatchProvider

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
        "tools",
        "seed",
        "stream",
        "service_tier",
    }
)


class OpenAIProvider(MonkeyPatchProvider):
    """OpenAI adapter with streaming, tool-call and full response-metadata capture."""

    name = "openai"
    capture_params = _CAPTURE_PARAMS

    @staticmethod
    def extract_output(response: Any) -> Any:
        try:
            choices = response.choices
            if choices:
                msg = choices[0].message
                out: Dict[str, Any] = {"role": msg.role, "content": msg.content}
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls and _is_iterable(tool_calls):
                    out["tool_calls"] = [_serialize_tool_call(tc) for tc in tool_calls]
                return out
        except (AttributeError, IndexError):
            pass
        return None

    @staticmethod
    def extract_meta(response: Any) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = _opt_int(getattr(details, "cached_tokens", None)) if details is not None else None
            completion_details = getattr(usage, "completion_tokens_details", None)
            reasoning = (
                _opt_int(getattr(completion_details, "reasoning_tokens", None))
                if completion_details is not None
                else None
            )
            meta["usage"] = {
                "prompt_tokens": _opt_int(getattr(usage, "prompt_tokens", 0)) or 0,
                "completion_tokens": _opt_int(getattr(usage, "completion_tokens", 0)) or 0,
                "total_tokens": _opt_int(getattr(usage, "total_tokens", 0)) or 0,
                **({"cached_tokens": cached} if cached is not None else {}),
                **({"reasoning_tokens": reasoning} if reasoning is not None else {}),
            }
        for attr in ("model", "id", "system_fingerprint", "service_tier"):
            try:
                val = getattr(response, attr, None)
                if isinstance(val, (str, int, float, bool)):
                    meta["response_model" if attr == "model" else f"response_{attr}" if attr == "id" else attr] = val
            except AttributeError:
                pass
        # finish_reason from first choice
        try:
            choices = response.choices
            if choices:
                fr = getattr(choices[0], "finish_reason", None)
                if isinstance(fr, str):
                    meta["finish_reason"] = fr
        except (AttributeError, IndexError):
            pass
        return meta

    @staticmethod
    def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
        try:
            choices = response.choices
            if not choices:
                return []
            msg = choices[0].message
            raw = getattr(msg, "tool_calls", None) or []
            if not _is_iterable(raw):
                return []
            return [_serialize_tool_call(tc) for tc in raw]
        except (AttributeError, IndexError):
            return []

    @staticmethod
    def aggregate_stream(chunks: list[Any]) -> Any:
        if not chunks:
            return None
        return _StreamedChatResponse.from_chunks(chunks)

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target
        self._patch_chat_completions(target)
        self._patch_responses(target)
        self._patch_embeddings(target)
        return target

    def _patch_chat_completions(self, target: Any) -> None:
        if not (hasattr(target, "chat") and hasattr(target.chat, "completions")):
            return
        completions = target.chat.completions
        orig = completions.create
        self._originals["chat.completions.create"] = orig
        completions.create = self._wrap_sync("openai.chat.completions.create", orig)
        if hasattr(completions, "acreate"):
            async_orig = completions.acreate
            self._originals["chat.completions.acreate"] = async_orig
            completions.acreate = self._wrap_async("openai.chat.completions.create", async_orig)

    def _patch_responses(self, target: Any) -> None:
        if not hasattr(target, "responses"):
            return
        responses = target.responses
        if hasattr(responses, "create"):
            orig = responses.create
            self._originals["responses.create"] = orig
            responses.create = self._wrap_sync("openai.responses.create", orig)

    def _patch_embeddings(self, target: Any) -> None:
        if not hasattr(target, "embeddings"):
            return
        embeddings = target.embeddings
        if hasattr(embeddings, "create"):
            orig = embeddings.create
            self._originals["embeddings.create"] = orig
            embeddings.create = self._wrap_sync("openai.embeddings.create", orig)


def _opt_int(val: Any) -> Any:
    """Best-effort ``int`` coercion. Returns ``None`` for non-numerics (e.g. Mock)."""
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


def _is_iterable(obj: Any) -> bool:
    """True for real sequences (list/tuple), false for Mocks or scalars.

    Using ``iter()`` would succeed for any Mock (which is iterable enough to
    return an empty iterator in some configurations) and still raise on
    others, so we whitelist concrete sequence types.
    """
    return isinstance(obj, (list, tuple))


def _serialize_tool_call(tc: Any) -> Dict[str, Any]:
    """Normalize both streaming ``ChoiceDeltaToolCall`` and non-streaming ``ToolCall``."""
    if isinstance(tc, dict):
        fn = tc.get("function", {}) or {}
        return {
            "id": tc.get("id"),
            "type": tc.get("type", "function"),
            "tool_name": fn.get("name"),
            "arguments": _maybe_load_json(fn.get("arguments")),
        }
    fn = getattr(tc, "function", None)
    return {
        "id": getattr(tc, "id", None),
        "type": getattr(tc, "type", "function"),
        "tool_name": getattr(fn, "name", None) if fn is not None else None,
        "arguments": _maybe_load_json(getattr(fn, "arguments", None) if fn is not None else None),
    }


def _maybe_load_json(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


class _StreamedChatResponse:
    """Minimal response shim assembled from OpenAI chat.completion.chunk objects."""

    class _Choice:
        __slots__ = ("message", "finish_reason")

        def __init__(self, role: str, content: str, tool_calls: list[Any], finish_reason: Any):
            self.message = _StreamedChatResponse._Message(role, content, tool_calls)
            self.finish_reason = finish_reason

    class _Message:
        __slots__ = ("role", "content", "tool_calls")

        def __init__(self, role: str, content: str, tool_calls: list[Any]):
            self.role = role
            self.content = content
            self.tool_calls = tool_calls or None

    def __init__(
        self,
        *,
        model: str | None,
        response_id: str | None,
        system_fingerprint: str | None,
        service_tier: str | None,
        choices: list["_StreamedChatResponse._Choice"],
        usage: Any,
    ):
        self.model = model
        self.id = response_id
        self.system_fingerprint = system_fingerprint
        self.service_tier = service_tier
        self.choices = choices
        self.usage = usage

    @classmethod
    def from_chunks(cls, chunks: list[Any]) -> "_StreamedChatResponse":
        role = "assistant"
        content_parts: list[str] = []
        tool_fragments: dict[int, dict[str, Any]] = {}
        finish_reason: Any = None
        model = None
        response_id = None
        system_fingerprint = None
        service_tier = None
        usage = None

        for chunk in chunks:
            model = getattr(chunk, "model", None) or model
            response_id = getattr(chunk, "id", None) or response_id
            system_fingerprint = getattr(chunk, "system_fingerprint", None) or system_fingerprint
            service_tier = getattr(chunk, "service_tier", None) or service_tier
            u = getattr(chunk, "usage", None)
            if u is not None:
                usage = u
            try:
                choices = chunk.choices
            except AttributeError:
                continue
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            fr = getattr(choices[0], "finish_reason", None)
            if fr is not None:
                finish_reason = fr
            if delta is None:
                continue
            piece = getattr(delta, "content", None)
            if piece:
                content_parts.append(piece)
            d_role = getattr(delta, "role", None)
            if d_role:
                role = d_role
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc, "index", 0) or 0
                slot = tool_fragments.setdefault(
                    idx, {"id": None, "type": "function", "function": {"name": None, "arguments": ""}}
                )
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["function"]["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["function"]["arguments"] += fn.arguments

        tool_calls = [tool_fragments[i] for i in sorted(tool_fragments)]
        choice = cls._Choice(role, "".join(content_parts), tool_calls, finish_reason)
        return cls(
            model=model,
            response_id=response_id,
            system_fingerprint=system_fingerprint,
            service_tier=service_tier,
            choices=[choice],
            usage=usage,
        )


# --- Convenience API ---


def instrument_openai(client: Any) -> OpenAIProvider:
    from .._registry import get, register

    existing = get("openai")
    if existing is not None:
        existing.disconnect()
    provider = OpenAIProvider()
    provider.connect(client)
    register("openai", provider)
    return provider


def uninstrument_openai() -> None:
    from .._registry import unregister

    unregister("openai")
