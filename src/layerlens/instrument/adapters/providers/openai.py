from __future__ import annotations

import time
import logging
from typing import Any, Dict

from .._base import AdapterInfo, BaseAdapter
from ._base_provider import emit_llm_events, emit_llm_error
from ..._context import _current_collector

log: logging.Logger = logging.getLogger(__name__)

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


class OpenAIProvider(BaseAdapter):
    def __init__(self) -> None:
        self._client: Any = None
        self._originals: Dict[str, Any] = {}

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target

        if hasattr(target, "chat") and hasattr(target.chat, "completions"):
            orig = target.chat.completions.create
            self._originals["chat.completions.create"] = orig
            target.chat.completions.create = self._wrap_sync(orig)

            if hasattr(target.chat.completions, "acreate"):
                async_orig = target.chat.completions.acreate
                self._originals["chat.completions.acreate"] = async_orig
                target.chat.completions.acreate = self._wrap_async(async_orig)

        return target

    def disconnect(self) -> None:
        if self._client is None:
            return
        for key, orig in self._originals.items():
            try:
                parts = key.split(".")
                obj = self._client
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], orig)
            except Exception:
                log.warning("Could not restore %s", key)
        self._client = None
        self._originals.clear()

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="openai",
            adapter_type="provider",
            connected=self._client is not None,
        )

    def _wrap_sync(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return original(*args, **kwargs)
            start = time.time()
            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                latency_ms = (time.time() - start) * 1000
                emit_llm_error("openai.chat.completions.create", exc, latency_ms)
                raise
            latency_ms = (time.time() - start) * 1000
            emit_llm_events(
                "openai.chat.completions.create", kwargs, response,
                _extract_output, _extract_response_meta, _CAPTURE_PARAMS, latency_ms,
            )
            return response

        return wrapped

    def _wrap_async(self, original: Any) -> Any:
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return await original(*args, **kwargs)
            start = time.time()
            try:
                response = await original(*args, **kwargs)
            except Exception as exc:
                latency_ms = (time.time() - start) * 1000
                emit_llm_error("openai.chat.completions.create", exc, latency_ms)
                raise
            latency_ms = (time.time() - start) * 1000
            emit_llm_events(
                "openai.chat.completions.create", kwargs, response,
                _extract_output, _extract_response_meta, _CAPTURE_PARAMS, latency_ms,
            )
            return response

        return wrapped


def _extract_output(response: Any) -> Any:
    try:
        choices = response.choices
        if choices:
            msg = choices[0].message
            return {"role": msg.role, "content": msg.content}
    except (AttributeError, IndexError):
        pass
    return None


def _extract_response_meta(response: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        usage = response.usage
        if usage is not None:
            meta["usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
    except AttributeError:
        pass
    try:
        meta["response_model"] = response.model
    except AttributeError:
        pass
    return meta


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
