from __future__ import annotations

import logging
from typing import Any, Dict

from .._base import AdapterInfo, BaseAdapter
from ._base_provider import fail_llm_span, create_llm_span, finish_llm_span

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
    }
)


class AnthropicProvider(BaseAdapter):
    def __init__(self) -> None:
        self._client: Any = None
        self._originals: Dict[str, Any] = {}

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target

        if hasattr(target, "messages"):
            orig = target.messages.create
            self._originals["messages.create"] = orig
            target.messages.create = self._wrap_sync(orig)

            if hasattr(target.messages, "acreate"):
                async_orig = target.messages.acreate
                self._originals["messages.acreate"] = async_orig
                target.messages.acreate = self._wrap_async(async_orig)

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
            name="anthropic",
            adapter_type="provider",
            connected=self._client is not None,
        )

    def _wrap_sync(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            span, token = create_llm_span("anthropic.messages.create", kwargs, _CAPTURE_PARAMS)
            if span is None:
                return original(*args, **kwargs)
            try:
                response = original(*args, **kwargs)
                finish_llm_span(span, token, response, _extract_output, _extract_response_meta)
                return response
            except Exception as exc:
                fail_llm_span(span, token, exc)
                raise

        return wrapped

    def _wrap_async(self, original: Any) -> Any:
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            span, token = create_llm_span("anthropic.messages.create", kwargs, _CAPTURE_PARAMS)
            if span is None:
                return await original(*args, **kwargs)
            try:
                response = await original(*args, **kwargs)
                finish_llm_span(span, token, response, _extract_output, _extract_response_meta)
                return response
            except Exception as exc:
                fail_llm_span(span, token, exc)
                raise

        return wrapped


def _extract_output(response: Any) -> Any:
    try:
        content = response.content
        if content:
            block = content[0]
            return {"type": block.type, "text": getattr(block, "text", None)}
    except (AttributeError, IndexError):
        pass
    return None


def _extract_response_meta(response: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        usage = response.usage
        if usage is not None:
            meta["usage"] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }
    except AttributeError:
        pass
    try:
        meta["response_model"] = response.model
    except AttributeError:
        pass
    try:
        meta["stop_reason"] = response.stop_reason
    except AttributeError:
        pass
    return meta


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
