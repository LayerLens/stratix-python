from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ._base_provider import fail_llm_span, create_llm_span, finish_llm_span

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


class OpenAIProvider:
    def __init__(self) -> None:
        self._client: Any = None
        self._originals: Dict[str, Any] = {}

    def connect_client(self, client: Any) -> Any:
        self._client = client

        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            orig = client.chat.completions.create
            self._originals["chat.completions.create"] = orig
            client.chat.completions.create = self._wrap_sync(orig)

            if hasattr(client.chat.completions, "acreate"):
                async_orig = client.chat.completions.acreate
                self._originals["chat.completions.acreate"] = async_orig
                client.chat.completions.acreate = self._wrap_async(async_orig)

        return client

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

    def _wrap_sync(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            span, token = create_llm_span("openai.chat.completions.create", kwargs, _CAPTURE_PARAMS)
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
            span, token = create_llm_span("openai.chat.completions.create", kwargs, _CAPTURE_PARAMS)
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

_provider_instance: Optional[OpenAIProvider] = None


def instrument_openai(client: Any) -> OpenAIProvider:
    global _provider_instance
    if _provider_instance is not None:
        _provider_instance.disconnect()
    _provider_instance = OpenAIProvider()
    _provider_instance.connect_client(client)
    return _provider_instance


def uninstrument_openai() -> None:
    global _provider_instance
    if _provider_instance is not None:
        _provider_instance.disconnect()
        _provider_instance = None
