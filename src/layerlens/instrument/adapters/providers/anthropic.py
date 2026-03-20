from __future__ import annotations

import logging
from typing import Any, Dict, Optional

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


class AnthropicProvider:
    def __init__(self) -> None:
        self._client: Any = None
        self._originals: Dict[str, Any] = {}

    def connect_client(self, client: Any) -> Any:
        self._client = client

        if hasattr(client, "messages"):
            orig = client.messages.create
            self._originals["messages.create"] = orig
            client.messages.create = self._wrap_sync(orig)

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

_provider_instance: Optional[AnthropicProvider] = None


def instrument_anthropic(client: Any) -> AnthropicProvider:
    global _provider_instance
    if _provider_instance is not None:
        _provider_instance.disconnect()
    _provider_instance = AnthropicProvider()
    _provider_instance.connect_client(client)
    return _provider_instance


def uninstrument_anthropic() -> None:
    global _provider_instance
    if _provider_instance is not None:
        _provider_instance.disconnect()
        _provider_instance = None
