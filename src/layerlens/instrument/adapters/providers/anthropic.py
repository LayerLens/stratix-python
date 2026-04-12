from __future__ import annotations

from typing import Any, Dict

from ._base_provider import MonkeyPatchProvider

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


class AnthropicProvider(MonkeyPatchProvider):
    name = "anthropic"
    capture_params = _CAPTURE_PARAMS

    @staticmethod
    def extract_output(response: Any) -> Any:
        try:
            content = response.content
            if content:
                block = content[0]
                return {"type": block.type, "text": getattr(block, "text", None)}
        except (AttributeError, IndexError):
            pass
        return None

    @staticmethod
    def extract_meta(response: Any) -> Dict[str, Any]:
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

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target

        if hasattr(target, "messages"):
            orig = target.messages.create
            self._originals["messages.create"] = orig
            target.messages.create = self._wrap_sync("anthropic.messages.create", orig)

            if hasattr(target.messages, "acreate"):
                async_orig = target.messages.acreate
                self._originals["messages.acreate"] = async_orig
                target.messages.acreate = self._wrap_async("anthropic.messages.create", async_orig)

        return target


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
