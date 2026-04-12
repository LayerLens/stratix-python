from __future__ import annotations

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
    }
)


class OpenAIProvider(MonkeyPatchProvider):
    name = "openai"
    capture_params = _CAPTURE_PARAMS

    @staticmethod
    def extract_output(response: Any) -> Any:
        try:
            choices = response.choices
            if choices:
                msg = choices[0].message
                return {"role": msg.role, "content": msg.content}
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

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target

        if hasattr(target, "chat") and hasattr(target.chat, "completions"):
            orig = target.chat.completions.create
            self._originals["chat.completions.create"] = orig
            target.chat.completions.create = self._wrap_sync("openai.chat.completions.create", orig)

            if hasattr(target.chat.completions, "acreate"):
                async_orig = target.chat.completions.acreate
                self._originals["chat.completions.acreate"] = async_orig
                target.chat.completions.acreate = self._wrap_async("openai.chat.completions.create", async_orig)

        return target


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
