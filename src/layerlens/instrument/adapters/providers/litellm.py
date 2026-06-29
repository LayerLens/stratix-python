from __future__ import annotations

from typing import Any, Dict

from .openai import OpenAIProvider
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
    }
)


class LiteLLMProvider(MonkeyPatchProvider):
    name = "litellm"
    capture_params = _CAPTURE_PARAMS

    @staticmethod
    def extract_output(response: Any) -> Any:
        return OpenAIProvider.extract_output(response)

    @staticmethod
    def extract_meta(response: Any) -> Dict[str, Any]:
        return OpenAIProvider.extract_meta(response)

    @staticmethod
    def aggregate_stream(chunks: list[Any]) -> Any:
        # litellm normalizes every provider's stream to the OpenAI chunk shape
        # (ModelResponse with .choices[].delta), so the OpenAI aggregator applies
        # verbatim. Without this override LiteLLMProvider inherited the base no-op
        # aggregate_stream (returns None) and a litellm.completion(stream=True)
        # emitted ZERO model.invoke / cost.record telemetry (G8 class, same bug
        # ollama hit — see test_litellm.TestStreaming).
        return OpenAIProvider.aggregate_stream(chunks)

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        try:
            import litellm
        except ImportError as err:
            raise ImportError(
                "The 'litellm' package is required for LiteLLM instrumentation. Install it with: pip install litellm"
            ) from err

        self._client = litellm

        if "completion" not in self._originals:
            orig_sync = litellm.completion
            self._originals["completion"] = orig_sync
            litellm.completion = self._wrap_sync("litellm.completion", orig_sync)

        if "acompletion" not in self._originals:
            orig_async = litellm.acompletion
            self._originals["acompletion"] = orig_async
            litellm.acompletion = self._wrap_async("litellm.acompletion", orig_async)

        return target


# --- Convenience API ---


def instrument_litellm() -> LiteLLMProvider:
    from .._registry import get, register

    existing = get("litellm")
    if existing is not None:
        existing.disconnect()
    provider = LiteLLMProvider()
    provider.connect()
    register("litellm", provider)
    return provider


def uninstrument_litellm() -> None:
    from .._registry import unregister

    unregister("litellm")
