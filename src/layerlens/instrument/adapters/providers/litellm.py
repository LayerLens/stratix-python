from __future__ import annotations

from typing import Any, Dict, Optional

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

#: Map LiteLLM ``provider/model`` route prefixes to our canonical provider id.
#: Prefixes not listed fall through to the prefix verbatim (custom providers).
_ROUTE_PREFIX_PROVIDERS: Dict[str, str] = {
    "bedrock": "bedrock",
    "azure": "azure",
    "vertex_ai": "google_vertex",
    "gemini": "google",
    "ollama": "ollama",
    "anthropic": "anthropic",
    "openai": "openai",
    "mistral": "mistral",
    "cohere": "cohere",
    "openrouter": "openrouter",
}


def _route_provider(model: str) -> str:
    """Classify the underlying provider for a LiteLLM model string (LAY-3455).

    LiteLLM routes by either an explicit ``provider/model`` prefix (e.g.
    ``bedrock/anthropic.claude-3-...``) or a bare model name whose shape implies
    the provider (e.g. ``gpt-4o`` -> openai). Returns ``"litellm"`` only when the
    bare name matches no known family.
    """
    if not isinstance(model, str) or not model:
        return "litellm"

    if "/" in model:
        prefix = model.split("/", 1)[0]
        return _ROUTE_PREFIX_PROVIDERS.get(prefix, prefix)

    low = model.lower()
    if low.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if low.startswith("claude"):
        return "anthropic"
    if low.startswith("gemini"):
        return "google"
    if low.startswith(("command", "cohere")):
        return "cohere"
    if low.startswith(("mistral", "mixtral")):
        return "mistral"
    if low.startswith(("llama", "meta")):
        return "meta"
    return "litellm"


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

    @staticmethod
    def classify_provider(event_name: str, kwargs: Dict[str, Any]) -> Optional[str]:  # noqa: ARG004
        model = kwargs.get("model")
        if not isinstance(model, str):
            return None
        return _route_provider(model)

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
