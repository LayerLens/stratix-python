"""Lazy public API for provider adapters.

Per the M3 ticket ACs (LAY-3453 Vertex / LAY-3454 Ollama) and the same PEP 562
``__getattr__`` pattern used for frameworks, the per-provider adapter classes
are imported lazily so ``pip install layerlens`` never pulls a provider SDK
into the default install. Each extra (``providers-vertex``, ``providers-ollama``,
``openai``, ``anthropic``, etc.) adds only the SDK that user actually needs.

Constants and the base class import eagerly because they have no heavy
dependencies of their own.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .pricing import PRICING, AZURE_PRICING, BEDROCK_PRICING, calculate_cost
from .token_usage import NormalizedTokenUsage
from ._base_provider import MonkeyPatchProvider

# Public-name → (sub-module, attribute). Add new entries when porting more
# provider adapters.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "OpenAIProvider": ("openai", "OpenAIProvider"),
    "AnthropicProvider": ("anthropic", "AnthropicProvider"),
    "AzureOpenAIProvider": ("azure_openai", "AzureOpenAIProvider"),
    "BedrockProvider": ("bedrock", "BedrockProvider"),
    "GoogleVertexProvider": ("google_vertex", "GoogleVertexProvider"),
    "OllamaProvider": ("ollama", "OllamaProvider"),
    "LiteLLMProvider": ("litellm", "LiteLLMProvider"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(f".{module_name}", package=__name__)
    return getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(list(_LAZY_EXPORTS.keys()) + list(globals().keys()))


if TYPE_CHECKING:
    # Re-export under TYPE_CHECKING so static analysers see the names without
    # forcing eager imports at runtime.
    from .ollama import OllamaProvider as OllamaProvider
    from .openai import OpenAIProvider as OpenAIProvider
    from .bedrock import BedrockProvider as BedrockProvider
    from .litellm import LiteLLMProvider as LiteLLMProvider
    from .anthropic import AnthropicProvider as AnthropicProvider
    from .azure_openai import AzureOpenAIProvider as AzureOpenAIProvider
    from .google_vertex import GoogleVertexProvider as GoogleVertexProvider


__all__ = [
    "AZURE_PRICING",
    "BEDROCK_PRICING",
    "MonkeyPatchProvider",
    "NormalizedTokenUsage",
    "PRICING",
    "calculate_cost",
    *_LAZY_EXPORTS.keys(),
]
