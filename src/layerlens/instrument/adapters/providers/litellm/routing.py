"""LiteLLM provider routing.

LiteLLM is a multi-provider router that dispatches a single
``litellm.completion(...)`` (or ``litellm.acompletion(...)``) call to one of
~100 underlying providers (OpenAI, Anthropic, Bedrock, Vertex, Cohere,
Ollama, Together, Groq, ...). The provider is selected from the
``model`` argument either via an explicit ``provider/model`` prefix
(``bedrock/anthropic.claude-3-5-sonnet``) or via heuristics on the bare
model name (``gpt-4o`` → OpenAI).

The adapter normalizes that routing decision into the canonical
``provider`` field on every emitted event so downstream telemetry
matches what the other LayerLens provider adapters emit. Pricing always
falls through to the canonical
:mod:`layerlens.instrument.adapters.providers._base.pricing` manifest —
LiteLLM contributes no new entries; it only routes.
"""

from __future__ import annotations

from typing import Dict

# Model-string prefix → canonical LayerLens provider name.
#
# These mirror the prefix scheme documented at
# https://docs.litellm.ai/docs/providers — the full prefix list is
# longer; this map covers the providers the platform recognises and
# prices today. Anything else lands in the heuristic block below or, as a
# final fallback, returns ``"unknown"``.
_PROVIDER_PREFIXES: Dict[str, str] = {
    "openai/": "openai",
    "anthropic/": "anthropic",
    "azure/": "azure_openai",
    "bedrock/": "aws_bedrock",
    "vertex_ai/": "google_vertex",
    "ollama/": "ollama",
    "cohere/": "cohere",
    "huggingface/": "huggingface",
    "together_ai/": "together_ai",
    "groq/": "groq",
}


def detect_provider(model_str: str) -> str:
    """Detect the underlying provider from a LiteLLM model string.

    Resolution order:

    1. Exact ``provider/...`` prefix match against :data:`_PROVIDER_PREFIXES`.
    2. Bare-model-name heuristics:

       * ``gpt-`` / ``o1`` / ``o3`` → ``openai``
       * ``claude-`` → ``anthropic``
       * ``gemini-`` → ``google_vertex``
       * ``llama`` → ``meta``
       * ``mistral`` → ``mistral``

    3. Fallback: ``"unknown"``.

    Args:
        model_str: The raw LiteLLM ``model`` argument, e.g.
            ``"openai/gpt-4o"``, ``"bedrock/anthropic.claude-3-5-sonnet"``,
            or just ``"gpt-4o"``.

    Returns:
        The canonical LayerLens provider name. Never raises.
    """
    if not model_str:
        return "unknown"
    for prefix, provider in _PROVIDER_PREFIXES.items():
        if model_str.startswith(prefix):
            return provider
    lower = model_str.lower()
    if lower.startswith("gpt-") or lower.startswith("o1") or lower.startswith("o3"):
        return "openai"
    if lower.startswith("claude-"):
        return "anthropic"
    if lower.startswith("gemini-"):
        return "google_vertex"
    if lower.startswith("llama"):
        return "meta"
    if lower.startswith("mistral"):
        return "mistral"
    return "unknown"


__all__ = ["detect_provider"]
