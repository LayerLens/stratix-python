"""
LLM Model Pricing

Maintains pricing tables (per-1K-token rates) for all supported models
and provides cost calculation with cached token adjustments.
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters.llm_providers.token_usage import NormalizedTokenUsage


# ---------------------------------------------------------------------------
# Pricing tables (per-1K-token rates: USD)
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.0100},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.0100},
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1": {"input": 0.015, "output": 0.060},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o3": {"input": 0.010, "output": 0.040},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    "o4-mini": {"input": 0.0011, "output": 0.0044},
    # Anthropic
    "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
    "claude-opus-4-20250115": {"input": 0.015, "output": 0.075},
    "claude-opus-4-6": {"input": 0.015, "output": 0.075},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
    "claude-haiku-3-5-20241022": {"input": 0.0008, "output": 0.004},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    # Google
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
    "gemini-2.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    # Meta (Ollama/Bedrock)
    "llama-3.3-70b": {"input": 0.00099, "output": 0.00099},
    "llama-3.1-70b": {"input": 0.00099, "output": 0.00099},
    "llama-3.1-8b": {"input": 0.00022, "output": 0.00022},
    # Mistral
    "mistral-large": {"input": 0.002, "output": 0.006},
    "mistral-small": {"input": 0.0002, "output": 0.0006},
}

AZURE_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.00275, "output": 0.011},
    "gpt-4o-mini": {"input": 0.000165, "output": 0.00066},
    "gpt-4-turbo": {"input": 0.011, "output": 0.033},
    "gpt-4": {"input": 0.033, "output": 0.066},
    "gpt-35-turbo": {"input": 0.00055, "output": 0.00165},
}

BEDROCK_PRICING: dict[str, dict[str, float]] = {
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-opus-20240229-v1:0": {"input": 0.015, "output": 0.075},
    "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
    "meta.llama3-1-70b-instruct-v1:0": {"input": 0.00099, "output": 0.00099},
    "meta.llama3-1-8b-instruct-v1:0": {"input": 0.00022, "output": 0.00022},
    "cohere.command-r-plus-v1:0": {"input": 0.003, "output": 0.015},
    "cohere.command-r-v1:0": {"input": 0.0005, "output": 0.0015},
}

def _cached_token_discount(model: str) -> float:
    """Determine the cached token rate as a fraction of input price.

    Different providers offer different cache discounts:
    - Anthropic: 90% discount (pay 10% of input rate)
    - Google: 75% discount (pay 25% of input rate)
    - OpenAI and others: 50% discount (pay 50% of input rate)
    """
    lower = model.lower()
    if lower.startswith("claude"):
        return 0.1
    if lower.startswith("gemini"):
        return 0.25
    return 0.5


def calculate_cost(
    model: str,
    usage: NormalizedTokenUsage,
    pricing_table: dict[str, dict[str, float]] | None = None,
) -> float | None:
    """
    Calculate the API cost in USD for a model invocation.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5-20250929")
        usage: Normalized token usage from the provider response
        pricing_table: Override pricing table (for Azure/Bedrock). Defaults to PRICING.

    Returns:
        Cost in USD, or None if model is not in the pricing table.
    """
    table = pricing_table or PRICING
    rates = table.get(model)
    if rates is None:
        return None

    input_rate = rates.get("input", 0.0)
    output_rate = rates.get("output", 0.0)

    prompt_tokens = usage.prompt_tokens
    cached = usage.cached_tokens or 0

    # Adjust for cached tokens: cached tokens use provider-specific discount
    non_cached = max(prompt_tokens - cached, 0)
    cached_rate = input_rate * _cached_token_discount(model)

    cost = (
        (non_cached * input_rate / 1000)
        + (cached * cached_rate / 1000)
        + (usage.completion_tokens * output_rate / 1000)
    )

    return round(cost, 8)
