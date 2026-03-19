"""Tests for LLM model pricing and cost calculation."""

import pytest
from layerlens.instrument.adapters.llm_providers.pricing import (
    AZURE_PRICING,
    BEDROCK_PRICING,
    PRICING,
    calculate_cost,
)
from layerlens.instrument.adapters.llm_providers.token_usage import NormalizedTokenUsage


class TestPricingTables:
    """Tests for pricing table contents."""

    def test_openai_models_present(self):
        assert "gpt-4o" in PRICING
        assert "gpt-4o-mini" in PRICING
        assert "o1" in PRICING

    def test_anthropic_models_present(self):
        assert "claude-sonnet-4-5-20250929" in PRICING
        assert "claude-opus-4-20250115" in PRICING
        assert "claude-haiku-3-5-20241022" in PRICING

    def test_google_models_present(self):
        assert "gemini-2.5-pro" in PRICING
        assert "gemini-2.0-flash" in PRICING

    def test_azure_pricing_separate(self):
        assert "gpt-4o" in AZURE_PRICING
        assert "gpt-4o-mini" in AZURE_PRICING

    def test_bedrock_pricing_separate(self):
        assert "anthropic.claude-3-5-sonnet-20241022-v2:0" in BEDROCK_PRICING
        assert "meta.llama3-1-70b-instruct-v1:0" in BEDROCK_PRICING

    def test_pricing_has_input_output_rates(self):
        for model, rates in PRICING.items():
            assert "input" in rates, f"Missing input rate for {model}"
            assert "output" in rates, f"Missing output rate for {model}"
            assert rates["input"] >= 0
            assert rates["output"] >= 0


class TestCalculateCost:
    """Tests for calculate_cost function."""

    def test_basic_cost_gpt4o(self):
        usage = NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        cost = calculate_cost("gpt-4o", usage)
        assert cost is not None
        # input: 1000 * 0.0025 / 1000 = 0.0025
        # output: 500 * 0.01 / 1000 = 0.005
        assert abs(cost - 0.0075) < 1e-6

    def test_basic_cost_claude_sonnet(self):
        usage = NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        cost = calculate_cost("claude-sonnet-4-5-20250929", usage)
        assert cost is not None
        # input: 1000 * 0.003 / 1000 = 0.003
        # output: 500 * 0.015 / 1000 = 0.0075
        assert abs(cost - 0.0105) < 1e-6

    def test_unknown_model_returns_none(self):
        usage = NormalizedTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        cost = calculate_cost("unknown-model-xyz", usage)
        assert cost is None

    def test_cached_tokens_reduce_cost(self):
        usage_no_cache = NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=0, total_tokens=1000)
        usage_with_cache = NormalizedTokenUsage(
            prompt_tokens=1000, completion_tokens=0, total_tokens=1000,
            cached_tokens=500,
        )
        cost_no_cache = calculate_cost("gpt-4o", usage_no_cache)
        cost_with_cache = calculate_cost("gpt-4o", usage_with_cache)
        assert cost_no_cache is not None
        assert cost_with_cache is not None
        assert cost_with_cache < cost_no_cache

    def test_zero_tokens(self):
        usage = NormalizedTokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        cost = calculate_cost("gpt-4o", usage)
        assert cost == 0.0

    def test_custom_pricing_table(self):
        custom = {"my-model": {"input": 0.01, "output": 0.02}}
        usage = NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000)
        cost = calculate_cost("my-model", usage, pricing_table=custom)
        assert cost is not None
        assert abs(cost - 0.03) < 1e-6

    def test_azure_pricing_table(self):
        usage = NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        cost = calculate_cost("gpt-4o", usage, pricing_table=AZURE_PRICING)
        assert cost is not None
        # Azure gpt-4o: input=0.00275, output=0.011
        expected = (1000 * 0.00275 / 1000) + (500 * 0.011 / 1000)
        assert abs(cost - expected) < 1e-6

    def test_bedrock_pricing_table(self):
        usage = NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        cost = calculate_cost(
            "anthropic.claude-3-5-sonnet-20241022-v2:0", usage,
            pricing_table=BEDROCK_PRICING,
        )
        assert cost is not None
        assert cost > 0

    def test_large_token_count(self):
        usage = NormalizedTokenUsage(
            prompt_tokens=100_000, completion_tokens=50_000, total_tokens=150_000,
        )
        cost = calculate_cost("gpt-4o", usage)
        assert cost is not None
        assert cost > 0

    def test_cached_more_than_prompt_clamps_to_zero(self):
        """Cached tokens > prompt tokens should not produce negative cost."""
        usage = NormalizedTokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
            cached_tokens=200,  # more than prompt
        )
        cost = calculate_cost("gpt-4o", usage)
        assert cost is not None
        assert cost >= 0

    def test_o1_model_cost(self):
        usage = NormalizedTokenUsage(
            prompt_tokens=500, completion_tokens=500, total_tokens=1000,
            reasoning_tokens=300,
        )
        cost = calculate_cost("o1", usage)
        assert cost is not None
        assert cost > 0

    def test_gemini_cost(self):
        usage = NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        cost = calculate_cost("gemini-2.0-flash", usage)
        assert cost is not None
        assert cost > 0

    def test_ollama_model_not_in_pricing(self):
        """Local Ollama models typically aren't in the pricing table."""
        usage = NormalizedTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        cost = calculate_cost("llama3.1:70b", usage)
        assert cost is None

    def test_cost_returns_float(self):
        usage = NormalizedTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        cost = calculate_cost("gpt-4o", usage)
        assert isinstance(cost, float)
