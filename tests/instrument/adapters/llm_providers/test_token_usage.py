"""Tests for NormalizedTokenUsage."""

import pytest
from layerlens.instrument.adapters.llm_providers.token_usage import NormalizedTokenUsage


class TestNormalizedTokenUsage:
    """Tests for NormalizedTokenUsage construction and methods."""

    def test_default_values(self):
        usage = NormalizedTokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cached_tokens is None
        assert usage.reasoning_tokens is None

    def test_basic_construction(self):
        usage = NormalizedTokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_with_cached_tokens(self):
        usage = NormalizedTokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cached_tokens=30,
        )
        assert usage.cached_tokens == 30

    def test_with_reasoning_tokens(self):
        usage = NormalizedTokenUsage(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            reasoning_tokens=150,
        )
        assert usage.reasoning_tokens == 150

    def test_compute_total(self):
        usage = NormalizedTokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=0,
        )
        computed = usage.compute_total()
        assert computed.total_tokens == 150
        assert computed.prompt_tokens == 100
        assert computed.completion_tokens == 50

    def test_compute_total_preserves_cached(self):
        usage = NormalizedTokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=20,
        )
        computed = usage.compute_total()
        assert computed.cached_tokens == 20
        assert computed.total_tokens == 150

    def test_model_dump(self):
        usage = NormalizedTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        d = usage.model_dump()
        assert d["prompt_tokens"] == 10
        assert d["completion_tokens"] == 5
        assert d["total_tokens"] == 15

    def test_zero_tokens(self):
        usage = NormalizedTokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        assert usage.compute_total().total_tokens == 0
