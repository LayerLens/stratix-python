from __future__ import annotations

import pytest

from layerlens.synthetic import (
    ProviderRegistry,
    StochasticProvider,
    SyntheticDataBuilder,
)


@pytest.fixture
def fresh_registry():
    r = ProviderRegistry()
    r.register(StochasticProvider(seed=1))
    return r


class TestBuilder:
    def test_list_templates(self, fresh_registry):
        builder = SyntheticDataBuilder(registry=fresh_registry)
        assert len(builder.list_templates()) >= 4
        rag = builder.list_templates(category="rag")
        assert all(t.category.value == "rag" for t in rag)

    def test_get_template_known_and_unknown(self, fresh_registry):
        builder = SyntheticDataBuilder(registry=fresh_registry)
        assert builder.get_template("rag.retrieval") is not None
        assert builder.get_template("does-not-exist") is None

    def test_estimate_cost_for_known(self, fresh_registry):
        builder = SyntheticDataBuilder(registry=fresh_registry)
        cost = builder.estimate_cost("rag.retrieval", count=10)
        assert cost["template_id"] == "rag.retrieval"
        assert cost["count"] == 10
        assert cost["provider_tier"] == "local"

    def test_estimate_cost_unknown_template_raises(self, fresh_registry):
        builder = SyntheticDataBuilder(registry=fresh_registry)
        with pytest.raises(ValueError):
            builder.estimate_cost("nope", count=1)

    def test_generate_happy_path(self, fresh_registry):
        builder = SyntheticDataBuilder(registry=fresh_registry)
        result = builder.generate("llm.chat.basic", count=4)
        assert result.errors == []
        assert len(result.traces) == 4

    def test_generate_clamps_count_to_template_bounds(self, fresh_registry):
        builder = SyntheticDataBuilder(registry=fresh_registry)
        result = builder.generate("llm.chat.basic", count=100_000)
        # Stochastic provider max_batch_size = 10000; template max_traces = 1000.
        assert len(result.traces) <= 1000

    def test_generate_unknown_template_returns_errors(self, fresh_registry):
        builder = SyntheticDataBuilder(registry=fresh_registry)
        result = builder.generate("unknown.tpl", count=1)
        assert result.traces == []
        assert result.errors

    def test_resolves_hinted_provider(self, fresh_registry):
        builder = SyntheticDataBuilder(registry=fresh_registry)
        result = builder.generate("llm.chat.basic", count=1)
        assert result.provider_id == "stochastic"

    def test_explicit_provider_id_honoured(self, fresh_registry):
        builder = SyntheticDataBuilder(registry=fresh_registry)
        result = builder.generate("rag.retrieval", count=1, provider_id="stochastic")
        assert result.provider_id == "stochastic"

    def test_missing_provider_returns_errors(self, fresh_registry):
        builder = SyntheticDataBuilder(registry=fresh_registry)
        result = builder.generate("rag.retrieval", count=1, provider_id="ghost")
        assert result.errors
        assert "no provider" in result.errors[0]
