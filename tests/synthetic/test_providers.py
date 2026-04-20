from __future__ import annotations

from layerlens.synthetic.providers import (
    ProviderTier,
    ProviderRegistry,
    ProviderCapability,
    StochasticProvider,
)
from layerlens.synthetic.templates import TEMPLATE_LIBRARY


class TestStochasticProvider:
    def test_generates_exact_count(self):
        provider = StochasticProvider(seed=7)
        result = provider.generate(
            template_id="llm.chat.basic",
            parameters={"model": "gpt-4o-mini", "prompt_tokens_avg": 100, "completion_tokens_avg": 50},
            count=5,
        )
        assert result.errors == []
        assert len(result.traces) == 5
        assert all(t.data["synthetic"] is True for t in result.traces)

    def test_determinism_with_seed(self):
        a = StochasticProvider(seed=42).generate(
            template_id="rag.retrieval",
            parameters={"model": "gpt-4o-mini", "top_k": 3},
            count=3,
        )
        b = StochasticProvider(seed=42).generate(
            template_id="rag.retrieval",
            parameters={"model": "gpt-4o-mini", "top_k": 3},
            count=3,
        )
        # Trace IDs are random but latency/usage are seeded — compare events:
        a_events = [t.data["events"] for t in a.traces]
        b_events = [t.data["events"] for t in b.traces]
        assert a_events == b_events

    def test_rag_template_adds_retrieval_event(self):
        result = StochasticProvider(seed=1).generate(
            template_id="rag.retrieval",
            parameters={"model": "gpt-4o-mini", "top_k": 5},
            count=1,
        )
        types = [e.get("type") for e in result.traces[0].data["events"]]
        assert "retrieval" in types
        assert "model.invoke" in types

    def test_multi_agent_template_adds_handoff(self):
        result = StochasticProvider(seed=1).generate(
            template_id="multi_agent.handoff",
            parameters={"agents": 3, "tools_per_run_max": 2},
            count=1,
        )
        types = [e.get("type") for e in result.traces[0].data["events"]]
        assert "agent.handoff" in types

    def test_unknown_template_returns_errors(self):
        result = StochasticProvider(seed=1).generate(template_id="nope.unknown", parameters={}, count=1)
        assert result.traces == []
        assert any("unknown template" in e for e in result.errors)

    def test_validate_parameters_catches_missing_required(self):
        provider = StochasticProvider(seed=1)
        # temporarily mutate a template parameter to be required
        tpl = TEMPLATE_LIBRARY["llm.chat.basic"]
        for param in tpl.parameters:
            if param.name == "model":
                original = param.required
                param.required = True
                try:
                    errors = provider.validate_parameters(tpl.id, {})
                    assert any("model" in e for e in errors)
                finally:
                    param.required = original
                break

    def test_estimate_cost_matches_ecu(self):
        provider = StochasticProvider()
        assert provider.estimate_cost("llm.chat.basic", 10) == provider.info.ecu_per_trace * 10

    def test_info_advertises_capabilities(self):
        provider = StochasticProvider()
        assert ProviderCapability.RAG_TRACES in provider.info.capabilities
        assert provider.info.tier == ProviderTier.LOCAL


class TestProviderRegistry:
    def test_singleton_instance(self):
        assert ProviderRegistry.instance() is ProviderRegistry.instance()

    def test_default_has_stochastic(self):
        registry = ProviderRegistry()
        assert registry.get("stochastic") is not None

    def test_get_unknown_returns_none(self):
        registry = ProviderRegistry()
        assert registry.get("does-not-exist") is None
        assert registry.get(None) is None

    def test_auto_select_matches_capabilities(self):
        registry = ProviderRegistry()
        p = registry.auto_select([ProviderCapability.RAG_TRACES])
        assert p is not None
        assert ProviderCapability.RAG_TRACES in p.info.capabilities

    def test_auto_select_returns_none_when_no_match(self):
        registry = ProviderRegistry()
        # stochastic provider doesn't advertise OTEL_SPANS
        assert registry.auto_select([ProviderCapability.OTEL_SPANS]) is None

    def test_list_returns_registered(self):
        registry = ProviderRegistry()
        infos = registry.list()
        assert any(i.id == "stochastic" for i in infos)
