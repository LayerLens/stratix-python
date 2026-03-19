"""Tests for scenario classes and registry."""

import pytest

from layerlens.instrument.simulators.clock import DeterministicClock
from layerlens.instrument.simulators.content.template_provider import TemplateContentProvider

from layerlens.instrument.simulators.scenarios import (
    get_scenario,
    list_scenarios,
)
from layerlens.instrument.simulators.scenarios.base import AgentProfile, BaseScenario


class TestScenarioRegistry:
    def test_list_scenarios_has_5(self):
        assert len(list_scenarios()) == 5

    def test_all_scenarios_retrievable(self):
        for name in list_scenarios():
            scenario = get_scenario(name)
            assert isinstance(scenario, BaseScenario)

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario("nonexistent")


class TestAllScenarios:
    @pytest.mark.parametrize("scenario_name", list_scenarios())
    def test_scenario_name(self, scenario_name):
        scenario = get_scenario(scenario_name)
        assert scenario.name == scenario_name

    @pytest.mark.parametrize("scenario_name", list_scenarios())
    def test_scenario_has_5_topics(self, scenario_name):
        scenario = get_scenario(scenario_name)
        assert len(scenario.topics) == 5

    @pytest.mark.parametrize("scenario_name", list_scenarios())
    def test_scenario_has_agents(self, scenario_name):
        scenario = get_scenario(scenario_name)
        assert len(scenario.agents) >= 1
        for agent in scenario.agents:
            assert isinstance(agent, AgentProfile)
            assert agent.name
            assert len(agent.tools) >= 1

    @pytest.mark.parametrize("scenario_name", list_scenarios())
    def test_generate_trace(self, scenario_name):
        scenario = get_scenario(scenario_name)
        clock = DeterministicClock(seed=42)
        content = TemplateContentProvider(seed=42)

        topic = scenario.topics[0]
        trace = scenario.generate_trace(
            topic=topic,
            provider="openai",
            model="gpt-4o",
            content_provider=content,
            clock=clock,
            seed=42,
        )
        assert trace.scenario == scenario_name
        assert trace.topic == topic
        assert trace.span_count >= 4  # agent + llm + tools + llm + eval

    @pytest.mark.parametrize("scenario_name", list_scenarios())
    def test_generate_with_content(self, scenario_name):
        scenario = get_scenario(scenario_name)
        clock = DeterministicClock(seed=42)
        content = TemplateContentProvider(seed=42)

        trace = scenario.generate_trace(
            topic=scenario.topics[0],
            provider="openai",
            model="gpt-4o",
            content_provider=content,
            clock=clock,
            include_content=True,
            seed=42,
        )
        llm_spans = trace.llm_spans
        assert len(llm_spans[0].input_messages) >= 1

    @pytest.mark.parametrize("scenario_name", list_scenarios())
    def test_deterministic_generation(self, scenario_name):
        scenario = get_scenario(scenario_name)

        def gen():
            clock = DeterministicClock(seed=42)
            content = TemplateContentProvider(seed=42)
            return scenario.generate_trace(
                topic=scenario.topics[0],
                provider="openai",
                model="gpt-4o",
                content_provider=content,
                clock=clock,
                seed=42,
            )

        t1 = gen()
        t2 = gen()
        assert t1.trace_id == t2.trace_id
        assert t1.span_count == t2.span_count


class TestAgentProfile:
    def test_defaults(self):
        profile = AgentProfile(name="Test_Agent")
        assert profile.name == "Test_Agent"
        assert profile.description == ""
        assert profile.tools == []
        assert profile.eval_dimensions == ["factual_accuracy"]

    def test_custom(self):
        profile = AgentProfile(
            name="Custom_Agent",
            description="A custom agent",
            tools=["tool_a", "tool_b"],
            eval_dimensions=["accuracy", "safety"],
        )
        assert profile.name == "Custom_Agent"
        assert len(profile.tools) == 2
        assert len(profile.eval_dimensions) == 2
