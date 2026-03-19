"""Tests for content templates and TemplateContentProvider."""

import pytest

from layerlens.instrument.simulators.content.template_provider import TemplateContentProvider
from layerlens.instrument.simulators.content.templates import SCENARIO_TEMPLATES


class TestScenarioTemplates:
    def test_all_5_scenarios_present(self):
        assert len(SCENARIO_TEMPLATES) == 5
        expected = {"customer_service", "sales", "order_management", "knowledge_faq", "it_helpdesk"}
        assert set(SCENARIO_TEMPLATES.keys()) == expected

    @pytest.mark.parametrize("scenario", list(SCENARIO_TEMPLATES.keys()))
    def test_scenario_has_required_keys(self, scenario):
        t = SCENARIO_TEMPLATES[scenario]
        assert "scenario" in t
        assert "agent_names" in t
        assert "system_prompts" in t
        assert "topics" in t
        assert t["scenario"] == scenario

    @pytest.mark.parametrize("scenario", list(SCENARIO_TEMPLATES.keys()))
    def test_scenario_has_5_topics(self, scenario):
        topics = SCENARIO_TEMPLATES[scenario]["topics"]
        assert len(topics) == 5

    @pytest.mark.parametrize("scenario", list(SCENARIO_TEMPLATES.keys()))
    def test_each_topic_has_messages(self, scenario):
        for topic, data in SCENARIO_TEMPLATES[scenario]["topics"].items():
            assert "user_messages" in data, f"{scenario}/{topic} missing user_messages"
            assert "agent_responses" in data, f"{scenario}/{topic} missing agent_responses"
            assert len(data["user_messages"]) >= 1
            assert len(data["agent_responses"]) >= 1

    @pytest.mark.parametrize("scenario", list(SCENARIO_TEMPLATES.keys()))
    def test_each_topic_has_tools(self, scenario):
        for topic, data in SCENARIO_TEMPLATES[scenario]["topics"].items():
            assert "tools" in data, f"{scenario}/{topic} missing tools"
            for tool_name, tool_data in data["tools"].items():
                assert "input" in tool_data, f"{scenario}/{topic}/{tool_name} missing input"
                assert "output" in tool_data, f"{scenario}/{topic}/{tool_name} missing output"


class TestTemplateContentProvider:
    def setup_method(self):
        self.provider = TemplateContentProvider(seed=42)

    def test_get_topics(self):
        topics = self.provider.get_topics("customer_service")
        assert len(topics) == 5
        assert "Shipping_Delay" in topics
        assert "Account_Access" in topics

    def test_get_agent_names(self):
        names = self.provider.get_agent_names("customer_service")
        assert len(names) >= 1
        assert "Case_Resolution_Agent" in names

    def test_get_tool_names(self):
        tools = self.provider.get_tool_names("customer_service", "Shipping_Delay")
        assert len(tools) >= 1
        assert "Get_Order_Details" in tools

    def test_get_user_message(self):
        msg = self.provider.get_user_message("customer_service", "Shipping_Delay")
        assert isinstance(msg, str)
        assert len(msg) > 10

    def test_get_user_message_turn_cycling(self):
        msg1 = self.provider.get_user_message("customer_service", "Shipping_Delay", turn=1)
        msg2 = self.provider.get_user_message("customer_service", "Shipping_Delay", turn=2)
        msg3 = self.provider.get_user_message("customer_service", "Shipping_Delay", turn=3)
        assert msg1 != msg2  # Different turns
        assert isinstance(msg3, str)

    def test_get_agent_response(self):
        resp = self.provider.get_agent_response("customer_service", "Shipping_Delay")
        assert isinstance(resp, str)
        assert len(resp) > 10

    def test_get_system_prompt(self):
        prompt = self.provider.get_system_prompt("customer_service", "Case_Resolution_Agent")
        assert "customer service" in prompt.lower()

    def test_get_system_prompt_fallback(self):
        prompt = self.provider.get_system_prompt("customer_service", "Unknown_Agent")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_tool_input(self):
        tool_input = self.provider.get_tool_input("Get_Order_Details", "Shipping_Delay")
        assert isinstance(tool_input, dict)
        assert "order_id" in tool_input

    def test_get_tool_output(self):
        tool_output = self.provider.get_tool_output("Get_Order_Details", "Shipping_Delay")
        assert isinstance(tool_output, dict)

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            self.provider.get_user_message("nonexistent", "topic")

    def test_unknown_topic_raises(self):
        with pytest.raises(ValueError, match="Unknown topic"):
            self.provider.get_user_message("customer_service", "Nonexistent_Topic")

    def test_interpolation(self):
        msg = self.provider.get_user_message("customer_service", "Shipping_Delay", turn=1)
        # Should not contain unresolved template placeholders
        assert "{order_id}" not in msg
        assert "{id}" not in msg

    @pytest.mark.parametrize("scenario", list(SCENARIO_TEMPLATES.keys()))
    def test_all_scenarios_work(self, scenario):
        topics = self.provider.get_topics(scenario)
        for topic in topics:
            msg = self.provider.get_user_message(scenario, topic)
            assert isinstance(msg, str) and len(msg) > 0
            resp = self.provider.get_agent_response(scenario, topic)
            assert isinstance(resp, str) and len(resp) > 0

    def test_deterministic_with_seed(self):
        p1 = TemplateContentProvider(seed=42)
        p2 = TemplateContentProvider(seed=42)
        # Template provider is deterministic by design (index-based, not random)
        msg1 = p1.get_user_message("customer_service", "Shipping_Delay")
        msg2 = p2.get_user_message("customer_service", "Shipping_Delay")
        assert msg1 == msg2

    def test_tool_input_fallback(self):
        tool_input = self.provider.get_tool_input("Unknown_Tool", "topic")
        assert isinstance(tool_input, dict)
        assert "action" in tool_input
