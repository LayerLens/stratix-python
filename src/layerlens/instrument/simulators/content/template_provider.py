"""Tier 2: Template-based content provider.

Generates content from parameterized templates per scenario without
requiring any external dependencies (no LLM calls).
"""

from __future__ import annotations

import random
from typing import Any

from .base import ContentProvider
from .templates import SCENARIO_TEMPLATES


class TemplateContentProvider(ContentProvider):
    """Template-based content provider (Tier 2).

    Uses parameterized templates from content/templates/ to generate
    realistic scenario content without external dependencies.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._counter = 0

    def _next_id(self) -> str:
        """Generate a sequential ID for template interpolation."""
        self._counter += 1
        return f"{self._counter:04d}"

    def _interpolate(self, text: str) -> str:
        """Replace template placeholders with generated values."""
        replacements = {
            "{id}": self._next_id(),
            "{order_id}": f"ORD-2024-{self._next_id()}",
            "{delivery_date}": "December 18, 2024",
            "{date}": "December 15, 2024",
            "{amount}": "149.99",
            "{credit_amount}": "$50.00",
            "{resolution}": "expedite your replacement and apply a loyalty credit",
        }
        result = text
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        return result

    def _get_template(self, scenario: str) -> dict[str, Any]:
        """Get template data for a scenario."""
        templates = SCENARIO_TEMPLATES.get(scenario)
        if not templates:
            raise ValueError(
                f"Unknown scenario: {scenario}. "
                f"Available: {list(SCENARIO_TEMPLATES.keys())}"
            )
        return templates

    def get_topics(self, scenario: str) -> list[str]:
        templates = self._get_template(scenario)
        return list(templates["topics"].keys())

    def get_agent_names(self, scenario: str) -> list[str]:
        templates = self._get_template(scenario)
        return templates["agent_names"]

    def get_tool_names(self, scenario: str, topic: str) -> list[str]:
        templates = self._get_template(scenario)
        topic_data = templates["topics"].get(topic, {})
        tools = topic_data.get("tools", {})
        return list(tools.keys())

    def get_user_message(self, scenario: str, topic: str, turn: int = 1) -> str:
        templates = self._get_template(scenario)
        topic_data = templates["topics"].get(topic)
        if not topic_data:
            available = list(templates["topics"].keys())
            raise ValueError(
                f"Unknown topic: {topic} for scenario {scenario}. Available: {available}"
            )
        messages = topic_data["user_messages"]
        idx = (turn - 1) % len(messages)
        return self._interpolate(messages[idx])

    def get_agent_response(
        self,
        scenario: str,
        topic: str,
        turn: int = 1,
        tool_results: dict[str, Any] | None = None,
    ) -> str:
        templates = self._get_template(scenario)
        topic_data = templates["topics"].get(topic)
        if not topic_data:
            return "I'll look into that for you right away."
        responses = topic_data["agent_responses"]
        idx = (turn - 1) % len(responses)
        return self._interpolate(responses[idx])

    def get_system_prompt(self, scenario: str, agent_name: str) -> str:
        templates = self._get_template(scenario)
        prompts = templates.get("system_prompts", {})
        if agent_name in prompts:
            return prompts[agent_name]
        # Fallback to first agent's prompt
        if prompts:
            return next(iter(prompts.values()))
        return f"You are a {scenario.replace('_', ' ')} agent named {agent_name}."

    def get_tool_input(self, action_name: str, topic: str) -> dict[str, Any]:
        """Get tool input from templates, preferring the matching topic."""
        # First try: find in the matching topic across all scenarios
        for templates in SCENARIO_TEMPLATES.values():
            topic_data = templates["topics"].get(topic, {})
            tools = topic_data.get("tools", {})
            if action_name in tools:
                raw_input = tools[action_name]["input"]
                return self._interpolate_dict(raw_input)
        # Second try: search all topics as fallback
        for templates in SCENARIO_TEMPLATES.values():
            for topic_data in templates["topics"].values():
                tools = topic_data.get("tools", {})
                if action_name in tools:
                    raw_input = tools[action_name]["input"]
                    return self._interpolate_dict(raw_input)
        # Fallback
        return {"action": action_name, "query": topic}

    def get_tool_output(self, action_name: str, topic: str) -> dict[str, Any]:
        """Get tool output from templates, preferring the matching topic."""
        # First try: find in the matching topic across all scenarios
        for templates in SCENARIO_TEMPLATES.values():
            topic_data = templates["topics"].get(topic, {})
            tools = topic_data.get("tools", {})
            if action_name in tools:
                raw_output = tools[action_name]["output"]
                return self._interpolate_dict(raw_output)
        # Second try: search all topics as fallback
        for templates in SCENARIO_TEMPLATES.values():
            for topic_data in templates["topics"].values():
                tools = topic_data.get("tools", {})
                if action_name in tools:
                    raw_output = tools[action_name]["output"]
                    return self._interpolate_dict(raw_output)
        # Fallback
        return {"result": "success", "action": action_name}

    def _interpolate_dict(self, data: Any) -> Any:
        """Recursively interpolate string values in a dict/list."""
        if isinstance(data, str):
            return self._interpolate(data)
        if isinstance(data, dict):
            return {k: self._interpolate_dict(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._interpolate_dict(item) for item in data]
        return data
