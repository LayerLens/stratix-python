"""Tier 3: LLM-generated content provider.

Uses an LLM API to generate realistic content for scenarios,
with disk caching to avoid redundant API calls.
"""

from __future__ import annotations

import json
from typing import Any

from .base import ContentProvider
from .cache import ContentCache
from .template_provider import TemplateContentProvider


class LLMContentProvider(ContentProvider):
    """LLM-generated content provider (Tier 3).

    Falls back to template provider if LLM is unavailable.
    Caches all LLM responses to disk for reproducibility.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        cache_enabled: bool = True,
        cache_path: str | None = None,
        api_key: str | None = None,
        seed: int | None = None,
    ):
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._cache = ContentCache(cache_dir=cache_path) if cache_enabled else None
        self._fallback = TemplateContentProvider(seed=seed)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI client (supports custom base_url for any OpenAI-compatible API)."""
        if self._client is not None:
            return self._client
        try:
            import openai

            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
            return self._client
        except (ImportError, Exception):
            return None

    def _generate(self, system_prompt: str, user_prompt: str) -> str | None:
        """Call the LLM API and cache the result."""
        # Check cache first
        if self._cache:
            cached = self._cache.get(
                model=self._model, system=system_prompt, user=user_prompt
            )
            if cached is not None:
                return cached

        client = self._get_client()
        if client is None:
            return None

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            content = response.choices[0].message.content

            # Cache the result
            if self._cache and content:
                self._cache.set(
                    content,
                    model=self._model,
                    system=system_prompt,
                    user=user_prompt,
                )
            return content
        except Exception:
            return None

    def get_user_message(self, scenario: str, topic: str, turn: int = 1) -> str:
        system = (
            f"Generate a realistic customer message for a {scenario.replace('_', ' ')} "
            f"scenario about {topic.replace('_', ' ')}. Turn {turn} of the conversation. "
            f"Write only the customer's message, nothing else."
        )
        user = f"Scenario: {scenario}, Topic: {topic}, Turn: {turn}"
        result = self._generate(system, user)
        if result:
            return result
        return self._fallback.get_user_message(scenario, topic, turn)

    def get_agent_response(
        self,
        scenario: str,
        topic: str,
        turn: int = 1,
        tool_results: dict[str, Any] | None = None,
    ) -> str:
        tool_context = ""
        if tool_results:
            tool_context = f"\nTool results: {json.dumps(tool_results)}"
        system = (
            f"Generate a realistic agent response for a {scenario.replace('_', ' ')} "
            f"scenario about {topic.replace('_', ' ')}. Turn {turn}.{tool_context} "
            f"Write only the agent's response, nothing else."
        )
        user = f"Scenario: {scenario}, Topic: {topic}, Turn: {turn}"
        result = self._generate(system, user)
        if result:
            return result
        return self._fallback.get_agent_response(scenario, topic, turn, tool_results)

    def get_system_prompt(self, scenario: str, agent_name: str) -> str:
        system = (
            "Generate a system prompt for an AI agent. "
            f"The agent is named {agent_name} in a {scenario.replace('_', ' ')} scenario. "
            "Write only the system prompt, nothing else."
        )
        user = f"Agent: {agent_name}, Scenario: {scenario}"
        result = self._generate(system, user)
        if result:
            return result
        return self._fallback.get_system_prompt(scenario, agent_name)

    def get_tool_input(self, action_name: str, topic: str) -> dict[str, Any]:
        return self._fallback.get_tool_input(action_name, topic)

    def get_tool_output(self, action_name: str, topic: str) -> dict[str, Any]:
        return self._fallback.get_tool_output(action_name, topic)

    def get_topics(self, scenario: str) -> list[str]:
        return self._fallback.get_topics(scenario)

    def get_agent_names(self, scenario: str) -> list[str]:
        return self._fallback.get_agent_names(scenario)

    def get_tool_names(self, scenario: str, topic: str) -> list[str]:
        return self._fallback.get_tool_names(scenario, topic)
