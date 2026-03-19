"""Abstract base class for content providers.

Content providers implement the 3-tier content generation strategy:
- Tier 1 (Seed): Load from agentforce-synthetic-data/ Langfuse JSONs
- Tier 2 (Template): Parameterized templates per scenario (no LLM)
- Tier 3 (LLM): Optional LLM generation with disk cache
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ContentProvider(ABC):
    """Abstract base for content generation.

    Each tier implements this interface to provide scenario-specific
    content for trace simulation.
    """

    @abstractmethod
    def get_user_message(self, scenario: str, topic: str, turn: int = 1) -> str:
        """Get a user message for the given scenario/topic/turn."""

    @abstractmethod
    def get_agent_response(
        self,
        scenario: str,
        topic: str,
        turn: int = 1,
        tool_results: dict[str, Any] | None = None,
    ) -> str:
        """Get an agent response for the given scenario/topic/turn."""

    @abstractmethod
    def get_system_prompt(self, scenario: str, agent_name: str) -> str:
        """Get the system prompt for an agent in a scenario."""

    @abstractmethod
    def get_tool_input(self, action_name: str, topic: str) -> dict[str, Any]:
        """Get tool input parameters for a given action."""

    @abstractmethod
    def get_tool_output(self, action_name: str, topic: str) -> dict[str, Any]:
        """Get tool output for a given action."""

    def get_topics(self, scenario: str) -> list[str]:
        """Get available topics for a scenario. Override for specific topic lists."""
        return []

    def get_agent_names(self, scenario: str) -> list[str]:
        """Get agent names for a scenario. Override for specific agent lists."""
        return []

    def get_tool_names(self, scenario: str, topic: str) -> list[str]:
        """Get tool names for a scenario/topic. Override for specific tool lists."""
        return []
