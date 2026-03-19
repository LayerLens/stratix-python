"""Base scenario classes.

Scenarios define what happens in a simulated trace: the agent profile,
available topics, typical span structure, and trace generation logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..clock import DeterministicClock
from ..content.base import ContentProvider
from ..span_model import SimulatedTrace
from ..trace_builder import TraceBuilder


@dataclass
class AgentProfile:
    """Profile of an agent in a scenario."""

    name: str
    description: str = ""
    tools: list[str] = field(default_factory=list)
    eval_dimensions: list[str] = field(default_factory=lambda: ["factual_accuracy"])


class BaseScenario(ABC):
    """Abstract base for scenario implementations.

    Each scenario defines the span structure and content
    for a particular business domain.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Scenario name (matches ScenarioName enum value)."""

    @property
    @abstractmethod
    def topics(self) -> list[str]:
        """Available topics for this scenario."""

    @property
    @abstractmethod
    def agents(self) -> list[AgentProfile]:
        """Agent profiles for this scenario."""

    def generate_trace(
        self,
        topic: str,
        provider: str,
        model: str,
        content_provider: ContentProvider,
        clock: DeterministicClock,
        include_content: bool = False,
        seed: int | None = None,
    ) -> SimulatedTrace:
        """Generate a single trace for this scenario.

        Default implementation creates a standard agent → LLM → tools → LLM → eval
        pattern. Override for scenario-specific structures.
        """
        agent = self.agents[0]
        builder = TraceBuilder(seed=seed)
        builder.with_scenario(self.name, topic=topic)

        # Agent span
        builder.add_agent_span(agent.name, description=agent.description)

        # First LLM call (planning/understanding)
        input_msgs: list[dict[str, Any]] = []
        output_msg: dict[str, Any] | None = None
        if include_content:
            system_prompt = content_provider.get_system_prompt(self.name, agent.name)
            user_msg = content_provider.get_user_message(self.name, topic, turn=1)
            agent_resp = content_provider.get_agent_response(self.name, topic, turn=1)
            input_msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            output_msg = {"role": "assistant", "content": agent_resp}

        builder.add_llm_span(
            provider=provider,
            model=model,
            prompt_tokens=clock.randint(150, 500),
            completion_tokens=clock.randint(100, 400),
            temperature=0.7,
            input_messages=input_msgs,
            output_message=output_msg,
        )

        # Tool calls
        tool_names = content_provider.get_tool_names(self.name, topic)
        for tool_name in tool_names[:2]:
            tool_input = (
                content_provider.get_tool_input(tool_name, topic)
                if include_content
                else None
            )
            tool_output = (
                content_provider.get_tool_output(tool_name, topic)
                if include_content
                else None
            )
            builder.add_tool_span(
                name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
            )

        # Second LLM call (response generation)
        builder.add_llm_span(
            provider=provider,
            model=model,
            prompt_tokens=clock.randint(300, 800),
            completion_tokens=clock.randint(150, 500),
            temperature=0.7,
        )

        # Evaluation
        for dimension in agent.eval_dimensions:
            score = clock.uniform(0.7, 1.0)
            builder.add_evaluation_span(
                dimension=dimension,
                score=round(score, 2),
            )

        return builder.build()
