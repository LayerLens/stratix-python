"""Multi-turn conversation builder.

Generates multi-turn conversations as sequences of traces
sharing the same session_id.
"""

from __future__ import annotations

from typing import Any

from .clock import DeterministicClock
from .config import ConversationConfig
from .content.base import ContentProvider
from .identifiers import IDGenerator
from .span_model import SimulatedTrace
from .trace_builder import TraceBuilder


class ConversationBuilder:
    """Build multi-turn conversations as linked traces."""

    def __init__(
        self,
        config: ConversationConfig,
        seed: int | None = None,
    ):
        self._config = config
        self._clock = DeterministicClock(seed=seed)
        self._ids = IDGenerator(seed=seed)

    def build_conversation(
        self,
        scenario: str,
        topic: str,
        provider: str,
        model: str,
        content_provider: ContentProvider,
        include_content: bool = False,
    ) -> list[SimulatedTrace]:
        """Generate a multi-turn conversation as linked traces."""
        if not self._config.enabled:
            return []

        num_turns = self._clock.randint(
            self._config.turns_min, self._config.turns_max
        )
        session_id = self._ids.session_id()
        traces: list[SimulatedTrace] = []

        agent_names = content_provider.get_agent_names(scenario)
        agent_name = agent_names[0] if agent_names else f"{scenario}_Agent"
        tool_names = content_provider.get_tool_names(scenario, topic)

        for turn in range(1, num_turns + 1):
            turn_seed = self._clock.randint(0, 2**31)
            builder = TraceBuilder(seed=turn_seed)
            builder.with_scenario(scenario, topic=topic)
            builder.with_session(session_id=session_id, turn=turn)

            # Agent span
            builder.add_agent_span(agent_name)

            # LLM call with turn-specific content
            input_msgs: list[dict[str, Any]] = []
            output_msg: dict[str, Any] | None = None

            if include_content:
                system_prompt = content_provider.get_system_prompt(scenario, agent_name)
                user_msg = content_provider.get_user_message(scenario, topic, turn=turn)
                agent_resp = content_provider.get_agent_response(
                    scenario, topic, turn=turn
                )
                input_msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ]
                output_msg = {"role": "assistant", "content": agent_resp}

            prompt_tokens = self._clock.randint(150, 500) + (turn * 50)  # Growing context
            completion_tokens = self._clock.randint(100, 400)

            builder.add_llm_span(
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                temperature=0.7,
                input_messages=input_msgs,
                output_message=output_msg,
            )

            # Tool calls (first and last turns)
            if turn == 1 or turn == num_turns:
                for tool_name in tool_names[:1]:
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

            # Second LLM call (response)
            builder.add_llm_span(
                provider=provider,
                model=model,
                prompt_tokens=self._clock.randint(200, 600),
                completion_tokens=self._clock.randint(100, 350),
                temperature=0.7,
            )

            # Evaluation on last turn
            if turn == num_turns:
                eval_score = self._clock.uniform(0.7, 1.0)
                builder.add_evaluation_span(
                    dimension="factual_accuracy",
                    score=round(eval_score, 2),
                )

            trace = builder.build()
            traces.append(trace)

        return traces
