"""
CrewAI Callback Handler

Routes CrewAI callback events to the CrewAIAdapter lifecycle hooks.
All methods wrap adapter calls in try/except to prevent tracing from
crashing the crew execution.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from layerlens.instrument.adapters.crewai.lifecycle import CrewAIAdapter

logger = logging.getLogger(__name__)


class STRATIXCrewCallback:
    """
    CrewAI callback handler that routes events to CrewAIAdapter.

    Implements the CrewAI callback protocol and translates framework
    callbacks into STRATIX lifecycle hook calls.
    """

    def __init__(self, adapter: CrewAIAdapter) -> None:
        self._adapter = adapter
        self._lock = threading.Lock()
        self._seen_agents: set[str] = set()
        self._task_counter: int = 0
        self._current_task_start_ns: int = 0

    # --- CrewAI callback methods ---

    def on_crew_start(self, crew: Any = None, inputs: Any = None) -> None:
        """Called when crew execution begins."""
        try:
            self._adapter.on_crew_start(crew_input=inputs)
        except Exception:
            logger.warning("Error in on_crew_start callback", exc_info=True)

    def on_crew_end(self, crew: Any = None, output: Any = None) -> None:
        """Called when crew execution completes."""
        try:
            self._adapter.on_crew_end(crew_output=output)
        except Exception:
            logger.warning("Error in on_crew_end callback", exc_info=True)

    def on_task_start(self, task: Any = None) -> None:
        """Called when a task begins execution."""
        try:
            with self._lock:
                self._task_counter += 1
                self._current_task_start_ns = time.time_ns()
                task_counter = self._task_counter

            description = getattr(task, "description", None) or ""
            expected_output = getattr(task, "expected_output", None)
            agent = getattr(task, "agent", None)
            agent_role = getattr(agent, "role", None) if agent else None

            # Emit agent config on first encounter
            if agent and agent_role:
                with self._lock:
                    seen = agent_role in self._seen_agents
                    if not seen:
                        self._seen_agents.add(agent_role)
                if not seen:
                    self._adapter._emit_agent_config(agent)

            self._adapter.on_task_start(
                task_description=description,
                agent_role=agent_role,
                expected_output=expected_output,
                task_order=task_counter,
            )
        except Exception:
            logger.warning("Error in on_task_start callback", exc_info=True)

    def on_task_end(self, task: Any = None, output: Any = None) -> None:
        """Called when a task completes."""
        try:
            agent = getattr(task, "agent", None) if task else None
            agent_role = getattr(agent, "role", None) if agent else None

            self._adapter.on_task_end(
                task_output=output,
                agent_role=agent_role,
                task_order=self._task_counter,
            )
        except Exception:
            logger.warning("Error in on_task_end callback", exc_info=True)

    def on_agent_action(self, agent: Any = None, action: Any = None) -> None:
        """Called when an agent takes an action."""
        try:
            role = getattr(agent, "role", None) if agent else None

            # Emit agent config on first encounter
            if agent and role:
                with self._lock:
                    seen = role in self._seen_agents
                    if not seen:
                        self._seen_agents.add(role)
                if not seen:
                    self._adapter._emit_agent_config(agent)
        except Exception:
            logger.warning("Error in on_agent_action callback", exc_info=True)

    def on_agent_end(self, agent: Any = None, output: Any = None) -> None:
        """Called when an agent finishes processing."""
        try:
            role = getattr(agent, "role", None) if agent else None
            self._adapter.emit_dict_event("agent.state.change", {
                "framework": "crewai",
                "agent_role": role,
                "event_subtype": "agent_complete",
                "output": self._adapter._safe_serialize(output),
            })
        except Exception:
            logger.warning("Error in on_agent_end callback", exc_info=True)

    def on_tool_use(
        self,
        agent: Any = None,
        tool_name: str = "",
        tool_input: Any = None,
        tool_output: Any = None,
    ) -> None:
        """Called when an agent uses a tool."""
        try:
            self._adapter.on_tool_use(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
            )
        except Exception:
            logger.warning("Error in on_tool_use callback", exc_info=True)

    def on_llm_call(self, agent: Any = None, response: Any = None) -> None:
        """Called when an LLM call completes."""
        try:
            provider = None
            model = None
            tokens_prompt = None
            tokens_completion = None

            if response is not None:
                # Try to extract model info from response
                model = (
                    getattr(response, "model", None)
                    or getattr(response, "model_name", None)
                )
                provider = self._detect_provider(response)

                # Token usage
                usage = getattr(response, "usage", None)
                if usage:
                    if isinstance(usage, dict):
                        tokens_prompt = usage.get("prompt_tokens")
                        tokens_completion = usage.get("completion_tokens")
                    else:
                        tokens_prompt = getattr(usage, "prompt_tokens", None)
                        tokens_completion = getattr(usage, "completion_tokens", None)

            self._adapter.on_llm_call(
                provider=provider,
                model=model,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
            )
        except Exception:
            logger.warning("Error in on_llm_call callback", exc_info=True)

    # --- Step/task callbacks (attached to crew) ---

    def on_step(self, step_output: Any = None) -> None:
        """
        CrewAI step_callback handler.

        Called after each agent step. Routes to appropriate handler.
        """
        try:
            # Extract tool usage from step output if present
            tool_name = getattr(step_output, "tool", None)
            if tool_name:
                tool_input = getattr(step_output, "tool_input", None)
                tool_output = getattr(step_output, "result", None)
                self._adapter.on_tool_use(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=tool_output,
                )

            # Check for delegation
            delegated_to = getattr(step_output, "delegated_to", None)
            if delegated_to:
                delegated_from = getattr(step_output, "agent", None)
                from_role = getattr(delegated_from, "role", "unknown") if delegated_from else "unknown"
                to_role = getattr(delegated_to, "role", str(delegated_to)) if delegated_to else "unknown"
                context = getattr(step_output, "result", None)
                self._adapter.on_delegation(from_role, to_role, context)
        except Exception:
            logger.warning("Error in on_step callback", exc_info=True)

    def on_task_complete(self, task_output: Any = None) -> None:
        """
        CrewAI task_callback handler.

        Called after each task completes.
        """
        try:
            self._adapter.on_task_end(task_output=task_output)
        except Exception:
            logger.warning("Error in on_task_complete callback", exc_info=True)

    # --- Internal helpers ---

    def _detect_provider(self, response: Any) -> str | None:
        """Detect LLM provider from response object."""
        try:
            class_name = type(response).__module__ or ""
            lower = class_name.lower()
            if "openai" in lower:
                return "openai"
            if "anthropic" in lower:
                return "anthropic"
            if "google" in lower:
                return "google"
            if "cohere" in lower:
                return "cohere"
        except Exception:
            pass
        return None
