"""
STRATIX LangChain Agent Instrumentation

Provides automatic instrumentation for LangChain agents.
"""

from __future__ import annotations

import time
import logging
from typing import TYPE_CHECKING, Any
from dataclasses import field, dataclass

from layerlens.instrument.adapters.frameworks.langchain.callbacks import LayerLensCallbackHandler

if TYPE_CHECKING:
    from layerlens.instrument.adapters._base.adapter import BaseAdapter

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """Represents a single step in agent execution."""

    step_number: int
    action: str | None = None
    action_input: Any | None = None
    observation: str | None = None
    timestamp_ns: int | None = None


@dataclass
class AgentExecution:
    """Tracks a complete agent execution."""

    agent_type: str
    start_time_ns: int
    end_time_ns: int | None = None
    input: str | dict[str, Any] | None = None
    output: Any | None = None
    steps: list[AgentStep] = field(default_factory=list)
    error: str | None = None


class TracedAgent:
    """
    Wrapper around a LangChain agent with STRATIX tracing.

    Captures:
    - Agent input/output
    - Intermediate reasoning steps
    - Tool calls during execution
    - LLM invocations

    Usage:
        from langchain.agents import create_react_agent  # type: ignore[import-untyped,unused-ignore]

        agent = create_react_agent(llm, tools, prompt)
        traced_agent = TracedAgent(agent, stratix_instance)

        # Use as normal
        result = traced_agent.invoke({"input": "What is the weather?"})
    """

    def __init__(
        self,
        agent: Any,
        stratix_instance: Any = None,
        adapter: BaseAdapter | None = None,
    ) -> None:
        """
        Initialize the traced agent.

        Args:
            agent: LangChain agent instance (AgentExecutor or similar)
            stratix_instance: STRATIX SDK instance (legacy)
            adapter: BaseAdapter instance (new-style)
        """
        self._agent = agent
        self._stratix = stratix_instance
        self._adapter = adapter
        self._handler = LayerLensCallbackHandler(
            stratix=adapter._stratix if adapter else None,
            capture_config=adapter.capture_config if adapter else None,
            stratix_instance=stratix_instance,
        )
        self._agent_type = type(agent).__name__
        self._executions: list[AgentExecution] = []
        self._current_execution: AgentExecution | None = None
        self._step_counter = 0

    def invoke(
        self,
        input: dict[str, Any] | str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Invoke the agent with tracing.

        Args:
            input: Agent input
            config: Optional config
            **kwargs: Additional arguments

        Returns:
            Agent output
        """
        execution = AgentExecution(
            agent_type=self._agent_type,
            start_time_ns=time.time_ns(),
            input=input,
        )
        self._executions.append(execution)
        self._current_execution = execution
        self._step_counter = 0

        # Emit agent input event
        self._emit_agent_input(input)

        # Inject callback handler
        callbacks = kwargs.get("callbacks", [])
        if self._handler not in callbacks:
            callbacks = list(callbacks) + [self._handler]
            kwargs["callbacks"] = callbacks

        try:
            result = self._agent.invoke(input, config, **kwargs)

            execution.end_time_ns = time.time_ns()
            execution.output = result

            # Emit agent output event
            self._emit_agent_output(execution)

            return result  # type: ignore[no-any-return]

        except Exception as e:
            execution.end_time_ns = time.time_ns()
            execution.error = str(e)
            self._emit_agent_output(execution)
            raise
        finally:
            self._current_execution = None

    async def ainvoke(
        self,
        input: dict[str, Any] | str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Async invoke the agent with tracing.

        Args:
            input: Agent input
            config: Optional config
            **kwargs: Additional arguments

        Returns:
            Agent output
        """
        execution = AgentExecution(
            agent_type=self._agent_type,
            start_time_ns=time.time_ns(),
            input=input,
        )
        self._executions.append(execution)
        self._current_execution = execution
        self._step_counter = 0

        self._emit_agent_input(input)

        callbacks = kwargs.get("callbacks", [])
        if self._handler not in callbacks:
            callbacks = list(callbacks) + [self._handler]
            kwargs["callbacks"] = callbacks

        try:
            result = await self._agent.ainvoke(input, config, **kwargs)

            execution.end_time_ns = time.time_ns()
            execution.output = result

            self._emit_agent_output(execution)

            return result  # type: ignore[no-any-return]

        except Exception as e:
            execution.end_time_ns = time.time_ns()
            execution.error = str(e)
            self._emit_agent_output(execution)
            raise
        finally:
            self._current_execution = None

    def run(self, *args: Any, **kwargs: Any) -> str:
        """
        Run the agent (deprecated method).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Agent output string
        """
        callbacks = kwargs.get("callbacks", [])
        if self._handler not in callbacks:
            callbacks = list(callbacks) + [self._handler]
            kwargs["callbacks"] = callbacks

        return self._agent.run(*args, **kwargs)  # type: ignore[no-any-return]

    def record_step(
        self,
        action: str | None = None,
        action_input: Any = None,
        observation: str | None = None,
    ) -> None:
        """
        Record an intermediate step.

        Called automatically by callback handler but can be
        called manually for custom step tracking.

        Args:
            action: The action taken
            action_input: Input to the action
            observation: Result of the action
        """
        if self._current_execution is None:
            return

        self._step_counter += 1
        step = AgentStep(
            step_number=self._step_counter,
            action=action,
            action_input=action_input,
            observation=observation,
            timestamp_ns=time.time_ns(),
        )
        self._current_execution.steps.append(step)

    def _emit_agent_input(self, input: Any) -> None:
        """Emit agent.input event."""
        payload = {
            "agent_type": self._agent_type,
            "input": input,
            "timestamp_ns": time.time_ns(),
        }

        if self._adapter is not None:
            try:
                from layerlens.instrument._vendored.events import (
                    MessageRole,
                    AgentInputEvent,
                )

                msg = str(input) if not isinstance(input, str) else input
                typed_payload = AgentInputEvent.create(message=msg, role=MessageRole.HUMAN)
                self._adapter.emit_event(typed_payload)
                return
            except Exception:
                logger.debug("Typed event emission failed, falling back to legacy", exc_info=True)

        if self._stratix and hasattr(self._stratix, "emit"):
            self._stratix.emit("agent.input", payload)

    def _emit_agent_output(self, execution: AgentExecution) -> None:
        """Emit agent.output event."""
        duration_ns = (execution.end_time_ns or 0) - execution.start_time_ns
        payload = {
            "agent_type": execution.agent_type,
            "input": execution.input,
            "output": execution.output,
            "num_steps": len(execution.steps),
            "duration_ns": duration_ns,
            "error": execution.error,
        }

        if self._adapter is not None:
            try:
                from layerlens.instrument._vendored.events import AgentOutputEvent

                msg = str(execution.output) if execution.output else ""
                typed_payload = AgentOutputEvent.create(message=msg)
                self._adapter.emit_event(typed_payload)
                return
            except Exception:
                logger.debug("Typed event emission failed, falling back to legacy", exc_info=True)

        if self._stratix and hasattr(self._stratix, "emit"):
            self._stratix.emit("agent.output", payload)

    @property
    def callback_handler(self) -> LayerLensCallbackHandler:
        """Get the callback handler."""
        return self._handler

    @property
    def executions(self) -> list[AgentExecution]:
        """Get all recorded executions."""
        return self._executions

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to underlying agent."""
        return getattr(self._agent, name)


def instrument_agent(
    agent: Any,
    stratix_instance: Any = None,
    adapter: BaseAdapter | None = None,
) -> TracedAgent:
    """
    Instrument a LangChain agent with STRATIX tracing.

    Args:
        agent: LangChain agent instance
        stratix_instance: STRATIX SDK instance
        adapter: BaseAdapter instance (new-style)

    Returns:
        TracedAgent wrapper
    """
    return TracedAgent(agent, stratix_instance, adapter=adapter)


class AgentTracer:
    """
    Tracer for multiple agent executions.

    Provides a unified view of agent activity across
    multiple invocations.
    """

    def __init__(self, stratix_instance: Any = None, adapter: BaseAdapter | None = None) -> None:
        """
        Initialize the agent tracer.

        Args:
            stratix_instance: STRATIX SDK instance
            adapter: BaseAdapter instance (new-style)
        """
        self._stratix = stratix_instance
        self._adapter = adapter
        self._agents: dict[str, TracedAgent] = {}
        self._all_executions: list[AgentExecution] = []

    def trace(self, agent: Any, name: str | None = None) -> TracedAgent:
        """
        Start tracing an agent.

        Args:
            agent: LangChain agent
            name: Optional name for the agent

        Returns:
            TracedAgent wrapper
        """
        agent_name = name or type(agent).__name__
        traced = TracedAgent(agent, self._stratix, adapter=self._adapter)
        self._agents[agent_name] = traced
        return traced

    def get_agent(self, name: str) -> TracedAgent | None:
        """Get a traced agent by name."""
        return self._agents.get(name)

    def get_all_executions(self) -> list[AgentExecution]:
        """Get all executions across all agents."""
        all_execs = []
        for agent in self._agents.values():
            all_execs.extend(agent.executions)
        return sorted(all_execs, key=lambda e: e.start_time_ns)

    def get_total_steps(self) -> int:
        """Get total number of steps across all executions."""
        return sum(len(e.steps) for agent in self._agents.values() for e in agent.executions)
