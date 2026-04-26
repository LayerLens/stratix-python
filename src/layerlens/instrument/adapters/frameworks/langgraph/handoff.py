"""
STRATIX LangGraph Handoff Detection

Detects and traces agent handoffs in multi-agent LangGraph workflows.
"""

from __future__ import annotations

import time
import logging
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass
from collections.abc import Callable

if TYPE_CHECKING:
    from layerlens.instrument.adapters._base.adapter import BaseAdapter

logger = logging.getLogger(__name__)


@dataclass
class AgentHandoff:
    """Represents a handoff between agents."""

    from_agent: str
    to_agent: str
    timestamp_ns: int
    context: dict[str, Any] | None = None
    reason: str | None = None


class HandoffDetector:
    """
    Detects agent handoffs in LangGraph multi-agent workflows.

    Handoffs occur when:
    - A supervisor routes to a different agent
    - Control transfers between agent nodes
    - An agent explicitly delegates to another

    Usage:
        detector = HandoffDetector(stratix_instance)

        # Register agents
        detector.register_agent("researcher")
        detector.register_agent("writer")

        # Check for handoff
        if detector.is_handoff("researcher", "writer", state):
            detector.emit_handoff("researcher", "writer", state)
    """

    def __init__(
        self,
        stratix_instance: Any = None,
        adapter: BaseAdapter | None = None,
    ) -> None:
        """
        Initialize the handoff detector.

        Args:
            stratix_instance: STRATIX SDK instance (legacy)
            adapter: BaseAdapter instance (new-style)
        """
        self._stratix = stratix_instance
        self._adapter = adapter
        self._registered_agents: set[str] = set()
        self._current_agent: str | None = None
        self._handoffs: list[AgentHandoff] = []

    def register_agent(self, agent_name: str) -> None:
        """
        Register an agent for handoff tracking.

        Args:
            agent_name: Name of the agent
        """
        self._registered_agents.add(agent_name)

    def register_agents(self, *agent_names: str) -> None:
        """
        Register multiple agents for handoff tracking.

        Args:
            *agent_names: Names of agents
        """
        for name in agent_names:
            self._registered_agents.add(name)

    def set_current_agent(self, agent_name: str) -> None:
        """
        Set the currently active agent.

        Args:
            agent_name: Name of the current agent
        """
        self._current_agent = agent_name

    def is_handoff(
        self,
        from_agent: str,
        to_agent: str,
        state: dict[str, Any] | None = None,
    ) -> bool:
        """
        Check if this represents a handoff.

        Args:
            from_agent: Source agent
            to_agent: Destination agent
            state: Current state (optional)

        Returns:
            True if this is a handoff
        """
        # Different agents = handoff
        return from_agent != to_agent

    def detect_handoff(
        self,
        next_agent: str,
        state: dict[str, Any] | None = None,
    ) -> AgentHandoff | None:
        """
        Detect if transitioning to next_agent is a handoff.

        Args:
            next_agent: The next agent to execute
            state: Current state

        Returns:
            AgentHandoff if detected, None otherwise
        """
        if self._current_agent and self._current_agent != next_agent:
            handoff = AgentHandoff(
                from_agent=self._current_agent,
                to_agent=next_agent,
                timestamp_ns=time.time_ns(),
                context=self._extract_context(state) if state else None,
            )
            self._handoffs.append(handoff)
            self._current_agent = next_agent
            self._emit_handoff(handoff)
            return handoff

        self._current_agent = next_agent
        return None

    def emit_handoff(
        self,
        from_agent: str,
        to_agent: str,
        state: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> AgentHandoff:
        """
        Explicitly emit a handoff event.

        Args:
            from_agent: Source agent
            to_agent: Destination agent
            state: Current state
            reason: Reason for handoff

        Returns:
            Created AgentHandoff
        """
        handoff = AgentHandoff(
            from_agent=from_agent,
            to_agent=to_agent,
            timestamp_ns=time.time_ns(),
            context=self._extract_context(state) if state else None,
            reason=reason,
        )
        self._handoffs.append(handoff)
        self._current_agent = to_agent
        self._emit_handoff(handoff)
        return handoff

    def _extract_context(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract relevant context from state for handoff tracking."""
        context = {}

        # Extract common handoff-related state keys
        for key in ["task", "current_task", "objective", "query", "messages"]:
            if key in state:
                value = state[key]
                # Truncate long values
                if isinstance(value, str) and len(value) > 500:
                    context[key] = value[:500] + "..."
                elif isinstance(value, list) and len(value) > 10:
                    context[key] = f"[{len(value)} items]"
                else:
                    context[key] = value

        return context

    def _emit_handoff(self, handoff: AgentHandoff) -> None:
        """Emit agent.handoff event via adapter (preferred) or legacy path."""
        payload_dict = {
            "from_agent": handoff.from_agent,
            "to_agent": handoff.to_agent,
            "timestamp_ns": handoff.timestamp_ns,
            "context": handoff.context,
            "reason": handoff.reason,
        }

        # New-style: route through adapter.emit_event
        if self._adapter is not None:
            try:
                import json
                import hashlib

                from layerlens.instrument._vendored.events import AgentHandoffEvent

                context_str = json.dumps(handoff.context or {}, sort_keys=True)
                context_hash = "sha256:" + hashlib.sha256(context_str.encode()).hexdigest()
                typed_payload = AgentHandoffEvent.create(
                    from_agent=handoff.from_agent,
                    to_agent=handoff.to_agent,
                    handoff_context_hash=context_hash,
                )
                self._adapter.emit_event(typed_payload)
                return
            except Exception:
                logger.debug("Typed event emission failed, falling back to legacy", exc_info=True)

        # Legacy fallback
        if self._stratix and hasattr(self._stratix, "emit"):
            self._stratix.emit("agent.handoff", payload_dict)


def detect_handoff(
    from_agent: str,
    to_agent: str,
    stratix_instance: Any = None,
    adapter: BaseAdapter | None = None,
    state: dict[str, Any] | None = None,
    reason: str | None = None,
) -> AgentHandoff | None:
    """
    Utility function to detect and emit a handoff event.

    Args:
        from_agent: Source agent
        to_agent: Destination agent
        stratix_instance: STRATIX SDK instance
        adapter: BaseAdapter instance (new-style)
        state: Current state
        reason: Reason for handoff

    Returns:
        AgentHandoff if detected, None if same agent
    """
    if from_agent == to_agent:
        return None

    detector = HandoffDetector(stratix_instance, adapter=adapter)
    return detector.emit_handoff(from_agent, to_agent, state, reason)


class SupervisorHandoffTracker:
    """
    Tracks handoffs in supervisor-style multi-agent architectures.

    In a supervisor architecture, a supervisor agent routes tasks
    to worker agents. This tracker monitors these transitions.

    Usage:
        tracker = SupervisorHandoffTracker(stratix_instance)

        # In supervisor node
        def supervisor(state):
            next_agent = decide_next_agent(state)
            tracker.route_to(next_agent, state)
            return {"next": next_agent}
    """

    def __init__(
        self,
        stratix_instance: Any = None,
        adapter: BaseAdapter | None = None,
        supervisor_name: str = "supervisor",
    ) -> None:
        """
        Initialize the supervisor tracker.

        Args:
            stratix_instance: STRATIX SDK instance
            adapter: BaseAdapter instance (new-style)
            supervisor_name: Name of the supervisor agent
        """
        self._detector = HandoffDetector(stratix_instance, adapter=adapter)
        self._supervisor_name = supervisor_name
        self._detector.register_agent(supervisor_name)
        self._detector.set_current_agent(supervisor_name)
        self._last_worker: str | None = None

    def register_worker(self, worker_name: str) -> None:
        """
        Register a worker agent.

        Args:
            worker_name: Name of the worker agent
        """
        self._detector.register_agent(worker_name)

    def route_to(
        self,
        worker_name: str,
        state: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> AgentHandoff:
        """
        Track routing from supervisor to worker.

        Args:
            worker_name: Worker to route to
            state: Current state
            reason: Reason for routing decision

        Returns:
            AgentHandoff event
        """
        from_agent = self._last_worker or self._supervisor_name
        handoff = self._detector.emit_handoff(
            from_agent=from_agent,
            to_agent=worker_name,
            state=state,
            reason=reason or f"Supervisor routed to {worker_name}",
        )
        self._last_worker = worker_name
        return handoff

    def return_to_supervisor(
        self,
        state: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> AgentHandoff | None:
        """
        Track return from worker to supervisor.

        Args:
            state: Current state
            reason: Reason for return

        Returns:
            AgentHandoff event or None if already at supervisor
        """
        if self._last_worker:
            handoff = self._detector.emit_handoff(
                from_agent=self._last_worker,
                to_agent=self._supervisor_name,
                state=state,
                reason=reason or "Worker completed, returning to supervisor",
            )
            self._last_worker = None
            return handoff
        return None


def create_handoff_aware_router(
    route_func: Callable[[dict[str, Any]], str],
    stratix_instance: Any = None,
    adapter: BaseAdapter | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Create a router function that tracks handoffs.

    Args:
        route_func: Function that takes state and returns next agent name
        stratix_instance: STRATIX SDK instance
        adapter: BaseAdapter instance (new-style)

    Returns:
        Router function that also emits handoff events
    """
    detector = HandoffDetector(stratix_instance, adapter=adapter)

    def router(state: dict[str, Any]) -> dict[str, Any]:
        next_agent = route_func(state)
        detector.detect_handoff(next_agent, state)
        return {"next": next_agent}

    return router
