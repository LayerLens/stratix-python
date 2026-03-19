"""
AutoGen Human-in-the-Loop Tracing

Traces human interactions through UserProxyAgent, capturing requests,
responses, and approval patterns.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from layerlens.instrument.adapters.autogen.lifecycle import AutoGenAdapter

logger = logging.getLogger(__name__)


class HumanProxyTracer:
    """
    Traces human interactions through UserProxyAgent.

    Wraps get_human_input() to capture human requests and responses,
    measure response latency, and detect approval patterns.
    """

    def __init__(self, adapter: AutoGenAdapter) -> None:
        self._adapter = adapter
        self._lock = threading.Lock()
        self._original_get_human_input: Callable | None = None
        self._interaction_count: int = 0

    @property
    def interaction_count(self) -> int:
        return self._interaction_count

    def wrap_agent(self, agent: Any) -> Any:
        """
        Wrap a UserProxyAgent with human interaction tracing.

        Args:
            agent: An AutoGen UserProxyAgent instance

        Returns:
            The wrapped agent (same object, modified in-place)
        """
        if hasattr(agent, "get_human_input"):
            self._original_get_human_input = agent.get_human_input
            agent.get_human_input = self._create_traced_get_human_input(
                agent, agent.get_human_input
            )
        agent._stratix_human_tracer = self
        return agent

    def _create_traced_get_human_input(
        self,
        agent: Any,
        original: Callable,
    ) -> Callable:
        """Create a traced version of get_human_input."""
        tracer = self

        def traced_get_human_input(prompt: str = "", **kwargs: Any) -> str:
            start_ns = time.time_ns()
            with tracer._lock:
                tracer._interaction_count += 1
                interaction_seq = tracer._interaction_count

            # Emit request event
            try:
                agent_name = getattr(agent, "name", str(agent))
                tracer._adapter.emit_dict_event("agent.input", {
                    "framework": "autogen",
                    "role": "HUMAN",
                    "input_type": "human_input_request",
                    "agent": agent_name,
                    "prompt": prompt[:500] if prompt else "",
                    "interaction_seq": interaction_seq,
                })
            except Exception:
                logger.warning("Error emitting human input request", exc_info=True)

            # Call original
            response = original(prompt, **kwargs)

            # Emit response event
            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                input_type = tracer._classify_input(response)
                tracer._adapter.emit_dict_event("agent.input", {
                    "framework": "autogen",
                    "role": "HUMAN",
                    "input_type": input_type,
                    "agent": agent_name,
                    "response_preview": response[:500] if response else "",
                    "response_latency_ms": elapsed_ms,
                    "interaction_seq": interaction_seq,
                })
            except Exception:
                logger.warning("Error emitting human input response", exc_info=True)

            return response

        traced_get_human_input._stratix_original = original
        return traced_get_human_input

    def _classify_input(self, response: str) -> str:
        """
        Classify the type of human input.

        Returns:
            Input type string: "approval", "rejection", "auto_reply",
            "custom_input", or "empty"
        """
        if not response:
            return "empty"

        lower = response.strip().lower()

        # Auto-reply detection
        if lower in ("", "exit"):
            return "auto_reply"

        # Approval patterns
        if lower in ("y", "yes", "approve", "ok", "okay", "sure", "proceed", "continue"):
            return "approval"

        # Rejection patterns
        if lower in ("n", "no", "reject", "deny", "stop", "cancel", "abort"):
            return "rejection"

        return "custom_input"
