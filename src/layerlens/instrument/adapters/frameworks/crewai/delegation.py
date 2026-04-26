"""
CrewAI Delegation Detection

Tracks delegation in hierarchical CrewAI processes and emits agent.handoff events.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from layerlens.instrument.adapters.frameworks.crewai.lifecycle import CrewAIAdapter

logger = logging.getLogger(__name__)


class CrewDelegationTracker:
    """Tracks delegation in hierarchical CrewAI processes."""

    def __init__(self, adapter: CrewAIAdapter) -> None:
        self._adapter = adapter
        self._lock = threading.Lock()
        self._delegation_count = 0

    @property
    def delegation_count(self) -> int:
        return self._delegation_count

    def track_delegation(
        self,
        from_agent: str,
        to_agent: str,
        context: Any = None,
    ) -> None:
        """
        Record a delegation from one agent to another.

        Emits an agent.handoff (cross-cutting, always enabled) event.

        Args:
            from_agent: Role/name of the delegating agent
            to_agent: Role/name of the delegate agent
            context: Optional context passed with the delegation
        """
        with self._lock:
            self._delegation_count += 1
            delegation_seq = self._delegation_count

        context_str = self._summarize_context(context)
        context_hash = self._hash_context(context_str)

        try:
            self._adapter.emit_dict_event(
                "agent.handoff",
                {
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "reason": "delegation",
                    "context_hash": context_hash,
                    "context_preview": context_str[:500] if context_str else None,
                    "delegation_seq": delegation_seq,
                },
            )
        except Exception:
            logger.warning("Failed to emit delegation handoff", exc_info=True)

    def _summarize_context(self, context: Any) -> str:
        """Safely summarize delegation context."""
        if context is None:
            return ""
        try:
            if isinstance(context, str):
                return context
            return str(context)
        except Exception:
            return "<unserializable>"

    def _hash_context(self, context_str: str) -> str:
        """SHA-256 hash of context string."""
        return hashlib.sha256(context_str.encode("utf-8", errors="replace")).hexdigest()
