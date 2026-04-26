"""
STRATIX LangChain Memory Tracing

Wraps LangChain memory to emit agent.state.change events.
"""

from __future__ import annotations

import json
import time
from typing import Any

from layerlens.instrument.adapters.frameworks.langchain.state import LangChainMemoryAdapter


class TracedMemory:
    """
    Wrapper around LangChain memory that emits state change events.

    Proxies all calls to the underlying memory while tracking changes
    and emitting agent.state.change events.

    Usage:
        from langchain.memory import ConversationBufferMemory  # type: ignore[import-untyped,unused-ignore]

        memory = ConversationBufferMemory()
        traced_memory = TracedMemory(memory, stratix_instance)

        # Use as normal
        traced_memory.save_context({"input": "hello"}, {"output": "hi"})
    """

    def __init__(
        self,
        memory: Any,
        stratix_instance: Any = None,
        memory_service: Any | None = None,
    ) -> None:
        """
        Initialize the traced memory.

        Args:
            memory: LangChain memory instance
            stratix_instance: STRATIX SDK instance
            memory_service: Optional AgentMemoryService for episodic memory storage.
                When provided, save_context() will persist a summary of each
                interaction as an episodic memory entry.
        """
        self._memory = memory
        self._stratix = stratix_instance
        self._memory_service = memory_service
        self._adapter = LangChainMemoryAdapter(memory)
        self._last_hash: str | None = None

    def save_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        """
        Save context to memory with state change tracking.

        Args:
            inputs: Input dictionary
            outputs: Output dictionary
        """
        # Snapshot before
        before_hash = self._adapter.get_hash()

        # Call underlying memory
        self._memory.save_context(inputs, outputs)

        # Snapshot after
        after_hash = self._adapter.get_hash()

        # Emit state change if changed
        if before_hash != after_hash:
            self._emit_state_change(before_hash, after_hash, "save_context")

        # Store episodic memory if memory_service is provided
        if self._memory_service is not None:
            self._store_episodic_memory(inputs, outputs)

        self._last_hash = after_hash

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Load memory variables.

        Args:
            inputs: Input dictionary

        Returns:
            Memory variables
        """
        return self._memory.load_memory_variables(inputs)  # type: ignore[no-any-return]

    def clear(self) -> None:
        """Clear memory with state change tracking."""
        before_hash = self._adapter.get_hash()

        self._memory.clear()

        after_hash = self._adapter.get_hash()

        if before_hash != after_hash:
            self._emit_state_change(before_hash, after_hash, "clear")

        self._last_hash = after_hash

    def _emit_state_change(
        self,
        before_hash: str,
        after_hash: str,
        trigger: str,
    ) -> None:
        """Emit agent.state.change event."""
        if self._stratix and hasattr(self._stratix, "emit"):
            self._stratix.emit(
                "agent.state.change",
                {
                    "memory_type": type(self._memory).__name__,
                    "before_hash": before_hash,
                    "after_hash": after_hash,
                    "trigger": trigger,
                    "timestamp_ns": time.time_ns(),
                },
            )

    def _store_episodic_memory(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        """Store a conversation turn as episodic memory via AgentMemoryService.

        Only called when ``memory_service`` was provided at construction time.
        Failures are logged and swallowed to avoid disrupting normal operation.
        """
        try:
            from layerlens.instrument._vendored.memory_models import MemoryEntry

            timestamp = int(time.time())
            summary = json.dumps(
                {"inputs": inputs, "outputs": outputs},
                default=str,
            )
            entry = MemoryEntry(
                org_id=getattr(self._stratix, "org_id", ""),
                agent_id=getattr(self._stratix, "agent_id", "langchain"),
                memory_type="episodic",
                key=f"conversation_{timestamp}",
                content=summary,
                importance=0.5,
                metadata={"source": "langchain_traced_memory"},
            )
            self._memory_service.store(entry)  # type: ignore[union-attr]
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "Failed to store episodic memory from save_context",
                exc_info=True,
            )

    @property
    def memory_variables(self) -> list[str]:
        """Get memory variable names."""
        return self._memory.memory_variables  # type: ignore[no-any-return]

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to underlying memory."""
        return getattr(self._memory, name)


def wrap_memory(
    memory: Any,
    stratix_instance: Any = None,
) -> TracedMemory:
    """
    Wrap a LangChain memory instance with STRATIX tracing.

    Args:
        memory: LangChain memory instance
        stratix_instance: STRATIX SDK instance

    Returns:
        TracedMemory wrapper
    """
    return TracedMemory(memory, stratix_instance)


class MemoryMutationTracker:
    """
    Tracks memory mutations for a conversation.

    Useful for tracking all memory changes across multiple
    LangChain invocations.
    """

    def __init__(self, stratix_instance: Any = None) -> None:
        """
        Initialize the mutation tracker.

        Args:
            stratix_instance: STRATIX SDK instance
        """
        self._stratix = stratix_instance
        self._mutations: list[dict[str, Any]] = []

    def track_memory(
        self,
        memory: Any,
        operation: str = "unknown",
    ) -> Any:
        """
        Create a context manager to track memory changes.

        Args:
            memory: LangChain memory instance
            operation: Description of the operation

        Returns:
            Context manager
        """
        return _MemoryTrackingContext(
            memory=memory,
            operation=operation,
            tracker=self,
            stratix=self._stratix,
        )

    def record_mutation(self, mutation: dict[str, Any]) -> None:
        """Record a mutation."""
        self._mutations.append(mutation)

    def get_mutations(self) -> list[dict[str, Any]]:
        """Get all recorded mutations."""
        return self._mutations

    def clear(self) -> None:
        """Clear recorded mutations."""
        self._mutations.clear()


class _MemoryTrackingContext:
    """Context manager for tracking memory changes."""

    def __init__(
        self,
        memory: Any,
        operation: str,
        tracker: MemoryMutationTracker,
        stratix: Any,
    ) -> None:
        self._memory = memory
        self._operation = operation
        self._tracker = tracker
        self._stratix = stratix
        self._adapter = LangChainMemoryAdapter(memory)
        self._before_snapshot = None

    def __enter__(self) -> _MemoryTrackingContext:
        self._before_snapshot = self._adapter.snapshot()  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        after_snapshot = self._adapter.snapshot()

        if self._adapter.has_changed(self._before_snapshot, after_snapshot):  # type: ignore[arg-type]
            diff = self._adapter.diff(self._before_snapshot, after_snapshot)  # type: ignore[arg-type]

            mutation = {
                "operation": self._operation,
                "before_hash": self._before_snapshot.hash,  # type: ignore[attr-defined]
                "after_hash": after_snapshot.hash,
                "diff": diff,
                "timestamp_ns": time.time_ns(),
            }

            self._tracker.record_mutation(mutation)

            # Emit event
            if self._stratix and hasattr(self._stratix, "emit"):
                self._stratix.emit(
                    "agent.state.change",
                    {
                        "memory_type": self._before_snapshot.memory_type,  # type: ignore[attr-defined]
                        "before_hash": self._before_snapshot.hash,  # type: ignore[attr-defined]
                        "after_hash": after_snapshot.hash,
                        "operation": self._operation,
                    },
                )
