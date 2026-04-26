"""
STRATIX LangChain Memory State Adapter

Adapts LangChain memory for STRATIX state tracking.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any
from dataclasses import dataclass


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""

    memory_type: str
    variables: dict[str, Any]
    hash: str
    timestamp_ns: int
    message_count: int | None = None


class LangChainMemoryAdapter:
    """
    State adapter for LangChain memory classes.

    Supports various LangChain memory types:
    - ConversationBufferMemory
    - ConversationSummaryMemory
    - ConversationBufferWindowMemory
    - Entity memory, etc.

    Usage:
        from langchain.memory import ConversationBufferMemory  # type: ignore[import-untyped,unused-ignore]

        memory = ConversationBufferMemory()
        adapter = LangChainMemoryAdapter(memory)

        # Take snapshot
        before = adapter.snapshot()

        # ... use memory ...

        # Check for changes
        after = adapter.snapshot()
        if adapter.has_changed(before, after):
            print("Memory changed!")
    """

    def __init__(self, memory: Any) -> None:
        """
        Initialize the memory adapter.

        Args:
            memory: LangChain memory instance
        """
        self._memory = memory
        self._memory_type = type(memory).__name__

    def snapshot(self) -> MemorySnapshot:
        """
        Create a snapshot of the current memory state.

        Returns:
            MemorySnapshot with hash for comparison
        """
        import time

        # Get memory variables
        variables = self._get_memory_variables()

        # Count messages if applicable
        message_count = self._count_messages()

        # Compute hash
        hash_value = self._compute_hash(variables)

        return MemorySnapshot(
            memory_type=self._memory_type,
            variables=variables,
            hash=hash_value,
            timestamp_ns=time.time_ns(),
            message_count=message_count,
        )

    def has_changed(self, before: MemorySnapshot, after: MemorySnapshot) -> bool:
        """
        Check if memory has changed between snapshots.

        Args:
            before: Snapshot before operation
            after: Snapshot after operation

        Returns:
            True if memory changed
        """
        return before.hash != after.hash

    def diff(self, before: MemorySnapshot, after: MemorySnapshot) -> dict[str, Any]:
        """
        Compute the difference between two snapshots.

        Args:
            before: Snapshot before operation
            after: Snapshot after operation

        Returns:
            Dictionary describing changes
        """
        added = {}
        removed = {}
        modified = {}

        before_vars = before.variables
        after_vars = after.variables

        before_keys = set(before_vars.keys())
        after_keys = set(after_vars.keys())

        # Added variables
        for key in after_keys - before_keys:
            added[key] = after_vars[key]

        # Removed variables
        for key in before_keys - after_keys:
            removed[key] = before_vars[key]

        # Modified variables
        for key in before_keys & after_keys:
            if before_vars[key] != after_vars[key]:
                modified[key] = {
                    "before": before_vars[key],
                    "after": after_vars[key],
                }

        # Message diff if applicable
        messages_added = None
        if before.message_count is not None and after.message_count is not None:  # noqa: SIM102
            if after.message_count > before.message_count:
                messages_added = after.message_count - before.message_count

        return {
            "added": added,
            "removed": removed,
            "modified": modified,
            "messages_added": messages_added,
        }

    def get_hash(self) -> str:
        """
        Get current memory hash without creating full snapshot.

        Returns:
            Hash string
        """
        variables = self._get_memory_variables()
        return self._compute_hash(variables)

    def _get_memory_variables(self) -> dict[str, Any]:
        """Get memory variables dictionary."""
        # Use load_memory_variables if available
        if hasattr(self._memory, "load_memory_variables"):
            try:
                return dict(self._memory.load_memory_variables({}))
            except Exception:
                pass

        # Fallback to chat_memory.messages
        if hasattr(self._memory, "chat_memory"):
            messages = getattr(self._memory.chat_memory, "messages", [])
            return {"messages": [self._serialize_message(m) for m in messages]}

        # Fallback to buffer attribute
        if hasattr(self._memory, "buffer"):
            return {"buffer": self._memory.buffer}

        return {}

    def _count_messages(self) -> int | None:
        """Count messages in memory."""
        if hasattr(self._memory, "chat_memory"):
            messages = getattr(self._memory.chat_memory, "messages", [])
            return len(messages)
        return None

    def _serialize_message(self, message: Any) -> dict[str, Any]:
        """Serialize a message for hashing."""
        if hasattr(message, "content") and hasattr(message, "type"):
            return {
                "type": message.type,
                "content": message.content,
            }
        return {"content": str(message)}

    def _compute_hash(self, variables: dict[str, Any]) -> str:
        """Compute SHA-256 hash of memory state."""
        try:
            serialized = json.dumps(variables, sort_keys=True, default=str)
        except TypeError:
            serialized = str(variables)

        return hashlib.sha256(serialized.encode()).hexdigest()
