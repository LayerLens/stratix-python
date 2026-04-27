"""Tests for LangChain TracedMemory integration with AgentMemoryService.

Ported as-is from
``ateam/tests/adapters/langchain/test_memory_service_integration.py``.

Validates that:

* ``TracedMemory`` stores episodic memories via ``memory_service`` when provided.
* Failures in ``memory_service.store()`` are swallowed (no disruption).
* ``MemoryEntry`` fields are populated correctly from conversation context.

Translation rules applied:
* ``stratix.sdk.python.adapters.langchain`` → ``layerlens.instrument.adapters.frameworks.langchain``
* ``MemoryEntry`` is sourced from the vendored shim
  (``layerlens.instrument._vendored.memory_models``); the adapter imports
  it from the same module, so ``TracedMemory`` continues to function
  end-to-end.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

from layerlens.instrument.adapters.frameworks.langchain.memory import TracedMemory


class MockMemory:
    """Minimal LangChain memory mock."""

    def __init__(self) -> None:
        self.chat_memory = MagicMock()
        self.chat_memory.messages = []
        self._context_saved: list[dict[str, Any]] = []
        self.memory_variables = ["history"]

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        self._context_saved.append({"inputs": inputs, "outputs": outputs})
        self.chat_memory.messages.append({"type": "human", "content": inputs.get("input", "")})
        self.chat_memory.messages.append({"type": "ai", "content": outputs.get("output", "")})

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"history": list(self.chat_memory.messages)}

    def clear(self) -> None:
        self.chat_memory.messages = []


class MockStratix:
    """Mock STRATIX instance with org_id and agent_id attributes."""

    def __init__(self, org_id: str = "test-org", agent_id: str = "langchain") -> None:
        self.org_id = org_id
        self.agent_id = agent_id
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})


class TestTracedMemoryServiceIntegration(unittest.TestCase):
    """Tests for TracedMemory + memory_service episodic storage."""

    def setUp(self) -> None:
        self.memory = MockMemory()
        self.stratix = MockStratix(org_id="org-123", agent_id="lc-agent")
        self.memory_service = MagicMock()

    def test_save_context_calls_memory_service_store(self) -> None:
        """save_context stores an episodic memory when memory_service is provided."""
        traced = TracedMemory(self.memory, self.stratix, memory_service=self.memory_service)
        traced.save_context({"input": "Hello"}, {"output": "Hi there!"})

        self.memory_service.store.assert_called_once()
        entry = self.memory_service.store.call_args[0][0]
        self.assertEqual(entry.org_id, "org-123")
        self.assertEqual(entry.agent_id, "lc-agent")
        self.assertEqual(entry.memory_type, "episodic")
        self.assertIn("Hello", entry.content)
        self.assertIn("Hi there!", entry.content)

    def test_save_context_episodic_entry_has_correct_metadata(self) -> None:
        """Episodic entry metadata includes source attribution."""
        traced = TracedMemory(self.memory, self.stratix, memory_service=self.memory_service)
        traced.save_context({"input": "test"}, {"output": "response"})

        entry = self.memory_service.store.call_args[0][0]
        self.assertEqual(entry.metadata.get("source"), "langchain_traced_memory")

    def test_save_context_episodic_entry_has_key_prefix(self) -> None:
        """Episodic entry key starts with 'conversation_' prefix."""
        traced = TracedMemory(self.memory, self.stratix, memory_service=self.memory_service)
        traced.save_context({"input": "test"}, {"output": "response"})

        entry = self.memory_service.store.call_args[0][0]
        self.assertTrue(entry.key.startswith("conversation_"))

    def test_save_context_episodic_entry_importance_is_default(self) -> None:
        """Episodic entries from conversation have default 0.5 importance."""
        traced = TracedMemory(self.memory, self.stratix, memory_service=self.memory_service)
        traced.save_context({"input": "a"}, {"output": "b"})

        entry = self.memory_service.store.call_args[0][0]
        self.assertAlmostEqual(entry.importance, 0.5)

    def test_save_context_without_memory_service_does_not_error(self) -> None:
        """save_context works fine when memory_service is None."""
        traced = TracedMemory(self.memory, self.stratix, memory_service=None)
        traced.save_context({"input": "no service"}, {"output": "ok"})

        # Just verify underlying memory was updated
        self.assertEqual(len(self.memory._context_saved), 1)

    def test_memory_service_store_failure_is_swallowed(self) -> None:
        """Failures in memory_service.store() do not propagate."""
        self.memory_service.store.side_effect = RuntimeError("DB connection failed")
        traced = TracedMemory(self.memory, self.stratix, memory_service=self.memory_service)

        # Should not raise
        traced.save_context({"input": "fail"}, {"output": "gracefully"})

        # Underlying memory still updated
        self.assertEqual(len(self.memory._context_saved), 1)

    def test_multiple_saves_each_store_episodic(self) -> None:
        """Each save_context call produces a separate episodic entry."""
        traced = TracedMemory(self.memory, self.stratix, memory_service=self.memory_service)
        traced.save_context({"input": "first"}, {"output": "1"})
        traced.save_context({"input": "second"}, {"output": "2"})

        self.assertEqual(self.memory_service.store.call_count, 2)

    def test_episodic_keys_are_unique(self) -> None:
        """Each episodic entry has a unique key (timestamp-based)."""
        traced = TracedMemory(self.memory, self.stratix, memory_service=self.memory_service)
        traced.save_context({"input": "a"}, {"output": "1"})
        traced.save_context({"input": "b"}, {"output": "2"})

        keys = [self.memory_service.store.call_args_list[i][0][0].key for i in range(2)]
        # Keys should be distinct (or at worst, same second is acceptable)
        # Just check format
        for k in keys:
            self.assertTrue(k.startswith("conversation_"))

    def test_state_change_event_still_emitted(self) -> None:
        """State change events are still emitted alongside memory storage."""
        traced = TracedMemory(self.memory, self.stratix, memory_service=self.memory_service)
        traced.save_context({"input": "hey"}, {"output": "yo"})

        state_events = [e for e in self.stratix.events if e["type"] == "agent.state.change"]
        self.assertGreater(len(state_events), 0)
        self.memory_service.store.assert_called_once()

    def test_no_org_id_uses_empty_string(self) -> None:
        """When stratix has no org_id, empty string is used."""
        stratix_no_org = MagicMock()
        del stratix_no_org.org_id  # Remove the attribute
        stratix_no_org.agent_id = "test"
        stratix_no_org.emit = MagicMock()

        traced = TracedMemory(self.memory, stratix_no_org, memory_service=self.memory_service)
        traced.save_context({"input": "x"}, {"output": "y"})

        entry = self.memory_service.store.call_args[0][0]
        self.assertEqual(entry.org_id, "")


if __name__ == "__main__":
    unittest.main()
