"""Tests for LayerLens LangChain Memory Tracing.

Ported as-is from ``ateam/tests/adapters/langchain/test_memory.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.langchain`` → ``layerlens.instrument.adapters.frameworks.langchain``
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters.frameworks.langchain.memory import (
    TracedMemory,
    MemoryMutationTracker,
    wrap_memory,
)


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class MockChatMemory:
    """Mock chat memory."""

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []


class MockMemory:
    """Mock LangChain memory."""

    def __init__(self) -> None:
        self.chat_memory = MockChatMemory()
        self._context_saved: list[dict[str, Any]] = []
        self.memory_variables = ["history"]

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        self._context_saved.append({"inputs": inputs, "outputs": outputs})
        # Simulate adding to messages
        self.chat_memory.messages.append({"type": "human", "content": inputs.get("input", "")})
        self.chat_memory.messages.append({"type": "ai", "content": outputs.get("output", "")})

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"history": list(self.chat_memory.messages)}

    def clear(self) -> None:
        self.chat_memory.messages = []


class TestTracedMemory:
    """Tests for TracedMemory."""

    def test_initialization(self) -> None:
        """Test TracedMemory initializes correctly."""
        memory = MockMemory()
        traced = TracedMemory(memory)

        assert traced._memory is memory

    def test_initialization_with_stratix(self) -> None:
        """Test initialization with STRATIX instance."""
        stratix = MockStratix()
        memory = MockMemory()
        traced = TracedMemory(memory, stratix)

        assert traced._stratix is stratix

    def test_save_context_calls_underlying(self) -> None:
        """Test save_context calls underlying memory."""
        memory = MockMemory()
        traced = TracedMemory(memory)

        traced.save_context(
            {"input": "Hello"},
            {"output": "Hi there!"},
        )

        assert len(memory._context_saved) == 1
        assert memory._context_saved[0]["inputs"] == {"input": "Hello"}

    def test_save_context_emits_state_change(self) -> None:
        """Test save_context emits state change event."""
        stratix = MockStratix()
        memory = MockMemory()
        traced = TracedMemory(memory, stratix)

        traced.save_context(
            {"input": "Hello"},
            {"output": "Hi!"},
        )

        events = stratix.get_events("agent.state.change")
        assert len(events) == 1
        assert events[0]["payload"]["memory_type"] == "MockMemory"
        assert events[0]["payload"]["trigger"] == "save_context"

    def test_load_memory_variables(self) -> None:
        """Test load_memory_variables proxies to underlying."""
        memory = MockMemory()
        memory.chat_memory.messages = [{"type": "human", "content": "test"}]
        traced = TracedMemory(memory)

        result = traced.load_memory_variables({})

        assert "history" in result
        assert len(result["history"]) == 1

    def test_clear_emits_state_change(self) -> None:
        """Test clear emits state change event."""
        stratix = MockStratix()
        memory = MockMemory()
        memory.chat_memory.messages = [{"type": "human", "content": "test"}]
        traced = TracedMemory(memory, stratix)

        traced.clear()

        events = stratix.get_events("agent.state.change")
        assert len(events) == 1
        assert events[0]["payload"]["trigger"] == "clear"

    def test_memory_variables_property(self) -> None:
        """Test memory_variables property."""
        memory = MockMemory()
        traced = TracedMemory(memory)

        assert traced.memory_variables == ["history"]

    def test_attribute_proxying(self) -> None:
        """Test attribute access is proxied."""
        memory = MockMemory()
        memory.custom_attr = "value"  # type: ignore[attr-defined]
        traced = TracedMemory(memory)

        assert traced.custom_attr == "value"


class TestWrapMemory:
    """Tests for wrap_memory function."""

    def test_creates_traced_memory(self) -> None:
        """Test creates TracedMemory instance."""
        memory = MockMemory()
        traced = wrap_memory(memory)

        assert isinstance(traced, TracedMemory)

    def test_passes_stratix_instance(self) -> None:
        """Test passes STRATIX instance."""
        stratix = MockStratix()
        memory = MockMemory()
        traced = wrap_memory(memory, stratix)

        assert traced._stratix is stratix


class TestMemoryMutationTracker:
    """Tests for MemoryMutationTracker."""

    def test_initialization(self) -> None:
        """Test tracker initializes correctly."""
        tracker = MemoryMutationTracker()

        assert tracker._mutations == []

    def test_track_memory_context_manager(self) -> None:
        """Test track_memory works as context manager."""
        tracker = MemoryMutationTracker()
        memory = MockMemory()

        with tracker.track_memory(memory, "test_operation"):
            memory.save_context({"input": "hi"}, {"output": "hello"})

        assert len(tracker._mutations) == 1
        assert tracker._mutations[0]["operation"] == "test_operation"

    def test_track_memory_records_changes(self) -> None:
        """Test track_memory records memory changes."""
        tracker = MemoryMutationTracker()
        memory = MockMemory()

        with tracker.track_memory(memory, "add_message"):
            memory.save_context({"input": "test"}, {"output": "response"})

        mutation = tracker._mutations[0]
        assert mutation["before_hash"] != mutation["after_hash"]

    def test_track_memory_no_change(self) -> None:
        """Test track_memory doesn't record when no change."""
        tracker = MemoryMutationTracker()
        memory = MockMemory()

        with tracker.track_memory(memory, "no_op"):
            # Don't modify memory
            pass

        assert len(tracker._mutations) == 0

    def test_track_memory_emits_event(self) -> None:
        """Test track_memory emits state change event."""
        stratix = MockStratix()
        tracker = MemoryMutationTracker(stratix)
        memory = MockMemory()

        with tracker.track_memory(memory, "modify"):
            memory.save_context({"input": "x"}, {"output": "y"})

        events = stratix.get_events("agent.state.change")
        assert len(events) == 1
        assert events[0]["payload"]["operation"] == "modify"

    def test_get_mutations(self) -> None:
        """Test get_mutations returns all mutations."""
        tracker = MemoryMutationTracker()
        memory = MockMemory()

        with tracker.track_memory(memory, "op1"):
            memory.save_context({"input": "a"}, {"output": "b"})

        with tracker.track_memory(memory, "op2"):
            memory.save_context({"input": "c"}, {"output": "d"})

        mutations = tracker.get_mutations()

        assert len(mutations) == 2
        assert mutations[0]["operation"] == "op1"
        assert mutations[1]["operation"] == "op2"

    def test_clear(self) -> None:
        """Test clear removes all mutations."""
        tracker = MemoryMutationTracker()
        memory = MockMemory()

        with tracker.track_memory(memory, "op"):
            memory.save_context({"input": "x"}, {"output": "y"})

        tracker.clear()

        assert len(tracker._mutations) == 0

    def test_record_mutation(self) -> None:
        """Test record_mutation adds mutation."""
        tracker = MemoryMutationTracker()

        tracker.record_mutation(
            {
                "operation": "test",
                "before_hash": "hash1",
                "after_hash": "hash2",
            }
        )

        assert len(tracker._mutations) == 1
        assert tracker._mutations[0]["operation"] == "test"
