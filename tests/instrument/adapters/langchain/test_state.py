"""Tests for STRATIX LangChain Memory State Adapter."""

import pytest

from layerlens.instrument.adapters.langchain.state import (
    LangChainMemoryAdapter,
    MemorySnapshot,
)


class MockChatMemory:
    """Mock chat memory."""

    def __init__(self, messages=None):
        self.messages = messages or []


class MockConversationMemory:
    """Mock conversation memory."""

    def __init__(self, messages=None):
        self.chat_memory = MockChatMemory(messages or [])
        self.memory_variables = ["history"]

    def load_memory_variables(self, inputs):
        messages = self.chat_memory.messages
        return {
            "history": [{"type": m["type"], "content": m["content"]} for m in messages]
        }


class MockBufferMemory:
    """Mock buffer memory without chat_memory."""

    def __init__(self, buffer=""):
        self.buffer = buffer


class MockMessage:
    """Mock message."""

    def __init__(self, type: str, content: str):
        self.type = type
        self.content = content


class TestLangChainMemoryAdapter:
    """Tests for LangChainMemoryAdapter."""

    def test_initialization(self):
        """Test adapter initializes correctly."""
        memory = MockConversationMemory()
        adapter = LangChainMemoryAdapter(memory)

        assert adapter._memory is memory
        assert adapter._memory_type == "MockConversationMemory"

    def test_snapshot_creates_valid_snapshot(self):
        """Test snapshot creates valid MemorySnapshot."""
        memory = MockConversationMemory()
        adapter = LangChainMemoryAdapter(memory)

        snapshot = adapter.snapshot()

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.memory_type == "MockConversationMemory"
        assert snapshot.hash is not None
        assert snapshot.timestamp_ns > 0

    def test_snapshot_hash_deterministic(self):
        """Test same memory produces same hash."""
        memory = MockConversationMemory([
            {"type": "human", "content": "Hello"}
        ])
        adapter = LangChainMemoryAdapter(memory)

        snapshot1 = adapter.snapshot()
        snapshot2 = adapter.snapshot()

        assert snapshot1.hash == snapshot2.hash

    def test_snapshot_hash_changes_with_content(self):
        """Test different content produces different hash."""
        memory1 = MockConversationMemory([
            {"type": "human", "content": "Hello"}
        ])
        memory2 = MockConversationMemory([
            {"type": "human", "content": "Goodbye"}
        ])

        adapter1 = LangChainMemoryAdapter(memory1)
        adapter2 = LangChainMemoryAdapter(memory2)

        snapshot1 = adapter1.snapshot()
        snapshot2 = adapter2.snapshot()

        assert snapshot1.hash != snapshot2.hash

    def test_has_changed_detects_changes(self):
        """Test has_changed detects memory modifications."""
        memory = MockConversationMemory([
            {"type": "human", "content": "Hello"}
        ])
        adapter = LangChainMemoryAdapter(memory)

        before = adapter.snapshot()

        # Modify memory
        memory.chat_memory.messages.append(
            {"type": "ai", "content": "Hi there!"}
        )

        after = adapter.snapshot()

        assert adapter.has_changed(before, after) is True

    def test_has_changed_detects_no_changes(self):
        """Test has_changed returns False for unchanged memory."""
        memory = MockConversationMemory([
            {"type": "human", "content": "Hello"}
        ])
        adapter = LangChainMemoryAdapter(memory)

        before = adapter.snapshot()
        after = adapter.snapshot()

        assert adapter.has_changed(before, after) is False

    def test_diff_detects_message_changes(self):
        """Test diff detects message changes."""
        memory = MockConversationMemory([
            {"type": "human", "content": "Hello"}
        ])
        adapter = LangChainMemoryAdapter(memory)

        before = adapter.snapshot()

        memory.chat_memory.messages.append(
            {"type": "ai", "content": "Hi!"}
        )

        after = adapter.snapshot()
        diff = adapter.diff(before, after)

        assert "modified" in diff or "messages_added" in diff

    def test_message_count_tracking(self):
        """Test message count is tracked in snapshot."""
        memory = MockConversationMemory([
            {"type": "human", "content": "Hello"},
            {"type": "ai", "content": "Hi!"},
        ])
        adapter = LangChainMemoryAdapter(memory)

        snapshot = adapter.snapshot()

        assert snapshot.message_count == 2

    def test_get_hash_without_snapshot(self):
        """Test get_hash computes hash directly."""
        memory = MockConversationMemory([
            {"type": "human", "content": "Test"}
        ])
        adapter = LangChainMemoryAdapter(memory)

        hash1 = adapter.get_hash()
        snapshot = adapter.snapshot()

        assert hash1 == snapshot.hash

    def test_handles_buffer_memory(self):
        """Test handling of buffer-based memory."""
        memory = MockBufferMemory("Some conversation buffer")
        adapter = LangChainMemoryAdapter(memory)

        snapshot = adapter.snapshot()

        assert "buffer" in snapshot.variables
        assert snapshot.variables["buffer"] == "Some conversation buffer"

    def test_handles_message_objects(self):
        """Test handling of message objects."""
        memory = MockConversationMemory()
        # Simulate message objects instead of dicts
        memory.chat_memory.messages = [
            MockMessage("human", "Hello"),
            MockMessage("ai", "Hi"),
        ]
        adapter = LangChainMemoryAdapter(memory)

        # Should not raise
        snapshot = adapter.snapshot()
        assert snapshot is not None


class TestMemorySnapshotDiff:
    """Tests for MemorySnapshot diff functionality."""

    def test_diff_detects_added_variables(self):
        """Test diff detects added variables."""
        memory = MockConversationMemory()
        adapter = LangChainMemoryAdapter(memory)

        before = MemorySnapshot(
            memory_type="test",
            variables={},
            hash="before",
            timestamp_ns=0,
        )

        after = MemorySnapshot(
            memory_type="test",
            variables={"new_var": "value"},
            hash="after",
            timestamp_ns=1,
        )

        diff = adapter.diff(before, after)

        assert "new_var" in diff["added"]

    def test_diff_detects_removed_variables(self):
        """Test diff detects removed variables."""
        memory = MockConversationMemory()
        adapter = LangChainMemoryAdapter(memory)

        before = MemorySnapshot(
            memory_type="test",
            variables={"old_var": "value"},
            hash="before",
            timestamp_ns=0,
        )

        after = MemorySnapshot(
            memory_type="test",
            variables={},
            hash="after",
            timestamp_ns=1,
        )

        diff = adapter.diff(before, after)

        assert "old_var" in diff["removed"]

    def test_diff_calculates_messages_added(self):
        """Test diff calculates number of messages added."""
        memory = MockConversationMemory()
        adapter = LangChainMemoryAdapter(memory)

        before = MemorySnapshot(
            memory_type="test",
            variables={},
            hash="before",
            timestamp_ns=0,
            message_count=2,
        )

        after = MemorySnapshot(
            memory_type="test",
            variables={},
            hash="after",
            timestamp_ns=1,
            message_count=5,
        )

        diff = adapter.diff(before, after)

        assert diff["messages_added"] == 3
