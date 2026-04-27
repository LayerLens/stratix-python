"""Tests for the LangGraph state adapter.

Ported from ``ateam/tests/adapters/langgraph/test_state.py``.

Every public symbol that ateam exercised exists on the stratix-python
adapter (``LangGraphStateAdapter``, ``MessageListAdapter``,
``StateSnapshot``), so this is a straight port with the import path
rewritten from ``stratix.sdk.python.adapters.langgraph`` to
``layerlens.instrument.adapters.frameworks.langgraph``.
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters.frameworks.langgraph.state import (
    StateSnapshot,
    MessageListAdapter,
    LangGraphStateAdapter,
)


class TestLangGraphStateAdapter:
    """Tests for LangGraphStateAdapter."""

    def test_snapshot_creates_valid_snapshot(self) -> None:
        """Test that snapshot creates a valid StateSnapshot."""
        adapter = LangGraphStateAdapter()
        state: dict[str, Any] = {"messages": [], "count": 0}

        snapshot = adapter.snapshot(state)

        assert isinstance(snapshot, StateSnapshot)
        assert snapshot.state == state
        assert snapshot.hash is not None
        assert snapshot.timestamp_ns > 0

    def test_snapshot_hash_deterministic(self) -> None:
        """Test that same state produces same hash."""
        adapter = LangGraphStateAdapter()
        state: dict[str, Any] = {"key": "value", "number": 42}

        snapshot1 = adapter.snapshot(state)
        snapshot2 = adapter.snapshot(state)

        assert snapshot1.hash == snapshot2.hash

    def test_snapshot_hash_changes_with_state(self) -> None:
        """Test that different state produces different hash."""
        adapter = LangGraphStateAdapter()

        snapshot1 = adapter.snapshot({"key": "value1"})
        snapshot2 = adapter.snapshot({"key": "value2"})

        assert snapshot1.hash != snapshot2.hash

    def test_has_changed_detects_changes(self) -> None:
        """Test has_changed detects state modifications."""
        adapter = LangGraphStateAdapter()

        before = adapter.snapshot({"count": 0})
        after = adapter.snapshot({"count": 1})

        assert adapter.has_changed(before, after) is True

    def test_has_changed_detects_no_changes(self) -> None:
        """Test has_changed returns False for identical states."""
        adapter = LangGraphStateAdapter()
        state: dict[str, Any] = {"count": 0}

        before = adapter.snapshot(state)
        after = adapter.snapshot(state)

        assert adapter.has_changed(before, after) is False

    def test_diff_detects_added_keys(self) -> None:
        """Test diff detects newly added keys."""
        adapter = LangGraphStateAdapter()

        before = adapter.snapshot({"a": 1})
        after = adapter.snapshot({"a": 1, "b": 2})

        diff = adapter.diff(before, after)

        assert "b" in diff["added"]
        assert diff["added"]["b"] == 2
        assert diff["removed"] == {}
        assert diff["modified"] == {}

    def test_diff_detects_removed_keys(self) -> None:
        """Test diff detects removed keys."""
        adapter = LangGraphStateAdapter()

        before = adapter.snapshot({"a": 1, "b": 2})
        after = adapter.snapshot({"a": 1})

        diff = adapter.diff(before, after)

        assert "b" in diff["removed"]
        assert diff["removed"]["b"] == 2
        assert diff["added"] == {}

    def test_diff_detects_modified_keys(self) -> None:
        """Test diff detects modified values."""
        adapter = LangGraphStateAdapter()

        before = adapter.snapshot({"a": 1})
        after = adapter.snapshot({"a": 2})

        diff = adapter.diff(before, after)

        assert "a" in diff["modified"]
        assert diff["modified"]["a"]["before"] == 1
        assert diff["modified"]["a"]["after"] == 2

    def test_include_keys_filter(self) -> None:
        """Test include_keys filters state."""
        adapter = LangGraphStateAdapter(include_keys=["important"])

        state: dict[str, Any] = {"important": "value", "unimportant": "ignored"}
        snapshot = adapter.snapshot(state)

        assert "important" in snapshot.state
        assert "unimportant" not in snapshot.state

    def test_exclude_keys_filter(self) -> None:
        """Test exclude_keys filters state."""
        adapter = LangGraphStateAdapter(exclude_keys=["secret"])

        state: dict[str, Any] = {"public": "value", "secret": "hidden"}
        snapshot = adapter.snapshot(state)

        assert "public" in snapshot.state
        assert "secret" not in snapshot.state

    def test_get_hash_without_snapshot(self) -> None:
        """Test get_hash computes hash directly."""
        adapter = LangGraphStateAdapter()
        state: dict[str, Any] = {"key": "value"}

        hash1 = adapter.get_hash(state)
        snapshot = adapter.snapshot(state)

        assert hash1 == snapshot.hash

    def test_handles_nested_dict(self) -> None:
        """Test handling of nested dictionaries."""
        adapter = LangGraphStateAdapter()

        state: dict[str, Any] = {"level1": {"level2": {"value": 42}}}

        snapshot = adapter.snapshot(state)

        assert snapshot.state == state

    def test_handles_list_values(self) -> None:
        """Test handling of list values."""
        adapter = LangGraphStateAdapter()

        state: dict[str, Any] = {"items": [1, 2, 3]}

        snapshot = adapter.snapshot(state)
        assert snapshot.state["items"] == [1, 2, 3]

    def test_handles_object_with_dict_attr(self) -> None:
        """Test handling of objects with __dict__."""
        adapter = LangGraphStateAdapter()

        class StateObj:
            def __init__(self) -> None:
                self.value = 42

        obj = StateObj()
        snapshot = adapter.snapshot(obj)

        assert snapshot.state["value"] == 42


class TestMessageListAdapter:
    """Tests for MessageListAdapter."""

    def test_tracks_message_count(self) -> None:
        """Test that message count is tracked."""
        adapter = MessageListAdapter(message_key="messages")

        state: dict[str, Any] = {"messages": ["msg1", "msg2"]}
        adapter.snapshot(state)

        assert adapter._last_message_count == 2

    def test_get_new_messages_returns_added(self) -> None:
        """Test get_new_messages returns newly added messages."""
        adapter = MessageListAdapter()

        before = adapter.snapshot({"messages": ["msg1"]})
        after = adapter.snapshot({"messages": ["msg1", "msg2", "msg3"]})

        new_messages = adapter.get_new_messages(before, after)

        assert new_messages == ["msg2", "msg3"]

    def test_get_new_messages_empty_when_no_new(self) -> None:
        """Test get_new_messages returns empty when no new messages."""
        adapter = MessageListAdapter()

        before = adapter.snapshot({"messages": ["msg1", "msg2"]})
        after = adapter.snapshot({"messages": ["msg1", "msg2"]})

        new_messages = adapter.get_new_messages(before, after)

        assert new_messages == []

    def test_custom_message_key(self) -> None:
        """Test custom message key configuration."""
        adapter = MessageListAdapter(message_key="chat_history")

        state: dict[str, Any] = {"chat_history": ["msg1", "msg2"]}
        adapter.snapshot(state)

        assert adapter._last_message_count == 2

    def test_handles_missing_message_key(self) -> None:
        """Test handling when message key is missing."""
        adapter = MessageListAdapter()

        before = adapter.snapshot({})
        after = adapter.snapshot({})

        new_messages = adapter.get_new_messages(before, after)

        assert new_messages == []

    def test_handles_non_list_messages(self) -> None:
        """Test handling when messages is not a list."""
        adapter = MessageListAdapter()

        before = adapter.snapshot({"messages": "not a list"})
        after = adapter.snapshot({"messages": "still not a list"})

        new_messages = adapter.get_new_messages(before, after)

        assert new_messages == []
