"""Tests for STRATIX Python SDK State Adapter."""

import pytest

from layerlens.instrument import STRATIX, DictStateAdapter


class TestDictStateAdapter:
    """Tests for the DictStateAdapter."""

    def test_snapshot(self):
        """Test taking a state snapshot."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {"counter": 0, "messages": []}
        adapter = DictStateAdapter(stratix, state)

        snapshot = adapter.snapshot()

        assert snapshot == {"counter": 0, "messages": []}

    def test_snapshot_is_deep_copy(self):
        """Test that snapshot is a deep copy."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {"data": [1, 2, 3]}
        adapter = DictStateAdapter(stratix, state)

        snapshot = adapter.snapshot()

        # Modify original
        state["data"].append(4)

        # Snapshot should be unchanged
        assert snapshot["data"] == [1, 2, 3]

    def test_get_state_keys(self):
        """Test getting state keys."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {"key1": "value1", "key2": "value2"}
        adapter = DictStateAdapter(stratix, state)

        keys = adapter.get_state_keys()

        assert set(keys) == {"key1", "key2"}

    def test_compute_hash(self):
        """Test computing state hash."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {"counter": 5}
        adapter = DictStateAdapter(stratix, state)

        hash1 = adapter.compute_hash(state)

        assert hash1.startswith("sha256:")
        assert len(hash1) == 71  # sha256: + 64 hex chars

    def test_hash_deterministic(self):
        """Test that hash is deterministic."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {"b": 2, "a": 1}  # Keys in random order
        adapter = DictStateAdapter(stratix, state)

        hash1 = adapter.compute_hash({"a": 1, "b": 2})
        hash2 = adapter.compute_hash({"b": 2, "a": 1})

        assert hash1 == hash2  # Should be same due to sort_keys

    def test_hash_different_for_different_state(self):
        """Test that different states produce different hashes."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {}
        adapter = DictStateAdapter(stratix, state)

        hash1 = adapter.compute_hash({"counter": 1})
        hash2 = adapter.compute_hash({"counter": 2})

        assert hash1 != hash2


class TestStateChangeCapture:
    """Tests for capturing state changes."""

    def test_capture_change_no_previous(self):
        """Test capturing first state change."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {"counter": 0}
        adapter = DictStateAdapter(stratix, state)

        # First capture
        event = adapter.capture_change("initial")

        assert event is not None
        assert event.event_type == "agent.state.change"
        assert event.state.after_hash.startswith("sha256:")

    def test_capture_change_detects_mutation(self):
        """Test that capture_change detects state mutation."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {"counter": 0}
        adapter = DictStateAdapter(stratix, state)

        # Initialize
        adapter.initialize()

        # Mutate state
        state["counter"] = 1

        # Capture change
        event = adapter.capture_change("increment")

        assert event is not None
        assert event.state.before_hash != event.state.after_hash

    def test_capture_change_no_change(self):
        """Test that capture_change returns None when state unchanged."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {"counter": 0}
        adapter = DictStateAdapter(stratix, state)

        # Initialize
        adapter.initialize()

        # No mutation - capture should return None
        event = adapter.capture_change("check")

        assert event is None


class TestStateChangeEmission:
    """Tests for emitting state change events."""

    def test_emit_change(self):
        """Test emitting state change event."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {"counter": 0}
        adapter = DictStateAdapter(stratix, state)

        ctx = stratix.start_trial()

        # Initialize adapter
        adapter.initialize()

        # Mutate state
        state["counter"] = 1

        # Emit change
        adapter.emit_change("increment")

        # Check event was emitted
        events = stratix.get_events()
        assert len(events) == 1

        event = events[0]
        assert event.payload.event_type == "agent.state.change"

    def test_emit_change_no_context(self):
        """Test that emit_change does nothing without context."""
        from layerlens.instrument._context import set_current_context, reset_context

        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {"counter": 0}
        adapter = DictStateAdapter(stratix, state)

        # Ensure no context
        token = set_current_context(None)
        try:
            adapter.initialize()
            state["counter"] = 1

            # Should not raise, just do nothing
            adapter.emit_change("increment")

            assert len(stratix.get_events()) == 0
        finally:
            reset_context(token)


class TestAdapterInitialization:
    """Tests for adapter initialization."""

    def test_initialize(self):
        """Test initializing the adapter."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {"counter": 0}
        adapter = DictStateAdapter(stratix, state)

        adapter.initialize()

        # Should have captured initial state
        assert adapter._last_snapshot is not None
        assert adapter._last_hash is not None

    def test_initialize_allows_change_detection(self):
        """Test that initialize enables change detection."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        state = {"counter": 0}
        adapter = DictStateAdapter(stratix, state)

        # Initialize
        adapter.initialize()
        initial_hash = adapter._last_hash

        # Mutate
        state["counter"] = 5

        # Capture change
        event = adapter.capture_change("update")

        assert event is not None
        assert event.state.before_hash == initial_hash
