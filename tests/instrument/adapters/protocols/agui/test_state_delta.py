"""
Tests for AG-UI state delta (JSON Patch) handling.
"""

import pytest

from layerlens.instrument.adapters.protocols.agui.state_handler import StateDeltaHandler


class TestStateDeltaHandler:
    def setup_method(self):
        self.handler = StateDeltaHandler()

    def test_apply_snapshot(self):
        before, after = self.handler.apply_snapshot({"count": 0, "name": "test"})
        assert before.startswith("sha256:")
        assert after.startswith("sha256:")
        assert before != after  # empty state → non-empty state

    def test_apply_snapshot_preserves_state(self):
        self.handler.apply_snapshot({"count": 0})
        state = self.handler.current_state
        assert state == {"count": 0}

    def test_apply_delta_add(self):
        self.handler.apply_snapshot({"existing": True})
        before, after = self.handler.apply_delta([
            {"op": "add", "path": "/new_field", "value": "hello"},
        ])
        assert before != after
        assert self.handler.current_state["new_field"] == "hello"

    def test_apply_delta_remove(self):
        self.handler.apply_snapshot({"field1": "a", "field2": "b"})
        self.handler.apply_delta([
            {"op": "remove", "path": "/field2"},
        ])
        assert "field2" not in self.handler.current_state
        assert "field1" in self.handler.current_state

    def test_apply_delta_replace(self):
        self.handler.apply_snapshot({"count": 0})
        self.handler.apply_delta([
            {"op": "replace", "path": "/count", "value": 42},
        ])
        assert self.handler.current_state["count"] == 42

    def test_apply_multiple_operations(self):
        self.handler.apply_snapshot({"a": 1, "b": 2})
        self.handler.apply_delta([
            {"op": "replace", "path": "/a", "value": 10},
            {"op": "add", "path": "/c", "value": 3},
            {"op": "remove", "path": "/b"},
        ])
        state = self.handler.current_state
        assert state == {"a": 10, "c": 3}

    def test_nested_path(self):
        self.handler.apply_snapshot({"parent": {"child": "old"}})
        self.handler.apply_delta([
            {"op": "replace", "path": "/parent/child", "value": "new"},
        ])
        assert self.handler.current_state["parent"]["child"] == "new"

    def test_json_pointer_unescaping(self):
        # RFC 6901: ~0 = ~, ~1 = /
        keys = StateDeltaHandler._parse_path("/foo~1bar/baz~0qux")
        assert keys == ["foo/bar", "baz~qux"]

    def test_reset(self):
        self.handler.apply_snapshot({"data": True})
        self.handler.reset()
        assert self.handler.current_state == {}

    def test_hash_consistency(self):
        h1 = StateDeltaHandler._hash_state({"a": 1, "b": 2})
        h2 = StateDeltaHandler._hash_state({"b": 2, "a": 1})
        assert h1 == h2  # sort_keys ensures consistent hashing

    def test_apply_delta_with_invalid_op(self):
        self.handler.apply_snapshot({"data": True})
        # Should log warning but not crash
        self.handler.apply_delta([
            {"op": "move", "path": "/data", "from": "/other"},
        ])
