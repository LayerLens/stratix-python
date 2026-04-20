from __future__ import annotations

from layerlens.replay.store import InMemoryReplayStore
from layerlens.replay.models import ReplayResult


def _result(original: str, replay: str) -> ReplayResult:
    return ReplayResult(original_trace_id=original, replay_trace_id=replay)


class TestInMemoryReplayStore:
    def test_save_and_get(self):
        store = InMemoryReplayStore()
        r = _result("t1", "r1")
        store.save(r)
        assert store.get("r1") is r
        assert store.get("missing") is None

    def test_list_for_original_groups(self):
        store = InMemoryReplayStore()
        store.save(_result("t1", "r1"))
        store.save(_result("t1", "r2"))
        store.save(_result("t2", "r3"))
        ids = [r.replay_trace_id for r in store.list_for_original("t1")]
        assert ids == ["r1", "r2"]
        assert store.list_for_original("unknown") == []

    def test_all_returns_every_result(self):
        store = InMemoryReplayStore()
        store.save(_result("t1", "r1"))
        store.save(_result("t2", "r2"))
        assert {r.replay_trace_id for r in store.all()} == {"r1", "r2"}

    def test_clear(self):
        store = InMemoryReplayStore()
        store.save(_result("t1", "r1"))
        store.clear()
        assert list(store.all()) == []
        assert store.list_for_original("t1") == []
