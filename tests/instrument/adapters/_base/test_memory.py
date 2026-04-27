"""Tests for the shared memory persistence module.

Pins the contract documented in
``src/layerlens/instrument/adapters/_base/memory.py``:

* :class:`MemorySnapshot` is content-addressable and immutable.
* :meth:`MemoryRecorder.snapshot` → :meth:`MemoryRecorder.restore`
  round-trips byte-exactly (replay safety).
* Bounded buffers evict deterministically.
* Multi-tenant isolation: tenant A's snapshot cannot be restored into
  tenant B's recorder.
* Thread-safe accumulation under concurrent ``record_turn`` callers.
* SHA-256 content hash is deterministic across processes / Python
  invocations (canonical JSON encoding).

Cross-pollination audit reference:
``A:/tmp/adapter-cross-pollination-audit.md`` §2.1 (memory persistence).
"""

from __future__ import annotations

import threading
from typing import List

import pytest

from layerlens.instrument.adapters._base.memory import (
    DEFAULT_MAX_EPISODIC,
    DEFAULT_MAX_SEMANTIC,
    DEFAULT_MAX_PROCEDURAL,
    MemoryRecorder,
    MemorySnapshot,
)

# ---------------------------------------------------------------------------
# Construction-time contract
# ---------------------------------------------------------------------------


def test_recorder_requires_non_empty_org_id() -> None:
    """Empty / whitespace org_id is rejected (multi-tenancy fail-fast)."""
    with pytest.raises(ValueError, match="non-empty org_id"):
        MemoryRecorder(org_id="")
    with pytest.raises(ValueError, match="non-empty org_id"):
        MemoryRecorder(org_id="   ")
    with pytest.raises(ValueError, match="non-empty org_id"):
        MemoryRecorder(org_id="   \t\n")


def test_recorder_rejects_non_string_org_id() -> None:
    """A non-string org_id is rejected at construction time."""
    with pytest.raises(ValueError, match="non-empty org_id"):
        MemoryRecorder(org_id=None)  # type: ignore[arg-type]


def test_recorder_rejects_zero_buffer_sizes() -> None:
    """Bounded buffers must allow at least one entry."""
    with pytest.raises(ValueError, match="bounded buffer"):
        MemoryRecorder(org_id="org-x", max_episodic=0)
    with pytest.raises(ValueError, match="bounded buffer"):
        MemoryRecorder(org_id="org-x", max_procedural=0)
    with pytest.raises(ValueError, match="bounded buffer"):
        MemoryRecorder(org_id="org-x", max_semantic=0)


def test_recorder_initial_state_is_empty() -> None:
    """A fresh recorder snapshots to all-empty buckets at turn 0."""
    rec = MemoryRecorder(org_id="org-init")
    snap = rec.snapshot()
    assert snap.turn_index == 0
    assert snap.episodic == []
    assert snap.procedural == []
    assert snap.semantic == {}
    assert snap.org_id == "org-init"
    # Hash is non-empty and deterministic for the empty state.
    assert isinstance(snap.content_hash, str)
    assert len(snap.content_hash) == 64  # SHA-256 hex digest length


# ---------------------------------------------------------------------------
# Snapshot determinism (content-hash invariant)
# ---------------------------------------------------------------------------


def test_snapshot_hash_deterministic_for_identical_content() -> None:
    """Two recorders with identical inputs produce identical content hashes."""
    rec_a = MemoryRecorder(org_id="org-deterministic")
    rec_b = MemoryRecorder(org_id="org-deterministic")

    for rec in (rec_a, rec_b):
        rec.record_turn(agent_name="researcher", input_data="hi", output_data="hello")
        rec.record_turn(agent_name="writer", input_data="topic", output_data="draft", tools=["search", "write"])
        rec.set_semantic("user_pref:lang", "en-US")

    snap_a = rec_a.snapshot()
    snap_b = rec_b.snapshot()

    assert snap_a.content_hash == snap_b.content_hash
    assert snap_a.turn_index == snap_b.turn_index == 2


def test_snapshot_hash_changes_when_org_id_differs() -> None:
    """Same content but different tenant → different hash (no collision)."""
    rec_a = MemoryRecorder(org_id="org-A")
    rec_b = MemoryRecorder(org_id="org-B")

    for rec in (rec_a, rec_b):
        rec.record_turn(agent_name="x", input_data="hi", output_data="ok")

    assert rec_a.snapshot().content_hash != rec_b.snapshot().content_hash


def test_snapshot_hash_changes_when_episodic_changes() -> None:
    """Adding a turn changes the snapshot hash."""
    rec = MemoryRecorder(org_id="org-x")
    h0 = rec.snapshot().content_hash
    rec.record_turn(agent_name="a", input_data="i", output_data="o")
    h1 = rec.snapshot().content_hash
    assert h0 != h1


def test_snapshot_immutability_after_recorder_mutation() -> None:
    """A previously-returned snapshot is unaffected by later recorder mutation."""
    rec = MemoryRecorder(org_id="org-immut")
    rec.record_turn(agent_name="x", input_data="hi", output_data="ok")
    snap_before = rec.snapshot()
    hash_before = snap_before.content_hash
    episodic_len_before = len(snap_before.episodic)

    # Mutate the recorder.
    rec.record_turn(agent_name="x", input_data="more", output_data="data")
    rec.set_semantic("key", "value")

    # Original snapshot is unchanged.
    assert snap_before.content_hash == hash_before
    assert len(snap_before.episodic) == episodic_len_before
    assert "key" not in snap_before.semantic


def test_snapshot_dict_roundtrip_preserves_hash() -> None:
    """to_dict() / from_dict() round-trip preserves the snapshot identity."""
    rec = MemoryRecorder(org_id="org-roundtrip")
    rec.record_turn(agent_name="a", input_data="i", output_data="o", tools=["t1", "t2"])
    rec.record_turn(agent_name="b", input_data="i2", output_data="o2", tools=["t2", "t3"])
    rec.set_semantic("k", "v")

    snap = rec.snapshot()
    serialised = snap.to_dict()
    restored = MemorySnapshot.from_dict(serialised)

    assert restored.content_hash == snap.content_hash
    assert restored.turn_index == snap.turn_index
    assert restored.episodic == snap.episodic
    assert restored.semantic == snap.semantic
    assert restored.org_id == snap.org_id


def test_snapshot_from_dict_rejects_missing_field() -> None:
    """from_dict raises when a required field is missing."""
    with pytest.raises(ValueError, match="missing required field"):
        MemorySnapshot.from_dict({"turn_index": 0, "episodic": [], "procedural": [], "semantic": {}})


# ---------------------------------------------------------------------------
# Replay round-trip (the core determinism guarantee)
# ---------------------------------------------------------------------------


def test_restore_reproduces_byte_exact_snapshot() -> None:
    """snapshot() → restore() → snapshot() yields identical content_hash."""
    src = MemoryRecorder(org_id="org-replay")
    for i in range(5):
        src.record_turn(agent_name=f"agent-{i % 2}", input_data=f"in-{i}", output_data=f"out-{i}")
    src.set_semantic("session_summary", "user asked about pricing tiers")
    snap = src.snapshot()

    # Fresh recorder, same tenant, restore from the snapshot, snapshot again.
    target = MemoryRecorder(org_id="org-replay")
    target.restore(snap)
    restored_snap = target.snapshot()

    assert restored_snap.content_hash == snap.content_hash
    assert restored_snap.turn_index == snap.turn_index
    assert restored_snap.episodic == snap.episodic
    assert restored_snap.procedural == snap.procedural
    assert restored_snap.semantic == snap.semantic


def test_restore_then_record_yields_deterministic_next_state() -> None:
    """After restoring identical snapshots into two recorders, the same
    next-turn produces identical next-snapshots (the replay-safety
    contract).

    Note: ``record_turn`` stamps a wall-clock ``timestamp_ns`` into the
    episodic entry, so two recorders run at different wall-clock times
    will drift in the ``timestamp_ns`` field of the *new* turn. That
    timestamp is part of the documented turn shape and intentionally
    ingested into the hash — replay engines suppress this drift by
    capturing the original ``timestamp_ns`` from the source trace and
    using it to seed the recorder's clock at restore time. The test
    here proves the deterministic-everything-else contract: aside
    from ``timestamp_ns``, the full state is byte-identical."""
    src = MemoryRecorder(org_id="org-det")
    src.record_turn(agent_name="a", input_data="i", output_data="o")
    snap = src.snapshot()

    rec_x = MemoryRecorder(org_id="org-det")
    rec_y = MemoryRecorder(org_id="org-det")
    rec_x.restore(snap)
    rec_y.restore(snap)

    rec_x.record_turn(agent_name="b", input_data="i2", output_data="o2", tools=["t1"])
    rec_y.record_turn(agent_name="b", input_data="i2", output_data="o2", tools=["t1"])

    snap_x = rec_x.snapshot()
    snap_y = rec_y.snapshot()
    # All-but-timestamp identity.
    assert snap_x.turn_index == snap_y.turn_index
    assert snap_x.semantic == snap_y.semantic
    assert snap_x.procedural == snap_y.procedural
    assert len(snap_x.episodic) == len(snap_y.episodic)
    for ex, ey in zip(snap_x.episodic, snap_y.episodic):
        assert ex["agent_name"] == ey["agent_name"]
        assert ex["input"] == ey["input"]
        assert ex["output"] == ey["output"]
        assert ex.get("tools") == ey.get("tools")
        assert ex["turn_index"] == ey["turn_index"]


def test_restore_rejects_cross_tenant_snapshot() -> None:
    """A snapshot from tenant A cannot be restored into a tenant-B recorder."""
    rec_a = MemoryRecorder(org_id="org-A")
    rec_a.record_turn(agent_name="x", input_data="hi", output_data="ok")
    snap_a = rec_a.snapshot()

    rec_b = MemoryRecorder(org_id="org-B")
    with pytest.raises(ValueError, match="Cross-tenant restore is prohibited"):
        rec_b.restore(snap_a)


def test_restore_rejects_tampered_snapshot() -> None:
    """A snapshot whose content_hash does not match its content is rejected."""
    rec = MemoryRecorder(org_id="org-tamper")
    rec.record_turn(agent_name="x", input_data="hi", output_data="ok")
    snap = rec.snapshot()

    # Build a tampered snapshot: same hash, mutated semantic content.
    tampered = MemorySnapshot(
        turn_index=snap.turn_index,
        episodic=list(snap.episodic),
        procedural=list(snap.procedural),
        semantic={"injected": "evil"},
        content_hash=snap.content_hash,  # Stale — does not cover the new semantic dict.
        org_id=snap.org_id,
    )
    target = MemoryRecorder(org_id="org-tamper")
    with pytest.raises(ValueError, match="content_hash mismatch"):
        target.restore(tampered)


# ---------------------------------------------------------------------------
# Bounded-buffer eviction
# ---------------------------------------------------------------------------


def test_episodic_buffer_evicts_oldest_fifo_at_cap() -> None:
    """Episodic buffer drops oldest turns when ``max_episodic`` is exceeded."""
    rec = MemoryRecorder(org_id="org-evict", max_episodic=3)
    for i in range(5):
        rec.record_turn(agent_name="x", input_data=f"in-{i}", output_data=f"out-{i}")

    snap = rec.snapshot()
    assert len(snap.episodic) == 3
    # Oldest two were dropped — turns 1 & 2. Surviving turns should be 3, 4, 5.
    assert [t["turn_index"] for t in snap.episodic] == [3, 4, 5]
    # Turn counter still monotonic — eviction does NOT roll back the counter.
    assert snap.turn_index == 5


def test_semantic_store_evicts_least_recently_set_at_cap() -> None:
    """Semantic store evicts the oldest-set entry when over cap."""
    rec = MemoryRecorder(org_id="org-sem", max_semantic=2)
    rec.set_semantic("k1", "v1")
    rec.set_semantic("k2", "v2")
    rec.set_semantic("k3", "v3")  # Should evict k1.

    snap = rec.snapshot()
    assert "k1" not in snap.semantic
    assert snap.semantic == {"k2": "v2", "k3": "v3"}


def test_semantic_overwrite_refreshes_lru_position() -> None:
    """Setting an existing key moves it to most-recent position."""
    rec = MemoryRecorder(org_id="org-sem-lru", max_semantic=2)
    rec.set_semantic("k1", "v1")
    rec.set_semantic("k2", "v2")
    # Refresh k1 → now k2 is the oldest.
    rec.set_semantic("k1", "v1-updated")
    rec.set_semantic("k3", "v3")  # Should evict k2.

    snap = rec.snapshot()
    assert "k2" not in snap.semantic
    assert snap.semantic == {"k1": "v1-updated", "k3": "v3"}


def test_semantic_set_rejects_empty_key() -> None:
    """An empty / whitespace key is rejected."""
    rec = MemoryRecorder(org_id="org-x")
    with pytest.raises(ValueError, match="non-empty key"):
        rec.set_semantic("", "v")
    with pytest.raises(ValueError, match="non-empty key"):
        rec.set_semantic("   ", "v")


def test_procedural_buffer_caps_distinct_patterns() -> None:
    """Procedural store is bounded by ``max_procedural``."""
    rec = MemoryRecorder(org_id="org-proc", max_procedural=2)
    # Generate >2 distinct procedural patterns — recurring tool sequences.
    for cycle in range(3):
        rec.record_turn(agent_name="a", input_data="i", output_data="o", tools=[f"t{cycle}-A"])
        rec.record_turn(agent_name="a", input_data="i", output_data="o", tools=[f"t{cycle}-B"])
        rec.record_turn(agent_name="a", input_data="i", output_data="o", tools=[f"t{cycle}-A"])
        rec.record_turn(agent_name="a", input_data="i", output_data="o", tools=[f"t{cycle}-B"])

    snap = rec.snapshot()
    assert len(snap.procedural) <= 2


# ---------------------------------------------------------------------------
# Procedural-pattern detection
# ---------------------------------------------------------------------------


def test_procedural_pattern_recurrence_increments_count() -> None:
    """Repeated tool sequences accumulate ``count``."""
    rec = MemoryRecorder(org_id="org-pat")
    # search → write happens twice in a row.
    for _ in range(3):
        rec.record_turn(agent_name="a", input_data="i", output_data="o", tools=["search"])
        rec.record_turn(agent_name="a", input_data="i", output_data="o", tools=["write"])

    snap = rec.snapshot()
    # We expect at least one pattern with count >= 2.
    assert snap.procedural, "expected at least one procedural pattern"
    counts = [p["count"] for p in snap.procedural]
    assert any(c >= 2 for c in counts), f"no recurring pattern detected; got {snap.procedural}"


def test_procedural_pattern_skips_turns_with_no_tools() -> None:
    """Two consecutive tool-less turns produce no procedural pattern."""
    rec = MemoryRecorder(org_id="org-no-tools")
    rec.record_turn(agent_name="a", input_data="i", output_data="o")
    rec.record_turn(agent_name="b", input_data="i", output_data="o")
    snap = rec.snapshot()
    assert snap.procedural == []


# ---------------------------------------------------------------------------
# Per-turn truncation (defence-in-depth, not a policy substitute)
# ---------------------------------------------------------------------------


def test_oversized_turn_value_is_capped() -> None:
    """A multi-megabyte string in a turn is hard-capped to prevent overflow."""
    rec = MemoryRecorder(org_id="org-cap")
    huge = "x" * 100_000  # 100 KB string.
    rec.record_turn(agent_name="a", input_data=huge, output_data=huge)

    snap = rec.snapshot()
    captured = snap.episodic[0]
    assert "<...truncated:orig_len=100000>" in captured["input"]
    assert "<...truncated:orig_len=100000>" in captured["output"]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_record_turn_calls_serialise_correctly() -> None:
    """Many threads recording turns simultaneously produce a consistent snapshot."""
    rec = MemoryRecorder(org_id="org-thread", max_episodic=10_000)

    def worker(start: int) -> None:
        for i in range(50):
            rec.record_turn(
                agent_name=f"t-{start}",
                input_data=f"i-{start}-{i}",
                output_data=f"o-{start}-{i}",
            )

    threads: List[threading.Thread] = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = rec.snapshot()
    assert snap.turn_index == 8 * 50
    assert len(snap.episodic) == 8 * 50
    # Every turn_index from 1..400 appears exactly once (no duplicates,
    # no gaps) — the lock guarantees serialisation.
    indices = sorted(t["turn_index"] for t in snap.episodic)
    assert indices == list(range(1, 8 * 50 + 1))


# ---------------------------------------------------------------------------
# Clear / reset
# ---------------------------------------------------------------------------


def test_clear_resets_state_but_preserves_org_binding() -> None:
    """``clear`` returns the recorder to empty state without releasing the tenant."""
    rec = MemoryRecorder(org_id="org-clear")
    rec.record_turn(agent_name="x", input_data="i", output_data="o")
    rec.set_semantic("k", "v")

    rec.clear()

    snap = rec.snapshot()
    assert snap.turn_index == 0
    assert snap.episodic == []
    assert snap.semantic == {}
    assert snap.org_id == "org-clear"  # Binding preserved.


# ---------------------------------------------------------------------------
# Default constants surface
# ---------------------------------------------------------------------------


def test_default_constants_are_positive() -> None:
    """Documented defaults are sensible non-zero values."""
    assert DEFAULT_MAX_EPISODIC > 0
    assert DEFAULT_MAX_PROCEDURAL > 0
    assert DEFAULT_MAX_SEMANTIC > 0


# ---------------------------------------------------------------------------
# Empty episodic / extra metadata edge cases
# ---------------------------------------------------------------------------


def test_extra_metadata_is_sorted_for_determinism() -> None:
    """Two callers passing the same ``extra`` dict in different key orders
    produce the same hash — the recorder sorts ``extra`` keys."""
    rec_a = MemoryRecorder(org_id="org-ex")
    rec_b = MemoryRecorder(org_id="org-ex")

    # Build dicts with deliberately different insertion order.
    extra_x = {"zeta": 1, "alpha": 2, "mu": 3}
    extra_y = {"alpha": 2, "mu": 3, "zeta": 1}

    rec_a.record_turn(agent_name="x", input_data="i", output_data="o", extra=extra_x)
    rec_b.record_turn(agent_name="x", input_data="i", output_data="o", extra=extra_y)

    assert rec_a.snapshot().content_hash == rec_b.snapshot().content_hash


def test_record_turn_returns_new_turn_index() -> None:
    """``record_turn`` returns the post-increment counter (caller convenience)."""
    rec = MemoryRecorder(org_id="org-ret")
    assert rec.record_turn(agent_name="x", input_data="i", output_data="o") == 1
    assert rec.record_turn(agent_name="x", input_data="i", output_data="o") == 2
    assert rec.turn_index == 2
