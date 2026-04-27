"""Per-tenant stream isolation tests for :class:`IngestionPipelineSink`.

Implements Gap 2 of the multi-tenancy hardening contract: a single
sink instance servicing multiple tenants must partition its buffer by
``org_id``, cap each tenant independently, and never let one tenant's
burst displace another's events.

Properties enforced
-------------------

* **Per-tenant buffers:** events from tenant A go to a different list
  slot than tenant B's. Flushing A never moves B's events.
* **Per-tenant cap (FIFO eviction):** when tenant A overflows the cap,
  A's OLDEST event is dropped — B's buffer (and its events) are
  untouched. The drop is recorded in :attr:`dropped_per_tenant` keyed
  by A's ``org_id``.
* **Per-tenant flush calls:** :meth:`flush` issues one ``ingest()``
  call per tenant with that tenant's ``org_id`` — never a mixed batch.
* **Concurrent isolation:** N threads emitting on N tenants in
  parallel produce per-tenant batches with no cross-contamination,
  even under contention with a small cap that triggers eviction.
* **Observability:** :meth:`buffer_size_per_tenant` reports a
  per-tenant snapshot suitable for the
  ``sink_per_tenant_buffer_size{org_id}`` gauge.

Background
----------
PR #118 added per-event ``org_id`` propagation but
:class:`IngestionPipelineSink` still kept a single buffer. A noisy
tenant could (a) starve quieter tenants by filling a global cap and
(b) cause flush batches to mix tenants — defeating downstream RLS.
This file pins the hardened contract.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

import pytest

from layerlens.instrument.adapters._base.sinks import IngestionPipelineSink


class _RecordingPipeline:
    """Pipeline double that captures every ``ingest()`` call.

    Each call appears as ``(events_list_copy, tenant_id)``. Thread-safe
    so the test suite can drive concurrent senders.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: List[Tuple[List[Dict[str, Any]], str]] = []

    def ingest(self, events: List[Dict[str, Any]], tenant_id: str) -> None:
        with self._lock:
            self._calls.append(([dict(e) for e in events], tenant_id))

    @property
    def calls(self) -> List[Tuple[List[Dict[str, Any]], str]]:
        with self._lock:
            return list(self._calls)

    def calls_for(self, tenant_id: str) -> List[Tuple[List[Dict[str, Any]], str]]:
        return [(events, tid) for events, tid in self.calls if tid == tenant_id]


def _emit(sink: IngestionPipelineSink, org_id: str, n: int, prefix: str = "ev") -> None:
    """Drive ``n`` events into ``sink`` bound to ``org_id``."""
    for i in range(n):
        sink.send(
            "tool.call",
            {"tool_name": f"{prefix}-{i}", "org_id": org_id},
            timestamp_ns=1_700_000_000_000_000_000 + i,
            org_id=org_id,
        )


# ---------------------------------------------------------------------------
# Cross-tenant burst isolation (immediate mode)
# ---------------------------------------------------------------------------


def test_immediate_mode_keys_each_event_by_org_id() -> None:
    """Immediate (unbuffered) mode routes per-event ``org_id`` to ``ingest``."""
    pipeline = _RecordingPipeline()
    sink = IngestionPipelineSink(pipeline=pipeline, buffered=False)

    _emit(sink, "org-A", 3)
    _emit(sink, "org-B", 2)

    a_calls = pipeline.calls_for("org-A")
    b_calls = pipeline.calls_for("org-B")
    assert len(a_calls) == 3
    assert len(b_calls) == 2
    # No mixed batch: every immediate call is for a single tenant.
    for events, tenant in pipeline.calls:
        assert len(events) == 1
        assert all(e["org_id"] == tenant for e in events)


# ---------------------------------------------------------------------------
# Per-tenant buffer (buffered mode)
# ---------------------------------------------------------------------------


def test_buffered_mode_partitions_buffers_per_tenant() -> None:
    """Each tenant has its own buffer; ``buffer_size_per_tenant`` reflects this."""
    pipeline = _RecordingPipeline()
    sink = IngestionPipelineSink(pipeline=pipeline, buffered=True)

    _emit(sink, "org-A", 5)
    _emit(sink, "org-B", 2)

    sizes = sink.buffer_size_per_tenant()
    assert sizes == {"org-A": 5, "org-B": 2}
    # Nothing flushed yet.
    assert pipeline.calls == []


def test_flush_emits_one_ingest_call_per_tenant() -> None:
    """Flushing a multi-tenant buffered sink yields exactly one batch per tenant."""
    pipeline = _RecordingPipeline()
    sink = IngestionPipelineSink(pipeline=pipeline, buffered=True)

    _emit(sink, "org-A", 4, prefix="A")
    _emit(sink, "org-B", 3, prefix="B")
    _emit(sink, "org-C", 2, prefix="C")

    sink.flush()

    by_tenant = {tenant: events for events, tenant in pipeline.calls}
    assert set(by_tenant.keys()) == {"org-A", "org-B", "org-C"}
    assert len(by_tenant["org-A"]) == 4
    assert len(by_tenant["org-B"]) == 3
    assert len(by_tenant["org-C"]) == 2

    # Every batch contains ONLY events for its tenant — no leak.
    for tenant, events in by_tenant.items():
        assert all(e["org_id"] == tenant for e in events), (
            f"batch for tenant {tenant} contained another tenant's event"
        )

    # Buffers are now empty.
    assert sink.buffer_size_per_tenant() == {"org-A": 0, "org-B": 0, "org-C": 0}


def test_per_tenant_cap_drops_only_overflowing_tenants_oldest_event() -> None:
    """A burst from tenant A FIFO-evicts ONLY A's oldest events."""
    pipeline = _RecordingPipeline()
    sink = IngestionPipelineSink(
        pipeline=pipeline,
        buffered=True,
        max_per_tenant_buffer_size=3,
    )

    # Tenant B sits below the cap.
    _emit(sink, "org-B", 2, prefix="B")
    # Tenant A overflows: cap=3, send 5 → 2 events dropped (the oldest).
    _emit(sink, "org-A", 5, prefix="A")

    sizes = sink.buffer_size_per_tenant()
    # B is unaffected by A's overflow.
    assert sizes["org-B"] == 2
    # A is at the cap (not above).
    assert sizes["org-A"] == 3
    # A's drop counter == 2; B's drop counter is absent or 0.
    drops = sink.dropped_per_tenant
    assert drops.get("org-A") == 2
    assert "org-B" not in drops or drops["org-B"] == 0

    # Flush — A gets exactly 3 events: A-2, A-3, A-4 (oldest two dropped).
    sink.flush()
    by_tenant = {tenant: events for events, tenant in pipeline.calls}
    a_names = [e["payload"]["tool_name"] for e in by_tenant["org-A"]]
    assert a_names == ["A-2", "A-3", "A-4"]
    # B's events are intact and untouched.
    b_names = [e["payload"]["tool_name"] for e in by_tenant["org-B"]]
    assert b_names == ["B-0", "B-1"]


def test_max_per_tenant_buffer_size_must_be_positive() -> None:
    """A non-positive cap is rejected at construction (no silent passthrough)."""
    with pytest.raises(ValueError, match="must be > 0"):
        IngestionPipelineSink(pipeline=_RecordingPipeline(), max_per_tenant_buffer_size=0)
    with pytest.raises(ValueError, match="must be > 0"):
        IngestionPipelineSink(pipeline=_RecordingPipeline(), max_per_tenant_buffer_size=-1)


# ---------------------------------------------------------------------------
# Concurrency: per-tenant burst isolation under contention
# ---------------------------------------------------------------------------


def test_concurrent_bursts_two_tenants_partition_correctly() -> None:
    """Two threads emitting under contention produce isolated per-tenant batches."""
    pipeline = _RecordingPipeline()
    sink = IngestionPipelineSink(
        pipeline=pipeline,
        buffered=True,
        max_per_tenant_buffer_size=10_000,
    )

    n = 500

    def _drive(org_id: str) -> None:
        for i in range(n):
            sink.send(
                "tool.call",
                {"tool_name": f"{org_id}-{i}", "org_id": org_id},
                timestamp_ns=1_700_000_000_000_000_000 + i,
                org_id=org_id,
            )

    t_a = threading.Thread(target=_drive, args=("org-A",))
    t_b = threading.Thread(target=_drive, args=("org-B",))
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)
    assert not t_a.is_alive() and not t_b.is_alive()

    sizes = sink.buffer_size_per_tenant()
    assert sizes["org-A"] == n
    assert sizes["org-B"] == n

    sink.flush()
    by_tenant: Dict[str, List[Dict[str, Any]]] = {}
    for events, tenant in pipeline.calls:
        by_tenant.setdefault(tenant, []).extend(events)

    # Every event in A's flush is tagged with A. No B leakage in either direction.
    assert len(by_tenant["org-A"]) == n
    assert len(by_tenant["org-B"]) == n
    assert all(e["org_id"] == "org-A" for e in by_tenant["org-A"])
    assert all(e["org_id"] == "org-B" for e in by_tenant["org-B"])


def test_noisy_tenant_does_not_evict_quiet_tenants_buffer() -> None:
    """One tenant flooding past the cap evicts ONLY its own events."""
    pipeline = _RecordingPipeline()
    cap = 50
    sink = IngestionPipelineSink(
        pipeline=pipeline,
        buffered=True,
        max_per_tenant_buffer_size=cap,
    )

    # Quiet tenant fills its buffer with a small burst.
    _emit(sink, "org-quiet", 5, prefix="Q")
    quiet_baseline = sink.buffer_size_per_tenant()["org-quiet"]
    assert quiet_baseline == 5

    # Noisy tenant in parallel drives 10x the cap, FIFO-evicts itself.
    noise = cap * 10

    def _flood() -> None:
        for i in range(noise):
            sink.send(
                "tool.call",
                {"tool_name": f"NOISE-{i}", "org_id": "org-noisy"},
                timestamp_ns=1_700_000_000_000_000_000 + i,
                org_id="org-noisy",
            )

    th = threading.Thread(target=_flood)
    th.start()
    th.join(timeout=15)
    assert not th.is_alive()

    # Quiet tenant's buffer is unchanged.
    assert sink.buffer_size_per_tenant()["org-quiet"] == quiet_baseline
    # Noisy tenant capped exactly at cap.
    assert sink.buffer_size_per_tenant()["org-noisy"] == cap
    # Drops counted ONLY against the noisy tenant.
    drops = sink.dropped_per_tenant
    assert drops.get("org-noisy") == noise - cap
    assert "org-quiet" not in drops or drops["org-quiet"] == 0


# ---------------------------------------------------------------------------
# Observability surface
# ---------------------------------------------------------------------------


def test_buffer_size_per_tenant_returns_defensive_copy() -> None:
    """Caller mutation of the returned dict does not affect sink state."""
    pipeline = _RecordingPipeline()
    sink = IngestionPipelineSink(pipeline=pipeline, buffered=True)
    _emit(sink, "org-X", 2)

    snapshot = sink.buffer_size_per_tenant()
    snapshot["org-X"] = 99999
    snapshot["org-OTHER"] = 42

    # Sink's internal state is unaffected.
    fresh = sink.buffer_size_per_tenant()
    assert fresh == {"org-X": 2}


def test_dropped_per_tenant_returns_defensive_copy() -> None:
    """Caller mutation of dropped counts dict does not affect sink state."""
    pipeline = _RecordingPipeline()
    sink = IngestionPipelineSink(
        pipeline=pipeline,
        buffered=True,
        max_per_tenant_buffer_size=2,
    )
    _emit(sink, "org-Q", 5)

    snapshot = sink.dropped_per_tenant
    snapshot["org-Q"] = 0
    snapshot["org-bogus"] = 42

    fresh = sink.dropped_per_tenant
    assert fresh == {"org-Q": 3}


def test_close_flushes_all_per_tenant_buffers() -> None:
    """``close()`` is equivalent to a final per-tenant flush across every tenant."""
    pipeline = _RecordingPipeline()
    sink = IngestionPipelineSink(pipeline=pipeline, buffered=True)

    _emit(sink, "org-A", 3)
    _emit(sink, "org-B", 1)

    sink.close()

    by_tenant = {tenant: events for events, tenant in pipeline.calls}
    assert sorted(by_tenant.keys()) == ["org-A", "org-B"]
    assert len(by_tenant["org-A"]) == 3
    assert len(by_tenant["org-B"]) == 1
    # Subsequent send becomes a no-op (sink is closed).
    _emit(sink, "org-C", 1)
    assert "org-C" not in {tenant for _, tenant in pipeline.calls}
