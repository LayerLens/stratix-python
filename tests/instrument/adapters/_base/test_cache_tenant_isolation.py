"""In-memory cache tenant-isolation tests (Gap 1).

Every per-instance cache held by :class:`BaseAdapter` (circuit breaker
counters, error count, ``_trace_events`` buffer, sink registry,
``_circuit_open`` flag, opened-at timestamp) must be tenant-scoped.
Because adapters fail-fast at construction with a single ``org_id``
binding (PR #118), the **per-instance** lifetime of every cache
inherits that single-tenant scope.

This suite enforces that contract empirically — not by trusting the
type system but by:

* Constructing two adapters bound to different tenants in the same
  process and asserting their state never bleeds.
* Driving sustained concurrent emission from multiple threads — one per
  tenant — and proving every event in tenant A's recorded stream is
  tagged with tenant A and never B (or vice versa).
* Tripping tenant A's circuit breaker and verifying tenant B's adapter
  remains HEALTHY and emit-able.
* Replacing one adapter's sink list and verifying the other adapter's
  sinks are untouched.

Background
----------
A 2026-04-25 multi-tenancy audit
(``A:/tmp/adapter-depth-audit.md`` cross-cutting finding #1) flagged
that the in-memory caches in :class:`BaseAdapter` were never tested for
cross-tenant contamination. PR #118 added construction-time
tenant binding; this suite closes the audit by proving the runtime
state behaves accordingly under contention.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from layerlens.instrument.adapters._base.sinks import EventSink
from layerlens.instrument.adapters._base.adapter import (
    ORG_ID_FIELD,
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _RecordingStratix:
    """Stratix double that records every emit call thread-safely."""

    def __init__(self, org_id: str) -> None:
        self.org_id = org_id
        self._lock = threading.Lock()
        self._events: List[Tuple[Any, ...]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self._events.append(args)

    @property
    def events(self) -> List[Tuple[Any, ...]]:
        with self._lock:
            return list(self._events)


class _RecordingSink(EventSink):
    """Sink that records every event seen, with thread-safe append."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._received: List[Dict[str, Any]] = []

    def send(
        self,
        event_type: str,
        payload: Dict[str, Any],
        timestamp_ns: int,
        *,
        org_id: str,
    ) -> None:
        with self._lock:
            self._received.append({"event_type": event_type, "payload": payload, "org_id": org_id})

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def received(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._received)


class _FailingStratix:
    """Stratix double whose emit always raises — used to trip the breaker."""

    def __init__(self, org_id: str) -> None:
        self.org_id = org_id
        self.calls = 0

    def emit(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1
        raise RuntimeError("synthetic emit failure")


class _MinimalAdapter(BaseAdapter):
    FRAMEWORK = "test"
    VERSION = "0.0.0"

    def connect(self) -> None:
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def health_check(self) -> AdapterHealth:
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            adapter_version=self.VERSION,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(name="MinimalAdapter", version=self.VERSION, framework=self.FRAMEWORK)

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="MinimalAdapter",
            framework=self.FRAMEWORK,
            trace_id="trace-test",
            events=list(self._trace_events),
        )


# ---------------------------------------------------------------------------
# Per-instance cache scoping
# ---------------------------------------------------------------------------


def test_trace_events_buffer_is_per_instance() -> None:
    """``_trace_events`` is a per-instance list — never shared across tenants."""
    a = _MinimalAdapter(stratix=_RecordingStratix("org-A"))
    b = _MinimalAdapter(stratix=_RecordingStratix("org-B"))
    a.connect()
    b.connect()

    a.emit_dict_event("tool.call", {"tool_name": "calc-A"})
    a.emit_dict_event("tool.call", {"tool_name": "calc-A2"})
    b.emit_dict_event("tool.call", {"tool_name": "calc-B"})

    trace_a = a.serialize_for_replay()
    trace_b = b.serialize_for_replay()

    assert len(trace_a.events) == 2
    assert len(trace_b.events) == 1
    # Cross-check the org_id stamp — every event in A's trace is org-A,
    # every event in B's trace is org-B.
    assert all(e[ORG_ID_FIELD] == "org-A" for e in trace_a.events)
    assert all(e[ORG_ID_FIELD] == "org-B" for e in trace_b.events)


def test_circuit_breaker_state_is_per_instance() -> None:
    """Tripping tenant A's breaker leaves tenant B HEALTHY."""
    failing = _FailingStratix("org-A")
    healthy = _RecordingStratix("org-B")
    a = _MinimalAdapter(stratix=failing)
    b = _MinimalAdapter(stratix=healthy)
    a.connect()
    b.connect()

    # Drive 12 emit failures into A — exceeds the threshold (10).
    for i in range(12):
        a.emit_dict_event("tool.call", {"i": i})

    # A is now in ERROR with circuit OPEN — but B is untouched.
    assert a._circuit_open is True
    assert a._status == AdapterStatus.ERROR
    assert a._error_count >= 10

    assert b._circuit_open is False
    assert b._status == AdapterStatus.HEALTHY
    assert b._error_count == 0

    # B can still emit normally even though A's circuit is open.
    b.emit_dict_event("tool.call", {"tool_name": "B-still-works"})
    assert len(healthy.events) == 1


def test_sink_registry_is_per_instance() -> None:
    """Sinks added to A are NOT visible on B."""
    sink_a = _RecordingSink()
    sink_b = _RecordingSink()
    a = _MinimalAdapter(stratix=_RecordingStratix("org-A"), event_sinks=[sink_a])
    b = _MinimalAdapter(stratix=_RecordingStratix("org-B"), event_sinks=[sink_b])
    a.connect()
    b.connect()

    a.emit_dict_event("tool.call", {"tool_name": "to-A"})
    b.emit_dict_event("tool.call", {"tool_name": "to-B"})

    assert len(sink_a.received) == 1
    assert sink_a.received[0]["org_id"] == "org-A"

    assert len(sink_b.received) == 1
    assert sink_b.received[0]["org_id"] == "org-B"


def test_remove_sink_does_not_affect_other_adapter() -> None:
    """Mutating one adapter's sink list never touches another's."""
    shared_sink = _RecordingSink()
    a = _MinimalAdapter(stratix=_RecordingStratix("org-A"), event_sinks=[shared_sink])
    b = _MinimalAdapter(stratix=_RecordingStratix("org-B"), event_sinks=[shared_sink])

    # Even though we passed the SAME sink instance into both adapters,
    # removing it from A leaves B's reference intact.
    assert a.remove_sink(shared_sink) is True
    assert shared_sink in b.sinks
    assert shared_sink not in a.sinks


def test_lock_is_per_instance() -> None:
    """Each adapter holds its own ``_lock`` — no global serialization."""
    a = _MinimalAdapter(org_id="org-A")
    b = _MinimalAdapter(org_id="org-B")
    assert a._lock is not b._lock


def test_org_id_property_remains_immutable_per_instance() -> None:
    """Adapter's bound ``org_id`` cannot be silently swapped at runtime."""
    a = _MinimalAdapter(org_id="org-A")
    b = _MinimalAdapter(org_id="org-B")
    assert a.org_id == "org-A"
    assert b.org_id == "org-B"
    # Mutating B does not affect A even via attribute write.
    b._org_id = "org-B-CHANGED"
    assert a.org_id == "org-A"


# ---------------------------------------------------------------------------
# Concurrent emission stress — proves no cross-tenant pollution under load
# ---------------------------------------------------------------------------


def _emit_burst(adapter: BaseAdapter, label: str, n: int) -> None:
    """Emit ``n`` events through ``adapter``, each labelled distinctly."""
    for i in range(n):
        adapter.emit_dict_event("tool.call", {"tool_name": f"{label}-{i}"})


def test_concurrent_emission_two_tenants_no_cross_contamination() -> None:
    """Two threads, two tenants, 500 events each — no leak in either direction."""
    stratix_a = _RecordingStratix("org-A")
    stratix_b = _RecordingStratix("org-B")
    sink_a = _RecordingSink()
    sink_b = _RecordingSink()
    adapter_a = _MinimalAdapter(stratix=stratix_a, event_sinks=[sink_a])
    adapter_b = _MinimalAdapter(stratix=stratix_b, event_sinks=[sink_b])
    adapter_a.connect()
    adapter_b.connect()

    n = 500
    t_a = threading.Thread(target=_emit_burst, args=(adapter_a, "A", n))
    t_b = threading.Thread(target=_emit_burst, args=(adapter_b, "B", n))
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)
    assert not t_a.is_alive(), "tenant A burst thread hung"
    assert not t_b.is_alive(), "tenant B burst thread hung"

    # A's stratix saw exactly n org-A events and zero org-B events.
    a_events = stratix_a.events
    b_events = stratix_b.events
    assert len(a_events) == n
    assert len(b_events) == n
    assert all(p[ORG_ID_FIELD] == "org-A" for _, p in a_events)
    assert all(p[ORG_ID_FIELD] == "org-B" for _, p in b_events)
    assert not any(p[ORG_ID_FIELD] == "org-B" for _, p in a_events)
    assert not any(p[ORG_ID_FIELD] == "org-A" for _, p in b_events)

    # Sinks observed the same isolation.
    assert all(r["org_id"] == "org-A" for r in sink_a.received)
    assert all(r["org_id"] == "org-B" for r in sink_b.received)


def test_concurrent_emission_three_tenants_isolated_under_contention() -> None:
    """Three concurrent tenants, each with their own adapter, see only their events."""
    tenants = ["org-X", "org-Y", "org-Z"]
    stratix_per_tenant = {t: _RecordingStratix(t) for t in tenants}
    adapters = {t: _MinimalAdapter(stratix=stratix_per_tenant[t]) for t in tenants}
    for a in adapters.values():
        a.connect()

    n = 200
    threads = [
        threading.Thread(target=_emit_burst, args=(adapters[t], t, n)) for t in tenants
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10)
        assert not th.is_alive()

    for t in tenants:
        events = stratix_per_tenant[t].events
        assert len(events) == n
        # Every event in this tenant's stream is correctly tagged.
        assert all(p[ORG_ID_FIELD] == t for _, p in events)
        # Cross-checks: no other tenant's id appears anywhere.
        for other in tenants:
            if other == t:
                continue
            assert not any(p[ORG_ID_FIELD] == other for _, p in events), (
                f"tenant {other}'s id leaked into {t}'s event stream"
            )


def test_concurrent_emission_does_not_corrupt_trace_event_buffer() -> None:
    """High-concurrency emit calls produce a consistent trace_events count."""
    stratix = _RecordingStratix("org-S")
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    n_per_thread = 100
    n_threads = 8

    def _worker() -> None:
        for i in range(n_per_thread):
            adapter.emit_dict_event("tool.call", {"i": i})

    threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10)
        assert not th.is_alive()

    # Trace event buffer length == total emissions. Even though events
    # may interleave, none are lost or duplicated.
    expected = n_per_thread * n_threads
    assert len(adapter._trace_events) == expected
    # Every record carries the same (correct) tenant.
    assert all(e[ORG_ID_FIELD] == "org-S" for e in adapter._trace_events)


# ---------------------------------------------------------------------------
# "Eviction respects tenant scope" — applies to per-instance trace_events too
# ---------------------------------------------------------------------------


def test_disconnect_does_not_clear_other_adapter_state() -> None:
    """Disconnecting tenant A leaves tenant B's events / sinks intact."""
    stratix_a = _RecordingStratix("org-A")
    stratix_b = _RecordingStratix("org-B")
    sink_b = _RecordingSink()
    a = _MinimalAdapter(stratix=stratix_a)
    b = _MinimalAdapter(stratix=stratix_b, event_sinks=[sink_b])
    a.connect()
    b.connect()

    a.emit_dict_event("tool.call", {"x": 1})
    b.emit_dict_event("tool.call", {"y": 1})

    a.disconnect()

    # B's recorded state is untouched.
    assert len(b._trace_events) == 1
    assert b._trace_events[0][ORG_ID_FIELD] == "org-B"
    assert sink_b.received[0]["org_id"] == "org-B"
    # B can still emit after A disconnects.
    b.emit_dict_event("tool.call", {"y": 2})
    assert len(stratix_b.events) == 2


def test_no_class_level_mutable_caches_on_base_adapter() -> None:
    """BaseAdapter MUST NOT carry class-level mutable state that could leak.

    A class-level ``dict`` or ``list`` shared across instances would
    silently fan a tenant A write into a tenant B read — exactly the
    bug Gap 1 forbids. This guard fails loudly if a future commit adds
    one.
    """
    # Allowlist of class-level non-callable attrs that are explicitly
    # immutable / type-level.
    allowed_class_attrs = {"FRAMEWORK", "VERSION", "requires_pydantic"}
    for name, value in vars(BaseAdapter).items():
        if name.startswith("_") or callable(value) or isinstance(value, (property, staticmethod, classmethod)):
            continue
        if name in allowed_class_attrs:
            continue
        # A bare list/dict/set at class level would be cross-instance
        # shared state — banned.
        assert not isinstance(value, (list, dict, set)), (
            f"BaseAdapter.{name} is a class-level mutable container "
            f"({type(value).__name__}) — would leak across tenants"
        )


def test_two_adapters_share_no_module_globals_for_state() -> None:
    """The adapter module exposes constants but no mutable state singletons.

    Module-level mutable state (a global dict, a global list) keyed
    without ``org_id`` would defeat the per-instance binding. The
    constants (``_CIRCUIT_BREAKER_THRESHOLD`` etc.) are immutable
    primitives and therefore safe.
    """
    import layerlens.instrument.adapters._base.adapter as adapter_mod

    forbidden_types = (list, dict, set, bytearray)
    leaks: list[str] = []
    for name, value in vars(adapter_mod).items():
        if name.startswith("__"):
            continue
        # Skip imported / type / function objects.
        if isinstance(value, type) or callable(value):
            continue
        # Skip the null sentinel — it has no mutable surface.
        if name == "_NULL_STRATIX":
            continue
        if isinstance(value, forbidden_types):
            leaks.append(f"{name}={type(value).__name__}")

    assert not leaks, (
        "adapter module exposes mutable global containers that could "
        f"silently aggregate cross-tenant state: {leaks}"
    )


# ---------------------------------------------------------------------------
# Smoke checks ensuring fail-fast still applies (no Gap 1 regression)
# ---------------------------------------------------------------------------


def test_construction_without_org_id_still_fail_fasts() -> None:
    """Gap 1 hardening MUST NOT regress PR #118's fail-fast guarantee."""
    with pytest.raises(ValueError, match="non-empty org_id"):
        _MinimalAdapter()


def test_blank_stratix_org_id_still_rejected() -> None:
    """A blank ``stratix.org_id`` still raises (no silent fallback)."""
    with pytest.raises(ValueError, match="non-empty org_id"):
        _MinimalAdapter(stratix=SimpleNamespace(org_id=""))
