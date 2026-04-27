"""Multi-tenant ``org_id`` propagation contract tests.

These tests pin the cross-cutting CLAUDE.md guarantee: every event
emitted from a :class:`BaseAdapter` carries the adapter's bound
``org_id``, the bound value cannot be omitted at construction, and
adapters constructed for one tenant cannot leak events into another.

Background: a 2026-04-25 depth audit
(``A:/tmp/adapter-depth-audit.md`` cross-cutting finding #3) flagged
that all 203 adapter emissions in the stratix-python SDK shipped
without ``org_id`` propagation. This suite enforces the fix.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from layerlens.instrument.adapters._base import (
    ORG_ID_FIELD,
    EventSink,
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
    """Stratix client double that records every emit call."""

    def __init__(self, org_id: str = "org-test") -> None:
        self.org_id = org_id
        self.events: List[Tuple[Any, ...]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        self.events.append(args)


class _RecordingSink(EventSink):
    """Sink that records every event it receives, including the
    explicit ``org_id`` keyword passed by the adapter dispatch path."""

    def __init__(self) -> None:
        self.received: List[Dict[str, Any]] = []
        self.flushed = 0
        self.closed = 0

    def send(
        self,
        event_type: str,
        payload: Dict[str, Any],
        timestamp_ns: int,
        *,
        org_id: str,
    ) -> None:
        self.received.append(
            {
                "event_type": event_type,
                "payload": payload,
                "timestamp_ns": timestamp_ns,
                "org_id": org_id,
            }
        )

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed += 1


class _MinimalAdapter(BaseAdapter):
    """Minimal concrete BaseAdapter used to exercise emission paths."""

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


class _TypedPayload:
    """Stand-in for a typed Pydantic payload with mutable attributes."""

    def __init__(self, event_type: str, **kwargs: Any) -> None:
        self.event_type = event_type
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Construction-time fail-fast
# ---------------------------------------------------------------------------


def test_init_raises_when_no_org_id_anywhere() -> None:
    """Adapter construction with neither stratix nor org_id raises."""
    with pytest.raises(ValueError, match="non-empty org_id"):
        _MinimalAdapter()


def test_init_raises_when_explicit_org_id_is_empty_string() -> None:
    """An explicit empty string org_id is rejected (no silent fallback)."""
    with pytest.raises(ValueError, match="non-empty org_id"):
        _MinimalAdapter(org_id="")


def test_init_raises_when_explicit_org_id_is_whitespace() -> None:
    """Whitespace-only org_id is rejected (treated as empty)."""
    with pytest.raises(ValueError, match="non-empty org_id"):
        _MinimalAdapter(org_id="   ")


def test_init_raises_when_stratix_org_id_is_blank() -> None:
    """A stratix client whose org_id field is blank raises."""
    blank = SimpleNamespace(org_id="")
    with pytest.raises(ValueError, match="non-empty org_id"):
        _MinimalAdapter(stratix=blank)


def test_init_accepts_explicit_org_id() -> None:
    """Explicit org_id kwarg satisfies the contract."""
    adapter = _MinimalAdapter(org_id="org-alpha")
    assert adapter.org_id == "org-alpha"


def test_init_resolves_org_id_from_stratix_org_id_attribute() -> None:
    """Resolution falls back to ``stratix.org_id`` when no explicit kwarg."""
    stratix = SimpleNamespace(org_id="org-from-stratix")
    adapter = _MinimalAdapter(stratix=stratix)
    assert adapter.org_id == "org-from-stratix"


def test_init_resolves_org_id_from_stratix_organization_id_attribute() -> None:
    """The public Stratix client uses ``organization_id``; resolve it."""
    stratix = SimpleNamespace(organization_id="org-from-public-client")
    adapter = _MinimalAdapter(stratix=stratix)
    assert adapter.org_id == "org-from-public-client"


def test_explicit_org_id_takes_precedence_over_stratix() -> None:
    """The explicit kwarg wins over any client-attached value."""
    stratix = SimpleNamespace(org_id="org-from-stratix")
    adapter = _MinimalAdapter(stratix=stratix, org_id="org-explicit")
    assert adapter.org_id == "org-explicit"


# ---------------------------------------------------------------------------
# Per-emission propagation
# ---------------------------------------------------------------------------


def test_emit_dict_event_stamps_org_id_into_payload() -> None:
    """``emit_dict_event`` injects org_id into the dict payload."""
    stratix = _RecordingStratix(org_id="org-alpha")
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    adapter.emit_dict_event("tool.call", {"tool_name": "calc", "args": {"x": 1}})

    assert len(stratix.events) == 1
    event_type, payload = stratix.events[0]
    assert event_type == "tool.call"
    assert payload[ORG_ID_FIELD] == "org-alpha"
    assert payload["tool_name"] == "calc"


def test_emit_dict_event_overwrites_caller_supplied_org_id() -> None:
    """A caller-supplied org_id (potentially wrong tenant) is overwritten
    with the adapter's bound tenant. This prevents cross-tenant leaks
    via misuse — the adapter's binding is the source of truth."""
    stratix = _RecordingStratix(org_id="org-alpha")
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    payload: Dict[str, Any] = {"org_id": "org-WRONG", "tool_name": "calc"}
    adapter.emit_dict_event("tool.call", payload)

    _, recorded = stratix.events[0]
    assert recorded[ORG_ID_FIELD] == "org-alpha", "adapter MUST overwrite caller-supplied org_id"


def test_emit_event_stamps_org_id_into_typed_payload() -> None:
    """``emit_event`` (typed path) sets ``payload.org_id`` on the model."""
    stratix = _RecordingStratix(org_id="org-beta")
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    payload = _TypedPayload("model.invoke", model="gpt-5")
    adapter.emit_event(payload)

    assert len(stratix.events) == 1
    (recorded,) = stratix.events[0]
    assert getattr(recorded, ORG_ID_FIELD) == "org-beta"


def test_trace_event_record_carries_org_id() -> None:
    """The replay trace record includes org_id at the envelope level."""
    stratix = _RecordingStratix(org_id="org-gamma")
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()
    adapter.emit_dict_event("agent.input", {"input": "hi"})

    trace = adapter.serialize_for_replay()
    assert len(trace.events) == 1
    record = trace.events[0]
    assert record[ORG_ID_FIELD] == "org-gamma"
    assert record["payload"][ORG_ID_FIELD] == "org-gamma"


def test_event_sinks_receive_explicit_org_id_kwarg() -> None:
    """Sinks see ``org_id`` both inside payload and as the kwarg."""
    stratix = _RecordingStratix(org_id="org-delta")
    sink = _RecordingSink()
    adapter = _MinimalAdapter(stratix=stratix, event_sinks=[sink])
    adapter.connect()

    adapter.emit_dict_event("tool.call", {"tool_name": "calc"})

    assert len(sink.received) == 1
    received = sink.received[0]
    assert received["org_id"] == "org-delta"
    assert received["payload"][ORG_ID_FIELD] == "org-delta"


# ---------------------------------------------------------------------------
# Cross-tenant isolation
# ---------------------------------------------------------------------------


def test_two_adapters_for_different_tenants_do_not_cross_contaminate() -> None:
    """Adapter A bound to tenant A never emits events tagged with tenant B."""
    stratix_a = _RecordingStratix(org_id="org-A")
    stratix_b = _RecordingStratix(org_id="org-B")

    adapter_a = _MinimalAdapter(stratix=stratix_a)
    adapter_b = _MinimalAdapter(stratix=stratix_b)
    adapter_a.connect()
    adapter_b.connect()

    adapter_a.emit_dict_event("tool.call", {"tool_name": "calc-a"})
    adapter_b.emit_dict_event("tool.call", {"tool_name": "calc-b"})

    assert all(p[ORG_ID_FIELD] == "org-A" for _, p in stratix_a.events)
    assert all(p[ORG_ID_FIELD] == "org-B" for _, p in stratix_b.events)
    # And vice versa: stratix_b never saw an org-A event.
    assert not any(p[ORG_ID_FIELD] == "org-A" for _, p in stratix_b.events)
    assert not any(p[ORG_ID_FIELD] == "org-B" for _, p in stratix_a.events)


def test_adapter_overwrites_payload_with_other_tenant_value() -> None:
    """Even if a caller injects another tenant's org_id directly into the
    payload, the adapter's bound tenant prevails. Defensive overwrite is
    the documented contract — see :meth:`BaseAdapter._stamp_org_id`."""
    stratix = _RecordingStratix(org_id="org-A")
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    adapter.emit_dict_event(
        "tool.call",
        {"org_id": "org-B-leaked", "tool_name": "should-be-stamped-as-A"},
    )

    _, payload = stratix.events[0]
    assert payload[ORG_ID_FIELD] == "org-A"


# ---------------------------------------------------------------------------
# Public surface stability
# ---------------------------------------------------------------------------


def test_org_id_field_constant_value() -> None:
    """The ORG_ID_FIELD constant is the documented payload field name."""
    assert ORG_ID_FIELD == "org_id"


def test_adapter_org_id_property_returns_resolved_value() -> None:
    """The ``adapter.org_id`` read-only property reports the bound tenant."""
    adapter = _MinimalAdapter(org_id="org-property-check")
    assert adapter.org_id == "org-property-check"
