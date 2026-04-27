"""OTel ↔ SDK ``org_id`` cross-correlation tests (Gap 3).

When a :class:`BaseAdapter` emits an event, the SDK's per-event
``org_id`` is stamped onto the **currently-active** OpenTelemetry span
as the ``layerlens.org_id`` attribute. This closes the cross-system
correlation gap: every distributed-trace span produced inside an
adapter call can be filtered by tenant.

Properties enforced
-------------------

* When OTel is installed AND a recording span is active, every
  ``emit_event`` / ``emit_dict_event`` sets ``layerlens.org_id`` on
  that span to the adapter's bound tenant.
* When the active span is the no-op span (no SDK installed / no
  TracerProvider configured), the attribute set is a no-op — the
  adapter's hot path never raises.
* When ``span.is_recording()`` returns False, the attribute set is
  skipped (paying for it on a sampled-out span is wasteful).
* Two adapters bound to different tenants emitting under the same
  parent span overwrite the attribute with each emit — the LAST emit's
  tenant wins per span (which is the expected semantics: a span scopes
  one logical operation by one tenant).
* Adapter errors do not block the OTel attribute set (defensive try /
  except around the OTel API).

Background
----------
The 2026-04-25 cross-cutting audit (gap #3) identified that the SDK's
``org_id`` was nowhere visible in OTel spans, making per-tenant
distributed-trace queries impossible from atlas-app and external
backends (Tempo / Jaeger / Honeycomb). This file pins the fix.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
    _set_current_span_org_id,
)

# ---------------------------------------------------------------------------
# OTel test harness — single global TracerProvider, per-test exporter
# ---------------------------------------------------------------------------
#
# ``trace.set_tracer_provider`` accepts the first call only (subsequent
# calls log "Overriding of current TracerProvider is not allowed" and
# silently ignore the new provider). We therefore install one provider
# at module load and attach a fresh InMemorySpanExporter per test via
# :func:`memory_exporter` — the exporter is the per-test surface, not
# the provider itself.

_PROVIDER = TracerProvider()
otel_trace.set_tracer_provider(_PROVIDER)


@pytest.fixture()
def memory_exporter() -> InMemorySpanExporter:
    """Attach a fresh in-memory exporter to the module-global provider."""
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    _PROVIDER.add_span_processor(processor)
    try:
        yield exporter
    finally:
        exporter.clear()
        # SDK does not expose a remove_span_processor; shut the
        # processor down so it stops receiving span events.
        processor.shutdown()


@pytest.fixture()
def tracer() -> otel_trace.Tracer:
    """Tracer scoped to the test module."""
    return _PROVIDER.get_tracer("layerlens.tests.otel_correlation")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _RecordingStratix:
    def __init__(self, org_id: str) -> None:
        self.org_id = org_id
        self.events: List[Tuple[Any, ...]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        self.events.append(args)


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


def _attrs(span: Any) -> Dict[str, Any]:
    """Extract the attributes dict from a captured span (cross-version safe)."""
    raw = getattr(span, "attributes", None) or {}
    return dict(raw)


# ---------------------------------------------------------------------------
# Cross-correlation under an active span
# ---------------------------------------------------------------------------


def test_emit_dict_event_stamps_org_id_on_active_span(
    memory_exporter: InMemorySpanExporter,
    tracer: otel_trace.Tracer,
) -> None:
    """``emit_dict_event`` sets ``layerlens.org_id`` on the active span."""
    stratix = _RecordingStratix("org-A")
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    with tracer.start_as_current_span("auth-check") as span:
        span.set_attribute("auth.org_id", "org-A")
        adapter.emit_dict_event("tool.call", {"tool_name": "calc"})

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = _attrs(spans[0])
    assert attrs.get("layerlens.org_id") == "org-A"
    # Pre-existing auth span attribute is preserved.
    assert attrs.get("auth.org_id") == "org-A"


def test_emit_event_stamps_org_id_on_active_span(
    memory_exporter: InMemorySpanExporter,
    tracer: otel_trace.Tracer,
) -> None:
    """``emit_event`` (typed payload path) also stamps the OTel span."""

    class _TypedPayload:
        def __init__(self) -> None:
            self.event_type = "model.invoke"
            self.model = "gpt-5"

    stratix = _RecordingStratix("org-B")
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    with tracer.start_as_current_span("invoke"):
        adapter.emit_event(_TypedPayload())

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert _attrs(spans[0]).get("layerlens.org_id") == "org-B"


def test_two_adapters_for_different_tenants_overwrite_per_span(
    memory_exporter: InMemorySpanExporter,
    tracer: otel_trace.Tracer,
) -> None:
    """Two adapters under the same parent span — last emit's tenant wins.

    One span scopes one logical operation by one tenant. A correctly-
    configured caller never multiplexes tenants under one span; this
    test simply pins the behaviour so misuse is observable rather than
    silently merging tenants.
    """
    a = _MinimalAdapter(stratix=_RecordingStratix("org-A"))
    b = _MinimalAdapter(stratix=_RecordingStratix("org-B"))
    a.connect()
    b.connect()

    with tracer.start_as_current_span("shared-parent"):
        a.emit_dict_event("tool.call", {"x": 1})
        b.emit_dict_event("tool.call", {"y": 1})

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert _attrs(spans[0]).get("layerlens.org_id") == "org-B"


def test_each_span_carries_its_own_tenant_attribute(
    memory_exporter: InMemorySpanExporter,
    tracer: otel_trace.Tracer,
) -> None:
    """Two sequential spans, two adapters: each span tagged with its own tenant."""
    a = _MinimalAdapter(stratix=_RecordingStratix("org-A"))
    b = _MinimalAdapter(stratix=_RecordingStratix("org-B"))
    a.connect()
    b.connect()

    with tracer.start_as_current_span("op-A"):
        a.emit_dict_event("tool.call", {"x": 1})
    with tracer.start_as_current_span("op-B"):
        b.emit_dict_event("tool.call", {"y": 1})

    spans = memory_exporter.get_finished_spans()
    by_name = {s.name: _attrs(s) for s in spans}
    assert by_name["op-A"].get("layerlens.org_id") == "org-A"
    assert by_name["op-B"].get("layerlens.org_id") == "org-B"


# ---------------------------------------------------------------------------
# No-op behaviour when no span / no recording span
# ---------------------------------------------------------------------------


def test_emit_outside_span_does_not_raise(
    memory_exporter: InMemorySpanExporter,
) -> None:
    """Emit with no active span finishes without raising and produces no spans."""
    stratix = _RecordingStratix("org-A")
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    # No span context — the OTel call is a no-op (INVALID_SPAN).
    adapter.emit_dict_event("tool.call", {"tool_name": "ok"})

    # OTel returns a tuple; convert for portable comparison.
    assert list(memory_exporter.get_finished_spans()) == []
    # SDK still saw the event.
    assert len(stratix.events) == 1


def test_set_current_span_org_id_is_safe_when_otel_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When OTel import fails, the helper silently no-ops."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError("simulated missing opentelemetry")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # No exception raised even without OTel.
    _set_current_span_org_id("org-A")


def test_non_recording_span_skipped(
    memory_exporter: InMemorySpanExporter,
) -> None:
    """A non-recording span (sampled out) is skipped — no attribute set."""

    class _FakeSpan:
        attributes: Dict[str, Any] = {}

        def is_recording(self) -> bool:
            return False

        def set_attribute(self, key: str, value: Any) -> None:
            # If we got here, the recording check failed — record it.
            type(self).attributes[key] = value

    fake_span = _FakeSpan()

    # Patch get_current_span to return our fake non-recording span.
    real_get = otel_trace.get_current_span
    try:
        otel_trace.get_current_span = lambda *args, **kwargs: fake_span  # type: ignore[assignment]
        _set_current_span_org_id("org-A")
    finally:
        otel_trace.get_current_span = real_get  # type: ignore[assignment]

    assert "layerlens.org_id" not in _FakeSpan.attributes


def test_otel_set_attribute_failure_does_not_block_emit(
    memory_exporter: InMemorySpanExporter,
    tracer: otel_trace.Tracer,
) -> None:
    """If set_attribute raises, the emit path still completes."""
    stratix = _RecordingStratix("org-A")
    adapter = _MinimalAdapter(stratix=stratix)
    adapter.connect()

    class _BoomSpan:
        def is_recording(self) -> bool:
            return True

        def set_attribute(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("boom")

    real_get = otel_trace.get_current_span
    try:
        otel_trace.get_current_span = lambda *args, **kwargs: _BoomSpan()  # type: ignore[assignment]
        # Must not raise.
        adapter.emit_dict_event("tool.call", {"tool_name": "ok"})
    finally:
        otel_trace.get_current_span = real_get  # type: ignore[assignment]

    # SDK still emitted the event despite the OTel failure.
    assert len(stratix.events) == 1
