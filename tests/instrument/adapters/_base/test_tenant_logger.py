"""Tenant-aware logging propagation tests (Gap 4).

Verifies that :class:`TenantContextLogAdapter` and the
:func:`get_tenant_logger` factory:

* Inject ``org_id`` into every record's ``extra`` dict so structured
  log handlers (JSON formatters, OTel log exporters) see it as a
  first-class field.
* Prepend ``[org_id=<value>] `` to the formatted message body so plain
  log lines also carry the tenant binding.
* Refuse construction without a non-empty ``org_id``
  (matches :class:`BaseAdapter`'s fail-fast contract).
* Are per-instance — two adapters bound to different tenants log with
  their respective bindings even when they share the same underlying
  logger name.
* Override caller-supplied ``extra={"org_id": ...}`` to prevent
  callers from impersonating another tenant in log records.
* Are wired into :class:`BaseAdapter` via the ``tlogger`` property so
  subclass code can drop in ``self.tlogger`` for ``logging.getLogger``.

Background
----------
Per the 2026-04-25 cross-cutting audit (gap #4), adapter log lines
omitted ``org_id`` entirely. Tenant-A and tenant-B circuit breaker
events were indistinguishable in shared log streams, making per-tenant
incident triage impossible. This file pins the fix.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters._base.logging import (
    TenantContextLogAdapter,
    get_tenant_logger,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _RecordingHandler(logging.Handler):
    """Capture every emitted ``LogRecord`` in a list for inspection."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _RecordingStratix:
    def __init__(self, org_id: str) -> None:
        self.org_id = org_id

    def emit(self, *args: Any, **kwargs: Any) -> None:
        pass


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


@pytest.fixture()
def handler() -> _RecordingHandler:
    """Attach a fresh capture handler to the root logger for the test."""
    h = _RecordingHandler()
    root = logging.getLogger()
    # Save and restore level so we don't pollute other tests.
    saved_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(h)
    try:
        yield h
    finally:
        root.removeHandler(h)
        root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# Construction-time fail-fast
# ---------------------------------------------------------------------------


def test_construction_rejects_empty_org_id() -> None:
    """An empty ``org_id`` is rejected at construction (CLAUDE.md fail-fast)."""
    base = logging.getLogger("layerlens.tests.tenant_logger.empty")
    with pytest.raises(ValueError, match="non-empty org_id"):
        TenantContextLogAdapter(base, "")


def test_construction_rejects_whitespace_org_id() -> None:
    """A whitespace-only ``org_id`` is rejected as well."""
    base = logging.getLogger("layerlens.tests.tenant_logger.ws")
    with pytest.raises(ValueError, match="non-empty org_id"):
        TenantContextLogAdapter(base, "   ")


def test_construction_rejects_non_string_org_id() -> None:
    """A non-string ``org_id`` is rejected."""
    base = logging.getLogger("layerlens.tests.tenant_logger.notstr")
    with pytest.raises(ValueError, match="non-empty org_id"):
        TenantContextLogAdapter(base, None)  # type: ignore[arg-type]


def test_org_id_property_returns_bound_value() -> None:
    """The read-only property reports the binding."""
    base = logging.getLogger("layerlens.tests.tenant_logger.prop")
    log = TenantContextLogAdapter(base, "org-prop")
    assert log.org_id == "org-prop"


# ---------------------------------------------------------------------------
# Record extras propagation
# ---------------------------------------------------------------------------


def test_extra_carries_org_id_on_every_record(handler: _RecordingHandler) -> None:
    """Every log call attaches ``org_id`` to the record extras."""
    log = get_tenant_logger("layerlens.tests.tenant_logger.extras", "org-A")

    log.warning("circuit breaker open")
    log.error("retry exhausted")

    assert len(handler.records) == 2
    for rec in handler.records:
        assert getattr(rec, "org_id", None) == "org-A", (
            f"record missing org_id extra: {rec.__dict__}"
        )


def test_message_is_prefixed_with_org_id(handler: _RecordingHandler) -> None:
    """The formatted message body includes the ``[org_id=...]`` prefix."""
    log = get_tenant_logger("layerlens.tests.tenant_logger.prefix", "org-B")
    log.info("emit succeeded")

    assert len(handler.records) == 1
    formatted = handler.records[0].getMessage()
    assert formatted.startswith("[org_id=org-B] ")
    assert "emit succeeded" in formatted


def test_caller_supplied_extra_cannot_overwrite_tenant(handler: _RecordingHandler) -> None:
    """A caller cannot pass ``extra={"org_id": "OTHER"}`` and impersonate a tenant."""
    log = get_tenant_logger("layerlens.tests.tenant_logger.spoof", "org-A")

    log.warning("attempted impersonation", extra={"org_id": "org-B-SPOOFED", "request_id": "r1"})

    assert len(handler.records) == 1
    rec = handler.records[0]
    assert rec.org_id == "org-A", "tenant binding overwritten by caller"
    # Other caller-supplied extras are preserved.
    assert rec.request_id == "r1"


# ---------------------------------------------------------------------------
# Per-instance binding (no leak across adapters / tenants)
# ---------------------------------------------------------------------------


def test_two_loggers_bound_to_different_tenants_do_not_leak(
    handler: _RecordingHandler,
) -> None:
    """Two TenantContextLogAdapter instances on the same logger keep distinct bindings."""
    log_a = get_tenant_logger("layerlens.tests.tenant_logger.shared", "org-A")
    log_b = get_tenant_logger("layerlens.tests.tenant_logger.shared", "org-B")

    log_a.info("from A")
    log_b.info("from B")

    assert len(handler.records) == 2
    by_msg: Dict[str, str] = {}
    for rec in handler.records:
        msg = rec.getMessage()
        if "from A" in msg:
            by_msg["A"] = rec.org_id
        elif "from B" in msg:
            by_msg["B"] = rec.org_id
    assert by_msg == {"A": "org-A", "B": "org-B"}


def test_get_tenant_logger_returns_distinct_adapter_instances() -> None:
    """Each call returns a fresh adapter even for the same logger name."""
    log_a = get_tenant_logger("layerlens.tests.tenant_logger.distinct", "org-A")
    log_b = get_tenant_logger("layerlens.tests.tenant_logger.distinct", "org-A")
    assert log_a is not log_b
    # But underlying logger is the same singleton — getLogger semantics.
    assert log_a.logger is log_b.logger


# ---------------------------------------------------------------------------
# BaseAdapter wiring
# ---------------------------------------------------------------------------


def test_base_adapter_exposes_tlogger_bound_to_its_org_id(handler: _RecordingHandler) -> None:
    """Adapter's ``tlogger`` is bound to the same ``org_id`` as ``adapter.org_id``."""
    adapter = _MinimalAdapter(stratix=_RecordingStratix("org-W"))
    assert isinstance(adapter.tlogger, TenantContextLogAdapter)
    assert adapter.tlogger.org_id == "org-W"

    adapter.tlogger.warning("hello from adapter")

    relevant = [r for r in handler.records if "hello from adapter" in r.getMessage()]
    assert len(relevant) == 1
    assert relevant[0].org_id == "org-W"


def test_two_adapters_have_distinct_tloggers(handler: _RecordingHandler) -> None:
    """Two adapters bound to different tenants produce separate log bindings."""
    a = _MinimalAdapter(stratix=_RecordingStratix("org-A"))
    b = _MinimalAdapter(stratix=_RecordingStratix("org-B"))

    a.tlogger.info("from-A-adapter")
    b.tlogger.info("from-B-adapter")

    a_recs = [r for r in handler.records if "from-A-adapter" in r.getMessage()]
    b_recs = [r for r in handler.records if "from-B-adapter" in r.getMessage()]
    assert len(a_recs) == 1
    assert len(b_recs) == 1
    assert a_recs[0].org_id == "org-A"
    assert b_recs[0].org_id == "org-B"


def test_tlogger_message_prefix_visible_in_formatted_output(handler: _RecordingHandler) -> None:
    """The ``[org_id=...]`` prefix is present in plain-text formatted output."""
    adapter = _MinimalAdapter(org_id="org-Z")
    adapter.tlogger.error("disconnected")

    relevant = [r for r in handler.records if "disconnected" in r.getMessage()]
    assert len(relevant) == 1
    assert relevant[0].getMessage().startswith("[org_id=org-Z] ")
