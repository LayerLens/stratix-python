"""Unit tests for `layerlens._telemetry` — the opt-in client telemetry.

Contract under test:

1. **Off by default.** Without ``LAYERLENS_TELEMETRY=on`` set, every
   helper is a no-op and never imports the OTel SDK.
2. **Failure-isolated.** When OTel is missing or any init step raises,
   subsequent calls quietly do nothing — telemetry MUST NOT break
   customer code.
3. **Allowlist on attributes.** Only `command`, `resource`, `outcome`,
   `status_code` may be passed; other keys are silently dropped.
4. **`shutdown()` is safe to call when telemetry was never enabled.**
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _reset_telemetry_module(monkeypatch):
    """Re-import the module so module-level singletons are fresh per test."""
    import layerlens._telemetry as t

    # Scrub global state.
    t._initialized = False
    t._meter = None
    t._counter_events = None
    t._hist_request_duration = None
    monkeypatch.delenv("LAYERLENS_TELEMETRY", raising=False)
    yield t
    t._initialized = False
    t._meter = None
    t._counter_events = None
    t._hist_request_duration = None


def test_disabled_by_default_is_silent_noop(_reset_telemetry_module):
    t = _reset_telemetry_module
    # Every helper must succeed and do nothing.
    t.event("sdk_python", "init")
    t.event("cli", "cmd_run", attributes={"command": "trace ls"})
    with t.timed("sdk_python", "trace_emit"):
        pass
    t.shutdown()

    assert t._meter is None
    assert t._counter_events is None
    assert t._initialized is True  # _try_init was called once and short-circuited


def test_event_with_telemetry_off_doesnt_import_otel(monkeypatch, _reset_telemetry_module):
    t = _reset_telemetry_module

    # Even if OTel is technically importable in the test env, the OFF state
    # must short-circuit BEFORE the import block.
    import sys
    sys.modules.pop("opentelemetry", None)
    t.event("sdk_python", "init")
    assert "opentelemetry" not in sys.modules


def test_attributes_allowlist(monkeypatch, _reset_telemetry_module):
    """Disallowed attribute keys are silently dropped, not raised."""
    t = _reset_telemetry_module
    monkeypatch.setenv("LAYERLENS_TELEMETRY", "on")

    # Stub OTel SDK so we can observe what reaches add().
    seen_attrs: dict = {}

    class _StubCounter:
        def add(self, value, attributes=None):
            seen_attrs.update(attributes or {})

    t._initialized = True
    t._meter = object()
    t._counter_events = _StubCounter()
    t._hist_request_duration = None

    t.event(
        "cli",
        "cmd_run",
        attributes={
            "command": "trace ls",
            "email": "user@example.com",  # MUST be dropped
            "ip": "10.0.0.1",  # MUST be dropped
            "outcome": "success",
        },
    )

    assert seen_attrs.get("surface") == "cli"
    assert seen_attrs.get("event") == "cmd_run"
    assert seen_attrs.get("command") == "trace ls"
    assert seen_attrs.get("outcome") == "success"
    assert "email" not in seen_attrs
    assert "ip" not in seen_attrs


def test_event_swallows_counter_errors(monkeypatch, _reset_telemetry_module):
    """If the underlying counter raises, telemetry MUST NOT propagate."""
    t = _reset_telemetry_module
    monkeypatch.setenv("LAYERLENS_TELEMETRY", "on")

    class _Boom:
        def add(self, *_a, **_kw):
            raise RuntimeError("backend down")

    t._initialized = True
    t._meter = object()
    t._counter_events = _Boom()
    t._hist_request_duration = None

    # No exception — silent failure is the contract.
    t.event("sdk_python", "init")


def test_timed_records_duration(monkeypatch, _reset_telemetry_module):
    t = _reset_telemetry_module
    monkeypatch.setenv("LAYERLENS_TELEMETRY", "on")

    class _Hist:
        seen: list = []

        def record(self, value, attributes=None):
            self.seen.append((value, attributes))

    class _Stub:
        seen: list = []

        def add(self, value, attributes=None):
            self.seen.append((value, attributes))

    hist = _Hist()
    counter = _Stub()
    t._initialized = True
    t._meter = object()
    t._counter_events = counter
    t._hist_request_duration = hist

    with t.timed("sdk_python", "trace_emit"):
        pass

    assert len(counter.seen) == 1
    assert len(hist.seen) == 1
    duration, attrs = hist.seen[0]
    assert duration >= 0
    assert attrs == {"surface": "sdk_python", "event": "trace_emit"}


def test_shutdown_is_safe_when_never_enabled(_reset_telemetry_module):
    """shutdown() must be safe even if telemetry never initialized."""
    t = _reset_telemetry_module
    t.shutdown()  # Must not raise.


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("on", True),
        ("ON", True),
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("off", False),
        ("0", False),
        ("", False),
        ("nope", False),
    ],
)
def test_enabled_recognizes_truthy_values(monkeypatch, _reset_telemetry_module, raw, expected):
    t = _reset_telemetry_module
    monkeypatch.setenv("LAYERLENS_TELEMETRY", raw)
    assert t._enabled() is expected
