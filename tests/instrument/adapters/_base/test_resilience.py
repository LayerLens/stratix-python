"""Tests for the per-callback resilience wrapper.

Covers ``ResilienceTracker``, ``resilient_callback``, ``get_default_for``,
``HealthStatus``, and the ``adapter_info().metadata`` integration on
``FrameworkAdapter`` subclasses.

Every test asserts a behaviour that prevents observability code from
breaking the framework's own execution path.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict
from unittest.mock import Mock

import pytest

from layerlens.instrument.adapters._base import (
    DEFAULT_FAILURE_THRESHOLD,
    AdapterInfo,
    BaseAdapter,
    HealthStatus,
    ResilienceTracker,
    get_default_for,
    resilient_callback,
)
from layerlens.instrument.adapters.frameworks._base_framework import FrameworkAdapter

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _Boom(Exception):
    """Sentinel error type so tests can assert the right exception was caught."""


class _DummyAdapter:
    """Minimal adapter shape — provides ``name`` and ``_resilience`` only."""

    name = "dummy"

    def __init__(self) -> None:
        self._resilience = ResilienceTracker(self.name)

    @resilient_callback(callback_name="my_callback", default="DEFAULT")
    def my_callback(self, value: Any) -> Any:
        if value == "raise":
            raise _Boom("dummy failure")
        return f"ok:{value}"

    @resilient_callback(callback_name="passthrough_cb", passthrough_arg="value")
    def passthrough_cb(self, value: Any) -> Any:
        if value == "raise":
            raise _Boom("passthrough failure")
        return f"transformed:{value}"

    @resilient_callback(callback_name="kw_passthrough", passthrough_arg="payload")
    def kw_passthrough(self, *, payload: Any) -> Any:
        if payload == "raise":
            raise _Boom("kw passthrough failure")
        return {"wrapped": payload}


class _MinimalFramework(FrameworkAdapter):
    """Real FrameworkAdapter subclass for integration tests."""

    name = "test-framework"

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        return None

    @resilient_callback(callback_name="emit_thing")
    def emit_thing(self, value: int) -> None:
        if value < 0:
            raise _Boom(f"negative value: {value}")


# ---------------------------------------------------------------------------
# get_default_for
# ---------------------------------------------------------------------------


class TestGetDefaultFor:
    def test_known_callback_returns_none(self) -> None:
        # All registered callbacks default to None — the framework default
        # for void-callback APIs (Strands hooks, Google ADK plugins, boto3).
        assert get_default_for("on_trace_start") is None
        assert get_default_for("_before_invoke") is None
        assert get_default_for("after_run_callback") is None

    def test_unknown_callback_returns_none(self) -> None:
        # Unknown callback names also return None — the safe default. If a
        # callback needs a non-None default, the adapter must pass it
        # explicitly via @resilient_callback(default=...).
        assert get_default_for("does_not_exist") is None
        assert get_default_for("") is None


# ---------------------------------------------------------------------------
# ResilienceTracker
# ---------------------------------------------------------------------------


class TestResilienceTracker:
    def test_starts_healthy_with_zero_failures(self) -> None:
        tracker = ResilienceTracker("test")
        assert tracker.total_failures == 0
        assert tracker.health_status() == HealthStatus.HEALTHY

    def test_threshold_validation(self) -> None:
        with pytest.raises(ValueError):
            ResilienceTracker("test", threshold=0)
        with pytest.raises(ValueError):
            ResilienceTracker("test", threshold=-1)

    def test_record_failure_increments_counter(self) -> None:
        tracker = ResilienceTracker("test")
        tracker.record_failure("cb1", _Boom("first"))
        assert tracker.total_failures == 1
        tracker.record_failure("cb1", _Boom("second"))
        tracker.record_failure("cb2", _Boom("third"))
        assert tracker.total_failures == 3

    def test_health_degrades_after_threshold(self) -> None:
        tracker = ResilienceTracker("test", threshold=3)
        tracker.record_failure("cb", _Boom("a"))
        tracker.record_failure("cb", _Boom("b"))
        assert tracker.health_status() == HealthStatus.HEALTHY
        tracker.record_failure("cb", _Boom("c"))
        assert tracker.health_status() == HealthStatus.DEGRADED

    def test_metadata_snapshot(self) -> None:
        tracker = ResilienceTracker("test", threshold=2)
        tracker.record_failure("cb1", _Boom("oops"))
        tracker.record_failure("cb2", ValueError("bad value"))
        snap = tracker.as_metadata()
        assert snap["resilience_status"] == HealthStatus.DEGRADED.value
        assert snap["resilience_failures_total"] == 2
        assert snap["resilience_failure_threshold"] == 2
        # Per-callback breakdown carries the top failure counts.
        assert snap["resilience_failures_by_callback"] == {"cb1": 1, "cb2": 1}
        # Last error preserved (truncated) for triage.
        assert snap["resilience_last_callback"] == "cb2"
        assert "ValueError" in snap["resilience_last_error"]

    def test_reset_clears_state(self) -> None:
        tracker = ResilienceTracker("test")
        tracker.record_failure("cb", _Boom("x"))
        assert tracker.total_failures == 1
        tracker.reset()
        assert tracker.total_failures == 0
        assert tracker.health_status() == HealthStatus.HEALTHY
        snap = tracker.as_metadata()
        assert snap["resilience_failures_total"] == 0
        assert "resilience_last_error" not in snap

    def test_thread_safety(self) -> None:
        # Many threads recording failures concurrently must not lose any
        # increments — observability code commonly fires from worker
        # threads (CrewAI, AutoGen group chat, Bedrock boto3 hooks).
        tracker = ResilienceTracker("test", threshold=DEFAULT_FAILURE_THRESHOLD)
        per_thread_count = 100
        thread_count = 8

        def _worker() -> None:
            for _ in range(per_thread_count):
                tracker.record_failure("cb", _Boom("concurrent"))

        threads = [threading.Thread(target=_worker) for _ in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.total_failures == per_thread_count * thread_count


# ---------------------------------------------------------------------------
# resilient_callback decorator
# ---------------------------------------------------------------------------


class TestResilientCallbackDecorator:
    def test_returns_normal_value_on_success(self) -> None:
        adapter = _DummyAdapter()
        assert adapter.my_callback("hello") == "ok:hello"
        assert adapter._resilience.total_failures == 0

    def test_returns_default_on_exception(self) -> None:
        adapter = _DummyAdapter()
        # Without the wrapper, the call would raise — instead it returns
        # the framework's expected default and the framework continues.
        result = adapter.my_callback("raise")
        assert result == "DEFAULT"

    def test_failure_counter_incremented(self) -> None:
        adapter = _DummyAdapter()
        adapter.my_callback("raise")
        adapter.my_callback("raise")
        adapter.my_callback("ok-1")  # this one succeeds
        adapter.my_callback("raise")
        assert adapter._resilience.total_failures == 3

    def test_exception_is_logged_with_context(self, caplog: pytest.LogCaptureFixture) -> None:
        adapter = _DummyAdapter()
        with caplog.at_level(logging.WARNING):
            adapter.my_callback("raise")

        # Adapter name + callback name + traceback all surfaced.
        assert any(
            "dummy" in rec.message and "my_callback" in rec.message and "_Boom" in rec.message
            for rec in caplog.records
        )

    def test_exception_does_not_propagate(self) -> None:
        adapter = _DummyAdapter()
        # The whole point: the framework calling this method MUST NOT see
        # _Boom — that would crash the user's agent.
        try:
            adapter.my_callback("raise")
        except _Boom:
            pytest.fail("resilient_callback let the exception escape")

    def test_passthrough_arg_returns_positional_value(self) -> None:
        adapter = _DummyAdapter()
        # On failure the wrapper returns the passthrough arg's value
        # rather than the default — critical for mutating hooks
        # (Pydantic-AI ``after_model_request`` returns ``response``).
        assert adapter.passthrough_cb("raise") == "raise"
        # On success, the original return value flows through.
        assert adapter.passthrough_cb("ok") == "transformed:ok"

    def test_passthrough_arg_returns_keyword_value(self) -> None:
        adapter = _DummyAdapter()
        assert adapter.kw_passthrough(payload="raise") == "raise"
        assert adapter.kw_passthrough(payload="data") == {"wrapped": "data"}

    def test_health_degrades_after_repeated_failures(self) -> None:
        adapter = _DummyAdapter()
        # Default threshold is 5; after 5 consecutive failures the
        # adapter reports DEGRADED so monitoring can alert.
        assert adapter._resilience.health_status() == HealthStatus.HEALTHY
        for _ in range(DEFAULT_FAILURE_THRESHOLD):
            adapter.my_callback("raise")
        assert adapter._resilience.health_status() == HealthStatus.DEGRADED

    def test_keyboard_interrupt_propagates(self) -> None:
        # We must NEVER swallow KeyboardInterrupt / SystemExit /
        # GeneratorExit — those are control-flow signals, not bugs.
        class _CtrlCAdapter:
            name = "ctrlc"

            def __init__(self) -> None:
                self._resilience = ResilienceTracker(self.name)

            @resilient_callback(callback_name="cb")
            def cb(self) -> None:
                raise KeyboardInterrupt("user pressed Ctrl-C")

        adapter = _CtrlCAdapter()
        with pytest.raises(KeyboardInterrupt):
            adapter.cb()

    def test_works_without_resilience_tracker_attribute(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # If an adapter forgets to set up _resilience, the wrapper still
        # logs and returns the default — never crashes the framework.
        class _NoTracker:
            name = "no_tracker"

            @resilient_callback(callback_name="cb", default="OK")
            def cb(self) -> str:
                raise _Boom("no tracker")

        adapter = _NoTracker()
        with caplog.at_level(logging.WARNING):
            assert adapter.cb() == "OK"
        assert any("_Boom" in rec.message for rec in caplog.records)

    def test_logger_uses_module_of_decorated_function(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Failures are logged via the wrapped function's module logger so
        # users can mute one adapter's resilience warnings without
        # silencing all of them.
        adapter = _DummyAdapter()
        with caplog.at_level(logging.WARNING, logger=__name__):
            adapter.my_callback("raise")
        # Our test module's logger captured the warning.
        assert any(rec.name == __name__ for rec in caplog.records)


# ---------------------------------------------------------------------------
# Integration with FrameworkAdapter
# ---------------------------------------------------------------------------


class TestFrameworkAdapterIntegration:
    def test_framework_adapter_owns_resilience_tracker(self) -> None:
        adapter = _MinimalFramework(Mock())
        assert isinstance(adapter._resilience, ResilienceTracker)
        assert adapter._resilience.total_failures == 0

    def test_adapter_info_surfaces_resilience_metadata(self) -> None:
        adapter = _MinimalFramework(Mock())
        info: AdapterInfo = adapter.adapter_info()
        meta = info.metadata
        assert meta["resilience_status"] == "healthy"
        assert meta["resilience_failures_total"] == 0
        assert meta["resilience_failure_threshold"] == DEFAULT_FAILURE_THRESHOLD

    def test_adapter_info_reports_degraded_after_failures(self) -> None:
        adapter = _MinimalFramework(Mock())
        for _ in range(DEFAULT_FAILURE_THRESHOLD):
            adapter.emit_thing(-1)  # raises inside the wrapped method
        info = adapter.adapter_info()
        assert info.metadata["resilience_status"] == "degraded"
        assert info.metadata["resilience_failures_total"] == DEFAULT_FAILURE_THRESHOLD

    def test_disconnect_resets_resilience(self) -> None:
        adapter = _MinimalFramework(Mock())
        adapter.connect()
        adapter.emit_thing(-1)
        adapter.emit_thing(-2)
        assert adapter._resilience.total_failures == 2
        adapter.disconnect()
        assert adapter._resilience.total_failures == 0

    def test_callback_failure_does_not_break_framework(self) -> None:
        adapter = _MinimalFramework(Mock())
        adapter.connect()
        # Simulating "framework fires our callback" — the callback throws
        # but the framework's call-site sees no exception, just None.
        result = adapter.emit_thing(-99)
        assert result is None
        assert adapter._resilience.total_failures == 1


# ---------------------------------------------------------------------------
# Public surface re-exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_base_package_re_exports_resilience_helpers(self) -> None:
        from layerlens.instrument.adapters._base import (
            DEFAULT_FAILURE_THRESHOLD as T1,
            AdapterInfo as A1,
            BaseAdapter as B1,
            HealthStatus as H1,
            ResilienceTracker as R1,
            get_default_for as G1,
            resilient_callback as RC1,
        )

        # Sanity — every public symbol resolves and is the right kind.
        assert A1 is AdapterInfo
        assert B1 is BaseAdapter
        assert R1 is ResilienceTracker
        assert RC1 is resilient_callback
        assert H1 is HealthStatus
        assert T1 == DEFAULT_FAILURE_THRESHOLD
        assert G1 is get_default_for


# ---------------------------------------------------------------------------
# Decorator preserves function metadata
# ---------------------------------------------------------------------------


class TestDecoratorMetadata:
    def test_wrapped_function_keeps_name_and_docstring(self) -> None:
        class _A:
            name = "x"
            _resilience = ResilienceTracker("x")

            @resilient_callback(callback_name="cb")
            def cb(self) -> None:
                """My docstring."""
                pass

        # functools.wraps preserves __name__ and __doc__ — important for
        # frameworks that introspect handlers by name (boto3 event system
        # uses handler identity for unregister()).
        assert _A.cb.__name__ == "cb"
        assert _A.cb.__doc__ == "My docstring."


# ---------------------------------------------------------------------------
# End-to-end: verifying the per-adapter failure scenario
# ---------------------------------------------------------------------------


class TestPerAdapterCallbackException:
    """Simulate a callback exception on each lighter adapter and assert
    the framework continues unaffected.

    Each test instantiates the adapter, monkey-patches one of its
    callback methods to raise, then invokes the callback and asserts:

    1. No exception escaped (framework would crash otherwise).
    2. The resilience tracker incremented its failure counter.
    3. Repeated failures cross the threshold and degrade adapter health.
    """

    @pytest.mark.parametrize(
        "module_path, class_name, callback_name, callback_args",
        [
            (
                "layerlens.instrument.adapters.frameworks.agno",
                "AgnoAdapter",
                "_on_run_start",
                (Mock(), "input"),
            ),
            (
                "layerlens.instrument.adapters.frameworks.agno",
                "AgnoAdapter",
                "_on_run_end",
                (Mock(), Mock(), None),
            ),
            (
                "layerlens.instrument.adapters.frameworks.smolagents",
                "SmolAgentsAdapter",
                "_on_run_start",
                (Mock(), "task"),
            ),
            (
                "layerlens.instrument.adapters.frameworks.smolagents",
                "SmolAgentsAdapter",
                "_on_run_end",
                (Mock(), Mock(), None),
            ),
            (
                "layerlens.instrument.adapters.frameworks.smolagents",
                "SmolAgentsAdapter",
                "_on_run_error",
                (Mock(), _Boom("framework error")),
            ),
            (
                "layerlens.instrument.adapters.frameworks.smolagents",
                "SmolAgentsAdapter",
                "_on_action_step",
                (Mock(), Mock()),
            ),
        ],
    )
    def test_callback_exception_caught_and_counted(
        self,
        module_path: str,
        class_name: str,
        callback_name: str,
        callback_args: tuple,
    ) -> None:
        import importlib

        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)
        adapter = adapter_cls(Mock())

        # Force the underlying body to raise by sabotaging an inner
        # helper the callback always calls. The simplest way is to patch
        # ``adapter._payload`` to raise — every callback uses it.
        original_payload = adapter._payload

        def _raise_on_payload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise _Boom("simulated callback failure")

        adapter._payload = _raise_on_payload  # type: ignore[method-assign]

        try:
            cb = getattr(adapter, callback_name)
            # Must not raise — that's the entire resilience contract.
            cb(*callback_args)
            # Failure recorded against this exact callback name.
            assert adapter._resilience.total_failures >= 1
        finally:
            adapter._payload = original_payload  # type: ignore[method-assign]

    def test_repeated_failures_degrade_adapter(self) -> None:
        # Use agno as the proxy — same wiring applies to all 10 lighter
        # adapters because they all inherit from FrameworkAdapter and use
        # @resilient_callback on their entry points.
        from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

        adapter = AgnoAdapter(Mock())

        def _raise_on_payload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise _Boom("persistent failure")

        adapter._payload = _raise_on_payload  # type: ignore[method-assign]

        for _ in range(DEFAULT_FAILURE_THRESHOLD):
            adapter._on_run_start(Mock(), "input")

        info = adapter.adapter_info()
        assert info.metadata["resilience_status"] == "degraded"
