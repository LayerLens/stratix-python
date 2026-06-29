"""Collector-seam runtime net (A4 / R5 / LAY-3627).

The schema-lock + secret-scan used to see only traces recorded via the
capture_trace / capture_framework_trace helpers (and the handful of suites that
opted in to record_for_schema_lock). The ~18 suites that build a TraceCollector
and read ``.events`` directly — and any suite that flushes without the helper —
bypassed the nets entirely (R5). The A4 seam installs an observer at the REAL
upload boundary (``TraceCollector.flush``), so EVERY flushing trace feeds the
nets, population-completely and independent of the load-bearing ``_sync_mode``.

This proves the seam is wired and BITES: an inner suite that builds a collector
and flushes a schema-violating trace WITHOUT any capture helper must still be
failed by the autouse net. Remove the ``set_trace_observer`` call from
``_enforce_schema_lock`` and the inner suite reports 0 errors (the trace escapes).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.invariant


class TestCollectorSeamCoversDirectFlush:
    def test_seam_fails_a_direct_flush_schema_violation(self, pytester) -> None:
        """A collector built + flushed directly (no capture_trace, no
        record_for_schema_lock) that uploads an unregistered event type must be
        caught by the autouse net via the A4 collector seam."""
        pytester.makeconftest("from tests.instrument.conftest import _enforce_schema_lock  # noqa: F401")
        pytester.makepyfile(
            test_inner="""
            from layerlens.instrument._collector import TraceCollector
            from layerlens.instrument._capture_config import CaptureConfig

            def test_direct_flush_unregistered_type():
                # No capture helper, no record_for_schema_lock — only flush().
                c = TraceCollector(object(), CaptureConfig(capture_content=True))
                c.emit("totally.unregistered.type", {"x": 1}, span_id="s1")
                c.flush()
            """
        )
        result = pytester.runpytest()
        # body passes; the autouse teardown ERRORs because the seam fed the
        # unregistered event to validate_events.
        result.assert_outcomes(passed=1, errors=1)

    def test_seam_passes_a_clean_direct_flush(self, pytester) -> None:
        pytester.makeconftest("from tests.instrument.conftest import _enforce_schema_lock  # noqa: F401")
        pytester.makepyfile(
            test_inner="""
            from layerlens.instrument._collector import TraceCollector
            from layerlens.instrument._capture_config import CaptureConfig

            def test_direct_flush_clean():
                c = TraceCollector(object(), CaptureConfig(capture_content=True))
                c.emit("agent.input", {"agent_name": "a"}, span_id="s1")
                c.flush()
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)
