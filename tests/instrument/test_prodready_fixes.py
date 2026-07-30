"""Bite-tests for the production-readiness assessment's cheap-and-obvious SDK fixes
(2026-06-27). Each was RED on pre-fix source (the bug bit, proven live) and is GREEN
after the fix. Tagged invariant so the gate enforces them. See the prod-readiness report.

- F-L7-002: a backend reject (empty trace_ids, no exception) must count as FAILURE.
- F-L5-001: a secret in span_name (event-envelope metadata) must be scrubbed.
- F-L1-003: emit() must not crash the host app on a non-JSON-native payload value.
"""

from __future__ import annotations

import decimal

import pytest

from layerlens.models import CreateTracesResponse
from layerlens.instrument._upload import UploadChannel
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig

pytestmark = pytest.mark.invariant


class _RejectClient:
    """Backend ACCEPTS the request (no exception) but REJECTS the trace (empty
    trace_ids). Built from the SDK's own response type for wire fidelity."""

    class _Traces:
        def upload(self, path):  # noqa: ANN001
            return CreateTracesResponse(trace_ids=[])

    traces = _Traces()


def test_empty_trace_ids_reject_counts_as_failure():
    """F-L7-002: a non-raising empty-trace_ids reject must trip the failure path so
    the circuit breaker is not blind to silent data loss."""
    ch = UploadChannel()
    for _ in range(3):
        ch._upload_sync(_RejectClient(), {"trace_id": "x", "events": [{"event_type": "agent.input"}]})
    assert ch._error_count >= 3, (
        f"reject (empty trace_ids) counted as success: error_count={ch._error_count} (F-L7-002)"
    )


def test_span_name_secret_is_scrubbed():
    """F-L5-001: a secret in span_name (envelope metadata) must be scrubbed, not
    shipped verbatim — it bypasses payload redaction+scrub otherwise."""
    secret = "sk-ant-api03-" + ("Z" * 40)
    col = TraceCollector(client=object(), config=CaptureConfig())
    col.emit("model.invoke", {"model": "gpt-4o-mini"}, span_id="s1", span_name=secret)
    span_name = col.events[-1].get("span_name") or ""
    assert secret not in span_name, f"secret leaked into event span_name: {span_name!r} (F-L5-001)"
    assert "[REDACTED-SECRET]" in span_name


@pytest.mark.parametrize("value", [decimal.Decimal("499.99"), b"\x00\x01rawbytes"])
def test_emit_does_not_crash_on_non_json_native(value):
    """F-L1-003: emit() must coerce a non-JSON-native payload value (Decimal/bytes)
    instead of raising out of the attestation hash path and crashing the host app."""
    col = TraceCollector(client=object(), config=CaptureConfig.full())
    try:
        col.emit("agent.input", {"v": value}, span_id="s1", span_name="t")
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"emit() crashed on a {type(value).__name__} payload value: {e!r} (F-L1-003)")
    assert len(col.events) == 1
