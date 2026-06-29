from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from ._secret_scan import scan_for_secrets
from ._event_schema import validate_events

# ---------------------------------------------------------------------------
# Schema lock (LAY-3583) — enforced OUTSIDE the swallowing upload path.
#
# The production upload path (``_upload._upload_sync`` / ``_worker_loop``) wraps
# ``client.traces.upload`` in a blanket ``except Exception`` and swallows it —
# correct best-effort-telemetry behaviour. But it means a ``validate_events``
# call made *inside* the mocked ``traces.upload`` side-effect is swallowed too,
# so the schema lock never failed a test (LAY-3613). Instead, the two shared
# capture helpers (``capture_trace`` here, ``capture_framework_trace`` in the
# frameworks conftest) record every uploaded event into this per-test buffer,
# and the ``_enforce_schema_lock`` autouse fixture validates it *after* the test
# body — where a violation actually fails the test (as a teardown ERROR).
#
# Scope: the two shared fixtures participate, and so do the suites with their own
# upload-capturing helpers — the per-framework ``test_concurrency_*``
# ``_collect_traces``, ``test_trace_context``'s local ``capture_trace``, and
# ``test_pydantic_ai``'s ad-hoc ``upload.side_effect`` all now call
# ``record_for_schema_lock`` too (validated across the hazardous-five venvs: the
# interleaved/corrupted traces those suites collect are still schema-valid, so
# the lock holds and the xfail(strict) outcomes are unchanged). Correctness
# relies on ``tests/conftest.py`` forcing ``_upload._sync_mode = True`` (autouse),
# so ``record_for_schema_lock`` runs synchronously on the main thread before
# teardown; without that invariant a background upload could append to a later
# test's buffer. A schema violation surfaces as a teardown ERROR (the test body
# is still reported PASSED) — the message names the offending ``[marker/type]``.
# ---------------------------------------------------------------------------
_pending_schema_events: List[Dict[str, Any]] = []


def record_for_schema_lock(events: List[Dict[str, Any]]) -> None:
    """Record uploaded events for post-test schema-lock validation."""
    _pending_schema_events.extend(events)


@pytest.fixture(autouse=True)
def _enforce_schema_lock():
    # A4 collector seam (LAY-3627): install an observer at the REAL upload
    # boundary (TraceCollector.flush) so EVERY flushing trace — not only the
    # capture_trace-helper suites — feeds the schema-lock + secret-scan, even
    # the ~18 suites that build a collector and read .events directly. This is
    # population-complete and independent of _sync_mode. Idempotent with the
    # capture helpers (double-recording an event just validates it twice).
    from layerlens.instrument import _collector as _collector_mod

    def _observe(payload: Dict[str, Any]) -> None:
        record_for_schema_lock(payload.get("events", []))

    _collector_mod.set_trace_observer(_observe)
    _pending_schema_events.clear()
    try:
        yield
    finally:
        _collector_mod.set_trace_observer(None)
    events = list(_pending_schema_events)
    _pending_schema_events.clear()
    if events:
        validate_events(events)
        # Credential-sprawl net: no secret-shaped value may reach an uploaded
        # event (orthogonal to capture_content; runs over every adapter suite).
        scan_for_secrets(events)


@pytest.fixture
def mock_client():
    client = Mock()
    client.traces = Mock()
    client.traces.upload = Mock()
    return client


@pytest.fixture
def capture_trace(mock_client):
    """Captures the uploaded trace payload for inspection.

    Returns a dict that gets populated with:
      - "trace_id": str
      - "events": list of event dicts
      - "capture_config": dict
      - "attestation": dict

    Uploaded events are recorded for the schema lock (validated after the test
    by ``_enforce_schema_lock`` — see the note above).
    """
    uploaded: Dict[str, Any] = {}

    def _capture(path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        # upload_trace wraps in a list
        payload = data[0]
        uploaded["trace_id"] = payload.get("trace_id")
        uploaded["events"] = payload.get("events", [])
        uploaded["capture_config"] = payload.get("capture_config", {})
        uploaded["attestation"] = payload.get("attestation", {})
        record_for_schema_lock(uploaded["events"])

    mock_client.traces.upload.side_effect = _capture
    return uploaded


def find_events(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    """Filter events by event_type."""
    return [e for e in events if e["event_type"] == event_type]


def find_event(events: List[Dict[str, Any]], event_type: str) -> Dict[str, Any]:
    """Find a single event by type. Raises if not found."""
    matches = find_events(events, event_type)
    assert matches, f"No event with type '{event_type}' found in {[e['event_type'] for e in events]}"
    return matches[0]
