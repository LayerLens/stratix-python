from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest


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
