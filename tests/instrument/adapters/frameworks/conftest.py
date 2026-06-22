from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import Mock

# Re-export from root conftest so framework tests can do `from .conftest import ...`
from ...conftest import find_event, find_events, record_for_schema_lock  # noqa: F401


def capture_framework_trace(mock_client: Mock) -> Dict[str, Any]:
    """Capture the uploaded trace payload from a framework adapter.

    Accumulates events across multiple flushes (some adapters use
    multiple collectors). Uploaded events are recorded for the schema lock,
    which is enforced after the test by the ``_enforce_schema_lock`` autouse
    fixture (validating inside the mocked upload is swallowed — see the note in
    the root ``conftest.py``; LAY-3613).
    """
    uploaded: Dict[str, Any] = {"events": []}

    def _capture(path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        payload = data[0]
        uploaded["trace_id"] = payload.get("trace_id")
        uploaded["events"].extend(payload.get("events", []))
        uploaded["capture_config"] = payload.get("capture_config", {})
        uploaded["attestation"] = payload.get("attestation", {})
        record_for_schema_lock(payload.get("events", []))

    mock_client.traces.upload.side_effect = _capture
    return uploaded
