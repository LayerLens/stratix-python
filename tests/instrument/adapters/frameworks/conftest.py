from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import Mock


# Re-export from root conftest so framework tests can do `from .conftest import ...`
from ...conftest import find_event, find_events  # noqa: F401


def capture_framework_trace(mock_client: Mock) -> Dict[str, Any]:
    """Capture the uploaded trace payload from a framework adapter.

    Accumulates events across multiple flushes (some adapters use
    multiple collectors).
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

    mock_client.traces.upload.side_effect = _capture
    return uploaded
