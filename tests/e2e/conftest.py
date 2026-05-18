"""Shared fixtures for end-to-end tests.

These tests run real framework code (langgraph graphs, crewai crews, etc.)
and verify that the layerlens instrumentation produces the expected
events for the full pipeline — instrument → invoke → flush → upload.

LLM calls are mocked at the boundary; everything else is the real library.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest


@pytest.fixture
def client_and_uploads():
    """A mock Stratix client that captures every trace upload as a parsed dict.

    Returns ``(client, uploads_list)``. ``uploads_list`` is mutated by the
    ``traces.upload(path)`` side-effect to contain the full trace payload
    from disk on every flush.
    """
    client = Mock()
    client.traces = Mock()
    uploads: List[Dict[str, Any]] = []

    def _capture(path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        # upload_trace wraps the trace payload in a list.
        uploads.append(data[0] if isinstance(data, list) else data)

    client.traces.upload.side_effect = _capture
    return client, uploads


def events_of(uploads: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    """Pull all events of a given type across every uploaded trace."""
    out = []
    for upload in uploads:
        for ev in upload.get("events", []) or []:
            if ev.get("event_type") == event_type:
                out.append(ev)
    return out


def first_event(uploads: List[Dict[str, Any]], event_type: str) -> Dict[str, Any]:
    """Return the first matching event or fail the test with a useful message."""
    matches = events_of(uploads, event_type)
    if not matches:
        all_types = sorted({e.get("event_type", "?") for u in uploads for e in u.get("events", [])})
        raise AssertionError(
            f"No event of type {event_type!r} found. Captured types across {len(uploads)} upload(s): {all_types}"
        )
    return matches[0]
