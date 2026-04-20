"""Shared utilities for adapter samples.

Each sample uses :func:`capture_events` to run a block under a local
``TraceCollector`` and print the events that fire, so you can eyeball what
instrumentation is capturing without hitting the live LayerLens API.
"""

from __future__ import annotations

import json
from typing import Any, Generator
from contextlib import contextmanager

from layerlens.instrument._context import _pop_span, _push_span, _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig


@contextmanager
def capture_events(name: str = "sample") -> Generator[TraceCollector, None, None]:
    """Run the block under a local TraceCollector and pretty-print events on exit."""

    class _StubClient:
        """TraceCollector requires a client; for samples we don't need a real one."""

        def __init__(self) -> None:
            self._base_url = "https://localhost/sample"

    collector = TraceCollector(_StubClient(), CaptureConfig.standard())
    root = "sample" + name[:8]
    col_token = _current_collector.set(collector)
    span_snapshot = _push_span(root, name)
    try:
        yield collector
    finally:
        _pop_span(span_snapshot)
        _current_collector.reset(col_token)
        _print_events(collector)


def _print_events(collector: TraceCollector) -> None:
    events = getattr(collector, "_events", [])
    print(f"\n--- captured {len(events)} events ---")
    for ev in events:
        print(json.dumps({"type": ev.get("event_type"), "payload": ev.get("payload")}, default=str)[:500])


def pretty(value: Any) -> str:
    return json.dumps(value, default=str, indent=2)
