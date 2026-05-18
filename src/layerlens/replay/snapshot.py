"""Persist and load replay-ready trace snapshots.

A snapshot is the dict produced by :meth:`TraceCollector.to_replay_dict`
— ``trace_id``, ``events``, ``capture_config``, ``attestation``. Snapshots
are plain JSON, so they round-trip cleanly to disk, blob storage, or
any transport that handles UTF-8.

Typical flow::

    from layerlens import Stratix
    from layerlens.instrument import trace_context
    from layerlens.replay.snapshot import dump_collector, load_snapshot, replay_events
    from layerlens.replay import ReplayController

    client = Stratix()

    # 1. Capture
    with trace_context(client) as collector:
        my_pipeline()
        dump_collector(collector, "/tmp/run-1.json")

    # 2. Later: load and replay
    snapshot = load_snapshot("/tmp/run-1.json")
    controller = ReplayController(replay_fn=my_pipeline)
    result = controller.replay(snapshot["trace_id"], ...)

    # Or: re-emit the captured events into a new collector
    new_collector = TraceCollector(client, capture_config)
    replay_events(snapshot, new_collector)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from pathlib import Path


def dump(payload: Dict[str, Any], path: str) -> str:
    """Write a snapshot payload to *path* as JSON. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    return str(p)


def dump_collector(collector: Any, path: str) -> str:
    """Convenience: serialize a :class:`TraceCollector` directly to *path*."""
    return dump(collector.to_replay_dict(), path)


def load_snapshot(path: str) -> Dict[str, Any]:
    """Read a snapshot back from disk."""
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Snapshot at {path} is not a JSON object")
    return data


def replay_events(snapshot: Dict[str, Any], target_collector: Any) -> int:
    """Re-emit ``snapshot["events"]`` into *target_collector*.

    Useful for re-hydrating a captured run into a fresh collector — for
    instance, when re-running attestation checks or feeding the events
    into a different sink. Returns the number of events re-emitted.

    Note: ``target_collector`` keeps its own ``trace_id`` and attestation
    chain — this is a fresh trace that happens to contain the same events,
    not a literal reincarnation of the original.
    """
    count = 0
    for event in snapshot.get("events", []):
        target_collector.emit(
            event["event_type"],
            event.get("payload") or {},
            span_id=event.get("span_id") or "",
            parent_span_id=event.get("parent_span_id"),
            span_name=event.get("span_name"),
        )
        count += 1
    return count


# ----------------------------------------------------------------------
# Adapter helpers — per-adapter "serialize for replay" pattern (ateam parity)
# ----------------------------------------------------------------------


def serialize_adapter(adapter: Any, collector: Optional[Any] = None) -> Dict[str, Any]:
    """Bundle adapter metadata + (optional) current trace into one dict.

    Mirrors ateam's per-adapter ``serialize_for_replay()`` pattern. The
    returned dict has ``adapter`` (the :class:`AdapterInfo`-as-dict) and
    optionally ``trace`` (the collector's :meth:`to_replay_dict` output).
    """
    info = adapter.adapter_info()
    out: Dict[str, Any] = {
        "adapter": {
            "name": info.name,
            "adapter_type": info.adapter_type,
            "version": getattr(info, "version", "0.1.0"),
            "metadata": dict(getattr(info, "metadata", {}) or {}),
        }
    }
    if collector is not None:
        out["trace"] = collector.to_replay_dict()
    return out
