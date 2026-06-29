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


class SnapshotIntegrityError(ValueError):
    """A loaded/replayed snapshot violates the on-disk contract.

    Raised when a snapshot is not a JSON object, its ``events`` is not a list,
    or a recorded event carries an unregistered ``event_type`` (fail-closed
    against a forged/garbage recorded trace — see :func:`replay_events`).
    """


def load_snapshot(path: str) -> Dict[str, Any]:
    """Read a snapshot back from disk, validating its shape (fail-closed).

    A snapshot is untrusted input — it may have been hand-edited, produced by an
    older/forged tool, or tampered with on the wire. We reject anything that is
    not a JSON object whose ``events`` (if present) is a list of objects, and we
    reject any event whose ``event_type`` is not a registered layerlens type.
    Replaying an unknown type would inject an arbitrary string straight into a
    fresh collector that then uploads it (fail-OPEN), so we refuse here too.
    """
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SnapshotIntegrityError(f"Snapshot at {path} is not a JSON object")
    events = data.get("events")
    if events is not None:
        if not isinstance(events, list):
            raise SnapshotIntegrityError(f"Snapshot at {path}: 'events' must be a list")
        for event in events:
            _validate_recorded_event(event, where=path)
    return data


def _validate_recorded_event(event: Any, *, where: str) -> str:
    """Validate one recorded event's shape + type; return its event_type.

    Fail-closed gate shared by :func:`load_snapshot` and :func:`replay_events`.
    Rejects a non-dict event, a missing/non-string ``event_type``, and any
    ``event_type`` not in the runtime registry (``known_event_types``). An
    unregistered type on the replay path is the fail-open hole this closes: it
    would otherwise be emitted verbatim into the target collector and uploaded.
    """
    # Lazy import: keeps the replay package importable without the instrument
    # package being initialised, and avoids any import cycle.
    from ..instrument._capture_config import known_event_types

    if not isinstance(event, dict):
        raise SnapshotIntegrityError(
            f"Snapshot at {where}: each event must be a JSON object, got {type(event).__name__}"
        )
    event_type = event.get("event_type")
    if not isinstance(event_type, str) or not event_type:
        raise SnapshotIntegrityError(f"Snapshot at {where}: event has no 'event_type' string")
    if event_type not in known_event_types():
        raise SnapshotIntegrityError(
            f"Snapshot at {where}: refusing to replay unregistered event_type {event_type!r} "
            "(fail-closed — a recorded trace may only carry registered layerlens event types; "
            "register a genuinely new type in src/layerlens/instrument/_capture_config.py)"
        )
    return event_type


def replay_events(snapshot: Dict[str, Any], target_collector: Any) -> int:
    """Re-emit ``snapshot["events"]`` into *target_collector* (fail-closed).

    Useful for re-hydrating a captured run into a fresh collector — for
    instance, when re-running attestation checks or feeding the events
    into a different sink. Returns the number of events re-emitted.

    SECURITY: a recorded ``event_type`` is NOT trusted. Each event is validated
    against the runtime event-type registry BEFORE it is emitted — an
    unregistered type raises :class:`SnapshotIntegrityError` rather than being
    injected verbatim into the collector (the old fail-OPEN behaviour). Every
    emitted event then passes through the collector's normal redact + secret
    scrub chokepoint (``TraceCollector.emit``), so a secret-bearing recorded
    payload is scrubbed on replay exactly as on a live emit.

    Note: ``target_collector`` keeps its own ``trace_id`` and attestation
    chain — this is a fresh trace that happens to contain the same events,
    not a literal reincarnation of the original.
    """
    count = 0
    for event in snapshot.get("events", []):
        event_type = _validate_recorded_event(event, where="<replay>")
        target_collector.emit(
            event_type,
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
