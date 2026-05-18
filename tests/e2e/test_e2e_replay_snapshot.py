"""End-to-end: capture a real trace, persist it, reload, and replay.

Exercises:
- @trace decorator emits agent.input / agent.output
- TraceCollector accumulates events with attestation
- dump_collector serialises to disk (no seal)
- load_snapshot reads it back
- replay_events re-emits into a fresh collector
"""

from __future__ import annotations

import json
from pathlib import Path

from layerlens.instrument import CaptureConfig, TraceCollector, span, trace, trace_context
from layerlens.replay.snapshot import (
    load_snapshot,
    replay_events,
    dump_collector,
    serialize_adapter,
)


def test_capture_dump_load_replay_roundtrip(client_and_uploads, tmp_path: Path):
    client, uploads = client_and_uploads

    snapshot_path = tmp_path / "trace.json"

    @trace(client, name="rag_pipeline")
    def my_pipeline(question: str) -> str:
        with span("retrieve"):
            pass
        with span("rerank"):
            pass
        return f"answer to: {question}"

    # 1. Run, capture, and dump mid-trace
    with trace_context(client) as collector:
        # Persist snapshot before flush — should NOT seal the chain.
        dump_collector(collector, str(snapshot_path))
        result = my_pipeline("what is up?")

    assert result == "answer to: what is up?"
    assert snapshot_path.exists()
    # The decorator flushed and uploaded a trace.
    assert len(uploads) >= 1

    # 2. Reload snapshot from disk
    snap = load_snapshot(str(snapshot_path))
    assert "trace_id" in snap
    assert isinstance(snap["events"], list)
    assert snap["capture_config"]["l1_agent_io"] is True

    # 3. Replay events into a fresh collector
    fresh = TraceCollector(client, CaptureConfig.standard())
    n = replay_events(snap, fresh)
    # Replay count should equal snapshot's event count
    assert n == len(snap["events"])
    # Fresh collector has its own trace_id, but the events match
    assert fresh.trace_id != snap["trace_id"]
    replayed_types = [e["event_type"] for e in fresh.events]
    snap_types = [e["event_type"] for e in snap["events"]]
    assert replayed_types == snap_types


def test_dump_then_emit_more_then_flush_preserves_history(client_and_uploads, tmp_path: Path):
    """Snapshotting mid-trace doesn't lock further emits, and the eventual
    flush still uploads the full final set."""
    client, uploads = client_and_uploads

    with trace_context(client) as collector:
        collector.emit("agent.input", {"name": "first"}, span_id="s1")
        dump_collector(collector, str(tmp_path / "snap-1.json"))
        # Keep going after the snapshot
        collector.emit("agent.input", {"name": "second"}, span_id="s2")
        collector.emit("agent.output", {"name": "second"}, span_id="s2")

    snap_1 = load_snapshot(str(tmp_path / "snap-1.json"))
    final_upload = uploads[-1]

    # Snapshot captured only the events that existed at dump time
    assert len(snap_1["events"]) == 1
    # Final upload has every event including those after the snapshot
    assert len(final_upload["events"]) == 3


def test_serialize_adapter_bundles_info_and_trace(client_and_uploads, tmp_path: Path):
    """serialize_adapter mirrors the per-adapter ateam pattern: it produces
    one dict containing adapter metadata + (optionally) the current trace."""
    from layerlens.instrument.adapters.frameworks._base_framework import FrameworkAdapter

    class _FakeAdapter(FrameworkAdapter):
        name = "fake_e2e"

        def _on_connect(self, target=None, **kwargs):
            pass

    client, _ = client_and_uploads
    adapter = _FakeAdapter(client)

    with trace_context(client) as collector:
        collector.emit("agent.input", {"name": "n"}, span_id="x")
        bundle = serialize_adapter(adapter, collector=collector)

    # Round-trip through JSON to prove the whole thing is serialisable.
    text = json.dumps(bundle, default=str)
    reloaded = json.loads(text)

    assert reloaded["adapter"]["name"] == "fake_e2e"
    assert reloaded["adapter"]["adapter_type"] == "framework"
    assert reloaded["trace"]["trace_id"] == collector.trace_id
    assert any(e["event_type"] == "agent.input" for e in reloaded["trace"]["events"])
