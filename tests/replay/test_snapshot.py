"""Tests for the snapshot module (persist + reload replay-ready traces)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from layerlens.instrument import CaptureConfig, TraceCollector
from layerlens.replay.snapshot import (
    dump,
    load_snapshot,
    replay_events,
    dump_collector,
    serialize_adapter,
)


def _make_collector(client):
    return TraceCollector(client, CaptureConfig.standard())


class TestDump:
    def test_dump_creates_file(self, tmp_path: Path):
        path = tmp_path / "snap.json"
        payload = {"trace_id": "abc", "events": [{"event_type": "agent.input", "payload": {}}]}
        result = dump(payload, str(path))
        assert result == str(path)
        assert path.exists()

    def test_dump_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "snap.json"
        dump({"x": 1}, str(nested))
        assert nested.exists()

    def test_dump_emits_valid_utf8_json(self, tmp_path: Path):
        path = tmp_path / "snap.json"
        dump({"name": "café"}, str(path))
        # round-trip
        with path.open(encoding="utf-8") as fh:
            assert json.load(fh)["name"] == "café"


class TestDumpCollector:
    def test_dumps_collector_to_replay_dict(self, tmp_path: Path):
        client = Mock()
        collector = _make_collector(client)
        collector.emit("agent.input", {"name": "test"}, span_id="s1", parent_span_id=None)

        path = tmp_path / "trace.json"
        dump_collector(collector, str(path))

        snap = load_snapshot(str(path))
        assert snap["trace_id"] == collector.trace_id
        assert len(snap["events"]) == 1
        assert snap["events"][0]["event_type"] == "agent.input"
        assert "capture_config" in snap
        assert "attestation" in snap

    def test_dump_does_not_seal_collector(self, tmp_path: Path):
        """Calling dump_collector should not stop further emits."""
        client = Mock()
        collector = _make_collector(client)
        collector.emit("agent.input", {}, span_id="s1")

        dump_collector(collector, str(tmp_path / "snap.json"))

        # Should still accept new emits afterward
        collector.emit("agent.output", {}, span_id="s2")
        assert len(collector.events) == 2


class TestReplayEvents:
    def test_replays_into_fresh_collector(self):
        client = Mock()
        src = _make_collector(client)
        src.emit("agent.input", {"x": 1}, span_id="a")
        src.emit("agent.output", {"y": 2}, span_id="b")

        # Serialize and replay into a fresh collector
        snapshot = src.to_replay_dict()
        dst = TraceCollector(client, CaptureConfig.standard())
        count = replay_events(snapshot, dst)

        assert count == 2
        dst_events = dst.events
        assert [e["event_type"] for e in dst_events] == ["agent.input", "agent.output"]
        assert dst_events[0]["payload"] == {"x": 1}
        # New collector has its own trace_id
        assert dst.trace_id != src.trace_id

    def test_handles_empty_snapshot(self):
        client = Mock()
        dst = _make_collector(client)
        count = replay_events({"events": []}, dst)
        assert count == 0


class TestSerializeAdapter:
    def test_returns_adapter_metadata(self):
        client = Mock()
        from layerlens.instrument.adapters._base import AdapterInfo

        adapter = Mock()
        adapter.adapter_info.return_value = AdapterInfo(
            name="test", adapter_type="framework", version="1.2.3", metadata={"key": "value"}
        )
        result = serialize_adapter(adapter)
        assert result["adapter"]["name"] == "test"
        assert result["adapter"]["adapter_type"] == "framework"
        assert result["adapter"]["version"] == "1.2.3"
        assert result["adapter"]["metadata"] == {"key": "value"}
        assert "trace" not in result

    def test_with_collector_includes_trace(self):
        client = Mock()
        collector = _make_collector(client)
        collector.emit("agent.input", {}, span_id="s1")
        from layerlens.instrument.adapters._base import AdapterInfo

        adapter = Mock()
        adapter.adapter_info.return_value = AdapterInfo(name="x", adapter_type="framework")

        result = serialize_adapter(adapter, collector=collector)
        assert "trace" in result
        assert result["trace"]["trace_id"] == collector.trace_id


class TestCollectorToReplayDict:
    def test_public_method_matches_internal(self):
        client = Mock()
        collector = _make_collector(client)
        collector.emit("agent.input", {}, span_id="s1")
        public = collector.to_replay_dict()
        # Same shape as the internal payload
        assert set(public.keys()) >= {"trace_id", "events", "capture_config", "attestation"}

    def test_round_trips_through_json(self):
        client = Mock()
        collector = _make_collector(client)
        collector.emit("agent.input", {"foo": "bar"}, span_id="s1")

        payload = collector.to_replay_dict()
        text = json.dumps(payload, default=str)
        reloaded = json.loads(text)
        assert reloaded["trace_id"] == collector.trace_id

    def test_events_property_is_snapshot(self):
        """Modifying the returned list shouldn't mutate the collector."""
        client = Mock()
        collector = _make_collector(client)
        collector.emit("agent.input", {}, span_id="s1")
        snapshot = collector.events
        snapshot.append({"event_type": "fake"})
        # Internal events untouched
        assert len(collector.events) == 1
