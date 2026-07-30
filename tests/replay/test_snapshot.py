"""Tests for the snapshot module (persist + reload replay-ready traces)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from layerlens.instrument import CaptureConfig, TraceCollector
from layerlens.replay.snapshot import (
    SnapshotIntegrityError,
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
        payload = {
            "trace_id": "abc",
            "events": [{"event_type": "agent.input", "payload": {}}],
        }
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
            name="test",
            adapter_type="framework",
            version="1.2.3",
            metadata={"key": "value"},
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
        assert set(public.keys()) >= {
            "trace_id",
            "events",
            "capture_config",
            "attestation",
        }

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


# ---------------------------------------------------------------------------
# Replay fail-CLOSED (A5 / SEC-2) — a recorded trace is UNTRUSTED input.
# ---------------------------------------------------------------------------
#
# replay_events() used to pass a recorded ``event_type`` VERBATIM to
# collector.emit with no membership check — a garbage/forged recorded trace
# injected an arbitrary type into a fresh collector that then uploaded it
# (fail-OPEN). load_snapshot only isinstance-checked dict. The fix rejects an
# unregistered event_type on both paths (fail-closed) and routes replayed events
# through the collector's normal redact + secret-scrub chokepoint.
#
# Real data: every snapshot below is produced by a REAL TraceCollector's emit +
# to_replay_dict (not a hand-rolled dict on the replay path), then one field is
# mutated to model the attacker-controlled / corrupted case.


def _real_snapshot_with(event_type: str, payload: dict) -> dict:
    """Build a snapshot dict via the REAL collector path, then force one event's
    event_type to *event_type* (which may be unregistered) — modelling a forged
    or hand-edited recorded trace without hand-rolling the whole structure."""
    src = _make_collector(Mock())
    src.emit("agent.input", payload, span_id="a")
    snap = src.to_replay_dict()
    # Mutate the recorded type to the (possibly unregistered) value under test.
    snap["events"][0]["event_type"] = event_type
    return snap


class TestReplayRejectsUnknownEventType:
    @pytest.mark.parametrize(
        "bogus_type",
        [
            "agent.exfil",  # plausible-looking but unregistered
            "evil.payload",
            "shell.exec",
            "model.invoke.EVIL",  # near-miss of a real type
            "",  # empty
        ],
    )
    def test_replay_events_rejects_unregistered_type(self, bogus_type):
        snap = _real_snapshot_with(bogus_type, {"x": 1})
        dst = TraceCollector(Mock(), CaptureConfig.standard())
        with pytest.raises(SnapshotIntegrityError):
            replay_events(snap, dst)
        # Fail closed: nothing was injected into the fresh collector.
        assert dst.events == []

    def test_load_snapshot_rejects_unregistered_type(self, tmp_path: Path):
        snap = _real_snapshot_with("agent.exfil", {"x": 1})
        path = tmp_path / "forged.json"
        dump(snap, str(path))
        with pytest.raises(SnapshotIntegrityError):
            load_snapshot(str(path))

    def test_load_snapshot_rejects_non_dict_event(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        dump({"trace_id": "t", "events": ["not-an-object"]}, str(path))
        with pytest.raises(SnapshotIntegrityError):
            load_snapshot(str(path))

    def test_load_snapshot_rejects_events_not_a_list(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        dump({"trace_id": "t", "events": {"event_type": "agent.input"}}, str(path))
        with pytest.raises(SnapshotIntegrityError):
            load_snapshot(str(path))

    def test_load_snapshot_rejects_non_object_root(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(SnapshotIntegrityError):
            load_snapshot(str(path))

    def test_registered_type_still_replays(self):
        """The gate must not over-block: a snapshot of registered types replays
        cleanly (this is the legitimate use)."""
        src = _make_collector(Mock())
        src.emit("agent.input", {"x": 1}, span_id="a")
        src.emit("model.invoke", {"model": "gpt-4o-mini"}, span_id="b")
        snap = src.to_replay_dict()
        dst = TraceCollector(Mock(), CaptureConfig.standard())
        count = replay_events(snap, dst)
        assert count == 2
        assert [e["event_type"] for e in dst.events] == ["agent.input", "model.invoke"]


class TestReplayScrubsSecrets:
    """A secret-bearing recorded payload is scrubbed on replay, because replay
    goes through the SAME collector chokepoint as a live emit."""

    def test_secret_in_replayed_payload_is_scrubbed(self):
        # capture_content=True so the error free-text survives redaction — proves
        # the SECRET SCRUB (not redaction) is what removes the key on replay.
        src = TraceCollector(Mock(), CaptureConfig.full())
        # agent.error keeps the free-text 'error' under full(); the scrubber must
        # still strip the embedded API key on replay.
        src.emit(
            "agent.error",
            {"error": "auth failed for sk-proj-ABCDEF0123456789ABCDEF", "error_type": "AuthError"},
            span_id="a",
        )
        snap = src.to_replay_dict()
        # Sanity: the recorded snapshot itself (full() capture) still carries the
        # error text — but the SECRET token inside it is already scrubbed at the
        # ORIGINAL emit (the chokepoint runs there too). Force the raw secret back
        # into the recorded payload to model a snapshot produced by an OLDER build
        # whose collector did not scrub, then prove replay re-scrubs it.
        snap["events"][0]["payload"]["error"] = "auth failed for sk-proj-ABCDEF0123456789ABCDEF"

        dst = TraceCollector(Mock(), CaptureConfig.full())
        replay_events(snap, dst)

        replayed_error = dst.events[0]["payload"]["error"]
        assert "sk-proj-ABCDEF0123456789ABCDEF" not in replayed_error
        assert "REDACTED-SECRET" in replayed_error
        # The category survives — replay does not blind observability.
        assert dst.events[0]["payload"]["error_type"] == "AuthError"
