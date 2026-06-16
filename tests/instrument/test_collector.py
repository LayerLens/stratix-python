"""TraceCollector lifecycle tests (LAY-3579 / T5).

Covers the MAX_EVENTS cap + truncated flag, sealed-after-flush semantics,
non-destructive ``to_replay_dict``, and a concurrent-emitters smoke test.
"""

from __future__ import annotations

import threading
from typing import Any

from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig


def _emit(collector: TraceCollector, n: int = 1, prefix: str = "s") -> None:
    for i in range(n):
        collector.emit(
            "model.invoke",
            {"model": "test-model", "latency_ms": 1},
            span_id=f"{prefix}{i:05d}",
        )


class TestMaxEventsCap:
    def test_cap_drops_excess_and_flags_truncation(
        self, monkeypatch: Any, mock_client: Any, capture_trace: Any
    ) -> None:
        monkeypatch.setattr(TraceCollector, "MAX_EVENTS", 5)
        collector = TraceCollector(mock_client, CaptureConfig())

        _emit(collector, 8)

        assert len(collector.events) == 5
        collector.flush()
        assert len(capture_trace["events"]) == 5

    def test_truncated_flag_in_payload(self, monkeypatch: Any, mock_client: Any) -> None:
        monkeypatch.setattr(TraceCollector, "MAX_EVENTS", 3)
        collector = TraceCollector(mock_client, CaptureConfig())
        _emit(collector, 4)

        replay = collector.to_replay_dict()
        assert replay["truncated"] is True
        assert replay["max_events"] == 3

    def test_no_truncated_flag_below_cap(self, mock_client: Any) -> None:
        collector = TraceCollector(mock_client, CaptureConfig())
        _emit(collector, 2)
        replay = collector.to_replay_dict()
        assert "truncated" not in replay


class TestSealing:
    def test_emit_after_flush_is_noop(self, mock_client: Any) -> None:
        collector = TraceCollector(mock_client, CaptureConfig())
        _emit(collector, 2)
        collector.flush()
        _emit(collector, 3, prefix="late")
        assert len(collector.events) == 2

    def test_second_flush_does_not_reupload(self, mock_client: Any) -> None:
        collector = TraceCollector(mock_client, CaptureConfig())
        _emit(collector, 1)
        collector.flush()
        collector.flush()
        assert mock_client.traces.upload.call_count == 1

    def test_flush_with_no_events_uploads_nothing(self, mock_client: Any) -> None:
        collector = TraceCollector(mock_client, CaptureConfig())
        collector.flush()
        assert mock_client.traces.upload.call_count == 0


class TestReplayDict:
    def test_to_replay_dict_is_non_destructive(self, mock_client: Any, capture_trace: Any) -> None:
        collector = TraceCollector(mock_client, CaptureConfig())
        _emit(collector, 2)

        replay = collector.to_replay_dict()
        assert len(replay["events"]) == 2
        assert replay["attestation"].get("root_hash")

        # Still usable afterwards: emit more, then flush uploads everything.
        _emit(collector, 1, prefix="more")
        collector.flush()
        assert len(capture_trace["events"]) == 3

    def test_replay_dict_of_empty_collector(self, mock_client: Any) -> None:
        collector = TraceCollector(mock_client, CaptureConfig())
        replay = collector.to_replay_dict()
        assert replay["events"] == []
        assert "root_hash" not in replay["attestation"]


class TestThreadSafety:
    def test_concurrent_emitters_keep_unique_ordered_sequence(self, mock_client: Any, capture_trace: Any) -> None:
        collector = TraceCollector(mock_client, CaptureConfig())
        n_threads, per_thread = 8, 50

        def worker(t: int) -> None:
            _emit(collector, per_thread, prefix=f"t{t}-")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        collector.flush()
        events = capture_trace["events"]
        assert len(events) == n_threads * per_thread
        seqs = [e["sequence_id"] for e in events]
        assert sorted(seqs) == list(range(1, n_threads * per_thread + 1))
        # Attestation chain must be intact over the concurrent emissions.
        assert capture_trace["attestation"].get("root_hash")
