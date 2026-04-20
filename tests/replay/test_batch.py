from __future__ import annotations

import time

from layerlens.replay.batch import BatchReplayer, BatchReplayRequest
from layerlens.replay.models import ReplayStatus
from layerlens.replay.controller import ReplayController

from .conftest import make_trace


class TestBatchReplayer:
    def test_runs_every_trace(self):
        def fn(t, req):
            return make_trace(output="ok")

        batch = BatchReplayer(ReplayController(fn)).run(
            [make_trace("t1"), make_trace("t2"), make_trace("t3")],
            BatchReplayRequest(concurrency=2),
        )
        assert batch.summary.total_traces == 3
        assert batch.summary.completed == 3
        assert batch.summary.failed == 0
        assert batch.summary.timed_out == 0
        assert batch.batch_id.startswith("batch_")

    def test_summary_tracks_changes(self):
        def fn(t, req):
            return make_trace(output="different")

        batch = BatchReplayer(ReplayController(fn)).run(
            [make_trace("t1", output="original"), make_trace("t2", output="original")],
            BatchReplayRequest(concurrency=1),
        )
        assert batch.summary.output_change_rate == 1.0
        assert 0.0 <= batch.summary.avg_output_similarity < 1.0

    def test_failures_counted(self):
        calls = {"n": 0}

        def fn(t, req):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise ValueError("bad")
            return make_trace(output="ok")

        batch = BatchReplayer(ReplayController(fn)).run(
            [make_trace(f"t{i}") for i in range(4)],
            BatchReplayRequest(concurrency=1),
        )
        assert batch.summary.completed + batch.summary.failed == 4
        assert batch.summary.failed >= 1

    def test_timeout(self):
        def fn(t, req):
            time.sleep(0.3)
            return make_trace(output="late")

        batch = BatchReplayer(ReplayController(fn)).run(
            [make_trace("t1")],
            BatchReplayRequest(concurrency=1, timeout_per_trace_ms=10.0),
        )
        statuses = {r.status for r in batch.results}
        assert ReplayStatus.TIMEOUT in statuses

    def test_cost_lookup_aggregated(self):
        costs = {"t1": 0.01, "t2": 0.02, "r1": 0.015, "r2": 0.025}

        def fn(t, req):
            # Mirror the original's id onto the replay so cost_lookup can tell them apart.
            return make_trace("r" + t.id[-1], output="x")

        batch = BatchReplayer(ReplayController(fn)).run(
            [make_trace("t1"), make_trace("t2")],
            BatchReplayRequest(concurrency=2),
            cost_lookup=lambda trace: costs.get(trace.id, 0.0),
        )
        # Each original costs 0.01 less than its replay → avg delta ≈ +0.005.
        assert batch.summary.avg_cost_diff_usd is not None
        assert 0.004 < batch.summary.avg_cost_diff_usd < 0.006
