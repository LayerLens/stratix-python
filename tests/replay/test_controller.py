from __future__ import annotations

import pytest

from layerlens.models.trace import Trace
from layerlens.replay.store import InMemoryReplayStore
from layerlens.replay.models import ReplayStatus, ReplayRequest
from layerlens.replay.controller import ReplayController

from .conftest import make_trace


class TestReplayController:
    def test_successful_replay_stores_result(self):
        original = make_trace(output="hi")
        replayed = make_trace(output="bye")

        def fn(t: Trace, req: ReplayRequest) -> Trace:
            assert req.trace_id == original.id
            return replayed

        store = InMemoryReplayStore()
        ctrl = ReplayController(fn, store=store)
        result = ctrl.run(
            original,
            ReplayRequest(trace_id=original.id, model_override="gpt-4o"),
        )
        assert result.status == ReplayStatus.COMPLETED
        assert result.diff.output_changed is True
        assert result.metadata["replay_type"] == "model_swap"
        assert store.get(result.replay_trace_id) is result

    def test_failed_replay_captures_error(self):
        def fn(t, req):
            raise RuntimeError("boom")

        ctrl = ReplayController(fn)
        result = ctrl.run(make_trace(), ReplayRequest(trace_id="t1"))
        assert result.status == ReplayStatus.FAILED
        assert "boom" in (result.error or "")
        # Even failed results land in the store for debugging.
        assert list(ctrl.store.all())[0].status == ReplayStatus.FAILED

    def test_cost_delta_when_callback_provided(self):
        def fn(t, req):
            return make_trace(output="x")

        result = ReplayController(fn).run(
            make_trace(),
            ReplayRequest(trace_id="t1"),
            cost_original=0.02,
            cost_replay_fn=lambda _t: 0.015,
        )
        assert result.diff.cost_diff_usd == pytest.approx(-0.005)

    def test_latency_lifted_from_replay_trace_data(self):
        def fn(t, req):
            return make_trace(output="x", latency_ms=250.0)

        result = ReplayController(fn).run(
            make_trace(),
            ReplayRequest(trace_id="t1"),
            latency_original_ms=200.0,
        )
        assert result.diff.latency_diff_ms == pytest.approx(50.0)

    def test_duration_ms_populated(self):
        def fn(t, req):
            return make_trace()

        result = ReplayController(fn).run(make_trace(), ReplayRequest(trace_id="t1"))
        assert result.duration_ms >= 0.0
