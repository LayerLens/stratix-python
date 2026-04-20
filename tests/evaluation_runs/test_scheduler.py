from __future__ import annotations

import time
import threading

import pytest

from layerlens.evaluation_runs import RunScheduler, EvaluationRun


def _fake_run(run_id: str = "r1") -> EvaluationRun:
    return EvaluationRun(id=run_id, dataset_id="d", dataset_version=1)


class TestRunScheduler:
    def test_rejects_non_positive_interval(self):
        sched = RunScheduler()
        with pytest.raises(ValueError):
            sched.schedule(_fake_run, interval_seconds=0)

    def test_list_and_get(self):
        sched = RunScheduler()
        s = sched.schedule(_fake_run, interval_seconds=10.0)
        try:
            assert sched.get(s.id) is s
            assert s in sched.list()
        finally:
            sched.cancel_all()

    def test_cancel_missing(self):
        sched = RunScheduler()
        assert sched.cancel("missing") is False

    def test_trigger_now_records_to_history(self):
        sched = RunScheduler()
        s = sched.schedule(_fake_run, interval_seconds=10.0)
        try:
            run = sched.trigger_now(s.id)
            assert run is not None
            assert s.last_run is run
            assert run in s.history
        finally:
            sched.cancel_all()

    def test_periodic_tick_runs_factory_multiple_times(self):
        counter = {"n": 0}
        evt = threading.Event()

        def factory():
            counter["n"] += 1
            if counter["n"] >= 3:
                evt.set()
            return _fake_run(f"r{counter['n']}")

        sched = RunScheduler()
        s = sched.schedule(factory, interval_seconds=0.02)
        try:
            # Wait up to 2s for 3 ticks to land.
            assert evt.wait(timeout=2.0), f"only got {counter['n']} ticks"
        finally:
            sched.cancel_all()
        assert len(s.history) >= 3

    def test_factory_exception_does_not_stop_loop(self):
        counter = {"n": 0}

        def factory():
            counter["n"] += 1
            if counter["n"] == 1:
                raise RuntimeError("bad")
            return _fake_run()

        sched = RunScheduler()
        s = sched.schedule(factory, interval_seconds=0.02)
        try:
            deadline = time.time() + 2.0
            while counter["n"] < 3 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            sched.cancel_all()
        assert counter["n"] >= 3
        # First tick raised and produced no history entry.
        assert len(s.history) >= 2

    def test_history_is_bounded(self):
        counter = {"n": 0}

        def factory():
            counter["n"] += 1
            return _fake_run(f"r{counter['n']}")

        sched = RunScheduler()
        s = sched.schedule(factory, interval_seconds=0.01)
        s.history_limit = 3
        try:
            deadline = time.time() + 2.0
            while counter["n"] < 6 and time.time() < deadline:
                time.sleep(0.01)
        finally:
            sched.cancel_all()
        assert len(s.history) <= 3

    def test_cancel_stops_future_ticks(self):
        counter = {"n": 0}

        def factory():
            counter["n"] += 1
            return _fake_run()

        sched = RunScheduler()
        s = sched.schedule(factory, interval_seconds=0.02)
        time.sleep(0.06)
        sched.cancel(s.id)
        snapshot = counter["n"]
        time.sleep(0.1)
        # At most one in-flight tick may land after cancellation, but growth
        # should stop quickly.
        assert counter["n"] - snapshot <= 1
