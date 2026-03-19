"""Tests for RunStore."""

import tempfile
import time

import pytest

from layerlens.instrument.simulators.run_store import RunRecord, RunStore


class TestRunRecord:
    def test_basic_creation(self):
        record = RunRecord(run_id="run_test01")
        assert record.run_id == "run_test01"
        assert record.status == "generating"
        assert record.trace_count == 0

    def test_duration(self):
        record = RunRecord(
            run_id="run_test01",
            start_time=1000.0,
            end_time=1005.0,
        )
        assert record.duration_seconds == 5.0

    def test_duration_running(self):
        record = RunRecord(
            run_id="run_test01",
            start_time=time.time() - 10.0,
        )
        assert record.duration_seconds >= 9.0

    def test_serialization(self):
        record = RunRecord(
            run_id="run_test01",
            config={"source_format": "openai"},
            trace_count=10,
        )
        data = record.model_dump(mode="json")
        restored = RunRecord(**data)
        assert restored.run_id == "run_test01"
        assert restored.config["source_format"] == "openai"


class TestRunStore:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = RunStore(store_dir=self._tmpdir)

    def test_store_dir_created(self):
        assert self.store.store_dir.exists()

    def test_save_and_get(self):
        record = RunRecord(
            run_id="run_abc123",
            config={"source_format": "openai"},
            trace_count=5,
            status="complete",
        )
        self.store.save(record)
        loaded = self.store.get("run_abc123")
        assert loaded is not None
        assert loaded.run_id == "run_abc123"
        assert loaded.trace_count == 5

    def test_get_nonexistent(self):
        assert self.store.get("nonexistent") is None

    def test_list_runs(self):
        for i in range(5):
            record = RunRecord(
                run_id=f"run_{i:04d}",
                start_time=float(i),
                status="complete",
            )
            self.store.save(record)

        runs = self.store.list_runs()
        assert len(runs) == 5
        # Sorted by start_time descending
        assert runs[0].run_id == "run_0004"

    def test_list_runs_with_limit(self):
        for i in range(10):
            self.store.save(RunRecord(run_id=f"run_{i}", start_time=float(i)))
        runs = self.store.list_runs(limit=3)
        assert len(runs) == 3

    def test_list_runs_with_status_filter(self):
        self.store.save(RunRecord(run_id="r1", status="complete", start_time=1.0))
        self.store.save(RunRecord(run_id="r2", status="failed", start_time=2.0))
        self.store.save(RunRecord(run_id="r3", status="complete", start_time=3.0))

        complete = self.store.list_runs(status="complete")
        assert len(complete) == 2

    def test_delete(self):
        self.store.save(RunRecord(run_id="run_del"))
        assert self.store.delete("run_del") is True
        assert self.store.get("run_del") is None
        assert self.store.delete("nonexistent") is False

    def test_update_status(self):
        self.store.save(RunRecord(run_id="run_upd", status="generating"))
        updated = self.store.update_status(
            "run_upd",
            status="complete",
            end_time=time.time(),
            validation_status="pass",
        )
        assert updated is not None
        assert updated.status == "complete"
        assert updated.validation_status == "pass"

    def test_update_nonexistent(self):
        assert self.store.update_status("nonexistent", "complete") is None

    def test_create_run(self):
        record = self.store.create_run(
            "run_new",
            config={"source_format": "openai", "count": 10},
        )
        assert record.run_id == "run_new"
        assert record.status == "generating"
        assert record.start_time > 0

        loaded = self.store.get("run_new")
        assert loaded is not None
        assert loaded.config["count"] == 10

    def test_complete_run(self):
        self.store.create_run("run_comp", config={})
        completed = self.store.complete_run(
            "run_comp",
            trace_count=50,
            span_count=250,
            total_tokens=10000,
            error_count=2,
            validation_status="pass",
        )
        assert completed is not None
        assert completed.status == "complete"
        assert completed.trace_count == 50
        assert completed.end_time is not None

    def test_get_summary_empty(self):
        summary = self.store.get_summary()
        assert summary["total_runs"] == 0
        assert summary["pass_rate"] == 0.0

    def test_get_summary(self):
        self.store.save(RunRecord(
            run_id="r1",
            config={"source_format": "openai", "scenario": "sales"},
            trace_count=10,
            total_tokens=5000,
            validation_status="pass",
            start_time=1.0,
        ))
        self.store.save(RunRecord(
            run_id="r2",
            config={"source_format": "anthropic", "scenario": "sales"},
            trace_count=20,
            total_tokens=8000,
            validation_status="pass",
            start_time=2.0,
        ))
        self.store.save(RunRecord(
            run_id="r3",
            config={"source_format": "openai", "scenario": "it_helpdesk"},
            trace_count=5,
            total_tokens=2000,
            validation_status="fail",
            start_time=3.0,
        ))

        summary = self.store.get_summary()
        assert summary["total_runs"] == 3
        assert summary["total_traces"] == 35
        assert summary["total_tokens"] == 15000
        assert summary["sources_used"] == 2
        assert summary["scenarios_used"] == 2
        assert summary["pass_rate"] == pytest.approx(66.7, abs=0.1)
