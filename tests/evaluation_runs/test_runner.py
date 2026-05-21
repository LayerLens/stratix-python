from __future__ import annotations

import pytest

from layerlens.datasets import Dataset, InMemoryDatasetStore
from layerlens.evaluation_runs import EvaluationRunner, EvaluationRunStatus


def _exact(actual, expected, _meta):
    return 1.0 if actual == expected else 0.0


def _length(actual, _expected, _meta):
    return min(len(str(actual)) / 10.0, 1.0)


@pytest.fixture
def populated_store():
    store = InMemoryDatasetStore()
    ds = store.create(Dataset(id="", name="qa"))
    store.import_items(
        ds.id,
        [
            {"id": "a", "input": 1, "expected_output": 2},
            {"id": "b", "input": 2, "expected_output": 4},
            {"id": "c", "input": 3, "expected_output": 6},
        ],
    )
    return store, ds.id


class TestRunner:
    def test_perfect_score(self, populated_store):
        store, ds_id = populated_store
        runner = EvaluationRunner(store)
        run = runner.run(
            dataset_id=ds_id,
            target=lambda x: x * 2,
            scorers={"exact": _exact},
        )
        assert run.status == EvaluationRunStatus.COMPLETED
        assert run.aggregate.pass_rate == 1.0
        assert run.aggregate.mean_scores["exact"] == 1.0
        assert run.aggregate.item_count == 3
        assert run.aggregate.error_count == 0
        assert run.aggregate.avg_latency_ms is not None

    def test_partial_failures(self, populated_store):
        store, ds_id = populated_store
        runner = EvaluationRunner(store)
        run = runner.run(
            dataset_id=ds_id,
            target=lambda x: x * 2 if x < 3 else 0,
            scorers={"exact": _exact},
        )
        # 2 of 3 pass → 0.666
        assert 0.6 < run.aggregate.pass_rate < 0.7

    def test_target_exceptions_captured(self, populated_store):
        store, ds_id = populated_store

        def broken(x):
            if x == 2:
                raise RuntimeError("boom")
            return x * 2

        run = EvaluationRunner(store).run(dataset_id=ds_id, target=broken, scorers={"exact": _exact})
        errored = [i for i in run.items if i.error is not None]
        assert len(errored) == 1
        assert "boom" in errored[0].error
        assert run.aggregate.error_count == 1

    def test_multiple_scorers_averaged(self, populated_store):
        store, ds_id = populated_store
        run = EvaluationRunner(store).run(
            dataset_id=ds_id,
            target=lambda x: x * 2,
            scorers={"exact": _exact, "length": _length},
        )
        assert set(run.aggregate.mean_scores) == {"exact", "length"}

    def test_unknown_dataset_fails_gracefully(self):
        run = EvaluationRunner(InMemoryDatasetStore()).run(
            dataset_id="missing",
            target=lambda x: x,
            scorers={"exact": _exact},
        )
        assert run.status == EvaluationRunStatus.FAILED
        assert "no items" in (run.error or "")

    def test_scorer_exceptions_do_not_break_run(self, populated_store):
        store, ds_id = populated_store

        def broken_scorer(_a, _e, _m):
            raise ValueError("nope")

        run = EvaluationRunner(store).run(
            dataset_id=ds_id,
            target=lambda x: x * 2,
            scorers={"broken": broken_scorer, "exact": _exact},
        )
        assert run.status == EvaluationRunStatus.COMPLETED
        # broken scorer contributes 0.0 to the mean.
        assert run.aggregate.mean_scores["broken"] == 0.0

    def test_on_item_callback(self, populated_store):
        store, ds_id = populated_store
        seen = []

        EvaluationRunner(store).run(
            dataset_id=ds_id,
            target=lambda x: x * 2,
            scorers={"exact": _exact},
            on_item=lambda item: seen.append(item.item_id),
        )
        assert seen == ["a", "b", "c"]

    def test_pass_threshold_honoured(self, populated_store):
        store, ds_id = populated_store
        # threshold at 0.95 — half-credit scorer should fail every item.
        runner = EvaluationRunner(store, pass_threshold=0.95)
        run = runner.run(
            dataset_id=ds_id,
            target=lambda x: x * 2,
            scorers={"half": lambda *_: 0.5},
        )
        assert run.aggregate.pass_rate == 0.0

    def test_metadata_pass_through(self, populated_store):
        store, ds_id = populated_store
        run = EvaluationRunner(store).run(
            dataset_id=ds_id,
            target=lambda x: x,
            scorers={"exact": _exact},
            metadata={"run_label": "smoke"},
        )
        assert run.metadata == {"run_label": "smoke"}
