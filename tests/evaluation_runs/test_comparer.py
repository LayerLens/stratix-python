from __future__ import annotations

from layerlens.evaluation_runs import (
    RunComparer,
    RunAggregate,
    EvaluationRun,
    EvaluationRunItem,
)


def _run(
    run_id: str,
    *,
    means: dict,
    pass_rate: float,
    items: list[tuple[str, bool]] | None = None,
    latency: float | None = None,
) -> EvaluationRun:
    return EvaluationRun(
        id=run_id,
        dataset_id="d",
        dataset_version=1,
        aggregate=RunAggregate(
            mean_scores=means,
            pass_rate=pass_rate,
            item_count=len(items) if items else 0,
            avg_latency_ms=latency,
        ),
        items=[EvaluationRunItem(item_id=iid, passed=passed) for iid, passed in (items or [])],
    )


class TestRunComparer:
    def test_improvement_not_regression(self):
        base = _run("b", means={"exact": 0.5}, pass_rate=0.5)
        cand = _run("c", means={"exact": 0.9}, pass_rate=0.9)
        cmp = RunComparer().compare(base, cand)
        assert cmp.is_regression is False
        assert cmp.improved_scorers == ["exact"]
        assert cmp.regressed_scorers == []
        assert cmp.score_deltas["exact"] > 0

    def test_regression_on_pass_rate(self):
        base = _run("b", means={"exact": 1.0}, pass_rate=1.0)
        cand = _run("c", means={"exact": 1.0}, pass_rate=0.5)
        cmp = RunComparer().compare(base, cand)
        assert cmp.is_regression is True
        assert cmp.pass_rate_delta == -0.5

    def test_tolerance_absorbs_noise(self):
        base = _run("b", means={"exact": 0.9}, pass_rate=0.9)
        cand = _run("c", means={"exact": 0.895}, pass_rate=0.895)
        cmp = RunComparer(score_tolerance=0.02, pass_rate_tolerance=0.02).compare(base, cand)
        assert cmp.is_regression is False
        assert cmp.regressed_scorers == []

    def test_per_item_regression_detection(self):
        base = _run(
            "b",
            means={"exact": 1.0},
            pass_rate=1.0,
            items=[("a", True), ("b", True)],
        )
        cand = _run(
            "c",
            means={"exact": 0.5},
            pass_rate=0.5,
            items=[("a", True), ("b", False)],
        )
        cmp = RunComparer().compare(base, cand)
        assert cmp.regressed_items == ["b"]
        assert cmp.recovered_items == []

    def test_recovery_detection(self):
        base = _run(
            "b",
            means={"exact": 0.5},
            pass_rate=0.5,
            items=[("a", True), ("b", False)],
        )
        cand = _run(
            "c",
            means={"exact": 1.0},
            pass_rate=1.0,
            items=[("a", True), ("b", True)],
        )
        cmp = RunComparer().compare(base, cand)
        assert cmp.recovered_items == ["b"]
        assert cmp.regressed_items == []

    def test_latency_delta(self):
        base = _run("b", means={}, pass_rate=1.0, latency=100.0)
        cand = _run("c", means={}, pass_rate=1.0, latency=140.0)
        cmp = RunComparer().compare(base, cand)
        assert cmp.latency_delta_ms == 40.0

    def test_latency_delta_none_when_missing(self):
        base = _run("b", means={}, pass_rate=1.0, latency=None)
        cand = _run("c", means={}, pass_rate=1.0, latency=140.0)
        assert RunComparer().compare(base, cand).latency_delta_ms is None

    def test_scorer_only_in_candidate_ignored(self):
        base = _run("b", means={"exact": 1.0}, pass_rate=1.0)
        cand = _run("c", means={"exact": 1.0, "new": 0.2}, pass_rate=1.0)
        cmp = RunComparer().compare(base, cand)
        assert "new" not in cmp.score_deltas
