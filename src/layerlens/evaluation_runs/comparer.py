"""Compare two evaluation runs and flag regressions."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field, BaseModel

from .models import EvaluationRun


class RunComparison(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    score_deltas: Dict[str, float] = Field(default_factory=dict)
    pass_rate_delta: float = 0.0
    latency_delta_ms: Optional[float] = None
    regressed_scorers: List[str] = Field(default_factory=list)
    improved_scorers: List[str] = Field(default_factory=list)
    regressed_items: List[str] = Field(
        default_factory=list,
        description="IDs of items that passed on baseline but failed on candidate.",
    )
    recovered_items: List[str] = Field(
        default_factory=list,
        description="IDs of items that failed on baseline but passed on candidate.",
    )
    is_regression: bool = False


class RunComparer:
    """Diff two :class:`EvaluationRun` objects within a tolerance."""

    def __init__(
        self,
        *,
        score_tolerance: float = 0.02,
        pass_rate_tolerance: float = 0.02,
    ) -> None:
        self._score_tol = score_tolerance
        self._pass_rate_tol = pass_rate_tolerance

    def compare(self, baseline: EvaluationRun, candidate: EvaluationRun) -> RunComparison:
        score_deltas: Dict[str, float] = {}
        regressed: List[str] = []
        improved: List[str] = []
        for name, base_mean in baseline.aggregate.mean_scores.items():
            cand_mean = candidate.aggregate.mean_scores.get(name)
            if cand_mean is None:
                continue
            delta = cand_mean - base_mean
            score_deltas[name] = delta
            if delta < -self._score_tol:
                regressed.append(name)
            elif delta > self._score_tol:
                improved.append(name)

        pass_rate_delta = candidate.aggregate.pass_rate - baseline.aggregate.pass_rate

        latency_delta: Optional[float] = None
        if baseline.aggregate.avg_latency_ms is not None and candidate.aggregate.avg_latency_ms is not None:
            latency_delta = candidate.aggregate.avg_latency_ms - baseline.aggregate.avg_latency_ms

        baseline_items = {i.item_id: i.passed for i in baseline.items}
        regressed_items = [i.item_id for i in candidate.items if baseline_items.get(i.item_id) and i.passed is False]
        recovered_items = [i.item_id for i in candidate.items if baseline_items.get(i.item_id) is False and i.passed]

        is_regression = bool(regressed) or pass_rate_delta < -self._pass_rate_tol

        return RunComparison(
            baseline_run_id=baseline.id,
            candidate_run_id=candidate.id,
            score_deltas=score_deltas,
            pass_rate_delta=pass_rate_delta,
            latency_delta_ms=latency_delta,
            regressed_scorers=regressed,
            improved_scorers=improved,
            regressed_items=regressed_items,
            recovered_items=recovered_items,
            is_regression=is_regression,
        )
