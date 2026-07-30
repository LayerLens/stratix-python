"""Execute a target function over every item in a dataset version."""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime, timezone

from .models import (
    ScorerFn,
    TargetFn,
    RunAggregate,
    EvaluationRun,
    EvaluationRunItem,
    EvaluationRunStatus,
)
from ..datasets import DatasetItem, DatasetStore

log = logging.getLogger(__name__)


class EvaluationRunner:
    """Run a ``TargetFn`` against a dataset, score each output, aggregate."""

    def __init__(
        self,
        dataset_store: DatasetStore,
        *,
        pass_threshold: float = 0.5,
    ) -> None:
        self._store = dataset_store
        self._pass_threshold = pass_threshold

    def run(
        self,
        *,
        dataset_id: str,
        target: TargetFn,
        scorers: Dict[str, ScorerFn],
        version: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        on_item: Optional[Callable[[EvaluationRunItem], None]] = None,
    ) -> EvaluationRun:
        items_iter = list(self._store.iter_items(dataset_id, version=version))
        dataset = self._store.get(dataset_id)
        latest = dataset.latest() if dataset else None
        resolved_version = version if version is not None else (latest.version if latest else 0)

        run = EvaluationRun(
            id=f"run_{uuid.uuid4().hex[:16]}",
            dataset_id=dataset_id,
            dataset_version=resolved_version,
            status=EvaluationRunStatus.RUNNING,
            metadata=dict(metadata or {}),
        )

        if not items_iter:
            run.status = EvaluationRunStatus.FAILED
            run.error = "dataset has no items for the requested version"
            run.completed_at = _now()
            return run

        for item in items_iter:
            run.items.append(self._execute_item(item, target, scorers))
            if on_item is not None:
                try:
                    on_item(run.items[-1])
                except Exception:  # pragma: no cover - callback defensively
                    log.debug("on_item callback raised", exc_info=True)

        run.aggregate = self._aggregate(run.items)
        run.status = EvaluationRunStatus.COMPLETED
        run.completed_at = _now()
        return run

    def _execute_item(
        self,
        item: DatasetItem,
        target: TargetFn,
        scorers: Dict[str, ScorerFn],
    ) -> EvaluationRunItem:
        run_item = EvaluationRunItem(
            item_id=item.id,
            input=item.input,
            expected_output=item.expected_output,
        )
        start = time.monotonic()
        try:
            run_item.actual_output = target(item.input)
        except Exception as exc:
            run_item.error = f"{type(exc).__name__}: {exc}"
            run_item.passed = False
            run_item.latency_ms = (time.monotonic() - start) * 1000
            return run_item
        run_item.latency_ms = (time.monotonic() - start) * 1000

        item_scores: Dict[str, float] = {}
        for name, scorer in scorers.items():
            try:
                item_scores[name] = float(scorer(run_item.actual_output, item.expected_output, item.metadata))
            except Exception as exc:
                log.debug("scorer %s raised on item %s: %s", name, item.id, exc)
                item_scores[name] = 0.0
        run_item.scores = item_scores

        if item_scores:
            mean = sum(item_scores.values()) / len(item_scores)
            run_item.passed = mean >= self._pass_threshold
        else:
            run_item.passed = run_item.error is None
        return run_item

    def _aggregate(self, items: List[EvaluationRunItem]) -> RunAggregate:
        if not items:
            return RunAggregate()
        score_totals: Dict[str, List[float]] = {}
        latencies: List[float] = []
        errors = 0
        passed = 0
        for it in items:
            if it.error is not None:
                errors += 1
            if it.passed:
                passed += 1
            if it.latency_ms is not None:
                latencies.append(it.latency_ms)
            for name, value in it.scores.items():
                score_totals.setdefault(name, []).append(value)
        means = {n: sum(v) / len(v) for n, v in score_totals.items() if v}
        return RunAggregate(
            mean_scores=means,
            pass_rate=passed / len(items),
            item_count=len(items),
            error_count=errors,
            avg_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
