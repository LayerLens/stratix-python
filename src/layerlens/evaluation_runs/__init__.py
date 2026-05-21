"""Systematic evaluation runs against managed datasets.

Connects :mod:`layerlens.datasets` to an evaluation executor and adds
the pieces needed for recurring quality gating:

* :class:`EvaluationRunner` runs a target function over every item in a
  dataset version, collecting per-item scores.
* :class:`RunScheduler` re-runs on an interval (thread-backed).
* :class:`RunComparer` diffs two completed runs and flags regressions
  against a configurable tolerance.
"""

from __future__ import annotations

from .models import (
    ScorerFn,
    TargetFn,
    RunAggregate,
    EvaluationRun,
    EvaluationRunItem,
    EvaluationRunStatus,
)
from .runner import EvaluationRunner
from .comparer import RunComparer, RunComparison
from .scheduler import RunScheduler, ScheduledRun

__all__ = [
    "EvaluationRun",
    "EvaluationRunItem",
    "EvaluationRunStatus",
    "EvaluationRunner",
    "RunAggregate",
    "RunComparer",
    "RunComparison",
    "RunScheduler",
    "ScheduledRun",
    "ScorerFn",
    "TargetFn",
]
