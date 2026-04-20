"""Evaluation run models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime, timezone

from pydantic import Field, BaseModel


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


TargetFn = Callable[[Any], Any]
"""Target under evaluation — takes an item's input, returns its output."""

ScorerFn = Callable[[Any, Any, Any], float]
"""Per-item scorer: ``(actual, expected, item_metadata) -> score in [0, 1]``."""


class EvaluationRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationRunItem(BaseModel):
    item_id: str
    input: Any = None
    expected_output: Any = None
    actual_output: Any = None
    scores: Dict[str, float] = Field(default_factory=dict)
    passed: Optional[bool] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class RunAggregate(BaseModel):
    mean_scores: Dict[str, float] = Field(default_factory=dict)
    pass_rate: float = 0.0
    item_count: int = 0
    error_count: int = 0
    avg_latency_ms: Optional[float] = None


class EvaluationRun(BaseModel):
    id: str
    dataset_id: str
    dataset_version: int
    status: EvaluationRunStatus = EvaluationRunStatus.PENDING
    created_at: str = Field(default_factory=_now)
    completed_at: Optional[str] = None
    items: List[EvaluationRunItem] = Field(default_factory=list)
    aggregate: RunAggregate = Field(default_factory=RunAggregate)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
