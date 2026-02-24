from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel

from .trace_evaluation import JudgeSnapshot


class OptimizationRunStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"


class OptimizationBudget(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class JudgeOptimizationRun(BaseModel):
    id: str
    organization_id: str
    project_id: str
    judge_id: str
    user_id: Optional[str] = None
    status: OptimizationRunStatus
    status_description: Optional[str] = None
    judge_snapshot: Optional[JudgeSnapshot] = None
    annotation_count: int = 0
    budget: OptimizationBudget = OptimizationBudget.MEDIUM
    baseline_accuracy: Optional[float] = None
    optimized_accuracy: Optional[float] = None
    original_goal: Optional[str] = None
    optimized_goal: Optional[str] = None
    logs: Optional[str] = None
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    applied_at: Optional[str] = None
    applied_version: Optional[int] = None
