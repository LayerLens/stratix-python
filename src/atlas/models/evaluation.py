from __future__ import annotations

from enum import Enum
from typing import Dict, Optional
from datetime import timedelta

from pydantic import Field, BaseModel, ConfigDict


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    FAILURE = "failure"
    IN_PROGRESS = "in-progress"
    PAUSED = "paused"
    SUCCESS = "success"
    TIMEOUT = "timeout"


class Evaluation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    status: EvaluationStatus
    submitted_at: int
    finished_at: int
    model_id: str
    benchmark_id: str = Field(..., alias="dataset_id")
    average_duration: int
    accuracy: float

    @property
    def is_finished(self) -> bool:
        """Return True if evaluation is done (success, failure, or timeout)."""
        return self.status in {
            EvaluationStatus.SUCCESS,
            EvaluationStatus.FAILURE,
            EvaluationStatus.TIMEOUT,
        }

    @property
    def is_success(self) -> bool:
        """Return True if evaluation completed successfully."""
        return self.status == EvaluationStatus.SUCCESS


class Result(BaseModel):
    subset: str
    prompt: str
    result: str
    truth: str
    duration: timedelta
    score: float
    metrics: Dict[str, Optional[float]]
