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


class Result(BaseModel):
    subset: str
    prompt: str
    result: str
    truth: str
    duration: timedelta
    score: float
    metrics: Dict[str, Optional[float]]
