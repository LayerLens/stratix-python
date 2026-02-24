from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Trace(BaseModel):
    id: str
    organization_id: str
    project_id: str
    created_at: str
    filename: str
    data: Dict[str, Any] = {}
    input: Optional[str] = None
    integration_id: Optional[str] = None


class TraceEvaluationSummary(BaseModel):
    judge_id: str
    judge_name: str
    judge_version: int
    created_at: str
    passed: Optional[bool] = None


class TraceWithEvaluations(Trace):
    evaluations_count: int = 0
    last_evaluations: List[TraceEvaluationSummary] = []
