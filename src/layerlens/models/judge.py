from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class JudgeVersion(BaseModel):
    version: int
    name: str
    evaluation_goal: str
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    model_company: Optional[str] = None
    updated_at: str
    updated_by: str


class Judge(BaseModel):
    id: str
    organization_id: str
    project_id: str
    name: str
    evaluation_goal: str
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    model_company: Optional[str] = None
    version: int
    run_count: int = 0
    created_at: str
    updated_at: str
    versions: List[JudgeVersion] = []
