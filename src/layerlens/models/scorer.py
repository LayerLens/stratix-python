from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Scorer(BaseModel):
    id: Optional[str] = None
    organization_id: Optional[str] = None
    project_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    model_key: Optional[str] = None
    model_company: Optional[str] = None
    prompt: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
