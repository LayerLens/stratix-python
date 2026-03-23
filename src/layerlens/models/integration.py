from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class Integration(BaseModel):
    id: str
    organization_id: str
    project_id: str
    name: str
    type: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[str] = None
    config: Dict[str, Any] = {}
