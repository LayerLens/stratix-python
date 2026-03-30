from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class Integration(BaseModel):
    id: str
    organization_id: str
    name: str
    type: Optional[str] = None
    host_url: Optional[str] = None
    active: Optional[bool] = None
    created_at: Optional[str] = None
    created_by: Optional[str] = None
    # Legacy/convenience aliases kept optional for backwards compatibility
    project_id: Optional[str] = None
    status: Optional[str] = None
    config: Dict[str, Any] = {}
