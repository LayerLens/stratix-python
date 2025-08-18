from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Organization(BaseModel):
    id: str
    name: str
    projects: Optional[List[Project]] = None


class Project(BaseModel):
    id: str
    name: str
