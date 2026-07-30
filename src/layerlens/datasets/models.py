"""Dataset models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from pydantic import Field, BaseModel


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetVisibility(str, Enum):
    PRIVATE = "private"
    ORGANIZATION = "organization"
    PUBLIC = "public"


class DatasetItem(BaseModel):
    id: str
    input: Any
    expected_output: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class DatasetVersion(BaseModel):
    """Immutable snapshot of a dataset's items."""

    version: int = Field(ge=1)
    created_at: str = Field(default_factory=_now)
    note: Optional[str] = None
    items: List[DatasetItem] = Field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.items)


class Dataset(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    visibility: DatasetVisibility = DatasetVisibility.PRIVATE
    tags: List[str] = Field(default_factory=list)
    organization_id: Optional[str] = None
    project_id: Optional[str] = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    current_version: int = 1
    versions: List[DatasetVersion] = Field(default_factory=list)

    def latest(self) -> Optional[DatasetVersion]:
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.version)

    def version(self, n: int) -> Optional[DatasetVersion]:
        for v in self.versions:
            if v.version == n:
                return v
        return None
