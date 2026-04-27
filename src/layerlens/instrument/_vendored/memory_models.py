"""Vendored snapshot of ``stratix.memory.models``.

Source: ``A:/github/layerlens/ateam/stratix/memory/models.py``
Source SHA: 7359c0e38d74e02aa1b27c34daef7a958abbd002

Compatibility shims applied for Python 3.9 + Pydantic 2:
- ``datetime.UTC`` (added in Python 3.11) replaced with the
  ``timezone.utc`` alias so ``datetime.now(UTC)`` keeps working.
- PEP-604 union syntax (``X | None``) on Pydantic field annotations
  rewritten as ``Optional[X]``.

Updates require re-vendoring — see ``__init__.py`` for the workflow.
"""

# STRATIX Agent Memory — Pydantic Models
#
# Data models for persistent long-term agent memory: entries, queries,
# consolidation results, and usage statistics.

from __future__ import annotations

from uuid import uuid4
from typing import Any, Literal, Optional
from datetime import datetime, timezone

from pydantic import Field, BaseModel

UTC = timezone.utc  # Python 3.11+ has datetime.UTC; alias for 3.9/3.10 compat.


class MemoryEntry(BaseModel):
    """A single memory record stored for an agent."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    org_id: str
    agent_id: str
    memory_type: Literal["episodic", "semantic", "procedural", "working"]
    namespace: str = "default"
    key: str
    content: str
    embedding_hash: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = 0
    last_accessed_at: Optional[str] = None
    expires_at: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class MemoryQuery(BaseModel):
    """Query parameters for memory retrieval."""

    org_id: str
    agent_id: str
    namespace: str = "default"
    memory_type: Optional[str] = None
    key_prefix: Optional[str] = None
    min_importance: float = 0.0
    limit: int = Field(default=20, le=100)
    include_expired: bool = False


class MemoryConsolidation(BaseModel):
    """Result of memory consolidation (summarization of old memories)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    org_id: str
    agent_id: str
    source_memory_ids: list[str]
    consolidated_content: str
    consolidation_method: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class MemoryStats(BaseModel):
    """Usage statistics for agent memory."""

    org_id: str
    agent_id: str
    total_entries: int
    by_type: dict[str, int]
    by_namespace: dict[str, int]
    avg_importance: float
    oldest_entry: Optional[str]
    newest_entry: Optional[str]
    storage_bytes: int


__all__ = [
    "MemoryEntry",
    "MemoryQuery",
    "MemoryConsolidation",
    "MemoryStats",
]
