"""
Langfuse Adapter Configuration Models

Pydantic models for Langfuse adapter configuration, sync state tracking,
and sync result reporting.
"""

from __future__ import annotations

from enum import Enum  # Python 3.11+ has StrEnum; using `(str, Enum)` for 3.9/3.10 compat.
from typing import Optional
from datetime import datetime

from pydantic import Field, BaseModel, field_validator


class SyncDirection(str, Enum):
    """Direction of synchronization."""

    IMPORT = "import"
    EXPORT = "export"
    BIDIRECTIONAL = "bidirectional"


class ConflictStrategy(str, Enum):
    """Strategy for resolving sync conflicts."""

    LAST_WRITE_WINS = "last-write-wins"
    MANUAL = "manual"


class LangfuseConfig(BaseModel):
    """Configuration for the Langfuse adapter."""

    public_key: str = Field(description="Langfuse public API key")
    secret_key: str = Field(description="Langfuse secret API key")
    host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse API host URL",
    )
    mode: SyncDirection = Field(
        default=SyncDirection.IMPORT,
        description="Sync mode: import, export, or bidirectional",
    )
    sync_interval_seconds: int = Field(
        default=3600,
        description="Auto-sync interval in seconds (0 = disabled)",
    )
    project_filter: Optional[str] = Field(
        default=None,
        description="Filter by Langfuse project name",
    )
    tag_filter: Optional[list[str]] = Field(
        default=None,
        description="Filter by trace tags",
    )
    since: Optional[datetime] = Field(
        default=None,
        description="Only sync traces after this timestamp",
    )
    conflict_strategy: ConflictStrategy = Field(
        default=ConflictStrategy.LAST_WRITE_WINS,
        description="Conflict resolution strategy",
    )
    max_retries: int = Field(default=3, description="Max retries per API call")
    page_size: int = Field(default=50, description="Page size for listing traces")

    @field_validator("host")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


class SyncState(BaseModel):
    """Tracks the state of a Langfuse sync session."""

    last_import_cursor: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the last imported trace",
    )
    last_export_cursor: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the last exported trace",
    )
    imported_trace_ids: set[str] = Field(
        default_factory=set,
        description="Set of Langfuse trace IDs that have been imported",
    )
    exported_trace_ids: set[str] = Field(
        default_factory=set,
        description="Set of STRATIX trace IDs that have been exported",
    )
    quarantined_trace_ids: dict[str, int] = Field(
        default_factory=dict,
        description="Trace IDs that have failed repeatedly, mapped to failure count",
    )

    def record_import(self, trace_id: str, updated_at: datetime) -> None:
        """Record a successful import."""
        self.imported_trace_ids.add(trace_id)
        if self.last_import_cursor is None or updated_at > self.last_import_cursor:
            self.last_import_cursor = updated_at
        # Clear from quarantine on success
        self.quarantined_trace_ids.pop(trace_id, None)

    def record_export(self, trace_id: str, updated_at: datetime) -> None:
        """Record a successful export."""
        self.exported_trace_ids.add(trace_id)
        if self.last_export_cursor is None or updated_at > self.last_export_cursor:
            self.last_export_cursor = updated_at

    def record_failure(self, trace_id: str, max_failures: int = 3) -> bool:
        """
        Record a failure for a trace. Returns True if the trace is now quarantined.
        """
        count = self.quarantined_trace_ids.get(trace_id, 0) + 1
        self.quarantined_trace_ids[trace_id] = count
        return count >= max_failures

    def is_quarantined(self, trace_id: str) -> bool:
        """Check if a trace is quarantined (3+ failures)."""
        return self.quarantined_trace_ids.get(trace_id, 0) >= 3

    def clear_quarantine(self, trace_id: str | None = None) -> None:
        """Clear quarantine for a specific trace or all traces."""
        if trace_id:
            self.quarantined_trace_ids.pop(trace_id, None)
        else:
            self.quarantined_trace_ids.clear()


class SyncResult(BaseModel):
    """Result of a sync operation."""

    direction: SyncDirection
    imported_count: int = Field(default=0)
    exported_count: int = Field(default=0)
    skipped_count: int = Field(default=0)
    failed_count: int = Field(default=0)
    quarantined_count: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    dry_run: bool = Field(default=False)
