"""Run store for persisting simulator run history.

Lightweight JSON-file run store that persists run metadata
to ~/.stratix/simulator/runs/. Enables the Audit screen
without requiring a database.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunRecord(BaseModel):
    """Persisted record of a simulator run."""

    run_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    start_time: float = 0.0
    end_time: float | None = None
    trace_count: int = 0
    span_count: int = 0
    total_tokens: int = 0
    error_count: int = 0
    validation_status: str = "pending"
    validation_details: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "generating"  # generating | complete | failed | cancelled

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class RunStore:
    """JSON-file-based run history store.

    Persists run records to individual JSON files in the store directory.
    Default location: ~/.stratix/simulator/runs/
    """

    def __init__(self, store_dir: str | None = None):
        if store_dir:
            self._store_dir = Path(store_dir)
        else:
            home = Path(os.environ.get("STRATIX_HOME", Path.home() / ".stratix"))
            self._store_dir = home / "simulator" / "runs"
        self._store_dir.mkdir(parents=True, exist_ok=True)

    @property
    def store_dir(self) -> Path:
        return self._store_dir

    def _run_path(self, run_id: str) -> Path:
        # Sanitize run_id to prevent path traversal
        safe_id = run_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        path = self._store_dir / f"{safe_id}.json"
        # Ensure resolved path stays within store directory
        if not path.resolve().is_relative_to(self._store_dir.resolve()):
            raise ValueError(f"Invalid run_id: {run_id}")
        return path

    def save(self, record: RunRecord) -> None:
        """Save or update a run record."""
        path = self._run_path(record.run_id)
        with open(path, "w") as f:
            json.dump(record.model_dump(mode="json"), f, indent=2)

    def get(self, run_id: str) -> RunRecord | None:
        """Load a run record by ID."""
        path = self._run_path(run_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return RunRecord(**data)

    def list_runs(
        self,
        limit: int = 50,
        status: str | None = None,
    ) -> list[RunRecord]:
        """List run records, sorted by start_time descending."""
        records: list[RunRecord] = []
        for path in self._store_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                record = RunRecord(**data)
                if status and record.status != status:
                    continue
                records.append(record)
            except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
                continue

        records.sort(key=lambda r: r.start_time, reverse=True)
        return records[:limit]

    def delete(self, run_id: str) -> bool:
        """Delete a run record."""
        path = self._run_path(run_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def update_status(
        self,
        run_id: str,
        status: str,
        end_time: float | None = None,
        validation_status: str | None = None,
        validation_details: list[dict[str, Any]] | None = None,
    ) -> RunRecord | None:
        """Update run status and optional fields."""
        record = self.get(run_id)
        if not record:
            return None
        record.status = status
        if end_time is not None:
            record.end_time = end_time
        if validation_status is not None:
            record.validation_status = validation_status
        if validation_details is not None:
            record.validation_details = validation_details
        self.save(record)
        return record

    def get_summary(self) -> dict[str, Any]:
        """Get aggregate summary stats for the audit screen."""
        runs = self.list_runs(limit=1000)
        total_traces = sum(r.trace_count for r in runs)
        total_tokens = sum(r.total_tokens for r in runs)
        pass_count = sum(
            1 for r in runs if r.validation_status in ("pass", "passed")
        )
        pass_rate = (pass_count / len(runs) * 100) if runs else 0.0

        sources_used = set()
        scenarios_used = set()
        for r in runs:
            cfg = r.config
            if "source_format" in cfg:
                sources_used.add(cfg["source_format"])
            if "scenario" in cfg:
                scenarios_used.add(cfg["scenario"])

        return {
            "total_runs": len(runs),
            "total_traces": total_traces,
            "total_tokens": total_tokens,
            "pass_rate": round(pass_rate, 1),
            "sources_used": len(sources_used),
            "scenarios_used": len(scenarios_used),
        }

    def create_run(
        self,
        run_id: str,
        config: dict[str, Any],
    ) -> RunRecord:
        """Create and persist a new run record."""
        record = RunRecord(
            run_id=run_id,
            config=config,
            start_time=time.time(),
            status="generating",
        )
        self.save(record)
        return record

    def complete_run(
        self,
        run_id: str,
        trace_count: int,
        span_count: int,
        total_tokens: int,
        error_count: int = 0,
        validation_status: str = "pending",
    ) -> RunRecord | None:
        """Mark a run as complete with final stats."""
        record = self.get(run_id)
        if not record:
            return None
        record.status = "complete"
        record.end_time = time.time()
        record.trace_count = trace_count
        record.span_count = span_count
        record.total_tokens = total_tokens
        record.error_count = error_count
        record.validation_status = validation_status
        self.save(record)
        return record
