"""
Langfuse Trace Importer

Fetches traces from Langfuse, maps them to STRATIX events, deduplicates,
and ingests via the STRATIX pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from layerlens.instrument.adapters.langfuse.client import LangfuseAPIClient, LangfuseAPIError
from layerlens.instrument.adapters.langfuse.config import SyncDirection, SyncResult, SyncState
from layerlens.instrument.adapters.langfuse.mapper import LangfuseToSTRATIXMapper

logger = logging.getLogger(__name__)


class TraceImporter:
    """
    Import pipeline for Langfuse -> STRATIX.

    Steps:
    1. List traces from Langfuse (with filters)
    2. Fetch full trace with observations
    3. Map to STRATIX events
    4. Deduplicate against previously imported traces
    5. Ingest via STRATIX emit or pipeline
    """

    def __init__(
        self,
        client: LangfuseAPIClient,
        state: SyncState,
    ) -> None:
        self._client = client
        self._state = state
        self._mapper = LangfuseToSTRATIXMapper()

    def import_traces(
        self,
        stratix: Any | None = None,
        since: datetime | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> SyncResult:
        """
        Import traces from Langfuse.

        Args:
            stratix: STRATIX instance for event emission (or pipeline).
            since: Only import traces after this timestamp.
            tags: Filter by trace tags.
            limit: Max number of traces to import.
            dry_run: If True, count but don't actually import.

        Returns:
            SyncResult with import statistics.
        """
        result = SyncResult(direction=SyncDirection.IMPORT, dry_run=dry_run)

        # Fetch trace list
        try:
            traces = self._client.get_all_traces(
                tags=tags,
                from_timestamp=since,
            )
        except LangfuseAPIError as e:
            result.errors.append(f"Failed to list traces: {e}")
            result.failed_count = 1
            return result

        if limit:
            traces = traces[:limit]

        for trace_summary in traces:
            trace_id = trace_summary.get("id", "")

            # Skip quarantined traces
            if self._state.is_quarantined(trace_id):
                result.quarantined_count += 1
                continue

            # Dedup: skip already imported (unless updated_at is newer)
            if trace_id in self._state.imported_trace_ids:
                result.skipped_count += 1
                continue

            # Skip traces exported by STRATIX (loop prevention)
            trace_tags = trace_summary.get("tags", []) or []
            if "stratix-exported" in trace_tags:
                result.skipped_count += 1
                continue

            if dry_run:
                result.imported_count += 1
                continue

            # Fetch full trace
            try:
                full_trace = self._client.get_trace(trace_id)
            except LangfuseAPIError as e:
                logger.warning("Failed to fetch trace %s: %s", trace_id, e)
                is_quarantined = self._state.record_failure(trace_id)
                if is_quarantined:
                    result.quarantined_count += 1
                result.failed_count += 1
                result.errors.append(f"Trace {trace_id}: {e}")
                continue

            # Map to STRATIX events
            try:
                events = self._mapper.map_trace(full_trace)
            except Exception as e:
                logger.warning("Failed to map trace %s: %s", trace_id, e)
                is_quarantined = self._state.record_failure(trace_id)
                if is_quarantined:
                    result.quarantined_count += 1
                result.failed_count += 1
                result.errors.append(f"Trace {trace_id} mapping: {e}")
                continue

            if not events:
                result.skipped_count += 1
                continue

            # Ingest events
            try:
                self._ingest_events(events, stratix)
            except Exception as e:
                logger.warning("Failed to ingest trace %s: %s", trace_id, e)
                is_quarantined = self._state.record_failure(trace_id)
                if is_quarantined:
                    result.quarantined_count += 1
                result.failed_count += 1
                result.errors.append(f"Trace {trace_id} ingestion: {e}")
                continue

            # Record success
            updated_at = self._parse_timestamp(
                full_trace.get("updatedAt", full_trace.get("timestamp"))
            )
            self._state.record_import(trace_id, updated_at)
            result.imported_count += 1

        return result

    def _ingest_events(
        self,
        events: list[dict[str, Any]],
        stratix: Any | None,
    ) -> None:
        """Ingest mapped events via STRATIX emit or pipeline."""
        if stratix is None or not bool(stratix):
            return

        for event in events:
            event_type = event.get("event_type", "")
            payload = event.get("payload", {})
            stratix.emit(event_type, payload)

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        """Parse a timestamp string to datetime, or return now."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        return datetime.now(timezone.utc)
