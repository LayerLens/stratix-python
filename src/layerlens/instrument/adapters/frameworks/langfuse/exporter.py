"""
Langfuse Trace Exporter

Reverse-maps STRATIX events to Langfuse traces and pushes them via the API.
"""

from __future__ import annotations

import uuid
import logging
from typing import Any
from datetime import datetime, timezone

UTC = timezone.utc  # Python 3.11+ has datetime.UTC; alias for 3.9/3.10 compat.

from layerlens.instrument.adapters.frameworks.langfuse.client import LangfuseAPIError, LangfuseAPIClient
from layerlens.instrument.adapters.frameworks.langfuse.config import SyncState, SyncResult, SyncDirection
from layerlens.instrument.adapters.frameworks.langfuse.mapper import LayerLensToLangfuseMapper

logger = logging.getLogger(__name__)


class TraceExporter:
    """
    Export pipeline for STRATIX -> Langfuse.

    Steps:
    1. Group STRATIX events by trace ID
    2. Reverse-map to Langfuse trace + observations
    3. Create trace and observations via Langfuse API
    4. Tag with 'stratix-exported' to prevent re-import
    """

    def __init__(
        self,
        client: LangfuseAPIClient,
        state: SyncState,
    ) -> None:
        self._client = client
        self._state = state
        self._mapper = LayerLensToLangfuseMapper()

    def export_traces(
        self,
        events_by_trace: dict[str, list[dict[str, Any]]],
        trace_ids: list[str] | None = None,
        dry_run: bool = False,
    ) -> SyncResult:
        """
        Export STRATIX traces to Langfuse.

        Args:
            events_by_trace: Dict mapping trace_id -> list of STRATIX event dicts.
            trace_ids: Optional filter — only export these trace IDs.
            dry_run: If True, count but don't actually export.

        Returns:
            SyncResult with export statistics.
        """
        result = SyncResult(direction=SyncDirection.EXPORT, dry_run=dry_run)

        ids_to_export = trace_ids or list(events_by_trace.keys())

        for trace_id in ids_to_export:
            events = events_by_trace.get(trace_id, [])
            if not events:
                result.skipped_count += 1
                continue

            # Loop prevention: skip traces that were imported from Langfuse
            if trace_id in self._state.imported_trace_ids:
                result.skipped_count += 1
                continue

            # Skip already exported
            if trace_id in self._state.exported_trace_ids:
                result.skipped_count += 1
                continue

            if dry_run:
                result.exported_count += 1
                continue

            # Map STRATIX events to Langfuse structure
            try:
                langfuse_data = self._mapper.map_events_to_trace(events, trace_id=trace_id)
            except Exception as e:
                logger.warning("Failed to map trace %s for export: %s", trace_id, e)
                result.failed_count += 1
                result.errors.append(f"Trace {trace_id} mapping: {e}")
                continue

            # Push to Langfuse
            try:
                self._push_to_langfuse(langfuse_data)
            except LangfuseAPIError as e:
                logger.warning("Failed to export trace %s: %s", trace_id, e)
                result.failed_count += 1
                result.errors.append(f"Trace {trace_id} export: {e}")
                continue

            # Record success
            self._state.record_export(trace_id, datetime.now(UTC))
            result.exported_count += 1

        return result

    def _push_to_langfuse(self, langfuse_data: dict[str, Any]) -> None:
        """Push a mapped trace + observations to Langfuse via batch ingestion."""
        trace_body = langfuse_data.get("trace", {})
        observations = langfuse_data.get("observations", [])

        # Build batch events
        batch: list[dict[str, Any]] = []
        now = datetime.now(UTC).isoformat()

        # Trace create event
        batch.append(
            {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "timestamp": now,
                "body": trace_body,
            }
        )

        # Observation create events
        for obs in observations:
            obs_type = obs.get("type", "SPAN").upper()
            event_type = "generation-create" if obs_type == "GENERATION" else "span-create"

            batch.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": event_type,
                    "timestamp": now,
                    "body": obs,
                }
            )

        self._client.ingestion_batch(batch)
