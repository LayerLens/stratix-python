"""
Langfuse Bidirectional Sync

Coordinates import and export with cursor tracking and conflict resolution.
"""

from __future__ import annotations

import logging
from typing import Any
from datetime import datetime

from layerlens.instrument.adapters.frameworks.langfuse.config import SyncState, SyncResult, SyncDirection
from layerlens.instrument.adapters.frameworks.langfuse.exporter import TraceExporter
from layerlens.instrument.adapters.frameworks.langfuse.importer import TraceImporter

logger = logging.getLogger(__name__)


class BidirectionalSync:
    """
    Orchestrates bidirectional sync between Langfuse and STRATIX.

    Uses cursor-based incremental sync to minimize API calls.
    """

    def __init__(
        self,
        importer: TraceImporter,
        exporter: TraceExporter,
        state: SyncState,
    ) -> None:
        self._importer = importer
        self._exporter = exporter
        self._state = state

    def run(
        self,
        stratix: Any | None = None,
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
        since: datetime | None = None,
        dry_run: bool = False,
        events_by_trace: dict[str, list[dict[str, Any]]] | None = None,
        tags: list[str] | None = None,
    ) -> SyncResult:
        """
        Run a sync cycle.

        Args:
            stratix: STRATIX instance for event emission.
            direction: Sync direction (import, export, or bidirectional).
            since: Override since timestamp.
            dry_run: If True, report what would happen without making changes.
            events_by_trace: STRATIX events for export (required for export/bidirectional).
            tags: Filter tags for import.

        Returns:
            Combined SyncResult.
        """
        result = SyncResult(direction=direction, dry_run=dry_run)

        # Import phase
        if direction in (SyncDirection.IMPORT, SyncDirection.BIDIRECTIONAL):
            effective_since = since or self._state.last_import_cursor
            import_result = self._importer.import_traces(
                stratix=stratix,
                since=effective_since,
                tags=tags,
                dry_run=dry_run,
            )
            result.imported_count = import_result.imported_count
            result.skipped_count += import_result.skipped_count
            result.failed_count += import_result.failed_count
            result.quarantined_count += import_result.quarantined_count
            result.errors.extend(import_result.errors)

        # Export phase
        if direction in (SyncDirection.EXPORT, SyncDirection.BIDIRECTIONAL):  # noqa: SIM102
            if events_by_trace:
                export_result = self._exporter.export_traces(
                    events_by_trace=events_by_trace,
                    dry_run=dry_run,
                )
                result.exported_count = export_result.exported_count
                result.skipped_count += export_result.skipped_count
                result.failed_count += export_result.failed_count
                result.errors.extend(export_result.errors)

        return result
