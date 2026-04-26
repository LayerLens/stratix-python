"""Langfuse Trace Exporter.

Reverse-maps STRATIX events to Langfuse traces and pushes them via the
API.

Hardened in PR ``feat/instrument-importer-hardening``:

* Exported batches go through
  :func:`layerlens.instrument.adapters._base.importer.retry_with_backoff`
  with rate-limit-aware delays, so a transient 429 / 5xx during push
  doesn't lose the trace.
* Trace IDs are validated against
  :data:`layerlens.instrument.adapters._base.importer.ID_PATTERN_UUID`
  before being sent to Langfuse — keeps the payload well-formed and
  prevents accidental relay of corrupt IDs.
"""

from __future__ import annotations

import uuid
import logging
from typing import Any
from datetime import datetime, timezone

UTC = timezone.utc  # Python 3.11+ has datetime.UTC; alias for 3.9/3.10 compat.

from layerlens.instrument.adapters._base.importer import (
    ID_PATTERN_UUID,
    RetryableHTTPError,
    validate_id,
    retry_with_backoff,
)
from layerlens.instrument.adapters.frameworks.langfuse.client import LangfuseAPIError, LangfuseAPIClient
from layerlens.instrument.adapters.frameworks.langfuse.config import SyncState, SyncResult, SyncDirection
from layerlens.instrument.adapters.frameworks.langfuse.mapper import LayerLensToLangfuseMapper

logger = logging.getLogger(__name__)


class TraceExporter:
    """Export pipeline for STRATIX → Langfuse.

    Steps:

    1. Group STRATIX events by trace ID.
    2. Validate each trace ID.
    3. Reverse-map to Langfuse trace + observations.
    4. Push the batch via the Langfuse ingestion API under
       :func:`retry_with_backoff` so a transient 429 / 5xx doesn't lose
       the trace.
    5. Tag with ``stratix-exported`` to prevent re-import.
    """

    def __init__(
        self,
        client: LangfuseAPIClient,
        state: SyncState,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 16.0,
    ) -> None:
        self._client = client
        self._state = state
        self._mapper = LayerLensToLangfuseMapper()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    def export_traces(
        self,
        events_by_trace: dict[str, list[dict[str, Any]]],
        trace_ids: list[str] | None = None,
        dry_run: bool = False,
    ) -> SyncResult:
        """Export STRATIX traces to Langfuse.

        Args:
            events_by_trace: Dict mapping trace_id → list of STRATIX
                event dicts.
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

            # Validate trace id BEFORE sending to upstream.
            if not validate_id(trace_id, ID_PATTERN_UUID):
                logger.warning(
                    "Langfuse exporter: skipping trace with invalid id %r", trace_id
                )
                result.skipped_count += 1
                result.errors.append(f"Invalid trace id: {trace_id!r}")
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

            # Push to Langfuse with retry
            try:
                self._push_with_retry(langfuse_data)
            except LangfuseAPIError as e:
                logger.warning("Failed to export trace %s: %s", trace_id, e)
                result.failed_count += 1
                result.errors.append(f"Trace {trace_id} export: {e}")
                continue

            # Record success
            self._state.record_export(trace_id, datetime.now(UTC))
            result.exported_count += 1

        return result

    def _push_with_retry(self, langfuse_data: dict[str, Any]) -> None:
        """Push the Langfuse batch under :func:`retry_with_backoff`."""

        def _do() -> None:
            try:
                self._push_to_langfuse(langfuse_data)
            except LangfuseAPIError as exc:
                status = exc.status_code
                if status is None or status == 429 or (500 <= status < 600):
                    raise RetryableHTTPError(
                        str(exc), status_code=status, rate_limit=None
                    ) from exc
                raise

        try:
            retry_with_backoff(
                _do,
                max_retries=self._max_retries,
                base_delay=self._base_delay,
                max_delay=self._max_delay,
            )
        except RetryableHTTPError as exc:
            raise LangfuseAPIError(str(exc), status_code=exc.status_code) from exc

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
