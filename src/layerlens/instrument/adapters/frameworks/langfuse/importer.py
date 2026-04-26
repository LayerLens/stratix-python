"""Langfuse Trace Importer.

Fetches traces from Langfuse, maps them to STRATIX events, deduplicates,
and ingests via the STRATIX pipeline.

Hardened in PR ``feat/instrument-importer-hardening``:

* Trace IDs are validated against
  :data:`layerlens.instrument.adapters._base.importer.ID_PATTERN_UUID`
  before being interpolated into the per-trace fetch URL — Langfuse
  trace IDs are UUIDs, and an upstream-supplied non-UUID could
  otherwise be relayed verbatim into the URL path.
* Per-trace fetches and per-trace ingest calls are wrapped in
  :func:`layerlens.instrument.adapters._base.importer.retry_with_backoff`
  so transient HTTP failures recover cleanly without quarantining the
  trace prematurely.
"""

from __future__ import annotations

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
from layerlens.instrument.adapters.frameworks.langfuse.mapper import LangfuseToLayerLensMapper

logger = logging.getLogger(__name__)


class TraceImporter:
    """Import pipeline for Langfuse → STRATIX.

    Steps:

    1. List traces from Langfuse (with filters), via the client's
       :meth:`get_all_traces` (paginated through the shared
       :func:`paginate` helper).
    2. Validate each trace ID against
       :data:`ID_PATTERN_UUID`.
    3. Fetch the full trace with observations (retry on transient
       failure).
    4. Map to STRATIX events.
    5. Deduplicate against previously-imported traces.
    6. Ingest via STRATIX emit (retry on transient failure).
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
        self._mapper = LangfuseToLayerLensMapper()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    def import_traces(
        self,
        stratix: Any | None = None,
        since: datetime | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> SyncResult:
        """Import traces from Langfuse.

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

        # Fetch trace list (no retry here — get_all_traces internally
        # uses the rate-limit-aware client._request which already
        # retries). A LangfuseAPIError surfaced here is terminal.
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

            # Reject malformed trace IDs before they reach the upstream
            # URL builder. Langfuse uses UUIDs; anything else is either
            # corruption or an injection attempt.
            if not validate_id(trace_id, ID_PATTERN_UUID):
                logger.warning(
                    "Langfuse importer: skipping trace with invalid id %r", trace_id
                )
                result.skipped_count += 1
                result.errors.append(f"Invalid trace id: {trace_id!r}")
                continue

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

            # Fetch full trace with retry/backoff (handles transient 5xx).
            try:
                full_trace = self._fetch_trace_with_retry(trace_id)
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

    def _fetch_trace_with_retry(self, trace_id: str) -> dict[str, Any]:
        """Fetch a single trace under :func:`retry_with_backoff`.

        The client's ``get_trace`` already retries via the shared
        retry helper for HTTP-level failures (429/5xx). We add a
        second retry layer here so a brief network blip on a single
        trace doesn't quarantine it after one attempt — quarantine
        should be reserved for traces that are *truly* unfetchable.
        """

        def _do() -> dict[str, Any]:
            try:
                return self._client.get_trace(trace_id)
            except LangfuseAPIError as exc:
                # Translate LangfuseAPIError into RetryableHTTPError ONLY
                # when the status is transient (network or 5xx). 4xx is
                # terminal — re-raise unchanged.
                status = exc.status_code
                if status is None or status == 429 or (500 <= status < 600):
                    raise RetryableHTTPError(
                        str(exc), status_code=status, rate_limit=None
                    ) from exc
                raise

        try:
            return retry_with_backoff(
                _do,
                max_retries=self._max_retries,
                base_delay=self._base_delay,
                max_delay=self._max_delay,
            )
        except RetryableHTTPError as exc:
            # Exhausted — surface as LangfuseAPIError for caller parity.
            raise LangfuseAPIError(str(exc), status_code=exc.status_code) from exc

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
        return datetime.now(UTC)
