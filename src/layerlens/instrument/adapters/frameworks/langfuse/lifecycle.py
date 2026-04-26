"""
Langfuse Adapter Lifecycle

Main LangfuseAdapter class extending BaseAdapter.
Manages connection, health, import/export, and sync operations.
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any
from datetime import datetime, timezone

UTC = timezone.utc  # Python 3.11+ has datetime.UTC; alias for 3.9/3.10 compat.

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat
from layerlens.instrument.adapters.frameworks.langfuse.sync import BidirectionalSync
from layerlens.instrument.adapters.frameworks.langfuse.client import LangfuseAPIError, LangfuseAPIClient
from layerlens.instrument.adapters.frameworks.langfuse.config import (
    SyncState,
    SyncResult,
    SyncDirection,
    LangfuseConfig,
)
from layerlens.instrument.adapters.frameworks.langfuse.exporter import TraceExporter
from layerlens.instrument.adapters.frameworks.langfuse.importer import TraceImporter

logger = logging.getLogger(__name__)


class LangfuseAdapter(BaseAdapter):
    """
    LayerLens adapter for Langfuse integration.

    Unlike other adapters that wrap running code in real-time, the Langfuse
    adapter is a data import/export pipeline that communicates with a remote
    Langfuse HTTP API to pull/push traces in batch.
    """

    FRAMEWORK = "langfuse"
    VERSION = "0.1.0"
    # The adapter's own config layer
    # (``frameworks/langfuse/config.py`` line 13) imports
    # ``from pydantic import field_validator`` — a v2-only decorator.
    # Pydantic v1 has ``validator``; ``field_validator`` was added in v2
    # (see pydantic v2 migration guide). Importing this adapter under v1
    # raises ``ImportError`` in config.py.
    requires_pydantic = PydanticCompat.V2_ONLY

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
        config: LangfuseConfig | None = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)
        self._config: LangfuseConfig | None = config
        self._client: LangfuseAPIClient | None = None
        self._sync_state = SyncState()
        self._importer: TraceImporter | None = None
        self._exporter: TraceExporter | None = None
        self._sync: BidirectionalSync | None = None
        self._last_health_check: datetime | None = None
        self._langfuse_healthy = False

    # --- BaseAdapter abstract methods ---

    def connect(self, config: LangfuseConfig | None = None) -> None:
        """
        Connect to the Langfuse API.

        Creates the HTTP client and validates credentials with a health check.
        """
        if config:
            self._config = config

        if self._config is None:
            # Connect without a config — adapter is usable but not connected to Langfuse
            self._connected = True
            self._status = AdapterStatus.HEALTHY
            return

        self._client = LangfuseAPIClient(
            public_key=self._config.public_key,
            secret_key=self._config.secret_key,
            host=self._config.host,
            max_retries=self._config.max_retries,
        )

        # Validate credentials
        try:
            self._client.health_check()
            self._langfuse_healthy = True
        except LangfuseAPIError as e:
            logger.warning("Langfuse health check failed: %s", e)
            self._langfuse_healthy = False

        # Initialize sub-components
        self._importer = TraceImporter(self._client, self._sync_state)
        self._exporter = TraceExporter(self._client, self._sync_state)
        self._sync = BidirectionalSync(
            importer=self._importer,
            exporter=self._exporter,
            state=self._sync_state,
        )

        self._connected = True
        self._status = AdapterStatus.HEALTHY if self._langfuse_healthy else AdapterStatus.DEGRADED
        self._last_health_check = datetime.now(UTC)

    def disconnect(self) -> None:
        """Disconnect from Langfuse."""
        self._client = None
        self._importer = None
        self._exporter = None
        self._sync = None
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        self._langfuse_healthy = False

    def health_check(self) -> AdapterHealth:
        """Return health status including Langfuse API reachability."""
        message = None
        if self._client and self._connected:
            try:
                self._client.health_check()
                self._langfuse_healthy = True
                message = "Langfuse API reachable"
            except LangfuseAPIError as e:
                self._langfuse_healthy = False
                message = f"Langfuse API unreachable: {e}"
                self._status = AdapterStatus.DEGRADED
        elif not self._config:
            message = "No Langfuse config — adapter connected without remote API"
        else:
            message = "Not connected"

        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            framework_version=None,
            adapter_version=self.VERSION,
            message=message,
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )

    def get_adapter_info(self) -> AdapterInfo:
        """Return metadata about this adapter."""
        return AdapterInfo(
            name="LangfuseAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=None,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.REPLAY,
            ],
            author="LayerLens Team",
            description="Bidirectional trace sync between LayerLens and Langfuse",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        """Serialize accumulated trace events for replay."""
        return ReplayableTrace(
            adapter_name="LangfuseAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            config=self._config.model_dump() if self._config else {},
            metadata={
                "sync_state": {
                    "imported": len(self._sync_state.imported_trace_ids),
                    "exported": len(self._sync_state.exported_trace_ids),
                    "quarantined": len(self._sync_state.quarantined_trace_ids),
                },
            },
        )

    # --- Import/Export/Sync API ---

    def import_traces(
        self,
        since: datetime | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> SyncResult:
        """
        Import traces from Langfuse into LayerLens.

        Args:
            since: Only import traces updated after this timestamp.
            tags: Filter by Langfuse trace tags.
            limit: Maximum number of traces to import.
            dry_run: If True, report what would be imported without importing.

        Returns:
            SyncResult with import statistics.
        """
        if self._importer is None:
            return SyncResult(
                direction=SyncDirection.IMPORT,
                errors=["Adapter not connected to Langfuse API"],
            )

        start_time = time.monotonic()
        effective_since = since or (self._config.since if self._config else None)
        effective_tags = tags or (self._config.tag_filter if self._config else None)

        result = self._importer.import_traces(
            stratix=self._stratix,
            since=effective_since,
            tags=effective_tags,
            limit=limit,
            dry_run=dry_run,
        )
        result.duration_ms = (time.monotonic() - start_time) * 1000
        return result

    def export_traces(
        self,
        events_by_trace: dict[str, list[dict[str, Any]]] | None = None,
        trace_ids: list[str] | None = None,
        dry_run: bool = False,
    ) -> SyncResult:
        """
        Export LayerLens traces to Langfuse.

        Args:
            events_by_trace: Dict mapping trace_id -> list of LayerLens event dicts.
            trace_ids: List of trace IDs to export (requires events_by_trace).
            dry_run: If True, report what would be exported without exporting.

        Returns:
            SyncResult with export statistics.
        """
        if self._exporter is None:
            return SyncResult(
                direction=SyncDirection.EXPORT,
                errors=["Adapter not connected to Langfuse API"],
            )

        start_time = time.monotonic()
        result = self._exporter.export_traces(
            events_by_trace=events_by_trace or {},
            trace_ids=trace_ids,
            dry_run=dry_run,
        )
        result.duration_ms = (time.monotonic() - start_time) * 1000
        return result

    def sync(
        self,
        direction: SyncDirection | None = None,
        since: datetime | None = None,
        dry_run: bool = False,
        events_by_trace: dict[str, list[dict[str, Any]]] | None = None,
    ) -> SyncResult:
        """
        Run a sync cycle in the configured direction.

        Args:
            direction: Override the configured sync direction.
            since: Override the since timestamp.
            dry_run: If True, report what would be synced without making changes.
            events_by_trace: Required for export/bidirectional — LayerLens events to export.

        Returns:
            SyncResult with combined statistics.
        """
        if self._sync is None:
            return SyncResult(
                direction=direction or SyncDirection.IMPORT,
                errors=["Adapter not connected to Langfuse API"],
            )

        effective_direction = direction or (
            self._config.mode if self._config else SyncDirection.IMPORT
        )
        start_time = time.monotonic()

        result = self._sync.run(
            stratix=self._stratix,
            direction=effective_direction,
            since=since,
            dry_run=dry_run,
            events_by_trace=events_by_trace or {},
            tags=self._config.tag_filter if self._config else None,
        )
        result.duration_ms = (time.monotonic() - start_time) * 1000
        return result

    # --- State access ---

    @property
    def sync_state(self) -> SyncState:
        """Return the current sync state."""
        return self._sync_state

    @property
    def config(self) -> LangfuseConfig | None:
        """Return the current configuration."""
        return self._config

    def get_status(self) -> dict[str, Any]:
        """Return a status summary for CLI/API use."""
        return {
            "connected": self._connected,
            "langfuse_healthy": self._langfuse_healthy,
            "host": self._config.host if self._config else None,
            "mode": self._config.mode.value if self._config else None,
            "imported_traces": len(self._sync_state.imported_trace_ids),
            "exported_traces": len(self._sync_state.exported_trace_ids),
            "quarantined_traces": len(self._sync_state.quarantined_trace_ids),
            "last_import_cursor": (
                self._sync_state.last_import_cursor.isoformat()
                if self._sync_state.last_import_cursor
                else None
            ),
            "last_export_cursor": (
                self._sync_state.last_export_cursor.isoformat()
                if self._sync_state.last_export_cursor
                else None
            ),
        }
