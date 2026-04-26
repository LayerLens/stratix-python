"""
AgentForce Adapter

BaseAdapter-compliant wrapper for AgentForce trace import.
Provides lifecycle management, circuit breaker protection,
CaptureConfig filtering, and health reporting.
"""

from __future__ import annotations

import uuid
import logging
from typing import Any

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
from layerlens.instrument.adapters.frameworks.agentforce.auth import (
    SalesforceAuthError,
    SalesforceConnection,
    SalesforceCredentials,
)
from layerlens.instrument.adapters.frameworks.agentforce.importer import ImportResult, AgentForceImporter
from layerlens.instrument.adapters.frameworks.agentforce.normalizer import AgentForceNormalizer

logger = logging.getLogger(__name__)


class AgentForceAdapter(BaseAdapter):
    """
    BaseAdapter wrapper for AgentForce trace import.

    Provides the standard LayerLens adapter lifecycle
    (connect / disconnect / health_check) around the AgentForce importer,
    routing imported events through the BaseAdapter circuit breaker and
    CaptureConfig pipeline.

    Usage::

        adapter = AgentForceAdapter(stratix=stratix, credentials=credentials)
        adapter.connect()
        result = adapter.import_sessions(start_date="2026-02-21")
        adapter.disconnect()
    """

    FRAMEWORK = "salesforce_agentforce"
    VERSION = "0.1.0"
    # ``frameworks/agentforce/models.py`` line 17 imports
    # ``from pydantic import Field, BaseModel`` only — both names exist
    # identically under v1 and v2. No v2-only decorators
    # (field_validator/model_validator) appear anywhere in the
    # agentforce subpackage. Salesforce Agentforce itself is a remote
    # REST API, not a Python library, so there is no framework-side
    # Pydantic dependency to constrain.
    requires_pydantic = PydanticCompat.V1_OR_V2

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
        credentials: SalesforceCredentials | None = None,
        connection: SalesforceConnection | None = None,
        batch_size: int = 200,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)
        self._credentials = credentials
        self._connection = connection
        self._normalizer = AgentForceNormalizer()
        self._importer: AgentForceImporter | None = None
        self._batch_size = batch_size

    def connect(self) -> None:
        """Authenticate with Salesforce and prepare the importer."""
        if self._connection is None:
            if self._credentials is None:
                raise SalesforceAuthError("Either 'credentials' or 'connection' must be provided")
            self._connection = SalesforceConnection(credentials=self._credentials)

        if self._credentials and self._credentials.is_expired:
            self._connection.authenticate()

        self._importer = AgentForceImporter(
            connection=self._connection,
            normalizer=self._normalizer,
            batch_size=self._batch_size,
        )

        self._connected = True
        self._status = AdapterStatus.HEALTHY
        logger.info("AgentForce adapter connected")

    def disconnect(self) -> None:
        """Disconnect and release resources."""
        self._importer = None
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        logger.info("AgentForce adapter disconnected")

    def health_check(self) -> AdapterHealth:
        """Return adapter health, including Salesforce connection status."""
        message = None
        if self._connection and self._credentials and self._credentials.is_expired:
            message = "Salesforce token expired, will re-authenticate on next operation"

        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            adapter_version=self.VERSION,
            message=message,
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="AgentForceAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            capabilities=[
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_TOOLS,
            ],
            description="LayerLens adapter for Salesforce AgentForce trace import",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="AgentForceAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={
                "capture_config": self._capture_config.model_dump(),
            },
        )

    def import_sessions(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        agent_type: str | None = None,
        channel_type: str | None = None,
        limit: int | None = None,
        last_import_timestamp: str | None = None,
    ) -> ImportResult:
        """
        Import AgentForce sessions and emit events through the adapter pipeline.

        Events are routed through ``emit_dict_event()`` for circuit breaker
        and CaptureConfig protection.

        Returns:
            ImportResult summary.
        """
        if not self._connected or not self._importer:
            raise RuntimeError("Adapter not connected. Call connect() first.")

        events, result = self._importer.import_sessions(
            start_date=start_date,
            end_date=end_date,
            agent_type=agent_type,
            channel_type=channel_type,
            limit=limit,
            last_import_timestamp=last_import_timestamp,
        )

        # Route each event through BaseAdapter pipeline
        emitted = 0
        for event in events:
            event_type = event.get("event_type", "")
            payload = event.get("payload", {})
            # Add identity and timestamp to payload for downstream consumers
            if "identity" in event:
                payload["_identity"] = event["identity"]
            if "timestamp" in event:
                payload["_timestamp"] = event["timestamp"]

            self.emit_dict_event(event_type, payload)
            emitted += 1

        result.events_generated = emitted
        return result
