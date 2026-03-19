"""
AgentForce Trace Importer

Imports AgentForce Session Tracing data from Salesforce Data Cloud
and normalizes it to STRATIX canonical events.

Supports:
- Batch import (date range filter)
- Incremental import (timestamp-based)
- Session, participant, interaction, step, and message extraction
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from layerlens.instrument.adapters.agentforce.auth import SalesforceConnection, SalesforceQueryError
from layerlens.instrument.adapters.agentforce.normalizer import AgentForceNormalizer

# Regex for validating ISO 8601 date strings (YYYY-MM-DD)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Regex for validating ISO 8601 timestamp strings
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
# Regex for Salesforce record IDs (15 or 18 char alphanumeric)
_SFID_RE = re.compile(r"^[a-zA-Z0-9]{15,18}$")

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Result of an AgentForce import operation."""
    sessions_imported: int = 0
    participants_imported: int = 0
    interactions_imported: int = 0
    steps_imported: int = 0
    messages_imported: int = 0
    events_generated: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_records(self) -> int:
        return (
            self.sessions_imported
            + self.participants_imported
            + self.interactions_imported
            + self.steps_imported
            + self.messages_imported
        )


class AgentForceImporter:
    """
    Import AgentForce traces from Salesforce Data Cloud.

    Usage:
        connection = SalesforceConnection(credentials)
        connection.authenticate()
        importer = AgentForceImporter(connection)
        events, result = importer.import_sessions(
            start_date="2026-02-21",
            end_date="2026-02-28",
        )
    """

    def __init__(
        self,
        connection: SalesforceConnection,
        normalizer: AgentForceNormalizer | None = None,
        batch_size: int = 200,
    ):
        self._connection = connection
        self._normalizer = normalizer or AgentForceNormalizer()
        self._batch_size = batch_size

    def import_sessions(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        agent_type: str | None = None,
        channel_type: str | None = None,
        limit: int | None = None,
        last_import_timestamp: str | None = None,
    ) -> tuple[list[dict[str, Any]], ImportResult]:
        """
        Import AgentForce sessions and all related records.

        Args:
            start_date: Import sessions starting from this date (ISO 8601)
            end_date: Import sessions up to this date (ISO 8601)
            agent_type: Filter by agent type (Employee, EinsteinSDR, EinsteinServiceAgent)
            channel_type: Filter by channel type
            limit: Maximum sessions to import
            last_import_timestamp: For incremental sync, only import after this timestamp

        Returns:
            Tuple of (list of STRATIX events, ImportResult summary)
        """
        result = ImportResult()
        all_events: list[dict[str, Any]] = []

        # Build session query with validated parameters
        conditions = []
        if start_date:
            self._validate_date(start_date)
            conditions.append(f"StartTimestamp >= {start_date}T00:00:00Z")
        if end_date:
            self._validate_date(end_date)
            conditions.append(f"StartTimestamp <= {end_date}T23:59:59Z")
        if last_import_timestamp:
            self._validate_timestamp(last_import_timestamp)
            conditions.append(f"StartTimestamp > {last_import_timestamp}")

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f" LIMIT {limit}" if limit else f" LIMIT {self._batch_size}"

        soql = (
            "SELECT Id, StartTimestamp, EndTimestamp, AiAgentChannelTypeId, "
            "AiAgentSessionEndType, VoiceCallId, MessagingSessionId, PreviousSessionId "
            f"FROM AIAgentSession{where} ORDER BY StartTimestamp ASC{limit_clause}"
        )

        try:
            sessions = self._connection.query(soql)
        except Exception as e:
            result.errors.append(f"Session query failed: {e}")
            return all_events, result

        if not sessions:
            return all_events, result

        session_ids = [s["Id"] for s in sessions]
        result.sessions_imported = len(sessions)

        # Normalize sessions
        for session in sessions:
            events = self._normalizer.normalize_session(session)
            all_events.extend(events)

        # Import participants
        participants = self._query_related(
            "AIAgentSessionParticipant",
            "AiAgentSessionId",
            session_ids,
            "Id, AiAgentSessionId, AiAgentTypeId, AiAgentApiName, "
            "AiAgentVersionApiName, ParticipantId, AiAgentSessionParticipantRoleId",
            result=result,
        )
        result.participants_imported = len(participants)
        for p in participants:
            all_events.append(self._normalizer.normalize_participant(p))

        # Import interactions
        interactions = self._query_related(
            "AIAgentInteraction",
            "AiAgentSessionId",
            session_ids,
            "Id, AiAgentSessionId, AiAgentInteractionTypeId, "
            "TelemetryTraceId, TelemetryTraceSpanId, TopicApiName, "
            "AttributeText, PrevInteractionId",
            order_by="Id ASC",
            result=result,
        )
        result.interactions_imported = len(interactions)
        for i in interactions:
            all_events.append(self._normalizer.normalize_interaction(i))

        if interactions:
            interaction_ids = [i["Id"] for i in interactions]

            # Import steps
            steps = self._query_related(
                "AIAgentInteractionStep",
                "AiAgentInteractionId",
                interaction_ids,
                "Id, AiAgentInteractionId, AiAgentInteractionStepTypeId, "
                "InputValueText, OutputValueText, ErrorMessageText, "
                "GenerationId, GenAiGatewayRequestId, GenAiGatewayResponseId, "
                "Name, TelemetryTraceSpanId",
                order_by="Id ASC",
                result=result,
            )
            result.steps_imported = len(steps)
            for s in steps:
                all_events.append(self._normalizer.normalize_step(s))

            # Import messages
            messages = self._query_related(
                "AIAgentInteractionMessage",
                "AiAgentInteractionId",
                interaction_ids,
                "Id, AiAgentInteractionId, AiAgentInteractionMessageTypeId, "
                "ContentText, AiAgentInteractionMsgContentTypeId, "
                "MessageSentTimestamp, ParentMessageId",
                order_by="MessageSentTimestamp ASC",
                result=result,
            )
            result.messages_imported = len(messages)
            for m in messages:
                all_events.append(self._normalizer.normalize_message(m))

        result.events_generated = len(all_events)
        logger.info(
            f"AgentForce import complete: {result.sessions_imported} sessions, "
            f"{result.events_generated} events generated"
        )
        return all_events, result

    def _query_related(
        self,
        object_name: str,
        foreign_key: str,
        parent_ids: list[str],
        fields: str,
        order_by: str | None = None,
        result: ImportResult | None = None,
    ) -> list[dict[str, Any]]:
        """Query related records in batches to respect SOQL limits."""
        all_records: list[dict[str, Any]] = []

        # Batch parent IDs to avoid SOQL IN clause limits
        for i in range(0, len(parent_ids), self._batch_size):
            batch = parent_ids[i:i + self._batch_size]
            # Escape IDs to prevent SOQL injection
            safe_ids = [self._escape_soql_id(pid) for pid in batch]
            ids_str = "', '".join(safe_ids)
            soql = f"SELECT {fields} FROM {object_name} WHERE {foreign_key} IN ('{ids_str}')"
            if order_by:
                soql += f" ORDER BY {order_by}"

            try:
                records = self._connection.query(soql)
                all_records.extend(records)
            except Exception as e:
                error_msg = f"Failed to query {object_name}: {e}"
                logger.error(error_msg)
                if result is not None:
                    result.errors.append(error_msg)

        return all_records

    @staticmethod
    def _validate_date(value: str) -> None:
        """Validate an ISO 8601 date string (YYYY-MM-DD)."""
        if not _DATE_RE.match(value):
            raise ValueError(
                f"Invalid date format: '{value}'. Expected YYYY-MM-DD."
            )

    @staticmethod
    def _validate_timestamp(value: str) -> None:
        """Validate an ISO 8601 timestamp string."""
        if not _TIMESTAMP_RE.match(value):
            raise ValueError(
                f"Invalid timestamp format: '{value}'. Expected ISO 8601."
            )

    @staticmethod
    def _escape_soql_id(value: str) -> str:
        """Escape a Salesforce ID for safe SOQL inclusion."""
        # Strip any SOQL special characters (single quotes, backslashes)
        return value.replace("\\", "").replace("'", "")
