"""
MCP Elicitation Handler

Handles MCP Elicitation extension events — server-initiated user input
requests and user responses. Manages the request/response event pair
with privacy-preserving hashing.
"""

from __future__ import annotations

import time
import uuid
import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ElicitationTracker:
    """
    Tracks active MCP elicitation interactions.

    Manages the lifecycle of elicitation request/response pairs,
    computing timing and generating unique identifiers.
    """

    def __init__(self) -> None:
        self._active: dict[str, float] = {}  # elicitation_id → start_time

    def start_request(
        self,
        server_name: str,
        schema: dict[str, Any] | None = None,
        title: str | None = None,
        elicitation_id: str | None = None,
    ) -> str:
        """
        Record the start of an elicitation request.

        Args:
            server_name: MCP server name.
            schema: JSON Schema for the requested input.
            title: Human-readable title.
            elicitation_id: Optional pre-assigned ID.

        Returns:
            The elicitation ID.
        """
        eid = elicitation_id or str(uuid.uuid4())
        self._active[eid] = time.monotonic()
        return eid

    def complete_response(
        self,
        elicitation_id: str,
        action: str,
        response: Any = None,
    ) -> float | None:
        """
        Record the completion of an elicitation response.

        Args:
            elicitation_id: The elicitation ID.
            action: User action (submit | cancel).
            response: User response (will be hashed, not stored in cleartext).

        Returns:
            Latency in milliseconds, or None if request not tracked.
        """
        start = self._active.pop(elicitation_id, None)
        if start is not None:
            return (time.monotonic() - start) * 1000
        return None

    def is_active(self, elicitation_id: str) -> bool:
        """Check if an elicitation is still awaiting response."""
        return elicitation_id in self._active

    @property
    def active_count(self) -> int:
        return len(self._active)

    def hash_response(self, response: Any) -> str:
        """Hash a user response for privacy-preserving storage."""
        response_str = str(response or "")
        h = hashlib.sha256(response_str.encode()).hexdigest()
        return f"sha256:{h}"

    def hash_schema(self, schema: dict[str, Any] | None) -> str:
        """Hash a request schema."""
        import json

        schema_str = json.dumps(schema or {}, sort_keys=True)
        h = hashlib.sha256(schema_str.encode()).hexdigest()
        return f"sha256:{h}"
