"""Track MCP elicitation request/response pairs.

``ElicitationTracker`` pairs up server-initiated ``elicit`` requests with
their user responses, preserving latency and privacy-preserving hashes so
the MCP adapter can emit ``mcp.elicitation`` events with per-request IDs
instead of treating each call as a one-off.
"""

from __future__ import annotations

import json
import time
import uuid
import hashlib
import logging
from typing import Any, Optional

log = logging.getLogger(__name__)


class ElicitationTracker:
    """Pairs MCP elicit request/response events and reports latency."""

    def __init__(self) -> None:
        self._active: dict[str, float] = {}

    def start_request(
        self,
        server_name: str,  # noqa: ARG002 — accepted for parity / future use
        schema: Optional[dict[str, Any]] = None,  # noqa: ARG002
        title: Optional[str] = None,  # noqa: ARG002
        elicitation_id: Optional[str] = None,
    ) -> str:
        eid = elicitation_id or uuid.uuid4().hex
        self._active[eid] = time.monotonic()
        return eid

    def complete_response(
        self,
        elicitation_id: str,
        action: str,  # noqa: ARG002 — accepted for downstream payload construction
        response: Any = None,  # noqa: ARG002
    ) -> Optional[float]:
        """Return elapsed ms from start_request, or None if the ID wasn't tracked."""
        start = self._active.pop(elicitation_id, None)
        if start is None:
            return None
        return (time.monotonic() - start) * 1000

    def is_active(self, elicitation_id: str) -> bool:
        return elicitation_id in self._active

    @property
    def active_count(self) -> int:
        return len(self._active)

    @staticmethod
    def hash_response(response: Any) -> str:
        return "sha256:" + hashlib.sha256(str(response or "").encode()).hexdigest()

    @staticmethod
    def hash_schema(schema: Optional[dict[str, Any]]) -> str:
        return "sha256:" + hashlib.sha256(json.dumps(schema or {}, sort_keys=True).encode()).hexdigest()
