"""Track MCP elicitation request/response pairs.

``ElicitationTracker`` pairs up server-initiated ``elicit`` requests with
their user responses, preserving latency and privacy-preserving hashes so
the MCP adapter can emit ``mcp.elicitation`` events with per-request IDs
instead of treating each call as a one-off.

Consent fidelity (D1): the real ``mcp.types.ElicitResult.action`` is one of
``accept`` / ``decline`` / ``cancel``. A refusal (decline/cancel) MUST be
distinguishable from an accept downstream, and MUST NOT carry a content-derived
hash of a payload the user never submitted (``ElicitResult.content`` is ``None``
for decline/cancel and for URL mode). The tracker therefore only ever hashes the
SUBMITTED form content of an ACCEPTED form-mode response.
"""

from __future__ import annotations

import json
import time
import uuid
import hashlib
import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

#: The real ElicitResult action vocabulary (mcp.types.ElicitResult.action).
ELICIT_ACTIONS = frozenset({"accept", "decline", "cancel"})


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
        action: str,  # noqa: ARG002 — the action is emitted by the caller, not here
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
    def normalize_action(action: Any) -> str:
        """Coerce a result's action onto the real vocabulary, failing CLOSED.

        An unknown / missing action is reported as ``"unknown"`` (NOT silently
        mapped to accept) so a refusal is never mislabelled as consent. The old
        code hardcoded ``"submit"`` — which is not even a real MCP action.
        """
        a = str(action).lower().strip() if action is not None else ""
        return a if a in ELICIT_ACTIONS else "unknown"

    @staticmethod
    def hash_content(content: Any) -> Optional[str]:
        """Hash the SUBMITTED form content (only for an accepted form-mode reply).

        Returns ``None`` when there is no submitted content (decline/cancel, or
        URL mode) — a refused/redirected elicitation hashes NOTHING. The hash is
        itself content-derived, so the emitting adapter gates it under
        ``capture_content=False``; this is a privacy-preserving stand-in only
        when content capture is on.
        """
        if content is None:
            return None
        return "sha256:" + hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def hash_schema(schema: Optional[dict[str, Any]]) -> str:
        return "sha256:" + hashlib.sha256(json.dumps(schema or {}, sort_keys=True).encode()).hexdigest()
