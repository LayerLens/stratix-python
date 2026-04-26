"""
ACP-Origin Pattern Normalizer

Detects and normalizes IBM Agent Communication Protocol (ACP) payloads
within A2A requests. ACP merged into A2A in August 2025; this normalizer
handles the legacy ACP structures by mapping them to A2A canonical format.

Detection uses a two-factor check:
1. Presence of X-ACP-Version HTTP header
2. Top-level 'acp' namespace key in the JSON-RPC payload

ACP-to-A2A field mapping:
  task_run.id          → task.id
  task_run.input.messages → task.history
  task_run.output.artifacts → task.artifacts
  task_run.status      → task.status.state  (running → working)
  task_run.metadata    → task.metadata
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ACP status → A2A status mapping
_ACP_STATUS_MAP: dict[str, str] = {
    "running": "working",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "pending": "submitted",
    "input_required": "input_required",
}


class ACPNormalizer:
    """
    Normalizes ACP-origin payloads into A2A canonical structures.

    Thread-safe, stateless normalizer. Can be shared across requests.
    """

    def detect_acp_origin(
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> bool:
        """
        Check if a payload originates from an ACP agent.

        Args:
            payload: JSON-RPC request payload.
            headers: HTTP headers (optional).

        Returns:
            True if ACP-origin indicators are detected.
        """
        # Factor 1: X-ACP-Version header
        if headers and ("X-ACP-Version" in headers or "x-acp-version" in headers):
            return True

        # Factor 2: 'acp' namespace in payload
        if "acp" in payload:
            return True

        # Factor 3: task_run structure (ACP-specific naming)
        params = payload.get("params", payload)
        return "task_run" in params

    def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize an ACP payload to A2A format.

        Args:
            payload: ACP-origin payload.

        Returns:
            Normalized payload in A2A canonical format.
        """
        result = dict(payload)

        # Extract params (JSON-RPC wrapping)
        params = result.get("params", result)

        # Normalize task_run → task
        if "task_run" in params:
            task_run = params.pop("task_run")
            task = self._normalize_task_run(task_run)
            params["task"] = task
            if "params" in result:
                result["params"] = params

        # Normalize ACP namespace metadata
        if "acp" in result:
            acp_meta = result.pop("acp")
            if "version" in acp_meta:
                result.setdefault("metadata", {})["acp_version"] = acp_meta["version"]

        return result

    def _normalize_task_run(self, task_run: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize an ACP task_run structure to A2A task format.

        Args:
            task_run: ACP task_run dict.

        Returns:
            A2A task dict.
        """
        task: dict[str, Any] = {}

        # task_run.id → task.id
        task["id"] = task_run.get("id", "")

        # task_run.input.messages → task.history
        input_data = task_run.get("input", {})
        if "messages" in input_data:
            task["history"] = input_data["messages"]

        # task_run.output.artifacts → task.artifacts
        output_data = task_run.get("output", {})
        if "artifacts" in output_data:
            task["artifacts"] = output_data["artifacts"]

        # task_run.status → task.status.state (with mapping)
        acp_status = task_run.get("status", "")
        if isinstance(acp_status, str):
            a2a_status = _ACP_STATUS_MAP.get(acp_status, acp_status)
            task["status"] = {"state": a2a_status}
        elif isinstance(acp_status, dict):
            state = acp_status.get("state", acp_status.get("status", ""))
            task["status"] = {"state": _ACP_STATUS_MAP.get(state, state)}  # type: ignore[arg-type]

        # task_run.metadata → task.metadata
        if "metadata" in task_run:
            task["metadata"] = task_run["metadata"]

        return task

    def detect_and_normalize(
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """
        Detect ACP origin and normalize if detected.

        Args:
            payload: Request payload.
            headers: HTTP headers.

        Returns:
            Tuple of (normalized_payload, is_acp).
        """
        is_acp = self.detect_acp_origin(payload, headers)
        if is_acp:
            normalized = self.normalize(payload)
            logger.debug("Normalized ACP-origin payload to A2A format")
            return normalized, True
        return payload, False
