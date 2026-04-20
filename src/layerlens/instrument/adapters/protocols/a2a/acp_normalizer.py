"""Normalize legacy ACP (IBM Agent Communication Protocol) payloads to A2A.

ACP merged into A2A in 2025. Servers that still emit ACP-shaped payloads
can be detected via the ``X-ACP-Version`` header or a top-level ``acp``
namespace; this helper detects and rewrites those payloads into the A2A
canonical shape so the A2A adapter can treat the two origins uniformly.

Mapping:
  ``task_run.id``              → ``task.id``
  ``task_run.input.messages``  → ``task.history``
  ``task_run.output.artifacts``→ ``task.artifacts``
  ``task_run.status``          → ``task.status.state`` (``running`` → ``working``)
  ``task_run.metadata``        → ``task.metadata``
"""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)


_ACP_STATUS_MAP: dict[str, str] = {
    "running": "working",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "pending": "submitted",
    "input_required": "input_required",
}


class ACPNormalizer:
    """Detects and rewrites ACP-origin payloads into A2A canonical form."""

    def detect_acp_origin(
        self,
        payload: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> bool:
        if headers and ("X-ACP-Version" in headers or "x-acp-version" in headers):
            return True
        if "acp" in payload:
            return True
        params = payload.get("params", payload)
        return isinstance(params, dict) and "task_run" in params

    def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        params = result.get("params", result)
        if isinstance(params, dict) and "task_run" in params:
            task_run = params.pop("task_run")
            params["task"] = self._normalize_task_run(task_run)
            if "params" in result:
                result["params"] = params

        if "acp" in result:
            acp_meta = result.pop("acp")
            if isinstance(acp_meta, dict) and "version" in acp_meta:
                result.setdefault("metadata", {})["acp_version"] = acp_meta["version"]
        return result

    def detect_and_normalize(
        self,
        payload: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> tuple[dict[str, Any], bool]:
        if self.detect_acp_origin(payload, headers):
            return self.normalize(payload), True
        return payload, False

    def _normalize_task_run(self, task_run: dict[str, Any]) -> dict[str, Any]:
        task: dict[str, Any] = {"id": task_run.get("id", "")}

        input_data = task_run.get("input", {})
        if isinstance(input_data, dict) and "messages" in input_data:
            task["history"] = input_data["messages"]

        output_data = task_run.get("output", {})
        if isinstance(output_data, dict) and "artifacts" in output_data:
            task["artifacts"] = output_data["artifacts"]

        status = task_run.get("status", "")
        if isinstance(status, str):
            task["status"] = {"state": _ACP_STATUS_MAP.get(status, status)}
        elif isinstance(status, dict):
            state = status.get("state", status.get("status", ""))
            task["status"] = {"state": _ACP_STATUS_MAP.get(state, state)}

        if "metadata" in task_run:
            task["metadata"] = task_run["metadata"]
        return task
