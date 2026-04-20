"""Helpers for tracing MCP App (interactive UI component) invocations.

MCP Apps are UI surfaces that a server exposes as callable "components"
— forms, confirmation dialogs, pickers. We hash their parameter / result
payloads so telemetry is deterministic without shipping user data, and
normalize the free-form type / result strings so dashboards aggregate
cleanly.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any, Dict, Optional

COMPONENT_TYPES = frozenset({"form", "confirmation", "picker", "custom"})
INTERACTION_RESULTS = frozenset({"submitted", "cancelled", "timeout"})


def _sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_parameters(parameters: Optional[Dict[str, Any]]) -> str:
    return _sha256(parameters or {})


def hash_result(result: Optional[Dict[str, Any]]) -> Optional[str]:
    if result is None:
        return None
    return _sha256(result)


def normalize_component_type(component_type: str) -> str:
    ct = (component_type or "").lower().strip()
    return ct if ct in COMPONENT_TYPES else "custom"


def normalize_interaction_result(result: str) -> str:
    r = (result or "").lower().strip()
    return r if r in INTERACTION_RESULTS else "submitted"


def build_invocation_payload(
    *,
    app_id: str,
    component_type: str,
    parameters: Optional[Dict[str, Any]] = None,
    server_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Canonical payload for an ``mcp.app.invoked`` event."""
    return {
        "app_id": app_id,
        "component_type": normalize_component_type(component_type),
        "parameters_hash": hash_parameters(parameters),
        "server_name": server_name,
    }


def build_interaction_payload(
    *,
    app_id: str,
    interaction_result: str,
    result: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Canonical payload for an ``mcp.app.interaction`` event."""
    payload: Dict[str, Any] = {
        "app_id": app_id,
        "interaction_result": normalize_interaction_result(interaction_result),
    }
    h = hash_result(result)
    if h is not None:
        payload["result_hash"] = h
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    return payload
