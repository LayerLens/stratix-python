"""
MCP App Invocation Handler

Captures MCP App (interactive UI component) invocations. MCP Apps
are UI components that can be invoked as tools — forms, confirmation
dialogs, pickers, etc.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


# Known MCP App component types
COMPONENT_TYPES = frozenset({"form", "confirmation", "picker", "custom"})

# Known interaction results
INTERACTION_RESULTS = frozenset({"submitted", "cancelled", "timeout"})


def hash_parameters(parameters: dict[str, Any] | None) -> str:
    """Hash MCP App invocation parameters."""
    import json
    params_str = json.dumps(parameters or {}, sort_keys=True, default=str)
    h = hashlib.sha256(params_str.encode()).hexdigest()
    return f"sha256:{h}"


def hash_result(result: dict[str, Any] | None) -> str | None:
    """Hash MCP App interaction result. Returns None if no result."""
    if result is None:
        return None
    import json
    result_str = json.dumps(result, sort_keys=True, default=str)
    h = hashlib.sha256(result_str.encode()).hexdigest()
    return f"sha256:{h}"


def normalize_component_type(component_type: str) -> str:
    """Normalize a component type string to a known type."""
    ct = component_type.lower().strip()
    if ct in COMPONENT_TYPES:
        return ct
    return "custom"


def normalize_interaction_result(result: str) -> str:
    """Normalize an interaction result string."""
    r = result.lower().strip()
    if r in INTERACTION_RESULTS:
        return r
    return "submitted"
