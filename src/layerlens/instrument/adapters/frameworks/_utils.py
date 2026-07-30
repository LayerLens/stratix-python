"""Shared utilities for framework adapters.

Centralises helpers that were previously copy-pasted across adapter
files: serialisation, span ID generation, and text truncation.
"""

from __future__ import annotations

import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Span IDs
# ---------------------------------------------------------------------------


def new_span_id() -> str:
    """Generate a short random span identifier."""
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def safe_serialize(value: Any) -> Any:
    """Best-effort conversion of *value* into a JSON-friendly form.

    Handles Pydantic models (``model_dump``), objects with ``to_dict``,
    dicts, lists/tuples, and falls back to ``str()``.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [safe_serialize(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): safe_serialize(v) for k, v in value.items()}
    return str(value)


# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------


def truncate(text: Any, max_len: int = 2000) -> Any:
    """Truncate *text* to *max_len* characters, appending ``'...'``.

    Returns *None* unchanged.  Non-string values are stringified first.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
