"""Shared base infrastructure for all instrument adapters.

This package re-exports the public surface of the base adapter contract
(``AdapterInfo``, ``BaseAdapter``) and the resilience helpers used by
framework adapters to wrap callbacks in try/except boundaries so that
observability code never breaks the user's framework.
"""

from __future__ import annotations

from ._core import AdapterInfo, BaseAdapter
from .resilience import (
    DEFAULT_FAILURE_THRESHOLD,
    HealthStatus,
    ResilienceTracker,
    get_default_for,
    resilient_callback,
)
from .state_filters import (
    DEFAULT_PII_EXCLUDE_KEYS,
    REDACTED_PLACEHOLDER,
    StateFilter,
    default_state_filter,
    filter_payload_fields,
    filter_state,
)

__all__ = [
    "AdapterInfo",
    "BaseAdapter",
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_PII_EXCLUDE_KEYS",
    "HealthStatus",
    "REDACTED_PLACEHOLDER",
    "ResilienceTracker",
    "StateFilter",
    "default_state_filter",
    "filter_payload_fields",
    "filter_state",
    "get_default_for",
    "resilient_callback",
]
