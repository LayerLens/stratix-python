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

__all__ = [
    "AdapterInfo",
    "BaseAdapter",
    "DEFAULT_FAILURE_THRESHOLD",
    "HealthStatus",
    "ResilienceTracker",
    "get_default_for",
    "resilient_callback",
]
