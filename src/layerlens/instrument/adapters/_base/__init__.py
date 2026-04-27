"""Shared base layer for all LayerLens adapters.

Re-exports the public surface so adapter modules and external callers
import from a single, stable path::

    from layerlens.instrument.adapters._base import BaseAdapter, CaptureConfig
"""

from __future__ import annotations

from layerlens.instrument.adapters._base.sinks import (
    EventSink,
    TraceStoreSink,
    IngestionPipelineSink,
)
from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.capture import (
    ALWAYS_ENABLED_EVENT_TYPES,
    CaptureConfig,
)
from layerlens.instrument.adapters._base.registry import AdapterRegistry
from layerlens.instrument.adapters._base.truncation import (
    DROP,
    DEFAULT_POLICY,
    DEFAULT_FIELD_CAPS,
    FieldTruncationPolicy,
    truncate_field,
    truncate_payload,
)
from layerlens.instrument.adapters._base.pydantic_compat import (
    PydanticCompat,
    requires_pydantic,
)

__all__ = [
    "ALWAYS_ENABLED_EVENT_TYPES",
    "AdapterCapability",
    "AdapterHealth",
    "AdapterInfo",
    "AdapterRegistry",
    "AdapterStatus",
    "BaseAdapter",
    "CaptureConfig",
    "DEFAULT_FIELD_CAPS",
    "DEFAULT_POLICY",
    "DROP",
    "EventSink",
    "FieldTruncationPolicy",
    "IngestionPipelineSink",
    "PydanticCompat",
    "ReplayableTrace",
    "TraceStoreSink",
    "requires_pydantic",
    "truncate_field",
    "truncate_payload",
]
