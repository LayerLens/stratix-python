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
from layerlens.instrument.adapters._base.errors import (
    MAX_MESSAGE_CHARS,
    SAFE_CONTEXT_KEYS,
    DEFAULT_EVENT_TYPE,
    MAX_TRACEBACK_CHARS,
    MAX_TRACEBACK_FRAMES,
    emit_error_event,
    build_error_payload,
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
    "DEFAULT_EVENT_TYPE",
    "EventSink",
    "IngestionPipelineSink",
    "MAX_MESSAGE_CHARS",
    "MAX_TRACEBACK_CHARS",
    "MAX_TRACEBACK_FRAMES",
    "PydanticCompat",
    "ReplayableTrace",
    "SAFE_CONTEXT_KEYS",
    "TraceStoreSink",
    "build_error_payload",
    "emit_error_event",
    "requires_pydantic",
]
