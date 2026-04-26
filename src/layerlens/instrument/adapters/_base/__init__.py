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
from layerlens.instrument.adapters._base.handoff import (
    DEFAULT_PREVIEW_MAX_CHARS,
    HandoffMetadata,
    HandoffSequencer,
    make_preview,
    compute_context_hash,
    build_handoff_payload,
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
    "DEFAULT_PREVIEW_MAX_CHARS",
    "EventSink",
    "HandoffMetadata",
    "HandoffSequencer",
    "IngestionPipelineSink",
    "PydanticCompat",
    "ReplayableTrace",
    "TraceStoreSink",
    "build_handoff_payload",
    "compute_context_hash",
    "make_preview",
    "requires_pydantic",
]
