"""LayerLens Instrument layer.

The ``instrument`` package houses framework, protocol, and LLM provider
adapters plus their shared base classes, registry, capture configuration,
and event-sink abstractions. Adapter code lives under
``layerlens.instrument.adapters``.

Importing ``layerlens.instrument`` MUST NOT import any optional adapter
dependency (langchain, crewai, anthropic, etc.). Adapter modules are
lazy-loaded from the registry the first time their framework is requested.

Convenience re-exports of the most commonly used base-layer types are
provided here so the typical adapter user can write::

    from layerlens.instrument import (
        BaseAdapter,
        AdapterRegistry,
        CaptureConfig,
    )

These are pure Python classes with only ``pydantic`` (already required)
as a dependency.
"""

from __future__ import annotations

from layerlens.instrument.adapters._base import (
    EventSink,
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    CaptureConfig,
    AdapterRegistry,
    ReplayableTrace,
    AdapterCapability,
)

__all__ = [
    "AdapterCapability",
    "AdapterHealth",
    "AdapterInfo",
    "AdapterRegistry",
    "AdapterStatus",
    "BaseAdapter",
    "CaptureConfig",
    "EventSink",
    "ReplayableTrace",
]
