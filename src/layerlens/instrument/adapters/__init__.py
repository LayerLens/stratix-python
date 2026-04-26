"""Adapter implementations and the shared base layer.

The ``_base`` subpackage contains the abstract :class:`BaseAdapter`,
:class:`AdapterRegistry`, :class:`CaptureConfig`, and :class:`EventSink`
classes that every concrete adapter depends on. Concrete adapters live
under ``frameworks/`` (LangChain, LangGraph, etc.), ``protocols/`` (A2A,
AGUI, MCP, etc.), and ``providers/`` (OpenAI, Anthropic, etc.).

The base layer has no optional dependencies — it works with only the
SDK's core ``pydantic`` requirement. Concrete adapters declare their own
optional ``[project.optional-dependencies]`` groups in ``pyproject.toml``.
"""

from __future__ import annotations

from layerlens.instrument.adapters._base import (
    EventSink,
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    CaptureConfig,
    TraceStoreSink,
    AdapterRegistry,
    ReplayableTrace,
    AdapterCapability,
    IngestionPipelineSink,
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
    "IngestionPipelineSink",
    "ReplayableTrace",
    "TraceStoreSink",
]
