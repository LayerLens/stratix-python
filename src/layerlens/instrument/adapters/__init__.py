"""
STRATIX Framework Adapters

Adapters for integrating STRATIX with various AI agent frameworks.
"""

from layerlens.instrument.adapters._base import (
    AdapterCapability,
    AdapterHealth,
    AdapterInfo,
    AdapterStatus,
    BaseAdapter,
    ReplayableTrace,
)
from layerlens.instrument.adapters._capture import CaptureConfig
from layerlens.instrument.adapters._registry import AdapterRegistry
from layerlens.instrument.adapters._sinks import (
    APIUploadSink,
    EventSink,
    LoggingSink,
)
from layerlens.instrument.adapters._trace_container import SerializedTrace

__all__ = [
    "BaseAdapter",
    "AdapterStatus",
    "AdapterHealth",
    "AdapterCapability",
    "AdapterInfo",
    "ReplayableTrace",
    "CaptureConfig",
    "AdapterRegistry",
    "SerializedTrace",
    "EventSink",
    "APIUploadSink",
    "LoggingSink",
]
