from __future__ import annotations

from ._w3c import inject_headers, extract_headers, new_traceparent
from ._emit import emit
from ._span import span
from ._collector import TraceCollector
from ._decorator import trace
from .adapters._base import AdapterInfo, BaseAdapter
from ._capture_config import CaptureConfig
from .adapters._registry import auto, discover_installed
from ._context_propagation import trace_context, get_trace_context

__all__ = [
    "AdapterInfo",
    "BaseAdapter",
    "CaptureConfig",
    "TraceCollector",
    "auto",
    "discover_installed",
    "emit",
    "extract_headers",
    "get_trace_context",
    "inject_headers",
    "new_traceparent",
    "span",
    "trace",
    "trace_context",
]
