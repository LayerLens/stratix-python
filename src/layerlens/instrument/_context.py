from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from contextvars import ContextVar

if TYPE_CHECKING:
    from ._types import SpanData
    from ._recorder import TraceRecorder

_current_recorder: ContextVar[Optional[TraceRecorder]] = ContextVar("_current_recorder", default=None)
_current_span: ContextVar[Optional[SpanData]] = ContextVar("_current_span", default=None)
