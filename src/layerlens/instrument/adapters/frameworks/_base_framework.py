from __future__ import annotations

import uuid
from uuid import UUID
from typing import Any, Dict, Optional, Tuple

from .._base import AdapterInfo, BaseAdapter
from ..._capture_config import CaptureConfig
from ..._collector import TraceCollector


class FrameworkTracer(BaseAdapter):
    """Base class for framework adapters that manage their own collector.

    Framework adapters (LangChain, LangGraph, etc.) receive callbacks
    from the framework rather than wrapping SDK methods. They maintain
    their own TraceCollector and map framework run_ids to span_ids.
    """

    _adapter_name: str = "framework"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        self._client: Any = None
        self._config = capture_config or CaptureConfig.standard()
        self._collector: Optional[TraceCollector] = None
        self._span_ids: Dict[str, str] = {}
        self._root_run_id: Optional[str] = None
        self.connect(client)

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target
        return target

    def disconnect(self) -> None:
        self._span_ids.clear()
        self._root_run_id = None
        self._collector = None

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name=self._adapter_name,
            adapter_type="framework",
            connected=self._client is not None,
        )

    def _ensure_collector(self) -> TraceCollector:
        if self._collector is None:
            self._collector = TraceCollector(self._client, self._config)
        return self._collector

    def _get_or_create_span_id(
        self, run_id: UUID, parent_run_id: Optional[UUID] = None
    ) -> Tuple[str, Optional[str]]:
        rid = str(run_id)
        if rid not in self._span_ids:
            self._span_ids[rid] = uuid.uuid4().hex[:16]
        span_id = self._span_ids[rid]
        parent_span_id = self._span_ids.get(str(parent_run_id)) if parent_run_id else None
        if self._root_run_id is None:
            self._root_run_id = rid
        return span_id, parent_span_id

    def _emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
    ) -> None:
        collector = self._ensure_collector()
        span_id, parent_span_id = self._get_or_create_span_id(run_id, parent_run_id)
        collector.emit(event_type, payload, span_id=span_id, parent_span_id=parent_span_id)

    def _maybe_flush(self, run_id: UUID) -> None:
        if str(run_id) == self._root_run_id and self._collector is not None:
            self._collector.flush()
            self._span_ids.clear()
            self._root_run_id = None
            self._collector = None
