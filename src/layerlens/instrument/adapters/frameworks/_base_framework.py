from __future__ import annotations

from uuid import UUID
from typing import Any, Dict, Optional

from .._base import AdapterInfo, BaseAdapter
from ..._types import SpanData
from ..._upload import upload_trace


class FrameworkTracer(BaseAdapter):
    """Base class for framework adapters that manage their own span tree.

    Provides run_id-based span tracking, parent-child linking, and
    automatic trace upload when the root span finishes.
    """

    _adapter_name: str = "framework"

    def __init__(self, client: Any) -> None:
        self._client: Any = None
        self._spans: Dict[str, SpanData] = {}
        self._root_run_id: Optional[str] = None
        self.connect(client)

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target
        return target

    def disconnect(self) -> None:
        self._spans.clear()
        self._root_run_id = None

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name=self._adapter_name,
            adapter_type="framework",
            connected=self._client is not None,
        )

    def _get_or_create_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        kind: str,
        input: Any = None,
    ) -> SpanData:
        rid = str(run_id)
        if rid in self._spans:
            return self._spans[rid]

        parent_span: Optional[SpanData] = None
        if parent_run_id is not None:
            parent_span = self._spans.get(str(parent_run_id))

        s = SpanData(
            name=name,
            kind=kind,
            parent_id=parent_span.span_id if parent_span else None,
            input=input,
        )
        self._spans[rid] = s

        if parent_span is not None:
            parent_span.children.append(s)

        if self._root_run_id is None:
            self._root_run_id = rid

        return s

    def _finish_span(self, run_id: UUID, output: Any = None, error: Optional[str] = None) -> None:
        rid = str(run_id)
        s = self._spans.get(rid)
        if s is None:
            return
        s.output = output
        s.finish(error=error)

        if rid == self._root_run_id:
            self._flush()

    def _flush(self) -> None:
        if self._root_run_id is None:
            return
        root = self._spans.get(self._root_run_id)
        if root is None:
            return

        upload_trace(self._client, root.to_dict())

        self._spans.clear()
        self._root_run_id = None
