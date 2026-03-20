from __future__ import annotations

from typing import Any, Optional

from ._types import SpanData
from ._upload import upload_trace, async_upload_trace


class TraceRecorder:
    def __init__(self, client: Any) -> None:
        self._client = client
        self.root: Optional[SpanData] = None

    def flush(self) -> None:
        if self.root is None:
            return
        upload_trace(self._client, self.root.to_dict())

    async def async_flush(self) -> None:
        if self.root is None:
            return
        await async_upload_trace(self._client, self.root.to_dict())
