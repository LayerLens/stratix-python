from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import field, dataclass


@dataclass
class SpanData:
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "ok"
    kind: str = "internal"
    input: Any = None
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[SpanData] = field(default_factory=list)

    def finish(self, error: Optional[str] = None) -> None:
        self.end_time = time.time()
        if error is not None:
            self.error = error
            self.status = "error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "kind": self.kind,
            "input": self.input,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }
