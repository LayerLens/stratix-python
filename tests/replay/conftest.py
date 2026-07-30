from __future__ import annotations

from typing import Any, Dict, List

import pytest

from layerlens.models.trace import Trace


def make_trace(
    trace_id: str = "t1",
    *,
    output: str = "hello",
    events: List[Dict[str, Any]] | None = None,
    latency_ms: float | None = None,
) -> Trace:
    data: Dict[str, Any] = {"output": output, "events": events or []}
    if latency_ms is not None:
        data["latency_ms"] = latency_ms
    return Trace(
        id=trace_id,
        organization_id="org",
        project_id="proj",
        created_at="2026-04-20T00:00:00Z",
        filename=f"{trace_id}.json",
        data=data,
    )


@pytest.fixture
def make_trace_factory():
    return make_trace
