"""Live seed + read-back helpers for the graph-contract harness (G7).

Seeds a lane's sealed events through a REAL ``Stratix()`` client to a running
atlas (``LAYERLENS_STRATIX_BASE_URL``) and reads the trace back.

Read-back note (the part code-reading never tells you): the SDK ``Trace`` model
does NOT expose the server-computed ``graph``/``agent`` fields — they are omitted
by the client schema. So the read-back goes through the RAW trace-detail HTTP
API, where the server returns ``graph`` + ``agent`` at the top level of ``data``.
"""

from __future__ import annotations

import os
import json
import time
import tempfile
from typing import Any, Dict, List, Tuple, Optional

import httpx

from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig


def seed_lane(client: Any, events: List[Dict[str, Any]], *, trace_id: Optional[str] = None) -> str:
    """Replay a lane's sealed events through a real collector + upload; return the
    server trace id. Re-emitting pre-captured events (not running an adapter) so
    ``to_replay_dict`` is the correct, faithful payload here."""
    collector = TraceCollector(client, CaptureConfig.standard())
    for e in events:
        collector.emit(e["event_type"], dict(e.get("payload", {})), span_id=e.get("span_id"))
    payload = collector.to_replay_dict()
    if trace_id is not None:
        payload["trace_id"] = trace_id
    fd, path = tempfile.mkstemp(suffix=".json", prefix="graph_contract_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump([payload], fh, default=str)
        result = client.traces.upload(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    trace_ids = getattr(result, "trace_ids", None) if result is not None else None
    return trace_ids[0] if trace_ids else str(payload.get("trace_id"))


def read_back_graph(
    client: Any, base_url: str, api_key: str, trace_id: str, *, retries: int = 10
) -> Tuple[Optional[dict], Optional[str]]:
    """RAW trace-detail GET -> (graph, agent). Polls for async indexing."""
    org = client.organization_id
    proj = client.project_id
    url = f"{base_url.rstrip('/')}/organizations/{org}/projects/{proj}/traces/{trace_id}"
    last: Optional[httpx.Response] = None
    for _ in range(retries):
        last = httpx.get(url, headers={"x-api-key": api_key}, timeout=30)
        if last.status_code == 200:
            body = last.json()
            data = body.get("data", body) if isinstance(body, dict) else {}
            return data.get("graph"), data.get("agent")
        time.sleep(1)
    if last is not None:
        last.raise_for_status()
    raise RuntimeError(f"trace {trace_id} never became readable")
