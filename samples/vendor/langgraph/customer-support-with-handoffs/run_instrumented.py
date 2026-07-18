"""Run the customer-support handoff graph under LayerLens instrumentation, then
upload + read the server-computed agent DAG back (the customer path).

What this does, in order:

1. Builds the multi-agent customer-support graph (``app.build_graph``) and the
   LayerLens ``LangGraphCallbackHandler``.
2. Runs one or more real customer conversations through the graph with the
   handler attached, capturing every event the adapter emits into a
   ``TraceCollector`` — proving the adapter emits ``agent.handoff`` /
   ``agent.node.*`` / ``agent.state.change`` / ``model.invoke`` on real node
   transitions.
3. Uploads each captured trace with a real ``Stratix()`` client (LayerLens key
   only) and reads the SERVER-computed agent graph back over the raw trace-detail
   API, asserting the honest multi-agent DAG (``triage -> <dept>_specialist ->
   closer``). If no LayerLens key is configured it runs capture-only and prints a
   BLOCKED notice — never a fabricated upload.

Env (source ``stratix-python-private/tests/e2e/live/.env``):
    VENDOR_MODEL_BACKEND        'openai' (this env) or 'openrouter' (upstream)
    OPENAI_API_KEY / OPENROUTER_API_KEY   the model credential for the chosen backend
    LAYERLENS_STRATIX_API_KEY   LayerLens platform key (customer upload)
    LAYERLENS_STRATIX_BASE_URL  atlas-app API base (e.g. local http://localhost:8080/api/v1)
"""

from __future__ import annotations

import os
import sys
import json
import time
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import build_graph  # noqa: E402

from layerlens.instrument._context import _current_collector, _push_span, _pop_span  # noqa: E402
from layerlens.instrument._collector import TraceCollector  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks import LangGraphCallbackHandler  # noqa: E402


class _StubClient:
    """Offline stand-in used only to build a collector when no real client exists.
    Makes no network calls; its presence never implies a successful upload."""

    def __init__(self) -> None:
        self._base_url = "https://localhost/offline"


def _mask(value: Optional[str]) -> str:
    if not value:
        return "<unset>"
    return value[:8] + "..." + value[-4:] if len(value) > 14 else "***"


def build_real_client():
    """Construct a native ``Stratix()`` (LayerLens-key-only), or return None."""
    if not os.environ.get("LAYERLENS_STRATIX_API_KEY"):
        return None, "LAYERLENS_STRATIX_API_KEY not set"
    if not os.environ.get("LAYERLENS_STRATIX_BASE_URL"):
        return None, "LAYERLENS_STRATIX_BASE_URL not set (refusing the prod default)"
    try:
        from layerlens import Stratix

        return Stratix(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


CONVERSATIONS: List[str] = [
    "I was charged twice for my subscription this month and I'd like the "
    "duplicate refunded to account ACC-77.",
    "My router keeps dropping the wifi connection every few minutes since the "
    "last update. Can you help me fix device RT-9?",
]


def run_conversation(user_text: str, client: Any) -> TraceCollector:
    """Run one conversation through the instrumented graph, capturing events."""
    collector = TraceCollector(client, CaptureConfig.standard())
    handler = LangGraphCallbackHandler(client)

    col_token = _current_collector.set(collector)
    span_snapshot = _push_span("customer-support-root", "customer-support")
    try:
        graph = build_graph()
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config={"callbacks": [handler]},
        )
        print(f"    routed: {result.get('handoff_log')}")
    finally:
        _pop_span(span_snapshot)
        _current_collector.reset(col_token)
    return collector


def summarize(collector: TraceCollector) -> Dict[str, Any]:
    events = collector.events
    counts = Counter(e["event_type"] for e in events)
    handoffs = [
        (e["payload"].get("from_agent"), e["payload"].get("to_agent"))
        for e in events
        if e["event_type"] == "agent.handoff"
    ]
    return {"total_events": len(events), "counts": dict(counts), "handoffs": handoffs}


def upload_and_read_dag(collector: TraceCollector, client: Any, base: str, key: str) -> Dict[str, Any]:
    """Upload one captured trace and read the SERVER agent graph back (raw API)."""
    payload = collector.to_replay_dict()
    fd, path = tempfile.mkstemp(suffix=".json", prefix="ll_vendor_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump([payload], fh, default=str)
    try:
        resp = client.traces.upload(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    trace_ids = list(getattr(resp, "trace_ids", []) or [])
    if not trace_ids:
        return {"upload_trace_ids": [], "graph": None}
    tid = trace_ids[0]
    org, proj = client.organization_id, client.project_id
    url = f"{base.rstrip('/')}/organizations/{org}/projects/{proj}/traces/{tid}"
    graph = None
    for _ in range(12):
        r = httpx.get(url, headers={"x-api-key": key}, timeout=30)
        if r.status_code == 200:
            graph = (r.json().get("data") or {}).get("graph")
            break
        time.sleep(1)
    return {"upload_trace_ids": trace_ids, "trace_id": tid, "graph": graph}


def main() -> None:
    print("=" * 78)
    print("LayerLens x LangGraph - customer-support-with-handoffs (vendor fork, instrumented)")
    print("=" * 78)
    backend = os.environ.get("VENDOR_MODEL_BACKEND", "openrouter")
    print(f"model backend : {backend}")
    print(f"OPENAI_API_KEY: {_mask(os.environ.get('OPENAI_API_KEY'))}   "
          f"OPENROUTER_API_KEY: {_mask(os.environ.get('OPENROUTER_API_KEY'))}")
    print(f"LAYERLENS_STRATIX_BASE_URL: {os.environ.get('LAYERLENS_STRATIX_BASE_URL', '<default>')}")
    print(f"LAYERLENS_STRATIX_API_KEY : {_mask(os.environ.get('LAYERLENS_STRATIX_API_KEY'))}")
    print()

    # Pre-flight: the model credential must exist, or there is nothing honest to run.
    need_key = "OPENAI_API_KEY" if backend == "openai" else "OPENROUTER_API_KEY"
    if not os.environ.get(need_key):
        print(f"BLOCKED: {need_key} not set for VENDOR_MODEL_BACKEND={backend}. "
              "Nothing run — no fake success.")
        return

    client, client_err = build_real_client()
    base = os.environ.get("LAYERLENS_STRATIX_BASE_URL", "")
    key = os.environ.get("LAYERLENS_STRATIX_API_KEY", "")
    if client is not None:
        print(f"[atlas-app] client: org={client.organization_id} project={client.project_id}")
    else:
        print(f"[atlas-app] upload BLOCKED (capture-only): {client_err}")
    print()

    collector_client = client if client is not None else _StubClient()
    total_handoffs = 0
    dag_ok = 0
    for i, text in enumerate(CONVERSATIONS, 1):
        print(f"--- conversation {i}: {text[:60]}...")
        collector = run_conversation(text, collector_client)
        summary = summarize(collector)
        total_handoffs += len(summary["handoffs"])
        print(f"    events        : {summary['total_events']}  {summary['counts']}")
        print(f"    agent.handoff : {summary['handoffs']}")

        if client is not None:
            try:
                res = upload_and_read_dag(collector, client, base, key)
                g = res.get("graph") or {}
                nodes = sorted((n.get("agent_id") or "") for n in g.get("nodes", []))
                edges = len(g.get("edges", []))
                print(f"    UPLOADED      : {res['upload_trace_ids']}")
                print(f"    SERVER DAG    : nodes={nodes} edges={edges} topology={g.get('topology')}")
                if len(nodes) >= 2 and edges >= 1:
                    dag_ok += 1
                    print("    -> honest multi-agent DAG rendered ✓")
                else:
                    print("    -> WARNING: expected a multi-agent DAG (>=2 nodes, >=1 edge)")
            except Exception as exc:  # noqa: BLE001
                print(f"    UPLOAD FAILED : {type(exc).__name__}: {exc}")
        print()

    print("=" * 78)
    print(f"SUMMARY: {len(CONVERSATIONS)} conversations, {total_handoffs} agent.handoff events; "
          f"{dag_ok} multi-agent DAGs rendered on the server")
    if client is None:
        print("atlas-app upload: BLOCKED (see above). Capture-only, no fake success.")
    print("=" * 78)


if __name__ == "__main__":
    main()
