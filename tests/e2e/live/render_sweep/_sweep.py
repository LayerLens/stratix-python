"""Read-back + full render-contract assertion for the customer-run sweep.

``read_back_detail`` fetches the raw trace-detail the FE renders from; the SDK
``Trace`` model omits the server-computed ``graph``/``agent``/``framework`` fields,
so the read-back goes through the RAW HTTP API (as graph_contract does). The
inner ``data.data.status`` is the byte-faithful uploaded status (the Status
column); the top-level ``framework``/``graph``/``agent`` are server-computed.

``check_render`` applies the per-kind render contract (W1 brief §3):

* multi-agent (>=2 expected nodes)  -> server node-set == expected, >=1 edge
  (a real DAG, not disconnected nodes), Agent column == 'multi-agent';
* single-agent (1 expected node)    -> server node-set == expected, Agent == id;
* empty-state (0 expected nodes)    -> no stored graph, Agent in (None, '');
* every kind                        -> Framework column filled, Status column
  filled, a populated waterfall (>=1 event) — providers render an honest
  empty-state + waterfall, never a fabricated DAG.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from ._render_oracle import kind_of, expected_agent_column


def read_back_detail(client: Any, base_url: str, api_key: str, trace_id: str, *, retries: int = 12) -> Dict[str, Any]:
    """RAW trace-detail GET -> the server ``data`` object. Polls for async
    indexing (the trace becomes readable a beat after upload)."""
    org = client.organization_id
    proj = client.project_id
    url = f"{base_url.rstrip('/')}/organizations/{org}/projects/{proj}/traces/{trace_id}"
    last: Optional[httpx.Response] = None
    for _ in range(retries):
        last = httpx.get(url, headers={"x-api-key": api_key}, timeout=30)
        if last.status_code == 200:
            body = last.json()
            return body.get("data", body) if isinstance(body, dict) else {}
        time.sleep(1)
    if last is not None:
        last.raise_for_status()
    raise RuntimeError(f"trace {trace_id} never became readable")


def server_node_ids(graph: Optional[dict]) -> List[str]:
    if not graph:
        return []
    return sorted((n.get("agent_id") or n.get("id") or "") for n in graph.get("nodes", []))


def check_render(detail: Dict[str, Any], expected_nodes: List[str]) -> List[str]:
    """Return a list of render-contract violations (empty == renders correctly)."""
    problems: List[str] = []
    graph = detail.get("graph") or {}
    nodes = server_node_ids(graph)
    edges = graph.get("edges") or []
    agent = detail.get("agent")
    framework = detail.get("framework")
    inner = detail.get("data") or {}
    status = inner.get("status") or detail.get("status")
    event_count = detail.get("event_count") or len(inner.get("events") or [])
    kind = kind_of(expected_nodes)

    # Core: the server's node-set matches what the fixture genuinely declares.
    if nodes != sorted(expected_nodes):
        problems.append(f"node-set {nodes} != fixture-derived {sorted(expected_nodes)}")

    # Agent column agrees with the graph projection.
    exp_agent = expected_agent_column(expected_nodes)
    if exp_agent is None:
        if agent not in (None, ""):
            problems.append(f"empty-state Agent column should be blank, got {agent!r}")
    elif agent != exp_agent:
        problems.append(f"Agent column {agent!r} != {exp_agent!r}")

    # A genuine multi-agent DAG has edges (>=2 nodes with none would be broken).
    if kind == "multi-agent" and len(edges) < 1:
        problems.append(f"multi-agent graph has {len(nodes)} nodes but NO edges (not a DAG)")

    # Empty-state must NOT carry a stored graph (FE shows the empty-state).
    if kind == "empty-state" and nodes:
        problems.append(f"empty-state trace unexpectedly stored a graph: {nodes}")

    # Every kind fills Framework + Status and has a populated waterfall.
    if not framework:
        problems.append("Framework column empty")
    if not status:
        problems.append("Status column empty")
    if not event_count or event_count < 1:
        problems.append("waterfall empty (no events)")

    return problems
