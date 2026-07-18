"""Faithful Python port of atlas ``services.InferAgentGraph`` node-identity
extraction (``apps/backend/services/graph_inference.go``, ``honestGuard=false`` —
the shipped product posture).

Given a trace's raw events, it computes the SET of agent node ids the SERVER
will render, so the render sweep can assert ``server-returned == fixture-derived``
per Family-B sample. This is the deterministic half of the dual oracle; its
fidelity is pinned by ``tests/e2e/test_render_oracle_fidelity.py`` — it must
reproduce every node-set in the live-proven ``graph_contract/oracle_expectations``
before it is trusted for the live sweep.

Why a port (not the SDK ``_identity`` resolver): the server admits identities the
SDK resolver would filter (``honestGuard=false`` is ateam parity — it surfaces
producer-declared ``agent_id``/``agent_name`` verbatim), so only a port of the
server rule predicts what actually renders.
"""

from __future__ import annotations

from typing import Any, Set, Dict, List

# graph_inference.go:58 nodeIdentityFields, in priority order. span_name is
# deliberately ABSENT (deviation 1: the SDK emits span_name in the event
# envelope, which ateam's ingest drops, so it never node-ifies an SDK trace).
NODE_IDENTITY_FIELDS = [
    "node",
    "node_name",
    "agent",
    "agent_name",
    "agent_id",
    "agent_role",
    "plugin_name",
    "component_name",
    "collaboratorAgentId",
    "submitter_agent_id",
]
# graph_inference.go:54 graphHiddenNodes.
HIDDEN: Set[str] = {"__unattributed__", "__start__"}
HANDOFF_TYPES = {"agent.handoff", "a2a.delegation"}


def _truthy(v: Any) -> bool:
    """Port of graph_inference.go truthy(): nil/false/"" are false; else true."""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v != ""
    return True


def resolve_event_agent(em: Dict[str, Any]) -> str:
    """Port of resolveEventAgent(honestGuard=false): the owning agent id, or
    ``__unattributed__`` when no identity field yields a value."""
    et = em.get("event_type") or ""
    p = em.get("payload") or {}
    if not isinstance(p, dict):
        p = {}

    # G2: the SDK's synthesized structural root is NOT an agent.
    if et == "trace.root" or _truthy(p.get("synthesized")):
        return "__unattributed__"

    # Handoff/delegation: the SOURCE agent owns the event.
    if et in HANDOFF_TYPES:
        frm = p.get("from_agent")
        if isinstance(frm, str) and frm.strip():
            return frm.strip()

    # First non-empty payload identity field (honestGuard=false -> just trim).
    for f in NODE_IDENTITY_FIELDS:
        v = p.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # ateam's metadata.span_name fallback — INERT for SDK traces (span_name is in
    # the envelope, never event metadata). Replicated for exactness.
    md = em.get("metadata")
    if isinstance(md, dict):
        sn = md.get("span_name")
        if isinstance(sn, str) and sn.strip():
            return sn.strip()

    return "__unattributed__"


def expected_node_ids(events: List[Dict[str, Any]]) -> List[str]:
    """The SORTED set of node ``agent_id``s the server renders for these events,
    including the G1 handoff/delegation endpoint-ensuring pass."""
    evs = sorted(events, key=lambda e: (e.get("sequence_id") or 0))
    seen: Set[str] = set()
    for e in evs:
        a = resolve_event_agent(e)
        if a not in HIDDEN:
            seen.add(a)
    # G1: both endpoints of every declared handoff/delegation become nodes.
    for e in evs:
        if (e.get("event_type") or "") not in HANDOFF_TYPES:
            continue
        p = e.get("payload") or {}
        if not isinstance(p, dict):
            continue
        for side in ("from_agent", "to_agent"):
            v = p.get(side)
            idv = v.strip() if isinstance(v, str) else ""
            if idv and idv not in HIDDEN:
                seen.add(idv)
    return sorted(seen)


def expected_agent_column(node_ids: List[str]):
    """agentColumn projection (graph_inference.go:427): 0 -> None (renders '—'),
    1 -> that id, >1 -> 'multi-agent'."""
    if not node_ids:
        return None
    if len(node_ids) == 1:
        return node_ids[0]
    return "multi-agent"


def kind_of(node_ids: List[str]) -> str:
    n = len(node_ids)
    return "empty-state" if n == 0 else ("single-agent" if n == 1 else "multi-agent")
