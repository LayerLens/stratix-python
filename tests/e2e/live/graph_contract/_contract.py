"""Shared assertion helpers for the agent-graph contract harness (G7).

The contract under test spans three layers:

    SDK adapter events  ->  atlas graph engine (services.InferAgentGraph)  ->  FE render

These helpers encode the invariants the harness asserts, in pure functions so the
deterministic parts (dual-oracle over the committed expectations) are testable
without a live stack, and the live parts reuse the same logic against a running
:8080.

The "dual oracle" is the heart of it. The shipped backend runs the engine with
``honestGuard=false`` (ateam parity — surfaces producer-declared identities
verbatim). The guarded oracle is the same engine with ``honestGuard=true`` (drops
generic class-name / api-method / model-as-agent identities). Enumerating the
nodes the shipped default admits that the guard would reject SURFACES the
honestGuard product divergence without changing any product behaviour — and we
assert that divergence is bounded to identities the SDK's own resolver
(``_identity``) also considers non-honest, i.e. the two layers agree on what
"honest" means.
"""

from __future__ import annotations

from typing import Set, Dict, List, Iterable, Optional

from layerlens.instrument._identity import (
    _API_METHOD_RE,
    _is_generic,
)


def divergence_nodes(shipped: Iterable[str], guarded: Iterable[str]) -> List[str]:
    """Nodes the shipped (honestGuard=false) engine admits that the guarded
    (honestGuard=true) engine rejects — the honestGuard divergence for one lane."""
    guarded_set = set(guarded)
    return sorted(n for n in shipped if n not in guarded_set)


def guard_only_removes(shipped: Iterable[str], guarded: Iterable[str]) -> bool:
    """The guard is a strict filter: it may drop nodes, never invent them."""
    return set(guarded) <= set(shipped)


def sdk_would_reject(node: str, models: Optional[Set[str]] = None) -> bool:
    """True iff the SDK's honest-identity resolver would also reject *node*.

    Mirrors ``_identity.honest_agent_identity``'s ``_honest`` guard: a generic
    class-name/placeholder, a dotted api-method label, or a model id used as an
    agent are all non-honest. This is the SDK side of the same honesty contract
    the server's ``honestAgentName`` guard enforces.
    """
    clean = node.strip().lower()
    if _is_generic(node):
        return True
    if _API_METHOD_RE.match(clean):
        return True
    if models and clean in {m.lower() for m in models}:
        return True
    return False


def graph_present_iff_topology(graph: Optional[dict], node_count: int) -> bool:
    """The server stores/serves a graph exactly when it inferred topology
    (>=1 node); an agentless trace must carry no graph (FE shows empty-state)."""
    has_graph = bool(graph) and len(graph.get("nodes", [])) > 0
    return has_graph == (node_count > 0)


def graph_node_ids(graph: Optional[dict]) -> List[str]:
    if not graph:
        return []
    return sorted(n.get("agent_id", "") for n in graph.get("nodes", []))


def expected_agent_column(node_ids: List[str]) -> str:
    """The count-aware Agent column the server projects from the graph:
    "" for 0 nodes, the single agent_id for 1, "multi-agent" for >1
    (atlas services.agentColumn)."""
    if len(node_ids) == 0:
        return ""
    if len(node_ids) == 1:
        return node_ids[0]
    return "multi-agent"


def load_oracle(path: str) -> Dict[str, dict]:
    import json

    with open(path) as f:
        return json.load(f)
