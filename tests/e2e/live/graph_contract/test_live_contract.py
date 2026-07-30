"""Live end-to-end agent-graph contract (G7).

Seeds each corpus lane's sealed events through a REAL ``Stratix()`` client to a
running atlas (``LAYERLENS_STRATIX_BASE_URL`` — local :8080 or a dev deploy) and
asserts the server-computed graph matches the shipped oracle:

  * ``graph`` is present IFF the lane has topology; an agentless trace stores no graph;
  * the graph node-set equals the oracle's shipped node-set;
  * the count-aware Agent column equals the graph projection ("" / id / "multi-agent");
  * re-uploading a topology trace with agentless events UNSETS the stale graph;
  * dual oracle: the running server (honestGuard=false) admits exactly the nodes the
    guarded oracle would keep PLUS generic class-names the SDK resolver also rejects —
    surfacing the honestGuard divergence against the LIVE server;
  * FE render probe: the served graph satisfies the render contract (present<->topology).

Gated: the whole ``tests/e2e/live`` tree is skipped unless ``LAYERLENS_LIVE=1``;
this file additionally skips unless a base URL + key are set (never the prod default).
The oracle + seed events are the same corpus the atlas honesty matrices assert on.
"""

from __future__ import annotations

import os
import json

import pytest

from ._live import seed_lane, read_back_graph
from ._contract import (
    load_oracle,
    graph_node_ids,
    divergence_nodes,
    sdk_would_reject,
    expected_agent_column,
    graph_present_iff_topology,
)

_HERE = os.path.dirname(__file__)
ORACLE = load_oracle(os.path.join(_HERE, "oracle_expectations.json"))
with open(os.path.join(_HERE, "seed_events.json")) as _f:
    SEED = json.load(_f)

_TOPOLOGY_LANES = sorted(k for k, v in ORACLE.items() if v["shipped_nodes"])
_EMPTY_LANES = sorted(k for k, v in ORACLE.items() if not v["shipped_nodes"])


@pytest.fixture(scope="module")
def live():
    """Real Stratix client + raw-API creds, or skip. Never the prod default URL."""
    base = os.environ.get("LAYERLENS_STRATIX_BASE_URL")
    key = os.environ.get("LAYERLENS_STRATIX_API_KEY")
    if not base or not key:
        pytest.skip("LAYERLENS_STRATIX_BASE_URL + LAYERLENS_STRATIX_API_KEY required (never the prod default)")
    from layerlens import Stratix

    client = Stratix()
    return client, base, key


@pytest.mark.parametrize("lane", _TOPOLOGY_LANES)
def test_server_graph_matches_shipped_oracle(lane, live):
    """Seed -> read back -> the server graph node-set + Agent column match the
    shipped oracle, and the graph is present because the lane has topology."""
    client, base, key = live
    shipped = ORACLE[lane]["shipped_nodes"]

    tid = seed_lane(client, SEED[lane])
    graph, agent = read_back_graph(client, base, key, tid)

    assert graph_present_iff_topology(graph, len(shipped)), f"{lane}: graph presence != topology"
    assert graph_node_ids(graph) == sorted(shipped), (
        f"{lane}: server node-set {graph_node_ids(graph)} != shipped oracle {sorted(shipped)}"
    )
    assert agent == expected_agent_column(shipped), (
        f"{lane}: Agent column {agent!r} != graph projection {expected_agent_column(shipped)!r}"
    )


@pytest.mark.parametrize("lane", _EMPTY_LANES)
def test_agentless_trace_stores_no_graph(lane, live):
    """A lane with no honest topology must NOT get a stored graph (FE empty-state)."""
    client, base, key = live
    tid = seed_lane(client, SEED[lane])
    graph, _agent = read_back_graph(client, base, key, tid)
    assert not (graph and graph.get("nodes")), f"{lane}: agentless trace unexpectedly stored a graph"


def test_live_dual_oracle_divergence_is_sdk_generic(live):
    """Against the LIVE server (honestGuard=false), every node it admits beyond the
    guarded oracle is one the SDK resolver also rejects — the honestGuard divergence,
    surfaced end-to-end, bounded to generic class-names (never a real agent)."""
    client, base, key = live
    seen_divergence = []
    for lane in _TOPOLOGY_LANES:
        guarded = ORACLE[lane]["guarded_nodes"]
        tid = seed_lane(client, SEED[lane])
        graph, _agent = read_back_graph(client, base, key, tid)
        server_nodes = graph_node_ids(graph)
        for node in divergence_nodes(server_nodes, guarded):
            seen_divergence.append((lane, node))
            assert sdk_would_reject(node), (
                f"{lane}: the live server surfaces {node!r} that the guard drops, but the SDK "
                f"resolver would NOT reject it — a real agent is being guarded or a fabrication admitted"
            )
    # The divergence genuinely exists on the live server (generic-name lanes).
    assert seen_divergence, "no live honestGuard divergence observed — is the running server the guarded build?"


def test_reupload_unsets_stale_graph(live):
    """Re-uploading a trace that WAS multi-agent with agentless events unsets the
    stored graph — no stale topology lingers.

    The server's forward-only upload idempotency (F5) UPSERTs keyed on the
    SDK-supplied ``trace_id`` (``sdk_trace_id``) and ``$unset``s ``graph``/``agent``
    when a re-upload no longer yields them (traces_service.go). So the two uploads
    MUST share the same SDK ``trace_id`` (not the server ``_id``) to target the
    same record."""
    import uuid

    client, base, key = live
    sdk_id = f"graphcontract-reupload-{uuid.uuid4().hex}"

    # First upload: real multi-agent topology under a controlled SDK trace id.
    tid = seed_lane(client, SEED["autogen-s2"], trace_id=sdk_id)
    graph, _ = read_back_graph(client, base, key, tid)
    assert graph and graph.get("nodes"), "precondition: autogen-s2 should have a graph"

    # Re-upload the SAME SDK trace id with a bare provider call (no agent identity)
    # -> the upsert replaces the record and $unsets the now-unsupported graph.
    agentless = [
        {"event_type": "model.invoke", "payload": {"model": "gpt-4o-mini", "framework": "openai"}, "span_id": "s1"},
        {"event_type": "cost.record", "payload": {"model": "gpt-4o-mini", "tokens_total": 5}, "span_id": "s1"},
    ]
    seed_lane(client, agentless, trace_id=sdk_id)
    graph2, _ = read_back_graph(client, base, key, tid)
    assert not (graph2 and graph2.get("nodes")), "stale graph not unset after agentless re-upload"


@pytest.mark.parametrize("lane", _TOPOLOGY_LANES)
def test_fe_render_contract_over_live_graph(lane, live):
    """FE render probe (reuses the G1 contract): the served graph would render its
    nodes (present<->topology). Confirms the server graph is FE-renderable, not blank."""
    client, base, key = live
    tid = seed_lane(client, SEED[lane])
    graph, _ = read_back_graph(client, base, key, tid)
    assert graph_present_iff_topology(graph, len(ORACLE[lane]["shipped_nodes"]))
    for node in graph.get("nodes", []):
        assert node.get("label"), f"{lane}: node {node.get('agent_id')} has no render label"
        assert node.get("agent_type"), f"{lane}: node {node.get('agent_id')} has no agent_type for colouring"
