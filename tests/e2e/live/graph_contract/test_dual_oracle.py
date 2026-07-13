"""Dual-oracle divergence test — the honestGuard product decision, surfaced (G7).

Deterministic half of the graph-contract harness (needs no live stack; it reads
the committed real-engine oracle). The atlas graph engine ships with
``honestGuard=false`` (ateam parity): it surfaces producer-declared identities
verbatim, INCLUDING generic framework class-names (``agno_agent``,
``ToolCallingAgent``, ``Strands Agents``). The guarded oracle (``honestGuard=true``)
drops those. This test enumerates that divergence and asserts it is bounded to
identities the SDK's OWN resolver (``_identity``) also considers non-honest — so
the two layers agree on what "honest" means, and the shipped default's extra
nodes are exactly the generic labels, never a real agent.

It does NOT change product behaviour (honestGuard stays false); it makes the
divergence a first-class, monitored fact.

``oracle_expectations.json`` is generated from the REAL engine
(atlas ``services.InferAgentGraph`` shipped vs ``inferAgentGraph(_, true)`` guarded)
over ``graph_honesty_fixtures.json`` — see README.md for provenance.
"""

from __future__ import annotations

import os

import pytest

from ._contract import (
    load_oracle,
    divergence_nodes,
    sdk_would_reject,
    guard_only_removes,
)

_ORACLE_PATH = os.path.join(os.path.dirname(__file__), "oracle_expectations.json")
ORACLE = load_oracle(_ORACLE_PATH)

# The generic class-name lanes the shipped default admits but the guard rejects
# (known divergence from the current corpus — asserted present, not exhaustive).
_KNOWN_DIVERGENT_LANES = {"agno-s1", "smolagents-s1", "strands-s1"}


@pytest.mark.parametrize("lane", sorted(ORACLE))
def test_guard_only_removes_never_invents(lane: str) -> None:
    """For every lane the guarded node-set is a SUBSET of the shipped node-set:
    the honesty guard is a strict filter, never a fabricator."""
    o = ORACLE[lane]
    assert guard_only_removes(o["shipped_nodes"], o["guarded_nodes"]), (
        f"{lane}: guarded {o['guarded_nodes']} is not a subset of shipped {o['shipped_nodes']} "
        f"— the guard invented a node"
    )


def test_divergence_is_bounded_to_sdk_nonhonest_identities() -> None:
    """Every node the shipped default admits that the guard rejects must be one
    the SDK's own resolver would ALSO reject (generic class-name / api-method /
    model-as-agent). This is the SDK<->server honesty agreement: the shipped
    default never surfaces a *real* agent that the guard would drop."""
    all_divergence: set[str] = set()
    for lane, o in ORACLE.items():
        for node in divergence_nodes(o["shipped_nodes"], o["guarded_nodes"]):
            all_divergence.add(node)
            assert sdk_would_reject(node), (
                f"{lane}: shipped surfaces {node!r} which the guard drops, but the SDK "
                f"resolver would NOT reject it — a real agent is being silently guarded, "
                f"or a fabrication is being silently admitted"
            )
    # Non-vacuous: the honestGuard divergence genuinely exists on this corpus.
    assert all_divergence, "no divergence found — the dual-oracle is vacuous (guard is a no-op?)"


def test_known_generic_lanes_diverge_with_blanked_column() -> None:
    """The generic-name lanes surface their class-name node + column under the
    shipped default, and blank out entirely under the guard — the exact,
    documented honestGuard divergence."""
    for lane in _KNOWN_DIVERGENT_LANES:
        assert lane in ORACLE, f"expected corpus lane {lane} missing"
        o = ORACLE[lane]
        div = divergence_nodes(o["shipped_nodes"], o["guarded_nodes"])
        assert div, f"{lane}: expected a shipped-vs-guarded divergence, found none"
        # Shipped surfaces the generic node as the Agent column; guard blanks it.
        assert o["shipped_column"] and o["shipped_column"] in o["shipped_nodes"], (
            f"{lane}: shipped column {o['shipped_column']!r} not backed by a node"
        )
        assert o["guarded_column"] == "", f"{lane}: guard should blank the column, got {o['guarded_column']!r}"
        for node in div:
            assert sdk_would_reject(node), f"{lane}: divergent node {node!r} is not SDK-generic"


def test_real_multi_agent_lanes_are_guard_invariant() -> None:
    """Lanes with genuine (honest) agent identities are IDENTICAL shipped vs
    guarded — the guard leaves real topology untouched. Catches an over-broad
    guard that would blank legitimate multi-agent graphs."""
    for lane, o in ORACLE.items():
        # A lane whose shipped nodes are all SDK-honest must be guard-invariant.
        if all(not sdk_would_reject(n) for n in o["shipped_nodes"]) and o["shipped_nodes"]:
            assert o["guarded_nodes"] == o["shipped_nodes"], (
                f"{lane}: honest lane changed under the guard ({o['shipped_nodes']} -> {o['guarded_nodes']})"
            )
            assert o["guarded_column"] == o["shipped_column"], f"{lane}: honest column changed under guard"
