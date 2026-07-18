"""Deterministic fidelity gate for the render-sweep oracle (runs in CI).

The customer-run render sweep (``tests/e2e/live/render_sweep/``) asserts
``server-returned graph == fixture-derived expectation``. That expectation comes
from ``_render_oracle`` — a Python port of the server's
``services.InferAgentGraph`` node-identity extraction. This test pins the port's
fidelity: it must reproduce EVERY node-set in the live-proven graph-contract
oracle (``oracle_expectations.json``, verified against the real :8080 server).

If ``graph_inference.go`` changes what renders, the committed graph-contract
oracle changes, and this test fails until the render-sweep port is updated to
match — so the sweep can never silently assert against a stale server model.

This lane is intentionally OUTSIDE ``tests/e2e/live`` so it runs in bare CI (no
live stack, no creds, no spend).
"""

from __future__ import annotations

import os
import json
import importlib.util

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_RENDER_SWEEP = os.path.join(_REPO, "tests", "e2e", "live", "render_sweep")
_GRAPH_CONTRACT = os.path.join(_REPO, "tests", "e2e", "live", "graph_contract")

# Load the render-sweep oracle BY FILE PATH — never via sys.path (avoids polluting
# the global import namespace with the render_sweep package's siblings).
_ospec = importlib.util.spec_from_file_location("_ll_render_oracle", os.path.join(_RENDER_SWEEP, "_render_oracle.py"))
_render_oracle = importlib.util.module_from_spec(_ospec)
_ospec.loader.exec_module(_render_oracle)
expected_node_ids = _render_oracle.expected_node_ids


def _load(name: str) -> dict:
    with open(os.path.join(_GRAPH_CONTRACT, name)) as f:
        return json.load(f)


_SEED = _load("seed_events.json")
_ORACLE = _load("oracle_expectations.json")


@pytest.mark.parametrize("lane", sorted(_SEED))
def test_render_oracle_reproduces_graph_contract_node_set(lane: str) -> None:
    """The render-sweep port computes exactly the live-proven node-set per lane."""
    assert lane in _ORACLE, f"{lane} present in seed_events but not oracle_expectations"
    got = expected_node_ids(_SEED[lane])
    want = sorted(_ORACLE[lane]["shipped_nodes"])
    assert got == want, f"{lane}: render-oracle {got} != live-proven oracle {want}"
