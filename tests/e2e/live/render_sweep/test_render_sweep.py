"""Live customer-run render sweep over every Family-B industry sample.

Uploads each sealed/recorded industry fixture the way a paying customer does
(``upload_recorded_trace``, LayerLens-key-ONLY) to a running atlas, reads the
server-computed trace detail back, and asserts the full per-kind render contract
(``_sweep.check_render``): the agent DAG where genuinely multi-agent, an honest
empty-state for providers/ingestion/non-agentic, plus the Agent/Framework/Status
columns and a populated waterfall.

Gated: the whole ``tests/e2e/live`` tree is skipped unless ``LAYERLENS_LIVE=1``;
this file additionally skips unless a local base URL + key are set (never the
prod default). Uploading a recorded fixture replays committed bytes — it does NOT
call any provider — so the sealed set (agentforce/vertex/azure/openrouter) is
swept too: they must still render + attest key-only.

Run:
    set -a; . tests/e2e/live/.env; set +a
    LAYERLENS_LIVE=1 pytest tests/e2e/live/render_sweep/ -q
"""

from __future__ import annotations

import os
import glob
import json
import importlib.util

import pytest

from ._sweep import check_render, read_back_detail
from ._render_oracle import kind_of, expected_node_ids

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_SAMPLES = os.path.join(_REPO, "samples")

# Load the real customer-path helper (samples/_helpers.py) BY FILE PATH — never via
# sys.path. samples/ holds namespace dirs (e.g. samples/mcp, no __init__.py) that,
# if put on sys.path at collection time, shadow installed packages (mcp, ...) and
# break unrelated test modules' imports.
_hspec = importlib.util.spec_from_file_location("_ll_sample_helpers", os.path.join(_SAMPLES, "_helpers.py"))
_helpers = importlib.util.module_from_spec(_hspec)
_hspec.loader.exec_module(_helpers)
recorded_trace_path = _helpers.recorded_trace_path
upload_recorded_trace = _helpers.upload_recorded_trace

_FIXTURES = os.path.join(_SAMPLES, "data", "traces", "industry")
STEMS = sorted(os.path.basename(p)[: -len(".jsonl")] for p in glob.glob(os.path.join(_FIXTURES, "*.jsonl")))

# The sealed fixtures (no provider cred): still uploaded + rendered from committed
# bytes; documented here so the report can label them (they are NOT live-run).
# agentforce/vertex/azure(manufacturing) stay sealed (no cred). openrouter stays
# sealed too: the provided OPENROUTER_API_KEY is dead (401 "User not found") so the
# gateway unseal is blocked — the recorded fixture still renders key-only.
SEALED_STEMS = {
    "salesforce_agentforce_order_status",
    "salesforce_agentforce_billing_escalation",
    "government_vertex_triage",
    "government_vertex_permit_tooluse",
    "manufacturing_predictive_maintenance",
    "manufacturing_maintenance_tooluse",
    "saas_openrouter_cost_routing",
}


@pytest.fixture(scope="module")
def live():
    """Real Stratix client + raw-API creds, or skip. Never the prod default URL."""
    base = os.environ.get("LAYERLENS_STRATIX_BASE_URL")
    key = os.environ.get("LAYERLENS_STRATIX_API_KEY")
    if not base or not key:
        pytest.skip("LAYERLENS_STRATIX_BASE_URL + LAYERLENS_STRATIX_API_KEY required (never the prod default)")
    from layerlens import Stratix

    return Stratix(), base, key


@pytest.mark.parametrize("stem", STEMS)
def test_family_b_sample_renders_as_customer(stem, live, render_report):
    client, base, key = live
    path = recorded_trace_path("industry", f"{stem}.jsonl")
    records = [json.loads(ln) for ln in open(path) if ln.strip()]
    expected = [expected_node_ids(r.get("events", [])) for r in records]

    ids = upload_recorded_trace(client, path)
    assert len(ids) == len(records), f"{stem}: {len(records)} fixture records -> {len(ids)} uploaded ids"

    all_problems = []
    for i, (tid, exp) in enumerate(zip(ids, expected)):
        detail = read_back_detail(client, base, key, tid)
        problems = check_render(detail, exp)
        graph = detail.get("graph") or {}
        render_report(
            {
                "stem": stem,
                "record": i,
                "trace_id": tid,
                "kind": kind_of(exp),
                "expected_nodes": exp,
                "server_nodes": sorted((n.get("agent_id") or "") for n in graph.get("nodes", [])),
                "edges": len(graph.get("edges", [])),
                "topology": graph.get("topology"),
                "agent": detail.get("agent"),
                "framework": detail.get("framework"),
                "status": (detail.get("data") or {}).get("status"),
                "event_count": detail.get("event_count"),
                "sealed": stem in SEALED_STEMS,
                "problems": problems,
            }
        )
        if problems:
            all_problems.append(f"record {i} ({kind_of(exp)}): " + "; ".join(problems))

    assert not all_problems, f"{stem} render violations:\n" + "\n".join(all_problems)
