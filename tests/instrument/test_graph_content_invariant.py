"""G6 — the inferred agent graph is identical with capture_content on vs off.

Agent identity and topology live in *metadata*, not content, so redaction
(``capture_content=False``, the privacy-by-default posture) must NOT change the
graph the server infers from a trace. If a content-strip ever ate an identity or
topology field, a redacted trace would render a different (or blank) graph than
the same trace captured with content — a silent, privacy-coupled regression.

This locks the invariant at the SDK collector boundary (the real
``TraceCollector.emit`` -> ``CaptureConfig.redact_payload`` path). The atlas
graph engine (``services.inferAgentGraph``) resolves nodes/edges ONLY from these
payload fields:

  * ``nodeIdentityFields`` = node / node_name / agent / agent_name / agent_id /
    agent_role / plugin_name / component_name / collaboratorAgentId /
    submitter_agent_id,
  * the handoff/delegation endpoints ``from_agent`` / ``to_agent`` /
    ``target_agent``,
  * and the synthesized ``agent.identity`` marker.

So the "graph projection" of a trace is exactly those fields. This test emits a
representative multi-agent trace twice — the two configs differ ONLY in
``capture_content`` (all L-layers on in both, so content is the only variable) —
and asserts:

  1. the per-event graph projection is byte-identical across both modes;
  2. the derived node-candidate set is identical across both modes;
  3. content DID differ (the SENTINEL free-text is present with content, gone
     without) — so the invariance above is non-vacuous, not "both stripped".
"""

from __future__ import annotations

import json
import dataclasses
from typing import Any, Dict, List, Tuple

from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig

SENTINEL = "SENTINEL-graph-content-canary-4111111111111111"

# All L-layers enabled in BOTH configs; they differ only in capture_content, so
# any projection difference is attributable to content redaction alone.
_WITH_CONTENT = CaptureConfig.full()
_NO_CONTENT = dataclasses.replace(_WITH_CONTENT, capture_content=False)

# The fields the server graph engine reads to build nodes + edges + the Agent
# column (atlas services/graph_inference.go: nodeIdentityFields + handoff/
# delegation endpoints). sender/receiver/participants are intentionally absent:
# the engine does not read them (observed-but-unused), so they are not part of
# the graph projection.
GRAPH_FIELDS: Tuple[str, ...] = (
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
    "from_agent",
    "to_agent",
    "target_agent",
)

# A representative multi-agent trace: a coordinator that hands off + delegates,
# a node-graph step, and identity markers. Every graph-carrier event also
# carries free-text content (input/messages/output/context/skill_description)
# so redaction has something to strip.
_TRACE: List[Tuple[str, Dict[str, Any]]] = [
    (
        "agent.identity",
        {"agent_name": "router", "source": "framework", "framework": "demo"},
    ),
    (
        "agent.input",
        {
            "agent_name": "router",
            "agent_id": "router",
            "agent_role": "coordinator",
            "framework": "demo",
            "input": f"route this: {SENTINEL}",
            "messages": [{"role": "user", "content": SENTINEL}],
        },
    ),
    (
        "agent.handoff",
        {
            "from_agent": "router",
            "to_agent": "billing",
            "reason": "delegate",
            "context": f"handoff context {SENTINEL}",
        },
    ),
    (
        "agent.node.enter",
        {"node": "billing_node", "input": f"node input {SENTINEL}"},
    ),
    (
        "a2a.delegation",
        {
            "from_agent": "billing",
            "to_agent": "refunds",
            "target_agent": "refunds",
            "skill": "issue_refund",
            "skill_description": f"refund skill {SENTINEL}",
            "context": f"delegation context {SENTINEL}",
        },
    ),
    (
        "agent.output",
        {"agent_name": "billing", "framework": "demo", "output": f"done {SENTINEL}"},
    ),
]


def _emit_trace(config: CaptureConfig) -> List[Dict[str, Any]]:
    """Emit the representative trace through a REAL collector under *config*."""
    collector = TraceCollector(object(), config)
    for i, (event_type, payload) in enumerate(_TRACE):
        # dict(payload) so the shared literal is never mutated by redaction.
        collector.emit(event_type, dict(payload), span_id=f"span-{i}")
    events: List[Dict[str, Any]] = collector.events
    return events


def _project(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of a payload the graph engine reads to build the graph."""
    return {k: payload[k] for k in GRAPH_FIELDS if k in payload}


def _node_candidates(events: List[Dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for ev in events:
        for value in _project(ev["payload"]).values():
            if isinstance(value, str) and value:
                ids.add(value)
    return ids


def _blob(events: List[Dict[str, Any]]) -> str:
    return json.dumps(events, default=str, sort_keys=True)


def test_graph_projection_identical_across_capture_content() -> None:
    """The engine-read identity + topology fields are byte-identical on vs off."""
    with_content = _emit_trace(_WITH_CONTENT)
    without_content = _emit_trace(_NO_CONTENT)

    # The two runs must emit the same events in the same order (only content
    # inside them may differ). If layer gating diverged this would trip first.
    assert [e["event_type"] for e in with_content] == [e["event_type"] for e in without_content], (
        "event set diverged across capture modes — graph inputs are not comparable"
    )

    for a, b in zip(with_content, without_content):
        assert _project(a["payload"]) == _project(b["payload"]), (
            f"graph projection for {a['event_type']} changed under "
            f"capture_content=False: {_project(a['payload'])} != {_project(b['payload'])}"
        )


def test_node_candidate_set_identical_across_capture_content() -> None:
    """The set of agent/node/edge identities the graph is built from is stable."""
    with_content = _node_candidates(_emit_trace(_WITH_CONTENT))
    without_content = _node_candidates(_emit_trace(_NO_CONTENT))

    assert with_content == without_content
    # Sanity: the trace really does declare the expected topology identities, so
    # an empty-vs-empty match can't pass this by accident.
    assert {"router", "billing", "billing_node", "refunds"} <= with_content


def test_agent_identity_marker_survives_redaction() -> None:
    """The synthesized agent.identity marker (the FE/server identity source) is
    preserved verbatim under capture_content=False."""
    ident_off = next(e for e in _emit_trace(_NO_CONTENT) if e["event_type"] == "agent.identity")
    assert ident_off["payload"].get("agent_name") == "router"


def test_content_actually_differs_so_the_invariant_is_not_vacuous() -> None:
    """Guard against a false-positive: content MUST be present with capture on and
    stripped with it off, or the identity-invariance above proves nothing."""
    with_content = _blob(_emit_trace(_WITH_CONTENT))
    without_content = _blob(_emit_trace(_NO_CONTENT))

    assert SENTINEL in with_content, "content was not captured even with capture_content=True"
    assert SENTINEL not in without_content, "free-text content leaked under capture_content=False"
