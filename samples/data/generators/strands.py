#!/usr/bin/env python3
"""ADP-W2 Family-B recorders for the AWS Strands adapter (framework=strands).

Records REAL ``strands.Agent`` runs (OpenAIModel ``gpt-4o-mini`` backend) under
the capture seam and writes sealed trace fixtures to
``samples/data/traces/industry/``. Nothing is fabricated: the Framework column
shows ``strands`` (the framework that actually ran), the tokens/cost are the
real per-cycle counts Strands lifted from the provider stream, and the Status
reflects the real run outcome.

- :func:`generate_strands_single` -- one named strands ``Agent`` answering a
  manufacturing quality-assurance question. A single honest agent node
  (Agent column ``quality-standards-agent``) + real ``model.invoke`` /
  ``cost.record`` events.

- :func:`generate_strands_multi` -- a REAL strands multi-agent ``Swarm`` where
  an ``intake-coordinator`` hands off (the built-in ``handoff_to_agent`` tool)
  to a ``defect-analyst`` who, when a process change is warranted, hands off to
  a ``corrective-action-engineer``. The whole swarm is captured as ONE trace via
  :func:`trace_context` (the per-agent runs reuse the shared collector, so none
  flush individually and every node's real ``model.invoke`` + the producer-
  declared ``agent.handoff`` edges land in the same sealed payload) -> the trace
  renders Agent column ``multi-agent``.

Framework deps (``strands`` / ``openai``) are imported FUNCTION-LOCAL so this
module imports in any venv; each recorder is a no-op-skip when they are absent.
The seam (``_write`` / ``_CAPTURE`` / ``OPENAI_MODEL`` / ``set_trace_observer``
/ ``_collector_mod`` / ``trace_context``) is reused from ``_generate_fixtures``.
"""
from __future__ import annotations

import os
import sys

# Make the seam (_generate_fixtures) + src importable regardless of entrypoint.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)  # samples/data
_SAMPLES = os.path.dirname(_DATA)  # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _DATA, _SAMPLES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from layerlens import Stratix  # noqa: E402
from layerlens.instrument import trace_context  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402

from _generate_fixtures import _CAPTURE, OPENAI_MODEL, _write  # noqa: E402

SINGLE_STEM = "manufacturing_strands_qa"
MULTI_STEM = "manufacturing_strands_defect_swarm"

# --------------------------------------------------------------------------
# Single agent: manufacturing quality-standards Q&A
# --------------------------------------------------------------------------
_SINGLE_AGENT = "quality-standards-agent"
_SINGLE_SYSTEM = (
    "You are quality-standards-agent, a manufacturing quality-assurance "
    "assistant. Given a production-line inspection result, decide whether the "
    "lot conforms to the stated spec/tolerance, assign a disposition "
    "(ACCEPT / REWORK / SCRAP / USE-AS-IS), and cite the governing standard. "
    "Answer concisely (under 150 words)."
)
_SINGLE_QUESTION = (
    "Incoming inspection, lot LOT-7742 (CNC-machined aluminum brackets, part "
    "AL-6061-BRK): measured mounting-hole diameter 10.28 mm; the drawing calls "
    "out 10.20 mm +0.05/-0.02 (i.e. 10.18-10.25 mm) per ASME Y14.5. Sample "
    "size 50, with 6 units above the upper limit. Is the lot within spec, and "
    "what is the disposition?"
)


def generate_strands_single(client: Stratix) -> None:
    """Record a single named strands Agent (OpenAIModel) manufacturing QA run."""
    from strands import Agent  # type: ignore[import-not-found]
    from strands.models.openai import OpenAIModel  # type: ignore[import-not-found]

    from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter

    model = OpenAIModel(model_id=OPENAI_MODEL, params={"max_tokens": 400})

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = StrandsAdapter(client, capture_config=_CAPTURE)
    try:
        adapter.connect()
        agent = Agent(
            model=model,
            hooks=[adapter],
            name=_SINGLE_AGENT,
            system_prompt=_SINGLE_SYSTEM,
            callback_handler=None,
        )
        agent(_SINGLE_QUESTION)
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for strands single")
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "manufacturing",
        "quality-assurance",
        "strands",
    ]

    events = payload.get("events", [])
    models = [e for e in events if e.get("event_type") == "model.invoke"]
    costs = [e for e in events if e.get("event_type") == "cost.record"]
    if not models or not costs:
        raise RuntimeError(
            "strands single trace missing real model.invoke/cost.record "
            "(models=%d costs=%d)" % (len(models), len(costs))
        )
    fw = models[0]["payload"].get("framework")
    agent_name = models[0]["payload"].get("agent_name")
    cost_usd = costs[0]["payload"].get("cost_usd")
    print(
        "  strands-single  %s  framework=%s agent=%s events=%d cost_usd=%s"
        % (SINGLE_STEM, fw, agent_name, len(events), cost_usd)
    )
    print("  ->", _write([payload], "industry", SINGLE_STEM), "\n")


# --------------------------------------------------------------------------
# Multi agent: manufacturing defect-triage Swarm (handoff_to_agent)
# --------------------------------------------------------------------------
_DEFECT_REPORT = (
    "Defect report DR-3391 (assembly line 4): 6% of TX-90 gearbox housings from "
    "today's run show hairline cracks at the mounting boss. The boss wall is "
    "die-cast A380 aluminum; the cracks appear only after the press-fit bearing "
    "insertion station. Determine the root cause, disposition today's affected "
    "run, and recommend a fix."
)


def _swarm_agent(name: str, system_prompt: str):
    from strands import Agent  # type: ignore[import-not-found]
    from strands.models.openai import OpenAIModel  # type: ignore[import-not-found]

    return Agent(
        name=name,
        model=OpenAIModel(model_id=OPENAI_MODEL, params={"max_tokens": 400}),
        system_prompt=system_prompt,
        callback_handler=None,
    )


def generate_strands_multi(client: Stratix) -> None:
    """Record a REAL strands Swarm with handoff_to_agent -> one multi-agent trace."""
    from strands.multiagent import Swarm  # type: ignore[import-not-found]

    from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter

    intake = _swarm_agent(
        "intake-coordinator",
        "You are intake-coordinator on a manufacturing quality line. You do NOT "
        "diagnose defects yourself. Read the defect report, then IMMEDIATELY hand "
        "off to the 'defect-analyst' agent by calling the handoff_to_agent tool, "
        "passing a one-sentence summary of the report as the message.",
    )
    analyst = _swarm_agent(
        "defect-analyst",
        "You are defect-analyst, a manufacturing root-cause specialist. Determine "
        "the most likely ROOT CAUSE of the reported defect and a disposition "
        "(REWORK / SCRAP / USE-AS-IS) for the affected run. Because this defect "
        "warrants a process change, hand off to the 'corrective-action-engineer' "
        "agent via the handoff_to_agent tool, passing your root-cause finding as "
        "the message.",
    )
    capa = _swarm_agent(
        "corrective-action-engineer",
        "You are corrective-action-engineer. Propose one concrete corrective and "
        "preventive action (CAPA) that removes the identified root cause so it "
        "cannot recur, in 2-3 sentences. This is the final step; do not hand off.",
    )

    adapter = StrandsAdapter(client, capture_config=_CAPTURE)
    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        adapter.connect()
        # One adapter instance registered on every node's hook registry.
        for ag in (intake, analyst, capa):
            ag.hooks.add_hook(adapter)
        swarm = Swarm(
            [intake, analyst, capa],
            entry_point=intake,
            max_handoffs=6,
            max_iterations=8,
        )
        # trace_context owns the collector: the per-agent Strands runs reuse it
        # (no per-agent flush), so the entire swarm -- every node's real
        # model.invoke + the agent.handoff edges -- flushes as ONE sealed trace.
        with trace_context(client, capture_config=_CAPTURE):
            swarm(_DEFECT_REPORT)
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for strands multi")
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "manufacturing",
        "defect-triage",
        "multi-agent",
    ]

    events = payload.get("events", [])
    handoffs = [e for e in events if e.get("event_type") == "agent.handoff"]
    if not handoffs:
        raise RuntimeError(
            "strands multi trace has NO agent.handoff edge -- the swarm did not "
            "hand off; refusing to ship a non-multi-agent fixture"
        )
    identities = sorted(
        {
            e["payload"].get("agent_name")
            for e in events
            if isinstance(e.get("payload"), dict) and e["payload"].get("agent_name")
        }
    )
    edges = sorted(
        "%s->%s" % (e["payload"].get("from_agent"), e["payload"].get("to_agent"))
        for e in handoffs
    )
    if len(identities) < 2:
        raise RuntimeError(
            "strands multi trace has <2 distinct agent identities: %s" % identities
        )
    print(
        "  strands-multi   %s  agents=%s handoffs=%s events=%d"
        % (MULTI_STEM, identities, edges, len(events))
    )
    print("  ->", _write([payload], "industry", MULTI_STEM), "\n")


if __name__ == "__main__":
    _client = Stratix()
    generate_strands_single(_client)
    generate_strands_multi(_client)
