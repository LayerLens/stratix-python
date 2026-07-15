"""ADP-W2 Family-B recorder for the ``bedrock_agents`` adapter (record-real-once).

Records a REAL AWS Bedrock ``InvokeAgent`` run against a *provisioned* Bedrock
Agent and writes it as a sealed real-trace fixture under
``samples/data/traces/industry/``.

* ``generate_bedrock_agents_single`` -> ``government_benefits_bedrock.jsonl``:
  a citizen-services question ("what is the difference between Medicare and
  Medicaid, and which is income-based?") sent to a real Bedrock Agent (Amazon
  Nova) via ``bedrock-agent-runtime.invoke_agent(enableTrace=True)``. The
  ``BedrockAgentsAdapter`` observes the ``completion`` EventStream as it is
  drained and emits the real ``environment.config`` / ``agent.input`` /
  ``model.invoke`` / ``cost.record`` / ``agent.output`` events. Framework column
  = ``bedrock_agents`` (the platform that really ran), Status = ok, and the
  token/cost fields are the real Nova usage. A single InvokeAgent turn declares
  NO producer-chosen agent *name* (the agentId is an opaque ARN-style id, and the
  adapter deliberately never fabricates one), so the Agent column renders the
  honest empty-state (``—``) — nothing is invented.

* ``generate_bedrock_agents_multi`` -> ``insurance_claims_bedrock_supervisor``:
  a supervisor Bedrock Agent (``agentCollaboration=SUPERVISOR``) delegating a
  claim to two collaborator agents via ``agentCollaboratorInvocation`` (which the
  adapter maps to real ``agent.handoff`` edges -> a multi-agent graph). This is
  DEFERRED in the shipped fixtures: the AWS account has no provisioned
  multi-collaborator supervisor agent, and provisioning one (supervisor +
  >=2 collaborator agents + ``bedrock:InvokeAgent``/``GetAgentAlias`` on the
  agent execution role) is out of scope for the record-real-once seam. The
  recorder below is real and self-documenting: set ``BEDROCK_SUPERVISOR_AGENT_ID``
  + ``BEDROCK_SUPERVISOR_AGENT_ALIAS_ID`` (a provisioned supervisor) and it
  records the genuine multi-hop handoff trace. Without them it raises a precise
  "deferred" error rather than fabricating a multi-agent fixture. (The adapter's
  ``agentCollaboratorInvocation -> agent.handoff`` path is unit-covered in
  ``tests/instrument/adapters/frameworks/test_bedrock_agents_doubles.py``.)

Both recorders drive the REAL ``BedrockAgentsAdapter``: it opens its own per-run
collector in the boto3 ``provide-client-params`` hook and flushes it via the
completion proxy when the customer finishes draining ``response["completion"]``
(``owns_collector`` path). That flush is observed through the ``_generate_
fixtures`` capture seam (``set_trace_observer`` + a no-op ``enqueue_upload``) so
the sealed payload is captured but never uploaded during generation. The samples
upload the captured fixture themselves at run time.
"""

from __future__ import annotations

import os
import sys
import uuid

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE

SINGLE_STEM = "government_benefits_bedrock"
MULTI_STEM = "insurance_claims_bedrock_supervisor"

# A real citizen-services / public-benefits question the provisioned general-
# knowledge Bedrock Agent answers substantively (verified live). Documents the
# scenario; the recorded trace is what the real InvokeAgent turn produced.
CITIZEN_QUESTION = (
    "I just turned 65 and I'm confused about government health coverage. In "
    "plain language, what is the difference between Medicare and Medicaid, and "
    "which one is based on income? Give me the key points I need to know."
)

# A claim needing BOTH a coverage determination and a fraud assessment, so a
# supervisor agent must delegate to both collaborators (used by the deferred
# multi recorder when a provisioned supervisor is available).
CLAIM_FOR_SUPERVISOR = (
    "Auto claim CLM-55810: policyholder reports their parked car was struck "
    "overnight in a lot, rear bumper cracked, no injuries, but the claim was "
    "filed 26 days after the reported date and there are two similar prior "
    "claims this year. Triage it: is this loss covered, and what is the fraud "
    "risk? Give a final triage tier."
)


# --------------------------------------------------------------------------
# Adapter-driven capture: the BedrockAgentsAdapter opens a per-run collector on
# the boto3 InvokeAgent hook and flushes it (owns_collector) when the customer
# drains response["completion"]. We register it, drive a REAL invoke_agent, drain
# the stream, and observe the flushed payload — mirroring the crewai/autogen/
# openai_agents self-flushing recorders in this package.
# --------------------------------------------------------------------------
def _capture_invoke_agent(
    client: Stratix, *, agent_id: str, alias_id: str, input_text: str, session_prefix: str
) -> dict:
    import boto3
    from layerlens.instrument.adapters.frameworks.bedrock_agents import (
        BedrockAgentsAdapter,
    )

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        rt = boto3.client(
            "bedrock-agent-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        adapter = BedrockAgentsAdapter(client, capture_config=_CAPTURE)
        adapter.connect(target=rt)
        try:
            response = rt.invoke_agent(
                agentId=agent_id,
                agentAliasId=alias_id,
                sessionId=f"{session_prefix}-" + uuid.uuid4().hex[:12],
                inputText=input_text,
                enableTrace=True,
            )
            # Drain the completion stream exactly as a customer would — this is
            # what drives the adapter's per-trace emission and the final flush.
            events = list(response["completion"])
            if not events:
                raise RuntimeError("bedrock_agents completion stream was empty")
        finally:
            adapter.disconnect()
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for bedrock_agents InvokeAgent")
    return payload


# --------------------------------------------------------------------------
# Single agent (government citizen-services / public-benefits explainer)
# --------------------------------------------------------------------------
def generate_bedrock_agents_single(client: Stratix) -> dict:
    """Record a single real Bedrock ``InvokeAgent`` citizen-services turn."""
    agent_id = os.environ.get("BEDROCK_AGENT_ID")
    alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID")
    if not (agent_id and alias_id):
        raise RuntimeError(
            "bedrock_agents single: set BEDROCK_AGENT_ID + BEDROCK_AGENT_ALIAS_ID "
            "(a provisioned Bedrock Agent) and AWS credentials to record."
        )

    payload = _capture_invoke_agent(
        client,
        agent_id=agent_id,
        alias_id=alias_id,
        input_text=CITIZEN_QUESTION,
        session_prefix="ll-w2-gov",
    )
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "government",
        "citizen-services",
        "bedrock-agents",
    ]
    events = payload.get("events", [])
    from collections import Counter

    counts = dict(Counter(e.get("event_type") for e in events))
    models = sorted(
        {(e.get("payload") or {}).get("model") for e in events
         if e.get("event_type") == "model.invoke"}
        - {None}
    )
    print(
        "  bedrock-agents single (citizen-services InvokeAgent)  "
        "events=%d models=%s counts=%s" % (len(events), models, counts)
    )
    print("  ->", _write([payload], "industry", SINGLE_STEM), "\n")
    return payload


# --------------------------------------------------------------------------
# Multi-agent: supervisor -> collaborator delegation (DEFERRED — see docstring)
# --------------------------------------------------------------------------
def generate_bedrock_agents_multi(client: Stratix) -> dict:
    """Record a genuine multi-agent supervisor->collaborator delegation run.

    DEFERRED unless a provisioned supervisor agent is supplied via
    ``BEDROCK_SUPERVISOR_AGENT_ID`` + ``BEDROCK_SUPERVISOR_AGENT_ALIAS_ID`` — see
    the module docstring. Never fabricates a multi-agent fixture.
    """
    sup_id = os.environ.get("BEDROCK_SUPERVISOR_AGENT_ID")
    sup_alias = os.environ.get("BEDROCK_SUPERVISOR_AGENT_ALIAS_ID")
    if not (sup_id and sup_alias):
        raise RuntimeError(
            "bedrock_agents multi DEFERRED: no provisioned multi-collaborator "
            "supervisor agent. Bedrock multi-agent collaboration requires a "
            "SUPERVISOR agent associated with >=2 collaborator agents "
            "(agentCollaboratorInvocation -> agent.handoff) plus "
            "bedrock:InvokeAgent/GetAgentAlias on the agent execution role; "
            "provisioning that is out of scope for the record-real-once seam. "
            "Set BEDROCK_SUPERVISOR_AGENT_ID + BEDROCK_SUPERVISOR_AGENT_ALIAS_ID "
            "(a provisioned supervisor) to record. The adapter's collaborator "
            "delegation -> agent.handoff path is unit-covered in "
            "test_bedrock_agents_doubles.py."
        )

    payload = _capture_invoke_agent(
        client,
        agent_id=sup_id,
        alias_id=sup_alias,
        input_text=CLAIM_FOR_SUPERVISOR,
        session_prefix="ll-w2-claims-sup",
    )
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "insurance",
        "claims-triage",
        "multi-agent",
    ]
    events = payload.get("events", [])
    handoffs = [
        (
            (e.get("payload") or {}).get("from_agent"),
            (e.get("payload") or {}).get("to_agent"),
        )
        for e in events
        if e.get("event_type") == "agent.handoff"
    ]
    print(
        "  bedrock-agents multi (supervisor->collaborator delegation)  "
        "events=%d handoffs=%s" % (len(events), handoffs)
    )
    print("  ->", _write([payload], "industry", MULTI_STEM), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_bedrock_agents_single(_client)
    try:
        generate_bedrock_agents_multi(_client)
    except RuntimeError as _exc:
        print("  [multi deferred]", _exc)
