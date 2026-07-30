"""ADP-W2 Family-B recorder for the ``autogen`` adapter (record-real-once).

Records TWO real, fully-instrumented AutoGen (autogen-agentchat >= 0.4) runs and
writes each as a sealed real-trace fixture under ``samples/data/traces/industry/``
(Telecom domain):

* ``generate_autogen_single`` -> ``telecom_autogen_triage.jsonl``: a single
  AutoGen ``AssistantAgent`` (``telecom_support_agent``, backed by
  ``OpenAIChatCompletionClient``) running a REAL tool-use turn — it calls the
  ``lookup_account`` function tool to fetch the customer's plan + recent charges,
  then answers the billing question grounded in the tool result
  (``reflect_on_tool_use=True`` so the model produces a final text answer after
  the tool result). Renders a single honest agent node (Agent column =
  ``telecom_support_agent``) with the real ``model.invoke`` / ``cost.record`` /
  ``tool.call`` events of the turn.

  The one agent is driven inside a real one-member ``RoundRobinGroupChat`` (rather
  than a bare ``AssistantAgent.run()``) BECAUSE the honest ``agent.identity`` is
  only emitted when autogen's ``SingleThreadedAgentRuntime`` assigns the agent a
  real ``AgentId`` — which happens inside a team, not for a standalone
  ``.run()``. A bare standalone run genuinely carries NO agent_id on its
  ``LLMCallEvent`` (verified), so it would honestly render Agent = empty-state;
  we do NOT stamp a name the runtime never assigned. The one-member team is a
  genuine autogen run whose single ``telecom_support_agent`` node + identity are
  the real ``AgentId`` the runtime assigned (team-UUID stripped by
  ``_autogen_agent_name``).

* ``generate_autogen_multi`` -> ``telecom_autogen_groupchat.jsonl``: a genuine
  multi-agent run — a real ``RoundRobinGroupChat`` support panel where three
  named agents (``triage_agent`` -> ``billing_specialist`` -> ``network_specialist``)
  each read the running conversation and contribute their turn on a mixed
  billing + connectivity complaint. The AutoGenAdapter taps autogen_core's
  module-global EVENT_LOGGER, so the trace carries THREE distinct honest agent
  identities — each with its OWN real ``model.invoke`` / ``cost.record`` (the
  per-agent LLM call the runtime logged). It renders as a multi-agent graph
  (Agent column ``multi-agent``) built from the >= 2 real agent nodes.

Both are recorded through the REAL ``AutoGenAdapter`` (a ``logging.Handler`` on
autogen_core's ``EVENT_LOGGER_NAME`` that self-flushes each run as its own trace
on ``disconnect()``); the flush is observed via the ``_generate_fixtures`` capture
seam (``set_trace_observer`` + a no-op ``enqueue_upload``) so the sealed payload —
real per-node ``model.invoke``/``cost.record`` (gpt-4o-mini pricing applied by the
adapter's ``_price_cost_record`` chokepoint, so ``cost.record.cost_usd`` is a real
non-None figure) + a real synthesized ``agent.identity`` + an intact attestation
chain — is captured but never uploaded during generation. The samples upload the
captured fixtures themselves at run time.

Nothing is fabricated: the Framework column shows the model the runtime really
called, the token/cost fields are real, and the multi-agent nodes are the real
per-agent turns the group chat produced (each agent's honest ``agent_name`` is the
team-runtime UUID stripped by ``_autogen_agent_name`` from the real ``AgentId``).

HONEST-BY-DESIGN NOTE — NO ``agent.handoff`` EDGES (autogen-agentchat 0.4-0.7):
    A real AutoGen team does NOT emit an honest agent->agent ``agent.handoff``
    edge through this adapter, and this recorder does not invent one. Modern
    autogen-agentchat routes every inter-agent message through the
    ``GroupChatManager`` + ``group_topic``/``output_topic`` pub-sub plumbing (and
    implements ``Swarm`` handoffs as a ``transfer_to_*`` TOOL call), so every
    ``MessageEvent`` the adapter sees has the manager/topic as one endpoint —
    which ``_autogen_agent_name`` correctly resolves to ``None`` (plumbing is not
    an agent). The adapter only fires ``agent.handoff`` when BOTH endpoints are
    real, distinct agents, which never happens in a real group-chat/swarm run.
    Empirically confirmed on both ``RoundRobinGroupChat`` and ``Swarm`` (0
    handoffs) — matching the shipped ``cowork/content_pipeline_team`` autogen
    fixture (0 handoffs) and the ``autogen-s2`` render oracle (which renders
    ``multi-agent`` from the >= 2 agent NODES, not from edges). The multi-agent
    topology here is therefore proven by the distinct real agent nodes, exactly
    as the atlas graph engine expects for autogen. (The adapter's ``agent.handoff``
    path is exercised by the unit tests, which construct direct agent->agent
    ``MessageEvent`` objects that the real runtime does not produce.)
"""

from __future__ import annotations

import json
import os
import sys

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config + model name).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# This module is named ``autogen.py``. When it is run directly, Python inserts its
# own directory at ``sys.path[0]``; drop it so a stray ``import autogen`` (the old
# pyautogen package name) can never resolve to this file instead of an installed
# package. A no-op when imported as ``generators.autogen``.
sys.path[:] = [_p for _p in sys.path if os.path.abspath(_p or ".") != _HERE]

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE
OPENAI_MODEL = _gf.OPENAI_MODEL


# --------------------------------------------------------------------------
# Adapter-driven capture: AutoGenAdapter is a logging.Handler on autogen_core's
# EVENT_LOGGER that self-flushes each run as its own trace on ``disconnect()``.
# We connect it, drive a REAL ``asyncio.run(coro)``, and observe the flushed
# payload via the shared capture seam (no background upload).
# --------------------------------------------------------------------------
def _capture_autogen(client: Stratix, coro) -> dict:
    import asyncio

    from layerlens.instrument.adapters.frameworks.autogen import AutoGenAdapter

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = AutoGenAdapter(client, capture_config=_CAPTURE)
    try:
        adapter.connect()
        asyncio.run(coro)
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for autogen run")
    return payload


def _summ(events: list) -> tuple:
    idents = sorted(
        {(e.get("payload") or {}).get("agent_name") for e in events
         if (e.get("payload") or {}).get("agent_name")}
        - {None}
    )
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    tools = sorted(
        {(e.get("payload") or {}).get("tool_name") for e in events
         if e.get("event_type") == "tool.call"}
        - {None}
    )
    handoffs = [
        ((e.get("payload") or {}).get("from_agent"), (e.get("payload") or {}).get("to_agent"))
        for e in events
        if e.get("event_type") == "agent.handoff"
    ]
    return idents, mi, cr, tools, handoffs


# --------------------------------------------------------------------------
# A small, realistic (non-sensitive) telecom account book the ``lookup_account``
# tool reads from, so the single agent's tool-use turn answers the billing
# question grounded in a genuine account record.
# --------------------------------------------------------------------------
_ACCOUNT_BOOK = {
    "ACCT-55231": {
        "account_id": "ACCT-55231",
        "plan": "Unlimited Fiber 1Gbps",
        "monthly_price_usd": 79.99,
        "autopay": True,
        "recent_charges": [
            {"date": "2026-06-01", "desc": "Monthly plan", "amount_usd": 79.99},
            {"date": "2026-06-14", "desc": "Monthly plan (duplicate)", "amount_usd": 79.99},
        ],
        "contract_end": "2027-03-01",
        "early_termination_fee_usd": 120,
    },
    "ACCT-90017": {
        "account_id": "ACCT-90017",
        "plan": "Mobile 5G 20GB",
        "monthly_price_usd": 45.00,
        "autopay": False,
        "recent_charges": [
            {"date": "2026-06-03", "desc": "Monthly plan", "amount_usd": 45.00},
            {"date": "2026-06-03", "desc": "Overage 2GB", "amount_usd": 20.00},
        ],
        "contract_end": None,
        "early_termination_fee_usd": 0,
    },
}


# --------------------------------------------------------------------------
# Single agent + a real ``lookup_account`` tool turn (telecom billing triage)
# --------------------------------------------------------------------------
def generate_autogen_single(client: Stratix) -> dict:
    """Record a single AutoGen ``telecom_support_agent`` running a real tool-use
    turn: it calls ``lookup_account`` to fetch the customer's plan + recent
    charges, then answers the billing question grounded in the tool result.

    Driven inside a one-member ``RoundRobinGroupChat`` so autogen's runtime
    assigns the agent a real ``AgentId`` and the honest ``agent.identity`` is
    emitted (see module docstring)."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    def lookup_account(account_id: str) -> str:
        """Look up a telecom customer's account: plan, monthly price, autopay,
        recent charges, and contract terms, by account_id (e.g. 'ACCT-55231')."""
        rec = _ACCOUNT_BOOK.get((account_id or "").strip())
        if rec is None:
            return json.dumps({"account_id": account_id, "found": False})
        return json.dumps(rec)

    mc = OpenAIChatCompletionClient(model=OPENAI_MODEL)
    agent = AssistantAgent(
        "telecom_support_agent",
        model_client=mc,
        tools=[lookup_account],
        reflect_on_tool_use=True,
        system_message=(
            "You are a telecom customer-support agent. For a billing question, "
            "FIRST call the lookup_account tool with the stated account_id to "
            "fetch the plan and recent charges, THEN answer the customer — "
            "identify any duplicate/incorrect charge, state the correct amount "
            "owed, and give a concrete next step — grounded ONLY in the account "
            "the tool returned. Answer concisely (under 120 words)."
        ),
    )
    team = RoundRobinGroupChat(
        [agent], termination_condition=MaxMessageTermination(2)
    )
    task = (
        "Account ACCT-55231: I think I was double-charged $79.99 on my last bill. "
        "Can you check my recent charges and tell me what I actually owe and how "
        "you'll fix it?"
    )
    payload = _capture_autogen(client, team.run(task=task))
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "telecom",
        "customer-support",
        "tool-use",
    ]
    events = payload.get("events", [])
    idents, mi, cr, tools, handoffs = _summ(events)
    print("  autogen single (telecom_support_agent, tool-use)  "
          "events=%d agents=%s tools=%s model.invoke=%d cost.record=%d"
          % (len(events), idents, tools, len(mi), len(cr)))
    print("  ->", _write([payload], "industry", "telecom_autogen_triage"), "\n")
    return payload


# --------------------------------------------------------------------------
# Multi-agent: a real RoundRobinGroupChat telecom support panel (3 named agents)
# --------------------------------------------------------------------------
def generate_autogen_multi(client: Stratix) -> dict:
    """Record a genuine multi-agent AutoGen ``RoundRobinGroupChat``: a telecom
    support panel (triage_agent -> billing_specialist -> network_specialist)
    collaborates on a mixed billing + connectivity complaint. Each agent takes a
    real turn -> three distinct honest agent nodes (renders ``multi-agent``)."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    mc = OpenAIChatCompletionClient(model=OPENAI_MODEL)
    triage_agent = AssistantAgent(
        "triage_agent",
        model_client=mc,
        system_message=(
            "You are the triage agent on a telecom support panel. In ONE sentence, "
            "restate the customer's complaint and say which parts are a billing "
            "issue and which are a connectivity issue."
        ),
    )
    billing_specialist = AssistantAgent(
        "billing_specialist",
        model_client=mc,
        system_message=(
            "You are the billing specialist. In one or two sentences, resolve the "
            "billing part of the complaint (identify the duplicate/incorrect charge "
            "and state the refund/adjustment)."
        ),
    )
    network_specialist = AssistantAgent(
        "network_specialist",
        model_client=mc,
        system_message=(
            "You are the network specialist. In two sentences, resolve the "
            "connectivity part of the complaint with concrete remediation steps, "
            "then give the customer a single combined next step."
        ),
    )
    team = RoundRobinGroupChat(
        [triage_agent, billing_specialist, network_specialist],
        termination_condition=MaxMessageTermination(4),
    )
    task = (
        "Customer complaint: I was double-charged $79.99 on my last bill AND my "
        "home internet keeps dropping every evening around 8pm. Please resolve "
        "both issues."
    )
    payload = _capture_autogen(client, team.run(task=task))
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "telecom",
        "customer-support",
        "multi-agent",
    ]
    events = payload.get("events", [])
    idents, mi, cr, tools, handoffs = _summ(events)
    print("  autogen multi (RoundRobinGroupChat: triage/billing/network)  "
          "events=%d agents=%s handoffs=%s model.invoke=%d cost.record=%d"
          % (len(events), idents, handoffs, len(mi), len(cr)))
    print("  ->", _write([payload], "industry", "telecom_autogen_groupchat"), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_autogen_single(_client)
    generate_autogen_multi(_client)
