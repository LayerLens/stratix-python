"""ADP-W2 Family-B recorders for the **ms_agent_framework** platform adapter.

Records two REAL, fully-instrumented runs of the ``MSAgentFrameworkAdapter``
(the LayerLens adapter for Microsoft Agent Framework — the
``semantic_kernel.agents`` chat surface: ``ChatCompletionAgent`` /
``AgentGroupChat``) and writes each sealed trace to
``samples/data/traces/industry/<stem>.jsonl``. Both fixtures are genuine runs of
the real adapter over real ``semantic_kernel.agents`` chats backed by a real
OpenAI model (gpt-4o-mini) — nothing is fabricated. The framework deps
(``semantic_kernel``) are imported FUNCTION-LOCALLY so this module imports in any
venv (SK is not in the base venv).

NOTE on adapter identity: the ms_agent_framework adapter wraps the agent/chat's
``invoke`` (async-generator) turn-stream via ``instrument_chat`` — a DIFFERENT
code path from the ``semantic_kernel`` adapter (which patches the kernel's
``_inner_get_chat_message_contents`` service call). This module exercises the
``MSAgentFrameworkAdapter`` path specifically.

Two lanes (Energy domain; de-conflicted from the W1 ``energy_grid_*`` stems):

* ``generate_ms_agent_framework_single`` -> ``energy_msagent_forecast``
  A single named ``ChatCompletionAgent`` (``grid-load-forecaster``) invoked
  directly via ``agent.invoke(messages=...)``. The adapter stamps the agent's
  declared name onto ``agent.input`` / ``agent.output`` — so the trace synthesizes
  a real ``agent.identity`` (Agent column = ``grid-load-forecaster``), a single
  1-node graph. Carries the real ``model.invoke`` + ``cost.record`` (token counts)
  of the run. Framework column = ``ms_agent_framework``.

* ``generate_ms_agent_framework_multi`` -> ``energy_msagent_ops``
  A GENUINE multi-agent ``AgentGroupChat``: three named ``ChatCompletionAgent``s
  (``grid-load-forecaster`` -> ``dispatch-optimizer`` -> ``reliability-auditor``)
  adjudicate a grid-contingency event in sequential round-robin turns. The adapter
  wraps the chat's ``invoke`` turn-stream and emits a real ``agent.handoff`` on
  each turn transition plus per-turn ``model.invoke`` / ``cost.record`` — a
  multi-agent DAG whose honest nodes are the three declared specialists (Agent
  column = ``multi-agent``).

HONESTY notes (documented, not bugs):
* The real SK ``message.metadata`` carries token ``usage`` but NO ``model`` id
  (the id lives on ``inner_content``), so ``model.invoke``/``cost.record`` honestly
  report ``model=None`` and ``cost_usd`` stays absent — the adapter emits token
  counts only, exactly as its recorded-shape unit test pins. Not fabricated.
* The group-chat container has no declared name, so the FIRST turn transition
  emits a handoff ``from_agent="AgentGroupChat"`` (a generic container the server
  honesty-guard drops); the substantive DAG is the honest
  forecaster -> optimizer -> auditor chain.
* Both lanes record the NON-streaming ``invoke`` path (``invoke_stream`` reuses the
  same wrapper and can double-emit across partial chunks) — the honest, complete one.
"""

from __future__ import annotations

import os
import sys
import json
import asyncio

# --- path bootstrap so this module runs standalone AND when imported by the
#     central _generate_fixtures.main() ---------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))          # samples/data/generators
_DATA = os.path.dirname(_HERE)                              # samples/data
_SAMPLES = os.path.dirname(_DATA)                           # samples
_REPO = os.path.dirname(_SAMPLES)                           # repo root
for _p in (os.path.join(_REPO, "src"), _DATA, _SAMPLES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from layerlens import Stratix  # noqa: E402  (re-exported for the seam signature)
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

# Reuse the shared record-real-once seam (_write / _CAPTURE) from the central
# fixture generator; fall back to a self-contained copy if it isn't importable.
try:
    from _generate_fixtures import _write, _CAPTURE  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - standalone fallback
    _CAPTURE = CaptureConfig.full()
    _TRACES = os.path.join(_DATA, "traces")

    def _write(payloads, category, stem):  # type: ignore[no-redef]
        out = os.path.join(_TRACES, category, f"{stem}.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            for p in payloads:
                f.write(json.dumps(p, default=str) + "\n")
        return out


OPENAI_MODEL = os.environ.get("SAMPLE_OPENAI_MODEL", "gpt-4o-mini")


def _capture_msaf(build_run, *, pick_marker: str) -> dict:
    """Run a real MS-Agent-Framework scenario under the observer seam (no
    background upload) and return the sealed trace payload.

    The adapter flushes one payload per run at ``_end_run``; we still collect all
    flushed payloads defensively and return the one carrying the marker event
    (``agent.output`` for the single run, ``agent.handoff`` for the multi run),
    falling back to the richest payload."""
    payloads: list = []
    set_trace_observer(lambda p: payloads.append(p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        build_run()
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    if not payloads:
        raise RuntimeError("no payload captured for ms_agent_framework run")
    for p in payloads:
        if any(e.get("event_type") == pick_marker for e in p.get("events", [])):
            return p
    return max(payloads, key=lambda p: len(p.get("events", [])))


def _summarize(payload: dict, label: str) -> None:
    events = payload.get("events", [])
    import collections
    counts = collections.Counter(e.get("event_type") for e in events)
    frameworks = sorted({(e.get("payload") or {}).get("framework")
                         for e in events if e.get("event_type") == "model.invoke"
                         and (e.get("payload") or {}).get("framework")})
    identity = next(((e.get("payload") or {}).get("agent_name")
                     for e in events if e.get("event_type") == "agent.identity"), None)
    handoffs = [((e.get("payload") or {}).get("from_agent"), (e.get("payload") or {}).get("to_agent"))
                for e in events if e.get("event_type") == "agent.handoff"]
    print("  %s  events=%d  model.invoke=%d cost.record=%d" % (
        label, len(events), counts.get("model.invoke", 0), counts.get("cost.record", 0)))
    print("    frameworks=%s  agent.identity=%r" % (frameworks, identity))
    if handoffs:
        print("    handoffs=%s" % (handoffs,))


# ---------------------------------------------------------------------------
# SINGLE: a single grid-load-forecaster ChatCompletionAgent (1-node graph)
# ---------------------------------------------------------------------------
_GRID_SNAPSHOT = (
    "ISO control-room grid snapshot (summer peak day, heat advisory):\n"
    "- Current demand: 38.4 GW at 16:00; load still climbing.\n"
    "- Day-ahead forecast evening peak: 42.1 GW around 18:30.\n"
    "- Available online + quick-start capacity: 44.0 GW.\n"
    "- Operating-reserve requirement: 3.0 GW (largest single contingency 1.2 GW).\n"
    "- One 600 MW combined-cycle unit on a forced derate; wind forecast falling after 19:00.\n"
    "Forecast the evening peak, assess whether the reserve margin meets the "
    "operating-reserve requirement, and recommend a specific proactive action."
)


def generate_ms_agent_framework_single(client: Stratix) -> None:
    """MS Agent Framework SINGLE (one named ChatCompletionAgent).

    A single ``ChatCompletionAgent`` named ``grid-load-forecaster`` assesses an
    ISO control-room grid snapshot and recommends a proactive operator action.
    Invoked directly via ``agent.invoke(messages=...)`` with the real
    ``MSAgentFrameworkAdapter`` attached, so the sealed trace carries the agent's
    declared name (Agent column = ``grid-load-forecaster``, a single 1-node graph)
    plus the real ``model.invoke`` + ``cost.record`` (token counts). Real OpenAI
    (gpt-4o-mini). Framework column = ``ms_agent_framework``."""
    from semantic_kernel import Kernel
    from semantic_kernel.agents import ChatCompletionAgent
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    from layerlens.instrument.adapters.frameworks.ms_agent_framework import (
        MSAgentFrameworkAdapter,
    )

    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id=OPENAI_MODEL))
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="grid-load-forecaster",
        instructions=(
            "You are grid-load-forecaster, a power-grid load-forecasting agent in an "
            "ISO/RTO control room. For the grid snapshot, forecast the evening peak "
            "demand, assess the reserve margin against the operating-reserve "
            "requirement, and recommend ONE specific proactive operator action "
            "(issue a demand-response call, commit a peaker, or import from a "
            "neighboring ISO). Be specific and concise (under 150 words)."
        ),
    )

    adapter = MSAgentFrameworkAdapter(client, capture_config=_CAPTURE)
    adapter.connect()
    adapter.instrument_chat(agent)

    async def _drive():
        async for message in agent.invoke(messages=_GRID_SNAPSHOT):
            print("    %s: %s" % (getattr(message, "name", "?"),
                                  str(getattr(message, "content", ""))[:80]))

    def _build_run():
        try:
            asyncio.run(_drive())
        finally:
            try:
                adapter.disconnect()
            except Exception:
                pass

    payload = _capture_msaf(_build_run, pick_marker="agent.output")
    payload["tags"] = [
        "layerlens-sample", "industry", "energy", "load-forecasting", "single-agent",
    ]
    _summarize(payload, "ms_agent_framework-single (ChatCompletionAgent)")
    print("  ->", _write([payload], "industry", "energy_msagent_forecast"), "\n")


# ---------------------------------------------------------------------------
# MULTI: grid-ops adjudication AgentGroupChat (genuine multi-agent DAG)
# ---------------------------------------------------------------------------
_CONTINGENCY = (
    "Grid contingency to adjudicate (summer peak, heat wave):\n"
    "- Forecast evening peak demand: 42.0 GW; available capacity 44.0 GW.\n"
    "- A 1.2 GW nuclear unit just tripped offline (N-1 event); reserve margin now thin.\n"
    "- Two peakers (0.4 GW each) available on 10-minute start; a 300 MW interchange "
    "import is offered from the neighboring ISO at a high price.\n"
    "- Transmission on the western corridor is at 92% of its stability limit.\n"
    "Adjudicate the dispatch response and confirm reliability compliance."
)


def generate_ms_agent_framework_multi(client: Stratix) -> None:
    """MS Agent Framework MULTI (genuine multi-agent ``AgentGroupChat``).

    Three named ``ChatCompletionAgent``s — ``grid-load-forecaster`` ->
    ``dispatch-optimizer`` -> ``reliability-auditor`` — adjudicate a grid
    contingency (an N-1 unit trip during a heat wave) in sequential round-robin
    turns of a real ``AgentGroupChat``. The ``MSAgentFrameworkAdapter`` wraps the
    chat's ``invoke`` turn-stream and emits a real ``agent.handoff`` on each turn
    transition plus per-turn ``model.invoke`` / ``cost.record`` — a multi-agent
    DAG whose honest nodes are the three declared specialists (Agent column =
    ``multi-agent``). Real OpenAI (gpt-4o-mini). Framework = ``ms_agent_framework``."""
    from semantic_kernel import Kernel
    from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
    from semantic_kernel.agents.strategies import (
        SequentialSelectionStrategy,
        DefaultTerminationStrategy,
    )
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    from layerlens.instrument.adapters.frameworks.ms_agent_framework import (
        MSAgentFrameworkAdapter,
    )

    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id=OPENAI_MODEL))
    forecaster = ChatCompletionAgent(
        kernel=kernel, name="grid-load-forecaster",
        instructions=("You are the grid load forecaster. State the net demand-vs-supply "
                      "position and the reserve margin after the contingency. 2 sentences."),
    )
    optimizer = ChatCompletionAgent(
        kernel=kernel, name="dispatch-optimizer",
        instructions=("You are the dispatch optimizer. Recommend a concrete least-cost "
                      "dispatch (which peakers / imports to commit) to restore reserves. "
                      "2 sentences."),
    )
    auditor = ChatCompletionAgent(
        kernel=kernel, name="reliability-auditor",
        instructions=("You are the reliability auditor. Confirm whether the proposed "
                      "dispatch keeps N-1 contingency and transmission limits within NERC "
                      "reliability standards, and flag any residual risk. 2 sentences."),
    )

    adapter = MSAgentFrameworkAdapter(client, capture_config=_CAPTURE)
    adapter.connect()

    async def _drive():
        chat = AgentGroupChat(
            agents=[forecaster, optimizer, auditor],
            selection_strategy=SequentialSelectionStrategy(),
            termination_strategy=DefaultTerminationStrategy(maximum_iterations=3),
        )
        adapter.instrument_chat(chat)
        await chat.add_chat_message(message=_CONTINGENCY)
        async for message in chat.invoke():
            print("    %s: %s" % (getattr(message, "name", "?"),
                                  str(getattr(message, "content", ""))[:70]))

    def _build_run():
        try:
            asyncio.run(_drive())
        finally:
            try:
                adapter.disconnect()
            except Exception:
                pass

    payload = _capture_msaf(_build_run, pick_marker="agent.handoff")
    payload["tags"] = [
        "layerlens-sample", "industry", "energy", "grid-operations", "multi-agent",
    ]
    _summarize(payload, "ms_agent_framework-multi (AgentGroupChat grid-ops)")
    print("  ->", _write([payload], "industry", "energy_msagent_ops"), "\n")


if __name__ == "__main__":
    _client = Stratix()
    generate_ms_agent_framework_single(_client)
    generate_ms_agent_framework_multi(_client)
