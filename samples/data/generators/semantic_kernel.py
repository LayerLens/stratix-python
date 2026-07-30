"""ADP-W2 Family-B recorders for the **semantic_kernel** framework adapter.

Records two REAL, fully-instrumented ``semantic-kernel`` runs and writes each
sealed trace to ``samples/data/traces/industry/<stem>.jsonl``. Both fixtures are
genuine runs of the real ``SemanticKernelAdapter`` over a real ``semantic_kernel``
``Kernel`` / ``AgentGroupChat`` backed by a real OpenAI model (gpt-4o-mini) —
nothing is fabricated. The framework deps (``semantic_kernel``) are imported
FUNCTION-LOCALLY so this module imports in any venv (SK is not in the base venv).

Two lanes (Healthcare domain; de-conflicted from the W1 ``healthcare_clinical*``
and ``clinical_consult_team`` stems):

* ``generate_semantic_kernel_single`` -> ``healthcare_sk_triage``
  A single SK ``Kernel`` clinical-intake triage assistant. A prompt-function
  (``triage.assess``) is invoked via ``kernel.invoke`` with
  ``FunctionChoiceBehavior.Auto`` so the model AUTO-INVOKES a real native plugin
  function (``ClinicalProtocols.lookup_triage_protocol``) through the SK filter
  API, then grounds an ESI acuity level + immediate next step in the returned
  protocol. Renders a single honest waterfall: real ``model.invoke`` (×2 rounds)
  + priced ``cost.record`` + the auto-invoked ``tool.call`` / ``tool.result`` +
  ``agent.code`` (prompt render). Framework column = ``semantic_kernel``.

  HONESTY: SK's single kernel-function path declares NO agent identity (it emits
  no ``agent.identity``/``agent_name``), so the Agent column renders honestly
  EMPTY (—) — like a provider trace. We do NOT invent an agent for it.

* ``generate_semantic_kernel_multi`` -> ``healthcare_sk_care_panel``
  A GENUINE multi-agent ``AgentGroupChat``: three named ``ChatCompletionAgent``s
  (``triage-nurse`` -> ``attending-physician`` -> ``clinical-pharmacist``) take
  turns over one ED case. The adapter wraps the chat's ``invoke`` turn-stream and
  emits honest per-turn ``agent.input`` / ``model.invoke`` / ``cost.record`` /
  ``agent.output`` (stamped with each agent's declared name) plus real
  ``agent.handoff`` edges on each turn transition — a 3-node multi-agent DAG with
  2 handoff edges (Agent column = ``multi-agent``).

Both lanes record the NON-streaming path: the SK adapter patches only the
non-streaming ``_inner_get_chat_message_contents`` (streaming chat completions
are uninstrumented and would drop all ``model.invoke``/``cost.record``), so the
non-streaming path is the honest, complete one.
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

import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens import Stratix  # noqa: E402  (re-exported for the seam signature)
from layerlens.instrument._collector import set_trace_observer  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402

# Reuse the shared record-real-once seam (_write / _CAPTURE) from the central
# fixture generator; fall back to a self-contained copy if it isn't importable.
try:
    from _generate_fixtures import _CAPTURE, _write  # type: ignore[attr-defined]
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


def _capture_sk(build_run, *, pick_marker: str) -> dict:
    """Run a real SK scenario under the observer seam (no background upload) and
    return the sealed trace payload.

    SK's ``_discover_plugins`` flushes plugin ``environment.config`` as its OWN
    short run at ``connect()`` time, separate from the substantive invoke/chat
    run — so we collect ALL flushed payloads and return the one carrying the
    marker event (``model.invoke`` for the single kernel run, ``agent.handoff``
    for the multi group-chat run), falling back to the richest payload.
    """
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
        raise RuntimeError("no payload captured for semantic_kernel run")
    for p in payloads:
        if any(e.get("event_type") == pick_marker for e in p.get("events", [])):
            return p
    return max(payloads, key=lambda p: len(p.get("events", [])))


def _summarize(payload: dict, label: str) -> None:
    events = payload.get("events", [])
    import collections
    counts = collections.Counter(e.get("event_type") for e in events)
    agents = sorted({(e.get("payload") or {}).get("agent_name")
                     for e in events if (e.get("payload") or {}).get("agent_name")})
    frameworks = sorted({(e.get("payload") or {}).get("framework")
                         for e in events if e.get("event_type") == "model.invoke"
                         and (e.get("payload") or {}).get("framework")})
    handoffs = [((e.get("payload") or {}).get("from_agent"), (e.get("payload") or {}).get("to_agent"))
                for e in events if e.get("event_type") == "agent.handoff"]
    tools = sorted({(e.get("payload") or {}).get("tool_name")
                    for e in events if e.get("event_type") in ("tool.call", "tool.result")
                    and (e.get("payload") or {}).get("tool_name")})
    print("  %s  events=%d  model.invoke=%d cost.record=%d" % (
        label, len(events), counts.get("model.invoke", 0), counts.get("cost.record", 0)))
    print("    frameworks=%s agents=%s" % (frameworks, agents))
    if tools:
        print("    tools=%s" % (tools,))
    if handoffs:
        print("    handoffs=%s" % (handoffs,))


# ---------------------------------------------------------------------------
# SINGLE: SK Kernel clinical-intake triage assistant (auto-invoked native tool)
# ---------------------------------------------------------------------------
_PRESENTATION = (
    "62yo male, exertional chest pressure radiating to the left arm, diaphoretic, "
    "BP 148 over 92, on lisinopril for hypertension; troponin pending."
)


def generate_semantic_kernel_single(client: Stratix) -> None:
    """SemanticKernel SINGLE (kernel-function + auto-invoked native tool).

    A single SK ``Kernel`` intake-triage assistant. A prompt-function
    (``triage.assess``) invoked via ``kernel.invoke`` with
    ``FunctionChoiceBehavior.Auto`` lets the model AUTO-INVOKE a real native
    plugin (``ClinicalProtocols.lookup_triage_protocol``) through the SK filter
    API, then ground an ESI acuity level + next step in the returned protocol.
    Real OpenAI (gpt-4o-mini), recorded under the real SemanticKernelAdapter ->
    a single honest waterfall (real ``model.invoke`` + priced ``cost.record`` +
    auto-invoked ``tool.call``/``tool.result`` + ``agent.code``). Framework =
    ``semantic_kernel``; Agent column honestly EMPTY (SK's kernel path declares
    no agent identity — not fabricated)."""
    from semantic_kernel import Kernel
    from semantic_kernel.functions import kernel_function
    from semantic_kernel.connectors.ai.open_ai import (
        OpenAIChatCompletion,
        OpenAIChatPromptExecutionSettings,
    )
    from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

    from layerlens.instrument.adapters.frameworks.semantic_kernel import SemanticKernelAdapter

    class ClinicalProtocols:
        @kernel_function(
            name="lookup_triage_protocol",
            description="Look up the ESI triage protocol and red flags for a chief complaint.",
        )
        def lookup_triage_protocol(self, chief_complaint: str) -> str:
            """REAL native tool: return the ESI protocol + red flags for a complaint."""
            return json.dumps({
                "complaint": chief_complaint,
                "esi_level": 2,
                "red_flags": ["radiation to arm", "diaphoresis", "exertional onset"],
                "guidance": (
                    "Immediate 12-lead ECG + troponin; place on continuous cardiac "
                    "monitor; aspirin 324mg chewed if no contraindication; page cardiology."
                ),
            })

    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id=OPENAI_MODEL))
    kernel.add_plugin(ClinicalProtocols(), "ClinicalProtocols")
    settings = OpenAIChatPromptExecutionSettings(
        max_tokens=350,
        function_choice_behavior=FunctionChoiceBehavior.Auto(maximum_auto_invoke_attempts=2),
    )
    # Patient inlined into the prompt (no ``{{$var}}``) to avoid SK's prompt-
    # injection content-encoder guard; keeps the run deterministic + bounded.
    triage = kernel.add_function(
        plugin_name="triage",
        function_name="assess",
        prompt=(
            "You are clinical-intake-assistant, an ED intake triage assistant. "
            "For the patient below, FIRST call lookup_triage_protocol with the chief "
            "complaint, THEN return an ESI acuity level (1-5) and the immediate next "
            "step, grounded ONLY in the returned protocol. Answer concisely.\n\n"
            "Patient: " + _PRESENTATION
        ),
        prompt_execution_settings=settings,
    )

    adapter = SemanticKernelAdapter(client, capture_config=_CAPTURE)

    def _build_run():
        adapter.connect(target=kernel)
        try:
            result = asyncio.run(kernel.invoke(triage))
            print("    triage output: %r" % (str(result)[:120],))
        finally:
            try:
                adapter.disconnect()
            except Exception:
                pass

    payload = _capture_sk(_build_run, pick_marker="model.invoke")
    payload["tags"] = [
        "layerlens-sample", "industry", "healthcare", "clinical-triage", "tool-use",
    ]
    # HONEST provenance: SK's single kernel-function path declares no agent
    # identity, so the Agent column renders empty (—) — this is a framework-level
    # tool-use waterfall, not an agent DAG. Not fabricated.
    payload["metadata"] = {
        "topology": "single-agent-kernel-tool-use",
        "agent_column": "empty",
        "reason": "SemanticKernel's kernel-function path emits no agent.identity; "
                  "the Agent column is honestly empty. Framework=semantic_kernel, "
                  "with a real model.invoke + auto-invoked native tool loop.",
    }
    _summarize(payload, "semantic_kernel-single (kernel + auto native tool)")
    print("  ->", _write([payload], "industry", "healthcare_sk_triage"), "\n")


# ---------------------------------------------------------------------------
# MULTI: SK AgentGroupChat clinical care panel (genuine multi-agent DAG)
# ---------------------------------------------------------------------------
_CASE = (
    "62yo male, exertional chest pressure, controlled hypertension on lisinopril, "
    "troponin pending. Provide a triage read, an attending physician assessment, "
    "and a medication-safety check."
)


def generate_semantic_kernel_multi(client: Stratix) -> None:
    """SemanticKernel MULTI (genuine multi-agent ``AgentGroupChat``).

    Three named ``ChatCompletionAgent``s — ``triage-nurse`` ->
    ``attending-physician`` -> ``clinical-pharmacist`` — take turns over one ED
    case in a real ``AgentGroupChat``. The SemanticKernelAdapter wraps the chat's
    ``invoke`` turn-stream and emits honest per-turn ``agent.input`` /
    ``model.invoke`` / ``cost.record`` / ``agent.output`` (stamped with each
    agent's declared name) plus a real ``agent.handoff`` on each turn transition —
    a 3-node multi-agent DAG with 2 honest handoff edges (Agent column =
    ``multi-agent``). Real OpenAI (gpt-4o-mini)."""
    from semantic_kernel import Kernel
    from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
    from semantic_kernel.agents.strategies import DefaultTerminationStrategy
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    from layerlens.instrument.adapters.frameworks.semantic_kernel import SemanticKernelAdapter

    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id=OPENAI_MODEL))
    nurse = ChatCompletionAgent(
        kernel=kernel, name="triage-nurse",
        instructions=("You are the triage nurse. Summarize the presentation and assign an "
                      "ESI acuity level (1-5). 2 sentences."),
    )
    physician = ChatCompletionAgent(
        kernel=kernel, name="attending-physician",
        instructions=("You are the attending physician. Give a focused assessment and the "
                      "immediate orders. 2 sentences."),
    )
    pharmacist = ChatCompletionAgent(
        kernel=kernel, name="clinical-pharmacist",
        instructions=("You are the clinical pharmacist. Flag any medication or interaction "
                      "concerns given the patient's lisinopril. 1-2 sentences."),
    )

    adapter = SemanticKernelAdapter(client, capture_config=_CAPTURE)

    async def _drive():
        chat = AgentGroupChat(
            agents=[nurse, physician, pharmacist],
            termination_strategy=DefaultTerminationStrategy(maximum_iterations=3),
        )
        adapter.connect(target=chat)
        await chat.add_chat_message(message=_CASE)
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

    payload = _capture_sk(_build_run, pick_marker="agent.handoff")
    payload["tags"] = [
        "layerlens-sample", "industry", "healthcare", "clinical-triage", "multi-agent",
    ]
    _summarize(payload, "semantic_kernel-multi (AgentGroupChat care panel)")
    print("  ->", _write([payload], "industry", "healthcare_sk_care_panel"), "\n")


if __name__ == "__main__":
    _client = Stratix()
    generate_semantic_kernel_single(_client)
    generate_semantic_kernel_multi(_client)
