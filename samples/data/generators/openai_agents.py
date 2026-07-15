"""ADP-W2 Family-B recorder for the ``openai_agents`` adapter (record-real-once).

Records TWO real, fully-instrumented OpenAI Agents SDK runs and writes each as a
sealed real-trace fixture under ``samples/data/traces/industry/``:

* ``generate_openai_agents_single`` -> ``retail_openai_agents_orders.jsonl``:
  a single retail ``order-support-agent`` that calls one real ``@function_tool``
  (``lookup_order``) and answers the shopper. Renders a single honest agent node
  (Agent column = ``order-support-agent``) with the real
  ``model.invoke`` / ``cost.record`` / ``tool.call`` / ``tool.result`` events of
  the two-step tool loop.

* ``generate_openai_agents_multi`` -> ``retail_openai_agents_triage.jsonl``: a
  genuine multi-agent run — a ``triage-agent`` (guarded by a real
  ``@input_guardrail``) hands off a product-return request to a
  ``returns-specialist`` (which calls a real ``check_return_eligibility`` tool).
  The handoff span records a real ``agent.handoff`` (triage-agent ->
  returns-specialist) and the guardrail records a real ``evaluation.result``, so
  the trace renders as a multi-agent DAG (>=2 agent nodes + a handoff edge).

Both are recorded through the REAL ``OpenAIAgentsAdapter`` (which *is* the SDK's
global ``TracingProcessor``): the adapter builds its own per-trace collector and
flushes it on ``on_trace_end``, and the flush is observed via the ``_generate_
fixtures`` capture seam (``set_trace_observer`` + a no-op ``enqueue_upload``) so
the sealed payload — real ``agent.identity`` (synthesized at flush from the
declared agent name) + intact attestation chain — is captured but never uploaded
during generation. The samples upload the captured fixtures themselves at run
time. Nothing is fabricated: the Framework column shows ``openai-agents`` (the
SDK that really ran), the token/cost fields are real, and the multi-agent nodes
and handoff edge are the real agents/handoff the SDK emitted.

IMPORTANT — the OpenAI Agents SDK ``response`` span (the default *Responses* API
path) maps to ``model.invoke`` WITHOUT a ``cost.record``; only the *chat
completions* ``generation`` span emits ``cost.record``. So both recorders select
``chat_completions`` via the SDK's own ``set_default_openai_api`` so the sealed
fixture carries real ``cost.record`` events — the honest, non-lossy path.
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

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE
OPENAI_MODEL = _gf.OPENAI_MODEL


# --------------------------------------------------------------------------
# Adapter-driven capture: the OpenAI Agents adapter IS the SDK's global
# TracingProcessor and flushes its own collector on trace-end. We register it,
# drive a REAL ``Runner.run_sync``, and observe the flushed payload — mirroring
# the crewai/autogen recorders in _generate_fixtures.py (self-flushing adapters).
# --------------------------------------------------------------------------
def _capture_openai_agents(client: Stratix, root_agent, prompt: str) -> dict:
    from agents import Runner, set_trace_processors
    from layerlens.instrument.adapters.frameworks.openai_agents import (
        OpenAIAgentsAdapter,
    )

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    # Suppress the SDK's default OpenAI backend trace exporter so generation does
    # not ship the recorded spans to OpenAI's tracing endpoint; only our adapter
    # receives the spans (model calls are unaffected — they do not go through the
    # trace processors).
    set_trace_processors([])
    adapter = OpenAIAgentsAdapter(client=client, capture_config=_CAPTURE)
    adapter.connect()
    try:
        Runner.run_sync(root_agent, prompt)
    finally:
        adapter.disconnect()
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for openai_agents run")
    return payload


# --------------------------------------------------------------------------
# Single agent + one tool (retail order support)
# --------------------------------------------------------------------------
def generate_openai_agents_single(client: Stratix) -> dict:
    """Record a single retail ``order-support-agent`` that calls one real tool."""
    from agents import Agent, function_tool, set_default_openai_api, ModelSettings

    # chat_completions -> generation spans -> real cost.record (see module docstring).
    set_default_openai_api("chat_completions")

    ORDER_DB = {
        "ORD-10432": {
            "order_id": "ORD-10432",
            "status": "in_transit",
            "carrier": "UPS",
            "eta": "2026-07-16",
            "shipped": True,
            "address_change_locked": True,
        },
        "ORD-77219": {
            "order_id": "ORD-77219",
            "status": "processing",
            "carrier": None,
            "eta": "2026-07-19",
            "shipped": False,
            "address_change_locked": False,
        },
    }

    @function_tool
    def lookup_order(order_id: str) -> str:
        """Look up the live status, carrier, ETA and whether the shipping
        address can still be changed for a customer order by its ID."""
        rec = ORDER_DB.get(
            order_id.strip().upper(),
            {"order_id": order_id, "found": False, "message": "No such order."},
        )
        return json.dumps(rec)

    agent = Agent(
        name="order-support-agent",
        instructions=(
            "You are order-support-agent for an online retailer. For the "
            "customer's question, FIRST call the lookup_order tool with the "
            "order ID, THEN answer their question grounded ONLY in the returned "
            "record (status, carrier, ETA, and whether the shipping address can "
            "still be changed). Answer concisely (under 80 words)."
        ),
        model=OPENAI_MODEL,
        tools=[lookup_order],
        model_settings=ModelSettings(max_tokens=220),
    )

    prompt = (
        "Where is my order ORD-10432 and can I still change the shipping address?"
    )
    payload = _capture_openai_agents(client, agent, prompt)
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "retail",
        "order-support",
        "tool-use",
    ]
    events = payload.get("events", [])
    tools = sorted(
        {(e.get("payload") or {}).get("tool_name") for e in events
         if e.get("event_type") == "tool.call"}
        - {None}
    )
    print("  openai-agents single (order-support-agent, tool-use)  "
          "events=%d tools=%s" % (len(events), tools))
    print("  ->", _write([payload], "industry", "retail_openai_agents_orders"), "\n")
    return payload


# --------------------------------------------------------------------------
# Multi-agent: triage -> specialist handoff + input guardrail (retail support)
# --------------------------------------------------------------------------
def generate_openai_agents_multi(client: Stratix) -> dict:
    """Record a genuine multi-agent retail support run: a guarded ``triage-agent``
    hands off a product-return request to a ``returns-specialist`` (real
    handoff + input guardrail + a specialist tool call)."""
    from agents import (
        Agent,
        function_tool,
        input_guardrail,
        GuardrailFunctionOutput,
        RunContextWrapper,
        set_default_openai_api,
        ModelSettings,
    )

    set_default_openai_api("chat_completions")

    @function_tool
    def check_return_eligibility(order_id: str) -> str:
        """Check whether a delivered order is within its return window and the
        refund amount the customer would receive."""
        return json.dumps(
            {
                "order_id": order_id.strip().upper(),
                "eligible": True,
                "return_window_days_left": 12,
                "refund_usd": 64.99,
                "reason_required": False,
                "method": "prepaid_label",
            }
        )

    @input_guardrail
    async def retail_relevance_guardrail(
        ctx: RunContextWrapper, agent: "Agent", input_data
    ) -> GuardrailFunctionOutput:
        """Real input guardrail: allow genuine retail-support requests, trip on a
        prompt-injection attempt. Records an ``evaluation.result`` either way."""
        text = input_data if isinstance(input_data, str) else str(input_data)
        injection = "ignore your instructions" in text.lower()
        return GuardrailFunctionOutput(
            output_info={"on_topic_retail_support": not injection},
            tripwire_triggered=injection,
        )

    returns_specialist = Agent(
        name="returns-specialist",
        instructions=(
            "You are returns-specialist. For a product-return request, FIRST "
            "call check_return_eligibility with the order ID, THEN tell the "
            "customer whether they can return it, the refund amount, and how "
            "(the return method). Answer concisely (under 80 words)."
        ),
        model=OPENAI_MODEL,
        tools=[check_return_eligibility],
        model_settings=ModelSettings(max_tokens=240),
    )
    triage = Agent(
        name="triage-agent",
        instructions=(
            "You are triage-agent for retail customer support. For any product "
            "return or refund request, hand off to returns-specialist — do not "
            "answer return questions yourself. Handle only routing."
        ),
        model=OPENAI_MODEL,
        handoffs=[returns_specialist],
        input_guardrails=[retail_relevance_guardrail],
        model_settings=ModelSettings(max_tokens=200),
    )

    prompt = (
        "I received a defective blender (order ORD-88120) and want to return it "
        "for a refund."
    )
    payload = _capture_openai_agents(client, triage, prompt)
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "retail",
        "customer-support",
        "multi-agent",
    ]
    events = payload.get("events", [])
    idents = sorted(
        {(e.get("payload") or {}).get("agent_name") for e in events
         if e.get("event_type") in ("agent.input", "agent.output")}
        - {None}
    )
    handoffs = [
        (
            (e.get("payload") or {}).get("from_agent"),
            (e.get("payload") or {}).get("to_agent"),
        )
        for e in events
        if e.get("event_type") == "agent.handoff"
    ]
    guardrails = [
        e for e in events if e.get("event_type") == "evaluation.result"
    ]
    print("  openai-agents multi (triage->returns handoff + guardrail)  "
          "events=%d agents=%s handoffs=%s guardrails=%d"
          % (len(events), idents, handoffs, len(guardrails)))
    print("  ->", _write([payload], "industry", "retail_openai_agents_triage"), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_openai_agents_single(_client)
    generate_openai_agents_multi(_client)
