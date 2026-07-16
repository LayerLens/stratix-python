"""ADP-W2 Family-B recorder for the ``litellm`` adapter (record-real-once).

LiteLLM is a **provider proxy**, not an agent framework: it emits ``model.invoke``
/ ``cost.record`` (framework=``litellm``) but NO ``agent.identity`` / handoff.
So both fixtures render an HONEST EMPTY-STATE — Agent column ``—`` + a span
waterfall — NOT an agent DAG. We do not invent an agent.

Records TWO real, fully-instrumented LiteLLM runs and writes each as a sealed
real-trace fixture under ``samples/data/traces/industry/``:

* ``generate_litellm_single`` -> ``retail_litellm_chat.jsonl``: a single retail
  customer-support turn answered through ``litellm.completion`` on its default
  route (OpenAI ``gpt-4o-mini``). One real ``model.invoke`` + one ``cost.record``
  (``provider=openai``, ``framework=litellm``). Renders empty-state + waterfall.

* ``generate_litellm_multi`` -> ``retail_litellm_gateway.jsonl``: LiteLLM's
  headline feature — a multi-provider routing gateway. A cost/capability router
  sends three real customer-support turns to THREE different providers
  (OpenAI ``gpt-4o-mini`` / Anthropic ``claude-haiku-4-5`` / Google
  ``gemini-2.5-flash``), all through the same ``litellm.completion`` seam. Each
  turn emits its own ``cost.record`` whose ``provider`` is the UNDERLYING
  provider that actually served it (``_route_provider``, LAY-3455), so the trace
  carries a genuine per-provider cost breakdown. This is a routing LOOP, not a
  multi-agent graph — it stays empty-state (no DAG); the value it proves is
  per-provider cost attribution, not agent topology.

Capture reuses the ``_generate_fixtures`` seam (``_write`` / ``_CAPTURE`` / the
model-name constants) but, because a provider has no ``@trace`` agent wrapper,
it builds its own collector directly (mirroring the live provider harness
``_collect``) and ``flush()``es it under the ``set_trace_observer`` +
no-op-``enqueue_upload`` seam. ``flush()`` synthesizes the content-free
``trace.root`` marker (span_name ``"trace"``, NO agent name) and seals the
attestation chain, so the sealed payload is self-rooting and verifies with a
LayerLens key alone. Nothing is fabricated: the Framework column is ``litellm``
(the proxy that really ran), the per-provider token/cost fields are real, and
there is deliberately no Agent because a proxy has none.
"""

from __future__ import annotations

import os
import sys
import uuid

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config + model names).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import (  # noqa: E402
    TraceCollector,
    set_trace_observer,
)
from layerlens.instrument._context import (  # noqa: E402
    _current_collector,
    _push_span,
    _pop_span,
)

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE
OPENAI_MODEL = _gf.OPENAI_MODEL          # gpt-4o-mini
ANTHROPIC_MODEL = _gf.ANTHROPIC_MODEL    # claude-haiku-4-5-20251001
# gemini-2.5-flash is a thinking model; disabling reasoning gives a normal chat
# completion (real output text, exact-priced in the bundled PRICING table).
GEMINI_MODEL = os.environ.get("SAMPLE_GEMINI_MODEL", "gemini/gemini-2.5-flash")

_SUPPORT_SYSTEM = (
    "You are the customer-support assistant for Northwind Outfitters, an online "
    "outdoor-gear retailer. Answer the shopper's question clearly and concisely "
    "(under 120 words). State the relevant policy (returns, exchanges, shipping, "
    "warranty, order tracking) accurately and end with one clear next step."
)


# --------------------------------------------------------------------------
# Provider capture: no @trace agent wrapper (a proxy has no agent). Build a
# collector, run the real litellm.completion call(s) under it, then flush() so
# the sealed payload gets a synthesized content-free trace.root (span_name
# "trace") and a finalized attestation chain — an honest empty-state trace.
# --------------------------------------------------------------------------
def _capture_litellm(client: Stratix, root_span_name: str, run_fn, tags: list) -> dict:
    import litellm  # noqa: E402
    from layerlens.instrument.adapters.providers.litellm import (
        instrument_litellm,
        uninstrument_litellm,
    )

    # Tolerate provider-specific kwargs (e.g. gemini reasoning_effort) instead of
    # erroring — litellm drops params a given provider does not accept.
    litellm.drop_params = True

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig_enqueue = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None

    collector = TraceCollector(client, _CAPTURE)
    col_token = _current_collector.set(collector)
    snap = _push_span(uuid.uuid4().hex[:16], root_span_name)
    instrument_litellm()
    try:
        run_fn(litellm)
    finally:
        uninstrument_litellm()
        _pop_span(snap)
        _current_collector.reset(col_token)
        # flush under the seam -> synthesize trace.root + seal + notify observer,
        # WITHOUT the background upload (suppressed above).
        collector.flush()
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig_enqueue

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for litellm run")
    payload["tags"] = list(tags)
    return payload


def _cost_by_provider(payload: dict) -> dict:
    out: dict = {}
    for e in payload.get("events", []):
        if e.get("event_type") != "cost.record":
            continue
        p = e.get("payload") or {}
        prov = p.get("provider") or "?"
        out[prov] = round(out.get(prov, 0.0) + float(p.get("cost_usd") or 0.0), 8)
    return out


# --------------------------------------------------------------------------
# Single: one retail customer-support turn on LiteLLM's default (OpenAI) route.
# --------------------------------------------------------------------------
def generate_litellm_single(client: Stratix) -> dict:
    """Record ONE retail support turn answered through ``litellm.completion``
    (default OpenAI route). Renders an honest empty-state (Agent ``—``,
    Framework ``litellm``) + a waterfall."""
    question = (
        "I ordered a rain jacket 10 days ago but it arrived a size too small. "
        "What are my options for exchanging it, and will I have to pay return "
        "shipping?"
    )

    def _run(litellm) -> None:
        litellm.completion(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SUPPORT_SYSTEM},
                {"role": "user", "content": question},
            ],
            max_tokens=250,
            temperature=0.2,
        )

    payload = _capture_litellm(
        client,
        "retail-support-gateway",
        _run,
        ["layerlens-sample", "industry", "retail", "customer-support", "litellm-gateway"],
    )
    events = payload.get("events", [])
    print(
        "  litellm single (retail support, default openai route)  "
        "events=%d cost_by_provider=%s"
        % (len(events), _cost_by_provider(payload))
    )
    print("  ->", _write([payload], "industry", "retail_litellm_chat"), "\n")
    return payload


# --------------------------------------------------------------------------
# Multi: a multi-provider routing gateway (per-provider cost.record).
# --------------------------------------------------------------------------
# Each turn declares the routing tier; the cost/capability router maps it to the
# provider that best fits (cheap FAQ -> OpenAI, nuanced policy reasoning ->
# Anthropic, quick factual lookup -> Google). All go through litellm.completion.
_ROUTING_TABLE = {
    "faq": (OPENAI_MODEL, {}),
    "reasoning": (ANTHROPIC_MODEL, {}),
    # gemini-2.5-flash is a thinking model -> disable reasoning for a normal,
    # cheap chat completion with real output text.
    "factual": (GEMINI_MODEL, {"reasoning_effort": "disable"}),
}

_GATEWAY_TURNS = [
    (
        "faq",
        "What are your standard shipping options and how long does each take to "
        "arrive within the continental US?",
    ),
    (
        "reasoning",
        "I received a tent as a gift 45 days ago (outside your 30-day window) but "
        "one of the poles was cracked in the box. I was traveling and could not "
        "report it sooner. Given your return and warranty policies, what can you "
        "actually do for me here?",
    ),
    (
        "factual",
        "How do I track order #NW-58217 and can I still change the delivery "
        "address before it ships?",
    ),
]


def generate_litellm_multi(client: Stratix) -> dict:
    """Record a LiteLLM multi-provider routing gateway: three real support turns
    routed to OpenAI / Anthropic / Google through one ``litellm.completion``
    seam, each with its own underlying-provider ``cost.record``. A routing LOOP
    (NOT multi-agent) — renders empty-state + a per-provider cost breakdown."""

    def _run(litellm) -> None:
        for tier, question in _GATEWAY_TURNS:
            model, extra = _ROUTING_TABLE[tier]
            litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": _SUPPORT_SYSTEM},
                    {"role": "user", "content": question},
                ],
                max_tokens=300,
                temperature=0.2,
                **extra,
            )

    payload = _capture_litellm(
        client,
        "retail-support-gateway",
        _run,
        [
            "layerlens-sample",
            "industry",
            "retail",
            "customer-support",
            "multi-provider",
            "litellm-gateway",
        ],
    )
    events = payload.get("events", [])
    providers = sorted(
        {(e.get("payload") or {}).get("provider") for e in events
         if e.get("event_type") == "cost.record"}
        - {None}
    )
    print(
        "  litellm multi (multi-provider routing gateway)  "
        "events=%d providers=%s cost_by_provider=%s"
        % (len(events), providers, _cost_by_provider(payload))
    )
    print("  ->", _write([payload], "industry", "retail_litellm_gateway"), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_litellm_single(_client)
    generate_litellm_multi(_client)
