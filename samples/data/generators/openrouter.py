"""ADP-W2 Family-B recorder for the ``openrouter`` adapter (record-real-once).

SEALED — BLOCKED, DEFERRED
--------------------------
No OpenRouter API key (``sk-or-…``) exists on any machine, so the gateway hop
could NOT be called live and this fixture does not claim it was. What IS real:

* **The model responses.** Both bodies are genuine live inferences, captured once
  and committed as the recorded corpus under
  ``tests/fixtures/recorded/openrouter/`` — this generator replays THOSE EXACT
  bodies (single source of truth, so the sample and the corpus can never drift):
  - free route  -> local ollama ``llama3:8b`` over its OpenAI-compatible endpoint.
    ``llama3:8b`` IS Meta Llama 3 8B Instruct, the same weights OpenRouter serves
    behind ``meta-llama/llama-3-8b-instruct:free``.
  - paid route  -> a REAL billed OpenAI ``gpt-4o-mini`` call (~$0.0001), which is
    exactly what OpenRouter's ``openai/gpt-4o-mini`` route proxies to.
  So every token count and every word of output in the sealed trace is real.
* **The whole adapter path.** A real ``openai.OpenAI`` client pointed at the real
  OpenRouter base URL does its real routing + deserialization over an
  ``httpx.MockTransport``, and the REAL ``OpenRouterProvider`` parses it. The
  ``model.invoke`` / ``cost.record`` events, the framework tag and the attestation
  chain are all genuine adapter output.

Sealed (and disclosed in ``metadata.sealed`` on the trace + in each fixture's
``provenance.sealed_reason``): the gateway network, the ``gen-sealed-…`` response
ids, and the re-slugging of each body to the OpenRouter route that names the same
model which actually produced the text. Re-record for real once a credential
exists.

WHAT IT RENDERS
---------------
OpenRouter is a **provider gateway**, not an agent framework: it emits
``model.invoke``/``cost.record`` (framework=``openrouter``) but NO
``agent.identity``/handoff. So the trace renders an HONEST EMPTY-STATE — Agent
column ``—`` + a span waterfall — NOT an agent DAG. We do not invent an agent.
(Same honest shape as the ``litellm`` sibling, for the same reason.)

THE SCENARIO (Software/SaaS)
----------------------------
``saas_openrouter_cost_routing`` — a multi-model **cost-routing** support
assistant for a B2B event-analytics API. A routine plan/limits FAQ is served by
the free model; the hard production incident (429s during a nightly backfill) is
escalated to a paid model. That cheap-first / escalate-on-complexity split is
OpenRouter's headline value, and it is what this trace proves.

THE COST STORY IS THE POINT (and it is honest)
----------------------------------------------
OpenRouter bills at its own rates, which no table we ship holds, so the gateway
is the SOLE authority (``provider_cost_only``). The two turns pin both branches
of that rule WITHOUT inventing a single cent:

* free route — usage accounting ON, gateway reports ``usage.cost = 0.0``. A
  ``:free`` slug genuinely bills $0.00, so that zero is a FACT, and it must reach
  ``cost.record`` stamped ``cost_source="provider"``.
* paid route — usage accounting OFF, no reported charge. We have no OpenRouter
  charge for that call and refuse to fabricate one, so the adapter emits real
  tokens on ``model.invoke`` and NO ``cost.record``. Pricing the routed
  ``openai/gpt-4o-mini`` slug from our own catalog would attach OpenAI list-rate
  dollars OpenRouter never charged.

The trace therefore carries a real, honest, partial cost picture — which is the
truth — rather than a complete fabricated one.
"""

from __future__ import annotations

import json
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

#: The committed recorded corpus — the SAME real bodies the adapter's
#: recorded-replay test asserts against. Read (never re-authored) so the sealed
#: sample trace and the regression gate can never drift apart.
_CORPUS = os.path.join(_REPO, "tests", "fixtures", "recorded", "openrouter")

_FREE_SLUG = "meta-llama/llama-3-8b-instruct:free"
_PAID_SLUG = "openai/gpt-4o-mini"

_SAAS_SYSTEM = (
    "You are the support assistant for Meridian Analytics, a B2B SaaS product "
    "that ships an event-analytics API. Answer the customer's question "
    "accurately and concisely (under 120 words). Be concrete about limits, "
    "status codes and next steps."
)

# The two real customer questions the recorded bodies actually answered.
_Q_FAQ = (
    "What is the rate limit on the Meridian events ingest API for the Growth "
    "plan, and what HTTP status do I get when I exceed it?"
)
_Q_ESCALATE = (
    "We're on the Growth plan and started getting 429s on /v1/events at about "
    "40k events/min during our nightly backfill, even though our steady-state "
    "traffic is well under the limit. Our client retries immediately on 429. "
    "Explain what is most likely happening and give us a concrete remediation "
    "plan for the backfill."
)


def _recorded_body(scenario: str) -> dict:
    """The REAL captured response body for a scenario (from the committed corpus)."""
    with open(os.path.join(_CORPUS, f"{scenario}.json")) as f:
        fixture = json.load(f)
    return fixture["interactions"][0]["response"]["json"]


# --------------------------------------------------------------------------
# The sealed gateway: a real openai SDK client at the real OpenRouter base URL
# over httpx.MockTransport, routing on the requested slug exactly as the real
# gateway would. Only the network hop is sealed.
# --------------------------------------------------------------------------
def _sealed_openrouter_client():
    import httpx
    from openai import OpenAI
    from layerlens.instrument.adapters.providers.openrouter import OPENROUTER_BASE_URL

    bodies = {
        _FREE_SLUG: _recorded_body("free_route"),
        _PAID_SLUG: _recorded_body("paid_route_no_accounting"),
    }

    def handler(request: "httpx.Request") -> "httpx.Response":
        requested = json.loads(request.content.decode())["model"]
        body = bodies.get(requested)
        if body is None:  # never silently serve the wrong model's body
            raise AssertionError(f"no recorded body for route {requested!r}")
        return httpx.Response(200, json=body)

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        # Sealed fixture -- never a real key (there is no OpenRouter account).
        api_key="sk-or-v1-sealed-no-openrouter-credential",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        default_headers={"HTTP-Referer": "https://meridian-analytics.example", "X-Title": "Meridian Support"},
    )


# --------------------------------------------------------------------------
# Provider capture: no @trace agent wrapper (a gateway has no agent). Build a
# collector, run the real instrumented calls under it, then flush() so the sealed
# payload gets a synthesized content-free trace.root (span_name "trace") and a
# finalized attestation chain -- an honest empty-state trace.
# --------------------------------------------------------------------------
def _capture_openrouter(client: Stratix, root_span_name: str, run_fn, tags: list) -> dict:
    from layerlens.instrument.adapters.providers.openrouter import (
        instrument_openrouter,
        uninstrument_openrouter,
    )

    openrouter_client = _sealed_openrouter_client()

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig_enqueue = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None

    collector = TraceCollector(client, _CAPTURE)
    col_token = _current_collector.set(collector)
    snap = _push_span(uuid.uuid4().hex[:16], root_span_name)
    instrument_openrouter(openrouter_client)
    try:
        run_fn(openrouter_client)
    finally:
        uninstrument_openrouter()
        _pop_span(snap)
        _current_collector.reset(col_token)
        # flush under the seam -> synthesize trace.root + seal + notify observer,
        # WITHOUT the background upload (suppressed above).
        collector.flush()
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig_enqueue

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for openrouter run")
    payload["tags"] = list(tags)
    payload["metadata"] = {
        "sealed": True,
        "provider": "openrouter",
        "reason": (
            "No OpenRouter credential exists, so the gateway hop is sealed behind an "
            "httpx.MockTransport. The REAL OpenRouterProvider + REAL openai SDK parse REAL "
            "captured model responses (ollama llama3:8b for the free route; a real billed "
            "OpenAI gpt-4o-mini call for the paid route) re-enveloped in OpenRouter's wire "
            "shape -- so the tokens, the output text, the events and the attestation chain "
            "are genuine. Deferred until an OpenRouter key is provisioned."
        ),
        "cost_honesty": (
            "cost_usd=0.0 on the free route is a FACT (a ':free' slug bills $0.00), not an "
            "estimate. The paid route reports NO cost because usage accounting was off and "
            "we refuse to invent a charge OpenRouter never billed."
        ),
        "latency_is_not_representative": (
            "model.invoke.latency_ms measures the LOCAL mock-transport replay (~1-20ms), NOT "
            "a real OpenRouter round-trip (~0.5-3s). It is a genuinely measured value of what "
            "actually ran, but it is an artifact of the sealed transport -- do not read it as "
            "gateway performance. The tokens, output text and cost are unaffected."
        ),
    }
    return payload


def _summary(payload: dict) -> str:
    events = payload.get("events", [])
    invokes = [e for e in events if e.get("event_type") == "model.invoke"]
    costs = [e for e in events if e.get("event_type") == "cost.record"]
    routes = [(e.get("payload") or {}).get("model") for e in invokes]
    priced = {
        (e.get("payload") or {}).get("model"): (e.get("payload") or {}).get("cost_usd")
        for e in costs
    }
    return (
        "events=%d model.invoke=%d routes=%s cost.record=%d priced=%s"
        % (len(events), len(invokes), routes, len(costs), priced)
    )


def generate_openrouter_single(client: Stratix) -> dict:
    """Record the SaaS multi-model cost-routing support run through the sealed
    OpenRouter gateway. Renders an honest empty-state (Agent ``—``, Framework
    ``openrouter``, Status ``ok``) + a two-call waterfall.

    Named for the ``_generate_fixtures._W2_ADAPTERS`` loader contract
    (``generate_<adapter>_{single,multi}``). There is deliberately no ``_multi``:
    the single trace already carries BOTH routes (free + paid escalation), which
    is the whole cost-routing story — a second fixture would add no new adapter
    behaviour and no new real body.
    """

    def _run(openrouter) -> None:
        # Tier 1 — routine plan/limits FAQ: the free model is enough. Usage
        # accounting ON, so the gateway reports its own (genuinely $0.00) charge.
        openrouter.chat.completions.create(
            model=_FREE_SLUG,
            messages=[
                {"role": "system", "content": _SAAS_SYSTEM},
                {"role": "user", "content": _Q_FAQ},
            ],
            temperature=0.2,
            extra_body={"usage": {"include": True}},
        )
        # Tier 2 — a live production incident: escalate to the paid route. Usage
        # accounting is OFF here, so OpenRouter reports no charge and the adapter
        # honestly records none (rather than pricing the slug from our catalog).
        openrouter.chat.completions.create(
            model=_PAID_SLUG,
            messages=[
                {"role": "system", "content": _SAAS_SYSTEM},
                {"role": "user", "content": _Q_ESCALATE},
            ],
            temperature=0.2,
        )

    payload = _capture_openrouter(
        client,
        "saas-support-cost-router",
        _run,
        [
            "layerlens-sample",
            "industry",
            "software-saas",
            "customer-support",
            "cost-routing",
            "openrouter-gateway",
            "sealed-fixture",
        ],
    )
    print("  openrouter cost-routing (saas support, SEALED gateway)  " + _summary(payload))
    print("  ->", _write([payload], "industry", "saas_openrouter_cost_routing"), "\n")
    return payload


#: Descriptive alias — the fixture stem this generator produces.
generate_saas_openrouter_cost_routing = generate_openrouter_single


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_openrouter_single(_client)
