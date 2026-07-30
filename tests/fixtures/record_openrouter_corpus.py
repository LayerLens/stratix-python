"""Offline capture harness for the SEALED ``openrouter`` recorded corpus.

⚠️  NOT a test and NEVER run in CI. Companion to ``record_corpus.py`` (same
gate, same helpers, same UPSTREAM-of-the-parser rule); kept as its own module
because OpenRouter is the one adapter whose corpus cannot be captured by
pointing a recording transport at the real endpoint.

WHY THIS IS SEPARATE — AND WHAT "SEALED" MEANS HERE
---------------------------------------------------
No OpenRouter credential (``sk-or-…``) exists on any machine, so the gateway
network hop CANNOT be called and this harness does not pretend otherwise. What
it does instead is capture the REAL model bodies from the REAL models that
OpenRouter's routes proxy to, and re-envelope each into OpenRouter's documented
wire shape:

* ``free_route`` -> a REAL local inference on ollama ``llama3:8b`` over its
  OpenAI-compatible endpoint. ``llama3:8b`` IS Meta Llama 3 8B Instruct — the
  same weights OpenRouter serves behind ``meta-llama/llama-3-8b-instruct:free``
  — so the slug names the model that genuinely produced the text. FREE.
* ``paid_route_no_accounting`` -> a REAL billed OpenAI ``gpt-4o-mini`` call
  (~$0.0001). OpenRouter's ``openai/gpt-4o-mini`` route proxies to this exact
  model, so again the slug names the model that genuinely produced the text.

So every token count and every word of output in the committed corpus is a real
tokenizer/model output. The ONLY sealed parts, disclosed in each fixture's
``provenance.sealed_reason``, are: the gateway network hop, the ``gen-sealed-…``
response id, the OpenRouter route slug, and — on the free route — the
``usage.cost``/``is_byok``/``cost_details`` block that only the real gateway
emits. Re-record for real once a credential is provisioned.

THE ONE FIELD WE ADD, AND WHY IT IS NOT A FABRICATION
-----------------------------------------------------
``usage.cost = 0.0`` on the free route is added because ollama's endpoint has no
such field — OpenRouter's does. It is not an invented price: a ``:free`` slug
genuinely bills $0.00, so the zero is a FACT, and it is the exact input needed
to gate the adapter's "a reported zero is kept" branch (a truthiness bug would
silently drop it). The paid route deliberately gets NO ``usage.cost``, because
usage accounting was off — that fixture gates the adapter's refusal to invent a
charge. Nothing else is added.

    set -a; . tests/e2e/live/.env; set +a
    LAYERLENS_RECORD=1 python tests/fixtures/record_openrouter_corpus.py
"""

from __future__ import annotations

import os
import sys
import copy
from typing import Any, Dict, List
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.fixtures.record_corpus import (  # noqa: E402
    _write,
    _provenance,
    _RecordingTransport,
)

#: The OpenRouter routes whose upstream models we capture for real.
FREE_SLUG = "meta-llama/llama-3-8b-instruct:free"
PAID_SLUG = "openai/gpt-4o-mini"

_SEALED_REASON = (
    "No OpenRouter API key exists (sk-or-...), so the gateway network could not be called. "
    "The bodies below are REAL model responses captured live (see real_body_source) and "
    "re-enveloped into OpenRouter's documented wire shape, then replayed through the REAL "
    "OpenRouterProvider over httpx.MockTransport. The adapter's parsing, event emission and "
    "cost handling are genuinely exercised; only the gateway hop is sealed. Deferred until an "
    "OpenRouter credential is provisioned."
)

# The real SaaS support scenario the corpus answers — kept identical to the
# sample generator's, so the regression gate and the shipped sample tell the
# same story from the same bodies.
SAAS_SYSTEM = (
    "You are the support assistant for Meridian Analytics, a B2B SaaS product "
    "that ships an event-analytics API. Answer the customer's question "
    "accurately and concisely (under 120 words). Be concrete about limits, "
    "status codes and next steps."
)
Q_FAQ = (
    "What is the rate limit on the Meridian events ingest API for the Growth "
    "plan, and what HTTP status do I get when I exceed it?"
)
Q_ESCALATE = (
    "We're on the Growth plan and started getting 429s on /v1/events at about "
    "40k events/min during our nightly backfill, even though our steady-state "
    "traffic is well under the limit. Our client retries immediately on 429. "
    "Explain what is most likely happening and give us a concrete remediation "
    "plan for the backfill."
)


def _openrouter_fixture(
    *,
    scenario: str,
    slug: str,
    sdk_version: str,
    real_body_source: str,
    notes: str,
    interaction: Dict[str, Any],
) -> Dict[str, Any]:
    """Wrap a captured interaction with SEALED provenance (never silently sealed)."""
    prov = _provenance("openrouter", sdk_version, slug, scenario)
    prov["sealed"] = True
    prov["sealed_reason"] = _SEALED_REASON
    prov["real_body_source"] = real_body_source
    prov["notes"] = notes
    return {"provenance": prov, "transport": "http", "interactions": [interaction]}


def _reslug(interaction: Dict[str, Any], *, slug: str, response_id: str) -> Dict[str, Any]:
    """Re-envelope a captured upstream body into OpenRouter's wire shape.

    Only the gateway-owned envelope fields are rewritten: the routed ``model``
    slug and the ``gen-…`` response id. The choices/usage/finish_reason the real
    model produced are passed through untouched.
    """
    out = copy.deepcopy(interaction)
    body = out["response"]["json"]
    body["model"] = slug
    body["id"] = response_id
    out["request"]["path"] = "/api/v1/chat/completions"
    return out


def capture_free_route() -> None:
    """REAL ollama llama3:8b inference -> the ``:free`` OpenRouter route."""
    import openai

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    rec = _RecordingTransport()
    # ollama's OpenAI-compatible endpoint: a real openai SDK, a real local model.
    client = openai.OpenAI(
        base_url=f"{host.rstrip('/')}/v1",
        api_key="ollama",  # ollama ignores the key; it is not a credential.
        http_client=httpx.Client(transport=rec, timeout=180.0),
    )
    client.chat.completions.create(
        model=os.environ.get("OLLAMA_MODEL", "llama3:8b"),
        messages=[
            {"role": "system", "content": SAAS_SYSTEM},
            {"role": "user", "content": Q_FAQ},
        ],
        temperature=0.2,
    )
    interaction = _reslug(
        rec.interactions[-1],
        slug=FREE_SLUG,
        response_id="gen-sealed-no-openrouter-credential-free-0001",
    )
    # OpenRouter reports usage accounting; ollama does not. A ':free' slug bills
    # $0.00 — the zero is a fact, and it is what gates the adapter's kept-zero
    # branch. This is the ONLY field added to the real body.
    interaction["response"]["json"]["usage"].update(
        {"cost": 0.0, "is_byok": False, "cost_details": {"upstream_inference_cost": 0.0}}
    )
    _write(
        "openrouter",
        "free_route",
        _openrouter_fixture(
            scenario="free_route",
            slug=FREE_SLUG,
            sdk_version=openai.__version__,
            real_body_source=(
                "REAL local inference: ollama llama3:8b via its OpenAI-compatible endpoint "
                "(POST %s/v1/chat/completions). llama3:8b IS Meta Llama 3 8B Instruct — the same "
                "weights OpenRouter serves behind the '%s' slug — so the slug names the model "
                "that genuinely produced this text." % (host.rstrip("/"), FREE_SLUG)
            ),
            notes=(
                "usage.cost = 0.0 is NOT invented: a ':free' OpenRouter slug genuinely bills "
                "$0.00, and the adapter deliberately KEEPS a reported zero (a zero is a fact) "
                "while rejecting bool/NaN/inf/negative. This fixture is the regression gate on "
                "that zero-kept branch. The cost/is_byok/cost_details block is the gateway's own "
                "accounting shape, added because ollama emits none; every other field is real."
            ),
            interaction=interaction,
        ),
    )


def capture_paid_route_no_accounting() -> None:
    """REAL billed OpenAI gpt-4o-mini call -> the paid OpenRouter route."""
    import openai

    rec = _RecordingTransport()
    client = openai.OpenAI(http_client=httpx.Client(transport=rec, timeout=60.0))
    client.chat.completions.create(
        model=os.environ.get("LL_OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SAAS_SYSTEM},
            {"role": "user", "content": Q_ESCALATE},
        ],
        temperature=0.2,
    )
    interaction = _reslug(
        rec.interactions[-1],
        slug=PAID_SLUG,
        response_id="gen-sealed-no-openrouter-credential-paid-0002",
    )
    # Deliberately NO usage.cost: the caller did not enable usage accounting, so
    # the gateway reports no charge and the adapter must invent none.
    _write(
        "openrouter",
        "paid_route_no_accounting",
        _openrouter_fixture(
            scenario="paid_route_no_accounting",
            slug=PAID_SLUG,
            sdk_version=openai.__version__,
            real_body_source=(
                "REAL live inference: OpenAI gpt-4o-mini (POST https://api.openai.com/v1/"
                "chat/completions, real key, billed ~$0.0001). OpenRouter's '%s' route proxies "
                "to this exact model, so the slug names the model that genuinely produced this "
                "text." % PAID_SLUG
            ),
            notes=(
                "NO usage.cost: the caller did not enable usage accounting "
                "(extra_body={'usage': {'include': True}}). We have no OpenRouter charge for "
                "this call and refuse to invent one, so the adapter must emit model.invoke with "
                "real tokens and NO cost.record (provider_cost_only). This fixture is the "
                "regression gate on that refusal branch — pricing the routed slug from our own "
                "catalog would attach a number nobody was billed."
            ),
            interaction=interaction,
        ),
    )


CAPTURES = {
    "free_route": capture_free_route,
    "paid_route_no_accounting": capture_paid_route_no_accounting,
}


def main(argv: List[str]) -> int:
    if os.environ.get("LAYERLENS_RECORD") != "1":
        print(
            "refusing to record: set LAYERLENS_RECORD=1 (offline capture, real creds, spend).",
            file=sys.stderr,
        )
        return 2
    targets = argv or list(CAPTURES)
    unknown = [t for t in targets if t not in CAPTURES]
    if unknown:
        print(f"unknown capture targets: {unknown}; known: {list(CAPTURES)}", file=sys.stderr)
        return 2
    for name in targets:
        print(f"capturing openrouter/{name} ...")
        CAPTURES[name]()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
