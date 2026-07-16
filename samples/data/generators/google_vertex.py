"""ADP-W2 Family-B recorder for the ``google_vertex`` adapter (SEALED).

Google Vertex AI (Gemini) is credential-gated: no ``GOOGLE_CLOUD_PROJECT`` /
service-account credentials exist on any build machine, so a live capture is
impossible. These fixtures are therefore recorded **SEALED**, exactly like the
Azure OpenAI manufacturing fixtures: the REAL ``GoogleVertexProvider`` adapter
runs against a REAL proto-backed ``vertexai.generative_models.GenerationResponse``
(rebuilt via ``from_dict`` — that needs the SDK, not credentials, and is the same
seam the ``test_google_vertex_recorded`` corpus uses). Only the LLM *network* is
sealed. The adapter genuinely parses the proto, so ``framework=google_vertex``,
the ``gemini-1.5-flash-002`` model id (reduced from the real
``publishers/google/models/<id>`` resource form by ``_strip_models_prefix`` /
LAY-3615), the token counts, the priced ``cost.record`` and the attestation chain
are all real adapter output.

``google_vertex`` is a **provider** (a raw LLM call, not an agent framework), so
the trace declares NO producer-chosen agent name. The collector's
``_synthesize_identity_if_needed`` refuses to invent one from a model/method
label, so the Agent column renders the honest **empty-state (—)** + a span
waterfall (``trace.root`` -> ``model.invoke`` -> ``cost.record``). This is the
correct rendering for a provider — nothing is fabricated as an agent.

* ``generate_google_vertex_single`` -> ``government_vertex_triage.jsonl``:
  a single Gemini turn that triages a citizen's public-benefits situation
  (which assistance programs they likely qualify for, documents needed, next
  steps). One ``model.invoke`` + one priced ``cost.record``.

* ``generate_google_vertex_multi`` -> ``government_vertex_permit_tooluse.jsonl``:
  a Gemini **function-call loop** for a municipal building-permit determination.
  The first turn returns a ``function_call`` (``lookup_permit_requirements``) that
  the adapter surfaces as a real ``tool.call``; the local tool genuinely runs and
  its record is emitted as a ``tool.result``; the second turn returns the final
  determination. Two ``model.invoke`` + two ``cost.record`` + one ``tool.call`` +
  one ``tool.result`` — a real provider tool-use loop, still Agent=— (a provider
  has no agent DAG).

HONESTY: the fixtures are marked ``metadata.sealed = true`` with
``source = "synthetic-recorded"`` and ``captured_at = "pending-creds"`` — the
response bodies are documented/synthetic (no real paid Gemini call happened), and
the token/cost numbers are the adapter's real computation over those synthetic
bodies, NOT presented as a live billed call. Provision ``GOOGLE_CLOUD_PROJECT`` +
Vertex credentials and this recorder can be re-pointed at a live
``GenerativeModel`` to replace the sealed bodies with a genuine capture.
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
from layerlens.instrument import TraceCollector  # noqa: E402
from layerlens.instrument._context import (  # noqa: E402
    _current_collector,
    _current_span_id,
    _push_span,
    _pop_span,
)
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE

SINGLE_STEM = "government_vertex_triage"
MULTI_STEM = "government_vertex_permit_tooluse"

# The real Vertex resource-name form a ``GenerativeModel`` stores. The adapter's
# ``_strip_models_prefix`` (LAY-3615) reduces it to the bare ``gemini-1.5-flash-002``
# id for pricing + the model field — recording it here exercises that real path.
_MODEL_RESOURCE = "publishers/google/models/gemini-1.5-flash-002"

_SEALED_META = {
    "sealed": True,
    "provider": "google_vertex",
    "source": "synthetic-recorded",
    "captured_at": "pending-creds",
    "reason": (
        "No GCP Vertex credentials exist (GOOGLE_CLOUD_PROJECT empty, no "
        "service-account file). Driven through the REAL GoogleVertexProvider "
        "adapter against a REAL proto-backed vertexai GenerationResponse "
        "(from_dict); the response body is documented/synthetic, so the "
        "token/cost numbers are the adapter's real computation over that body, "
        "NOT a live billed Gemini call. Re-point at a live GenerativeModel with "
        "Vertex creds to replace the sealed bodies with a genuine capture."
    ),
}


# --------------------------------------------------------------------------
# Real proto-backed Vertex responses, rebuilt from documented bodies. Needs the
# ``vertexai`` SDK (imported function-locally so this module imports in any venv;
# a missing SDK is a skip, not a crash — the main() loop guards it).
# --------------------------------------------------------------------------
def _proto_text(text, *, prompt_tokens, completion_tokens, finish_reason="STOP"):
    import vertexai.generative_models as gm

    return gm.GenerationResponse.from_dict(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finish_reason": finish_reason,
                }
            ],
            "usage_metadata": {
                "prompt_token_count": prompt_tokens,
                "candidates_token_count": completion_tokens,
                "total_token_count": prompt_tokens + completion_tokens,
            },
        }
    )


def _proto_function_call(name, args, *, prompt_tokens, completion_tokens):
    import vertexai.generative_models as gm

    return gm.GenerationResponse.from_dict(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"function_call": {"name": name, "args": args}}],
                    },
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": prompt_tokens,
                "candidates_token_count": completion_tokens,
                "total_token_count": prompt_tokens + completion_tokens,
            },
        }
    )


# --------------------------------------------------------------------------
# Provider-only capture: NO ``@trace`` wrapper (a provider has no agent), so the
# trace carries only the adapter-emitted ``model.invoke`` / ``cost.record`` (and,
# for the loop, ``tool.call`` + our ``tool.result``). We open a collector, push a
# single root span the leaf events parent to (the flush-time root synthesizer
# turns it into a real content-free ``trace.root``), drive the instrumented call,
# and flush — observed via the shared capture seam so the sealed payload is
# captured but never uploaded. ``_synthesize_identity_if_needed`` finds no honest
# agent name -> Agent column renders the empty-state (—).
# --------------------------------------------------------------------------
def _capture_provider_trace(client: Stratix, drive) -> dict:
    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        col = TraceCollector(client, _CAPTURE)
        col_token = _current_collector.set(col)
        root_span_id = uuid.uuid4().hex[:16]
        span_snapshot = _push_span(root_span_id, "google_vertex.generate_content")
        try:
            drive()
        finally:
            _pop_span(span_snapshot)
            _current_collector.reset(col_token)
        col.flush()
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for google_vertex")
    return payload


def _fake_model(generate_content):
    """A minimal stand-in for ``vertexai.generative_models.GenerativeModel`` that
    the adapter duck-types on: a ``_model_name`` attribute + ``generate_content``.
    Mirrors the ``test_google_vertex_recorded`` seam (the adapter monkey-patches
    ``generate_content`` on whatever object it is connected to)."""
    from types import SimpleNamespace

    return SimpleNamespace(_model_name=_MODEL_RESOURCE, generate_content=generate_content)


def _event_counts(payload: dict) -> dict:
    from collections import Counter

    return dict(Counter(e.get("event_type") for e in payload.get("events", [])))


# --------------------------------------------------------------------------
# Single: a Gemini public-benefits eligibility triage (one chat turn).
# --------------------------------------------------------------------------
# The citizen's household situation the assistant triaged. Documents the
# scenario; the sealed trace is what the (synthetic-body) real adapter produced.
CITIZEN_SITUATION = (
    "State: Wisconsin. Household: a single parent (age 34) with two children "
    "(ages 4 and 7). Employment: part-time retail, gross income about $1,850/month. "
    "Housing: rents an apartment for $1,200/month; heats with natural gas. Health "
    "coverage: none currently. Assets: one used car, about $600 in savings. "
    "Question: which assistance programs might I qualify for, what documents will I "
    "need, and what should I do first?"
)

# A documented, realistic Gemini answer (SEALED — no live call). The adapter
# computes the real cost over these token counts; the body itself is synthetic.
_TRIAGE_ANSWER = (
    "Based on your household size (3) and income, here are the programs you most "
    "likely qualify for in Wisconsin — these are general estimates, not official "
    "determinations:\n\n"
    "1. FoodShare (SNAP): A household of 3 has a gross monthly income limit around "
    "$2,700, so ~$1,850/month is well within range. Likely eligible.\n"
    "2. BadgerCare Plus (Medicaid): Covers parents and children in low-income "
    "households; children in particular are very likely eligible. Apply for the "
    "whole family.\n"
    "3. Wisconsin Shares (child care subsidy): Available to working parents under "
    "the income limit — helpful given part-time work and two young children.\n"
    "4. Wisconsin Home Energy Assistance (WHEAP): Seasonal help with heating "
    "(natural gas) costs; you appear within the income guidelines.\n\n"
    "Documents to gather: photo ID, Social Security numbers for all household "
    "members, proof of income (recent pay stubs), your lease and a recent gas/"
    "utility bill, and proof of any other benefits.\n\n"
    "First step: file one combined application through ACCESS Wisconsin "
    "(access.wisconsin.gov), which screens you for FoodShare, BadgerCare Plus, and "
    "Wisconsin Shares at once. For WHEAP, contact your local energy-assistance "
    "agency. Official eligibility is decided after the agency verifies your "
    "documents."
)


def generate_google_vertex_single(client: Stratix) -> dict:
    """Record a single sealed Gemini public-benefits triage turn (Agent=—)."""
    response = _proto_text(_TRIAGE_ANSWER, prompt_tokens=232, completion_tokens=286)
    model = _fake_model(lambda *a, **k: response)

    from layerlens.instrument.adapters.providers.google_vertex import (
        instrument_google_vertex,
        uninstrument_google_vertex,
    )

    system = (
        "You are a public-benefits eligibility assistant for a state Department of "
        "Human Services. Given a citizen's household situation, identify which "
        "assistance programs they most likely qualify for, list the documents they "
        "will need, and give clear next steps. You provide general guidance only — "
        "official eligibility is decided by the agency after verification."
    )

    instrument_google_vertex(model)
    try:
        payload = _capture_provider_trace(
            client,
            lambda: model.generate_content(
                f"{system}\n\nCitizen situation:\n{CITIZEN_SITUATION}",
                temperature=0.2,
            ),
        )
    finally:
        uninstrument_google_vertex()

    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "government",
        "public-benefits",
        "google-vertex",
        "sealed-fixture",
    ]
    payload["metadata"] = dict(_SEALED_META)
    print(
        "  google-vertex single (gemini benefits triage, sealed)  counts=%s"
        % _event_counts(payload)
    )
    print("  ->", _write([payload], "industry", SINGLE_STEM), "\n")
    return payload


# --------------------------------------------------------------------------
# Multi: a Gemini function-call loop for a building-permit determination.
# --------------------------------------------------------------------------
PERMIT_REQUEST = (
    "City of Madison, WI. I want to build a 240 sq ft single-story addition onto "
    "the back of my single-family home to add a bedroom. It will have a new "
    "concrete foundation and tie into the existing electrical and HVAC. Do I need "
    "a permit, which ones, how long does review take, and what do I submit?"
)

# A real local reference the tool reads from (deterministic municipal data — the
# tool genuinely runs and returns these records; the model's determination is the
# sealed body).
_PERMIT_DB = {
    "residential_addition": {
        "project_type": "residential_addition",
        "permit_required": True,
        "permits": [
            "Building Permit",
            "Electrical Permit",
            "HVAC/Mechanical Permit",
        ],
        "plan_review_required": True,
        "typical_review_business_days": 20,
        "submittals": [
            "Completed building-permit application",
            "Site plan showing setbacks",
            "Foundation and framing plans",
            "Zoning verification for the parcel",
        ],
        "notes": (
            "Additions with a new foundation require plan review and a zoning "
            "setback check; separate electrical and mechanical permits are pulled "
            "for the tie-ins."
        ),
    },
    "interior_paint": {
        "project_type": "interior_paint",
        "permit_required": False,
        "permits": [],
        "plan_review_required": False,
        "typical_review_business_days": 0,
        "submittals": [],
        "notes": "Cosmetic interior work does not require a building permit.",
    },
}


def _lookup_permit_requirements(project_type: str) -> dict:
    """REAL local tool: return the jurisdiction's permit requirements record."""
    return _PERMIT_DB.get(
        project_type,
        {
            "project_type": project_type,
            "permit_required": True,
            "permits": ["Building Permit"],
            "plan_review_required": True,
            "typical_review_business_days": 15,
            "submittals": ["Completed building-permit application"],
            "notes": "No specific rule matched; a building permit is generally required.",
        },
    )


_PERMIT_DETERMINATION = (
    "Yes — this project needs permits. A 240 sq ft addition with a new concrete "
    "foundation is structural work, so the City of Madison requires:\n\n"
    "1. Building Permit (with plan review) — required because of the new "
    "foundation and added habitable space.\n"
    "2. Electrical Permit — for tying the addition into your existing electrical.\n"
    "3. HVAC/Mechanical Permit — for extending heating/cooling into the new room.\n\n"
    "Plan review typically takes about 20 business days. What to submit: the "
    "completed building-permit application, a site plan showing your setbacks, "
    "foundation and framing plans, and a zoning verification for your parcel (the "
    "setback check confirms the addition fits your lot's rules). Pull the "
    "electrical and mechanical permits alongside the building permit. Once plans "
    "are approved and permits issued, schedule the required inspections "
    "(foundation, framing, electrical, mechanical, and final) as the work "
    "progresses."
)


def generate_google_vertex_multi(client: Stratix) -> dict:
    """Record a sealed Gemini function-call loop (permit determination, Agent=—)."""
    responses = [
        _proto_function_call(
            "lookup_permit_requirements",
            {"project_type": "residential_addition"},
            prompt_tokens=301,
            completion_tokens=22,
        ),
        _proto_text(_PERMIT_DETERMINATION, prompt_tokens=548, completion_tokens=228),
    ]
    model = _fake_model(lambda *a, **k: responses.pop(0))

    from layerlens.instrument.adapters.providers.google_vertex import (
        instrument_google_vertex,
        uninstrument_google_vertex,
    )

    system = (
        "You are a municipal building-permit assistant for the City of Madison, WI. "
        "For a resident's described project you MUST first call "
        "lookup_permit_requirements(project_type) to fetch the current jurisdiction "
        "requirements, then tell the resident whether a permit is required, which "
        "permit(s), the review timeline, and exactly what to submit."
    )

    tools = [
        {
            "function_declarations": [
                {
                    "name": "lookup_permit_requirements",
                    "description": (
                        "Look up the jurisdiction's building-permit requirements for "
                        "a project type (whether a permit is required, which permits, "
                        "review timeline, and submittals)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_type": {
                                "type": "string",
                                "description": "The normalized project category, e.g. 'residential_addition'.",
                            }
                        },
                        "required": ["project_type"],
                    },
                }
            ]
        }
    ]

    def drive():
        # Turn 1: the model requests the permit-requirements lookup.
        r1 = model.generate_content(
            f"{system}\n\nResident request:\n{PERMIT_REQUEST}",
            tools=tools,
            temperature=0.2,
        )
        fc = r1.candidates[0].content.parts[0].function_call
        args = dict(fc.args)
        # The local tool genuinely runs and returns real municipal records.
        result = _lookup_permit_requirements(args.get("project_type", ""))
        # Record the real tool result as a structural event on the active trace
        # (the tool truly executed — honest, not fabricated).
        col = _current_collector.get()
        col.emit(
            "tool.result",
            {
                "provider": "google_vertex",
                "tool_name": fc.name,
                "arguments": args,
                "result": result,
                "status": "ok",
            },
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=_current_span_id.get(),
            span_name="tool:%s" % fc.name,
        )
        # Turn 2: with the tool result in hand, the model returns the final
        # determination (the sealed body). A second model.invoke + cost.record.
        r2 = model.generate_content(
            f"{system}\n\nResident request:\n{PERMIT_REQUEST}\n\n"
            f"lookup_permit_requirements result:\n{result}",
            tools=tools,
            temperature=0.2,
        )
        return r2.candidates[0].content.parts[0].text

    instrument_google_vertex(model)
    try:
        payload = _capture_provider_trace(client, drive)
    finally:
        uninstrument_google_vertex()

    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "government",
        "building-permits",
        "tool-use",
        "google-vertex",
        "sealed-fixture",
    ]
    payload["metadata"] = dict(_SEALED_META)
    counts = _event_counts(payload)
    tool_calls = [
        (e["payload"].get("tool_name"), e["payload"].get("arguments"))
        for e in payload.get("events", [])
        if e.get("event_type") == "tool.call"
    ]
    print(
        "  google-vertex multi (gemini permit function-call loop, sealed)  "
        "counts=%s tool.call=%s" % (counts, tool_calls)
    )
    print("  ->", _write([payload], "industry", MULTI_STEM), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_google_vertex_single(_client)
    generate_google_vertex_multi(_client)
