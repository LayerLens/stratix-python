"""ADP-PORT Family-B recorder for the ported **marvin** framework adapter.

Records ONE real, fully-instrumented Marvin 3.x run — a Real-Estate MLS listing
normalizer — and seals TWO artifacts out of that SINGLE run:

1. ``samples/data/traces/industry/realestate_marvin_listing_extract.jsonl``
   The sealed real trace (the Family-B sample uploads this).
2. ``tests/fixtures/recorded/marvin/{listing_cast,features_extract}.json``
   The RAW upstream OpenAI chat.completion bodies the run actually received —
   the recorded corpus ``tests/instrument/adapters/frameworks/test_marvin_recorded.py``
   replays through the real ``MarvinAdapter``.

Recording both from one run is the point: the corpus fixture is literally the
transport response that produced the sealed trace, so "does the adapter still
parse a REAL response" and "does the sample render a REAL trace" can never drift
apart. It follows ``tests/fixtures/record_corpus.py``'s rule — *record UPSTREAM
of the parser, assert DOWNSTREAM of it* — via the same ``http_client=`` seam the
replay test injects ``httpx.MockTransport`` through.

THE SCENARIO (Real Estate — MLS listing intake)
-----------------------------------------------
A listing coordinator pastes an agent's freeform write-up into the MLS. The
``listing-extraction-agent`` normalizes it into the structured record the
database needs (``marvin.cast`` -> ``PropertyListing``: beds, baths, sqft, list
price, year built, garage, lot size, property type) and then pulls the
marketable feature/amenity list off the same prose (``marvin.extract``). Two
real Marvin primitives, one agent, one trace.

TRANSPORT — WHY NOT OLLAMA (the free lane was tried first and genuinely cannot
serve this): Marvin 3.x runs on pydantic-ai, and every Marvin primitive gets its
structured result through a pydantic-ai **output tool call**. The local
``llama3:8b`` (and ``codegemma``) report ``capabilities: ["completion"]`` — no
tools — and Ollama rejects the request outright::

    status_code: 400, model_name: llama3:8b, body: {'message':
    'registry.ollama.ai/library/llama3:8b does not support tools', ...}

So this lane records against real OpenAI ``gpt-4o-mini`` (two short requests,
well under a cent).

WHAT THE SEALED TRACE HONESTLY CARRIES (and what it does NOT)
------------------------------------------------------------
* ``agent.input`` / ``agent.output`` / ``agent.identity`` from ``@trace`` — the
  Agent column fills with ``listing-extraction-agent`` and the Status column
  with the real run outcome.
* One ``tool.call`` + one ``model.invoke`` per primitive, ``framework: marvin``
  (the Framework column), carrying the developer-declared ``agent_name``, the
  resolved ``response_model``, the real args/response, and real latency.
* **NO ``cost.record``, and no token counts.** That is not a gap in this
  recording — it is the adapter's honest omission: Marvin surfaces no usage on
  its primitives, so there are no tokens at this layer and the pricing hook has
  nothing real to price. A token/cost figure here would be fabricated. The
  underlying pydantic-ai/OpenAI layer is NOT instrumented in this run, so
  nothing deeper reports them either (the corpus fixture below is where the real
  ``usage`` block from the same run is preserved verbatim).
* ``model`` is the model the developer really configured on the
  ``marvin.Agent`` (``gpt-4o-mini``), read off ``Agent.model`` — not a
  placeholder, and NOT the dated ``gpt-4o-mini-2024-07-18`` the provider
  response echoes (the adapter never reads a model back off a response body).

HONEST STRUCTURAL LIMIT — FRAGMENTED SPAN TREE (a real finding, recorded as-is):
Marvin's primitives are ambient module functions, so ``FrameworkAdapter._begin_run``
opens a fresh run scope per call and parents that call's events on a root span the
adapter never emits an event for. Standalone (the adapter's documented ambient
usage) the collector's ``_synthesize_root_if_needed`` roots that single dangling
parent and the tree is clean. Inside the enclosing ``@trace`` needed here for the
Status column, there are TWO dangling parents (one per primitive), which the
synthesizer deliberately declines to merge — so the sealed trace has 3 tree roots
and the FE flags it ``fragmented`` (it still renders every span, and Agent /
Framework / Status all fill). The events are complete and real; only the
marvin-root -> @trace-root linkage is missing. Left as-is rather than papered
over: hand-emitting the missing link in this generator would be inventing
structure the SDK did not produce.
"""

from __future__ import annotations

import os
import sys
import json
import datetime as _dt
from typing import Any, Dict, List, Optional

# --- path bootstrap so this module runs standalone AND when imported by the
#     central _generate_fixtures.main() loader ---------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))          # samples/data/generators
_DATA = os.path.dirname(_HERE)                              # samples/data
_SAMPLES = os.path.dirname(_DATA)                           # samples
_REPO = os.path.dirname(_SAMPLES)                           # repo root
for _p in (os.path.join(_REPO, "src"), _DATA, _SAMPLES):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# This file is named ``marvin.py``. A bare-script launch puts its own directory
# on sys.path[0], which would SHADOW the real ``marvin`` package for the
# function-local ``import marvin``. Drop our own dir (a no-op in the integrated
# ``generators.marvin`` import path).
sys.path[:] = [_q for _q in sys.path if os.path.abspath(_q or ".") != _HERE]

import httpx  # noqa: E402

import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens import Stratix  # noqa: E402
from layerlens.instrument import trace  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402

# Reuse the shared record-real-once seam (_write / _CAPTURE / OPENAI_MODEL) from
# the central fixture generator; fall back to a self-contained copy so this
# module still records standalone if _generate_fixtures isn't importable.
try:
    from _generate_fixtures import _CAPTURE, OPENAI_MODEL, _write  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - standalone fallback
    _CAPTURE = CaptureConfig.full()
    OPENAI_MODEL = os.environ.get("SAMPLE_OPENAI_MODEL", "gpt-4o-mini")
    _TRACES = os.path.join(_DATA, "traces")

    def _write(payloads, category, stem):  # type: ignore[no-redef]
        out = os.path.join(_TRACES, category, f"{stem}.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            for p in payloads:
                f.write(json.dumps(p, default=str) + "\n")
        return out


# ``import marvin`` calls ensure_db_tables_exist() at module scope, CREATING a
# SQLite database as an import side effect. Point it at a throwaway file BEFORE
# any marvin import so regenerating a fixture never touches the developer's real
# Marvin database.
def _isolate_marvin_db() -> None:
    if "MARVIN_DATABASE_URL" not in os.environ:
        import tempfile

        os.environ["MARVIN_DATABASE_URL"] = "sqlite+aiosqlite:///" + os.path.join(
            tempfile.mkdtemp(prefix="layerlens-marvin-gen-"), "marvin.db"
        )


# ---------------------------------------------------------------------------
# The recorded corpus half — capture the RAW upstream bodies of this same run.
# Mirrors tests/fixtures/record_corpus.py's _RecordingTransport, in its ASYNC
# form: pydantic-ai's OpenAI client is async-only, so the seam marvin reaches
# the network through is an ``httpx.AsyncClient``.
# ---------------------------------------------------------------------------
_KEEP_HEADERS = frozenset({"content-type", "x-request-id", "request-id", "openai-version"})


class _RecordingAsyncTransport(httpx.AsyncBaseTransport):
    """Perform the REAL request, capture the response body, and hand back a fresh
    response the SDK can still read (the original stream is consumed by us)."""

    def __init__(self) -> None:
        self._real = httpx.AsyncHTTPTransport()
        self.interactions: List[Dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        response = await self._real.handle_async_request(request)
        raw = await response.aread()
        await response.aclose()
        body_json: Any = None
        body_text: Optional[str] = None
        try:
            body_json = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            body_text = raw.decode("utf-8", "replace")
        headers = {k: v for k, v in response.headers.items() if k.lower() in _KEEP_HEADERS}
        resp_record: Dict[str, Any] = {"status_code": response.status_code, "headers": headers}
        if body_json is not None:
            resp_record["json"] = body_json
        else:
            resp_record["text"] = body_text
        self.interactions.append(
            {
                "request": {"method": request.method, "path": request.url.path},
                "response": resp_record,
            }
        )
        return httpx.Response(
            response.status_code,
            content=raw,
            headers=response.headers,
            request=request,
        )


def _write_corpus(scenario: str, interactions: List[Dict[str, Any]], *, model: str) -> str:
    """Seal one recorded-corpus fixture under tests/fixtures/recorded/marvin/.

    Scrubbed + provenance-stamped with ``tests/instrument/_recorded.py``'s own
    helpers, so ``tests/test_recorded_corpus.py``'s leak-scan and provenance
    guards apply to these exactly like every other committed fixture.
    """
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    import marvin  # pyright: ignore[reportMissingImports]

    from tests.instrument._recorded import RECORDED_ROOT, scrub  # noqa: E402

    fixture = {
        "provenance": {
            # ``provider`` names the upstream whose raw body this is (OpenAI's
            # chat.completion), which is what the replay deserializes; the
            # framework under test is marvin, hence the directory.
            "provider": "openai",
            "sdk_version": "marvin %s" % getattr(marvin, "__version__", "unknown"),
            "model": model,
            "scenario": scenario,
            "captured_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        },
        "transport": "http",
        "interactions": interactions,
    }
    out = RECORDED_ROOT / "marvin" / ("%s.json" % scenario)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(scrub(fixture), f, indent=2, sort_keys=False)
        f.write("\n")
    return str(out)


# ---------------------------------------------------------------------------
# The Real-Estate scenario — a genuine freeform MLS listing write-up.
# ---------------------------------------------------------------------------
LISTING_ID = "MLS-4471-OAKRIDGE"

LISTING_DESCRIPTION = (
    "Welcome to 1428 Oakridge Lane, a beautifully maintained 1997 craftsman-style "
    "single-family home tucked into the sought-after Oakridge Park neighborhood of "
    "Round Rock. Offered at $749,000, this 2,340 square foot residence gives you four "
    "generous bedrooms and two and a half bathrooms, including a main-floor primary "
    "suite with a spa-inspired walk-in shower and dual vanities. The chef's kitchen was "
    "fully renovated in 2023 with quartz countertops, a gas range, and a walk-in pantry, "
    "and it opens onto a light-filled great room anchored by a wood-burning fireplace. "
    "Enjoy hardwood floors throughout the main level, a dedicated home office, and a "
    "finished bonus room over the attached two-car garage. Outside, the 0.28 acre lot is "
    "fully fenced and backs to a greenbelt, with a covered patio, mature oaks, and an "
    "in-ground sprinkler system. Recent updates include a 2022 roof and a new 16-SEER "
    "HVAC system. Zoned to the highly rated Oakridge Elementary, and just minutes from "
    "the tollway. HOA dues are $45/month and cover the neighborhood pool and trails."
)

_AGENT_NAME = "listing-extraction-agent"

_AGENT_INSTRUCTIONS = (
    "You are listing-extraction-agent, an MLS listing-intake assistant for a real-estate "
    "brokerage. You normalize an agent's freeform property write-up into the structured "
    "fields the MLS database requires. Use ONLY what the description actually states — "
    "never infer, round, or invent a value that is not written."
)

_CAST_INSTRUCTIONS = (
    "Extract the structured MLS record for this single property from the listing "
    "description. Every field must come from the text."
)

_EXTRACT_INSTRUCTIONS = (
    "Each distinct, marketable feature or amenity of the property that a buyer would "
    "search on (e.g. 'wood-burning fireplace', 'covered patio'). Short noun phrases, "
    "taken from the description only."
)


def _property_listing_model():
    """The MLS record schema. Function-local so the module imports without pydantic."""
    from pydantic import Field, BaseModel

    class PropertyListing(BaseModel):
        """A structured MLS record normalized from a freeform listing description."""

        street_address: str = Field(description="Street address of the property.")
        property_type: str = Field(description="Property type, e.g. 'single-family'.")
        bedrooms: int = Field(description="Number of bedrooms.")
        bathrooms: float = Field(description="Number of bathrooms (half-baths count as 0.5).")
        square_feet: int = Field(description="Interior living area in square feet.")
        list_price_usd: int = Field(description="Asking price in USD.")
        year_built: int = Field(description="Year the home was built.")
        garage_spaces: int = Field(description="Number of garage parking spaces.")
        lot_size_acres: float = Field(description="Lot size in acres.")
        hoa_monthly_usd: float = Field(description="Monthly HOA dues in USD.")

    return PropertyListing


def _marvin_agent(rec: _RecordingAsyncTransport):
    """A REAL ``marvin.Agent`` with a developer-declared name, backed by real
    OpenAI over the recording transport (the same ``http_client=`` seam the
    replay test injects a MockTransport through)."""
    import marvin  # pyright: ignore[reportMissingImports]
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    # Marvin's rich console handler renders a live panel per call — pure noise
    # in a recording run.
    marvin.settings.enable_default_print_handler = False

    provider = OpenAIProvider(http_client=httpx.AsyncClient(transport=rec))
    model = OpenAIChatModel(OPENAI_MODEL, provider=provider)
    # ``model=`` on the Agent is the developer's explicit, real configuration —
    # the adapter reads it off ``Agent.model`` (no Marvin 3.x primitive takes a
    # ``model=`` kwarg, so this is the only honest per-call seam).
    return marvin.Agent(name=_AGENT_NAME, model=model, instructions=_AGENT_INSTRUCTIONS)


def generate_marvin_single(client: Stratix) -> dict:
    """Record the REAL Marvin MLS listing-intake run and seal both artifacts.

    Drives two real Marvin primitives (``cast`` -> ``PropertyListing``, then
    ``extract`` -> the feature list) through ONE named ``marvin.Agent`` inside a
    single ``@trace`` scope, with the real ``MarvinAdapter`` patched onto the
    real ``marvin`` module. The adapter reuses the decorator's collector (its
    ``_begin_run`` sees ``_current_collector``), so both primitives land in ONE
    trace alongside the decorator's ``agent.input``/``agent.output``.
    """
    _isolate_marvin_db()

    import marvin  # pyright: ignore[reportMissingImports]

    from layerlens.instrument.adapters.frameworks.marvin import MarvinAdapter

    PropertyListing = _property_listing_model()
    rec = _RecordingAsyncTransport()
    agent = _marvin_agent(rec)

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = MarvinAdapter(client, capture_config=_CAPTURE)
    try:
        adapter.connect(target=marvin)

        @trace(client, name=_AGENT_NAME, capture_config=_CAPTURE)
        def _normalize_listing(description: str) -> dict:
            record = marvin.cast(
                description,
                target=PropertyListing,
                instructions=_CAST_INSTRUCTIONS,
                agent=agent,
            )
            features = marvin.extract(
                description,
                target=str,
                instructions=_EXTRACT_INSTRUCTIONS,
                agent=agent,
            )
            return {
                "listing_id": LISTING_ID,
                "record": record.model_dump(),
                "features": list(features),
            }

        result = _normalize_listing(LISTING_DESCRIPTION)
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for marvin listing-extract run")
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "real-estate",
        "listing-extraction",
        "structured-extraction",
    ]

    # ---- seal the corpus half of the SAME run --------------------------------
    # Two primitives -> two upstream chat.completions. Split them into two
    # single-interaction fixtures so each replay test is an independent,
    # deterministic one-request run (the shape capture_openai() uses).
    if len(rec.interactions) < 2:
        raise RuntimeError(
            "expected >=2 recorded OpenAI interactions (cast + extract), got %d — "
            "refusing to seal a corpus that does not match the run"
            % len(rec.interactions)
        )
    corpus_paths = [
        _write_corpus("listing_cast", rec.interactions[0:1], model=OPENAI_MODEL),
        _write_corpus("features_extract", rec.interactions[1:2], model=OPENAI_MODEL),
    ]

    # ---- honest provenance print --------------------------------------------
    events = payload.get("events", [])
    agents = sorted(
        {(e.get("payload") or {}).get("agent_name") for e in events
         if (e.get("payload") or {}).get("agent_name")}
    )
    frameworks = sorted(
        {(e.get("payload") or {}).get("framework") for e in events
         if (e.get("payload") or {}).get("framework")}
    )
    models = sorted(
        {(e.get("payload") or {}).get("model") for e in events
         if e.get("event_type") == "model.invoke" and (e.get("payload") or {}).get("model")}
    )
    tc = [e for e in events if e.get("event_type") == "tool.call"]
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    ident = [e for e in events if e.get("event_type") == "agent.identity"]
    print(
        "  marvin-single (real-estate MLS listing intake)  agents=%s frameworks=%s models=%s"
        % (agents, frameworks, models)
    )
    print(
        "    tool.call=%d model.invoke=%d cost.record=%d (0 is HONEST — marvin surfaces no usage) "
        "agent.identity=%d" % (len(tc), len(mi), len(cr), len(ident))
    )
    print("    record=%r" % (json.dumps(result["record"])[:160],))
    print("    features=%r" % (result["features"][:6],))
    for p in corpus_paths:
        print("    corpus ->", p)
    print("  ->", _write([payload], "industry", "realestate_marvin_listing_extract"), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_marvin_single(_client)
