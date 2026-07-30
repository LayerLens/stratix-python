"""ADP-PORT Family-B recorder for the **instructor** framework adapter (Legal).

Records ONE REAL, fully-instrumented ``instructor`` run and seals it to
``samples/data/traces/industry/legal_instructor_contract_extract.jsonl``. The
fixture is a genuine run of the real ``InstructorAdapter`` over a real
``instructor.from_openai(OpenAI())`` patched client backed by a real OpenAI model
(gpt-4o-mini) — nothing is fabricated. The framework deps (``instructor``,
``openai``, ``pydantic``) are imported FUNCTION-LOCALLY so this module imports in
any venv.

ONE lane (Legal domain; de-conflicted from the W1 ``legal_contracts`` /
``legal_research`` / ``legal_haystack_*`` / ``legal_contract_rag`` stems):

* ``generate_instructor_single`` -> ``legal_instructor_contract_extract``
  A NAMED contract-abstraction assistant (``contract-metadata-extractor``)
  extracts the structured deal terms a legal-ops team abstracts off every
  inbound MSA — the parties, the effective date and initial term, the governing
  law / venue, and the renewal mechanics (auto-renew, renewal term, non-renewal
  notice window) — from a real commercial Master Services Agreement excerpt into
  a Pydantic ``ContractMetadata`` ``response_model``. This is instructor's whole
  reason to exist (a validated Pydantic object off a provider tool call), and it
  is the real day-one task of contract abstraction.

RECORD REAL **ONCE** — both artifacts come from this single API call:

1. the sealed Family-B trace (``samples/data/traces/industry/*.jsonl``) — the
   real adapter's emitted events, which the industry sample uploads; and
2. the recorded-corpus fixture (``tests/fixtures/recorded/instructor/
   contract_extract.json``) — the raw OpenAI ``chat.completion`` transport body
   captured UPSTREAM of instructor's parser, which
   ``tests/instrument/adapters/frameworks/test_instructor_recorded.py`` replays
   in CI (no creds, no network, no spend).

The run drives the real ``openai`` client through a ``_RecordingTransport`` (the
same seam ``tests/fixtures/record_corpus.py`` uses): the request really goes to
OpenAI, and the response body is captured on its way into instructor's real
``Mode.TOOLS`` parser. That keeps the corpus fixture honest by construction —
it is the provider's raw body, the one thing we do not control.

WHY OPENAI AND NOT OLLAMA: instructor's default ``Mode.TOOLS`` needs reliable
provider tool-calling to return a schema-valid ``ContractMetadata``; llama3:8b
does not do this dependably, and a retry-until-it-parses loop would be a
different (dishonest) scenario. The real spend is a fraction of a cent.

HONESTY / Agent column: instructor declares NO agent identity of its own — the
adapter's only honest source is a caller-DECLARED name, so this lane passes
``agent_name="contract-metadata-extractor"`` explicitly to
``adapter.connect(...)``. Nothing is synthesized from the framework label.
"""

from __future__ import annotations

import os
import sys
import json

# --- path bootstrap so this module runs standalone AND when imported by the
#     central _generate_fixtures.main() ---------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))          # samples/data/generators
_DATA = os.path.dirname(_HERE)                              # samples/data
_SAMPLES = os.path.dirname(_DATA)                           # samples
_REPO = os.path.dirname(_SAMPLES)                           # repo root
for _p in (os.path.join(_REPO, "src"), _DATA, _SAMPLES, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# This file is named ``instructor.py``; if a bare-script launch put its own
# directory on sys.path it would SHADOW the real ``instructor`` package. Drop our
# own dir so the function-local ``import instructor`` always resolves to the
# installed framework (a no-op in the integrated ``generators.instructor`` path).
sys.path[:] = [_q for _q in sys.path if os.path.abspath(_q) != _HERE]

from layerlens import Stratix  # noqa: E402  (re-exported for the seam signature)
from layerlens.instrument import trace  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

# Reuse the shared record-real-once seam (_write / _CAPTURE / OPENAI_MODEL) from
# the central fixture generator; fall back to a self-contained copy so this
# module still records standalone if _generate_fixtures isn't importable.
try:
    from _generate_fixtures import _write, _CAPTURE, OPENAI_MODEL  # type: ignore[attr-defined]
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


# ---------------------------------------------------------------------------
# The REAL contract under abstraction (a commercial MSA excerpt).
# ---------------------------------------------------------------------------
# A genuine-shape Master Services Agreement excerpt carrying exactly the clauses
# a legal-ops abstraction pass reads: the preamble (parties + incorporation),
# Term (effective date, initial term, auto-renewal, notice window), and Governing
# Law (choice of law + exclusive venue). Every field the response_model asks for
# is STATED in the text — so the extraction is checkable, and a hallucinated
# value is a real, visible failure rather than an unfalsifiable guess.
_CONTRACT_TEXT = """\
MASTER SERVICES AGREEMENT

This Master Services Agreement (this "Agreement") is entered into as of
March 14, 2026 (the "Effective Date") by and between NORTHWIND ANALYTICS, INC.,
a Delaware corporation with its principal place of business at 400 Harrison
Avenue, Boston, Massachusetts 02118 ("Provider"), and MERIDIAN HEALTH PARTNERS,
LLC, a New York limited liability company with its principal place of business
at 1220 Sixth Avenue, New York, New York 10020 ("Customer"). Provider and
Customer are each a "Party" and together the "Parties."

1. SERVICES. Provider shall perform the data-engineering and analytics services
described in one or more Statements of Work executed by the Parties and
incorporated herein by reference.

...

7. TERM AND TERMINATION.

7.1 Initial Term. This Agreement commences on the Effective Date and, unless
earlier terminated in accordance with this Section 7, continues for an initial
term of thirty-six (36) months (the "Initial Term").

7.2 Renewal. Upon expiration of the Initial Term, this Agreement shall
automatically renew for successive renewal terms of twelve (12) months each
(each, a "Renewal Term"), unless either Party delivers written notice of its
intent not to renew at least ninety (90) days prior to the end of the
then-current term.

7.3 Termination for Convenience. Customer may terminate any Statement of Work
for convenience upon sixty (60) days' prior written notice; such termination
shall not terminate this Agreement.

...

14. GOVERNING LAW; VENUE. This Agreement and any dispute arising out of or
relating to it shall be governed by and construed in accordance with the laws of
the State of New York, without regard to its conflict-of-laws principles. The
Parties irrevocably submit to the exclusive jurisdiction of the state and
federal courts located in New York County, New York.

15. ENTIRE AGREEMENT. This Agreement, together with all Statements of Work,
constitutes the entire agreement between the Parties with respect to its subject
matter and supersedes all prior or contemporaneous understandings.
"""

_SYSTEM_PROMPT = (
    "You are contract-metadata-extractor, a legal-operations contract-abstraction "
    "assistant. Read the agreement and abstract its key deal terms into the "
    "structured schema: the contracting parties (with their contract-defined role "
    "and state of incorporation), the effective date, the initial term, the "
    "governing law and exclusive venue, and the renewal mechanics. Use ONLY what "
    "the agreement actually states — never infer, assume, or invent a term. If the "
    "agreement does not state a field, leave it null."
)


def _capture_instructor(
    client: Stratix,
    *,
    agent_name: str,
    response_model,
    system: str,
    user: str,
    max_tokens: int = 700,
):
    """Run ONE real ``instructor`` create() under ``@trace`` + the real
    ``InstructorAdapter`` + the observer seam (background upload suppressed),
    through a recording transport that captures the raw OpenAI body on its way
    into instructor's parser.

    ``@trace`` + ``instrument_<framework>`` is the documented customer pattern,
    and the two compose into ONE trace: the adapter's ``_begin_run`` reuses the
    collector ``@trace`` already put on the context. That layering is what makes
    every rendered column honest:

    * ``@trace`` contributes the real run lifecycle — ``agent.input`` (the real
      contract), ``agent.output`` (the real extracted object + the real run
      outcome ``status``), ``agent.identity`` (the caller-declared name). The
      **Status column derives from ``agent.output.status``** (see the atlas
      ``extractTraceMetadata`` status branch), so without the ``@trace`` boundary
      Status would render EMPTY: ``InstructorAdapter`` emits ``status`` on
      ``model.invoke`` only, and nothing reads it there.
    * the adapter contributes the real ``model.invoke`` (model / tokens /
      ``response_model`` / provider) and the priced ``cost.record``.

    Returns ``(payload, extracted, interactions, openai_version, instructor_version)``:
    the sealed trace payload, the validated Pydantic object, and the captured
    upstream HTTP interactions for the recorded corpus.
    """
    import openai
    import httpx
    import instructor

    from layerlens.instrument.adapters.frameworks.instructor import InstructorAdapter

    # The recording seam: performs the REAL request and captures the response
    # body (the same transport tests/fixtures/record_corpus.py records with), so
    # the corpus fixture is the provider's raw body — captured UPSTREAM of
    # instructor's Mode.TOOLS parser, which then really parses it in this run.
    from tests.fixtures.record_corpus import _RecordingTransport  # type: ignore[import-not-found]

    rec = _RecordingTransport()
    raw = openai.OpenAI(http_client=httpx.Client(transport=rec, timeout=60.0))
    patched = instructor.from_openai(raw)

    holder: dict = {}
    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = InstructorAdapter(client, capture_config=_CAPTURE)
    try:
        # agent_name is the ONLY honest identity source for instructor (it
        # declares none of its own) — so the caller declares it explicitly, both
        # to the adapter (stamps model.invoke) and to @trace (the run boundary).
        adapter.connect(target=patched, agent_name=agent_name)

        @trace(client, name=agent_name, capture_config=_CAPTURE)
        def _abstract(_contract: str):
            extracted = patched.chat.completions.create(
                model=OPENAI_MODEL,
                response_model=response_model,
                max_retries=2,
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": _contract},
                ],
            )
            holder["extracted"] = extracted
            # Return the validated object's real fields so agent.output carries
            # the genuine structured extraction as JSON (not a repr string).
            return extracted.model_dump()

        _abstract(user)
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for instructor run")
    return payload, holder["extracted"], rec.interactions, openai.__version__, instructor.__version__


def _write_corpus(interactions, *, openai_version: str) -> None:
    """Seal the raw OpenAI body this run produced as the recorded-corpus fixture.

    Same run, same body — the replay test in CI drives instructor's real
    ``Mode.TOOLS`` parser over exactly the response that parsed here for real.
    Best-effort: a corpus write failure must not lose the Family-B trace.
    """
    try:
        from tests.fixtures.record_corpus import _write as _write_fixture, _http_fixture  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - corpus harness unavailable
        print("    [skip] recorded-corpus write: %s: %s" % (type(exc).__name__, exc))
        return
    if not interactions:
        print("    [skip] recorded-corpus write: the run captured no HTTP interaction")
        return
    # The final interaction is the completed tool call instructor validated.
    _write_fixture(
        "instructor",
        "contract_extract",
        _http_fixture("openai", openai_version, OPENAI_MODEL, "contract_extract", interactions[-1:]),
    )


def generate_instructor_single(client: Stratix) -> None:
    """Instructor SINGLE (Legal contract-metadata extraction).

    A NAMED ``contract-metadata-extractor`` abstracts a real Master Services
    Agreement excerpt into a validated Pydantic ``ContractMetadata`` via a real
    ``instructor.from_openai(OpenAI())`` patched client on gpt-4o-mini, recorded
    under the real ``InstructorAdapter`` -> one honest agent node (Agent column =
    ``contract-metadata-extractor``, Framework = ``instructor``) with a real
    ``model.invoke`` (real model + real tokens + real ``response_model``) and a
    priced ``cost.record``.
    """
    from typing import List, Optional

    from pydantic import BaseModel, Field

    class Party(BaseModel):
        """A contracting party as the agreement's preamble defines it."""

        name: str = Field(description="Legal entity name of the party, as written in the agreement.")
        role: str = Field(description="The party's contract-defined role, e.g. 'Provider' or 'Customer'.")
        entity_type: Optional[str] = Field(
            default=None,
            description="Entity form as stated, e.g. 'Delaware corporation', 'New York limited liability company'.",
        )

    class ContractMetadata(BaseModel):
        """The deal terms a legal-ops team abstracts off an inbound agreement."""

        agreement_title: str = Field(description="The agreement's title, e.g. 'Master Services Agreement'.")
        parties: List[Party] = Field(description="Every contracting party named in the preamble.")
        effective_date: str = Field(description="The stated Effective Date, as written in the agreement.")
        initial_term_months: int = Field(description="Length of the Initial Term in months.")
        governing_law: str = Field(description="The jurisdiction whose law governs, e.g. 'State of New York'.")
        exclusive_venue: Optional[str] = Field(
            default=None, description="The exclusive forum for disputes, if the agreement states one."
        )
        auto_renews: bool = Field(description="True if the agreement automatically renews at the end of a term.")
        renewal_term_months: Optional[int] = Field(
            default=None, description="Length of each Renewal Term in months, if stated."
        )
        non_renewal_notice_days: Optional[int] = Field(
            default=None,
            description="Days of advance written notice required to stop a renewal, if stated.",
        )

    payload, extracted, interactions, openai_version, instructor_version = _capture_instructor(
        client,
        agent_name="contract-metadata-extractor",
        response_model=ContractMetadata,
        system=_SYSTEM_PROMPT,
        user="Abstract the deal terms from this agreement:\n\n%s" % _CONTRACT_TEXT,
        max_tokens=700,
    )
    payload["tags"] = [
        "layerlens-sample", "industry", "legal", "contract-abstraction", "typed-extraction",
    ]

    events = payload.get("events", [])
    agents = sorted({(e.get("payload") or {}).get("agent_name")
                     for e in events if (e.get("payload") or {}).get("agent_name")})
    frameworks = sorted({(e.get("payload") or {}).get("framework")
                         for e in events if (e.get("payload") or {}).get("framework")})
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    print("  instructor-single (legal contract abstraction)  instructor=%s agents=%s frameworks=%s "
          "model.invoke=%d cost.record=%d" % (instructor_version, agents, frameworks, len(mi), len(cr)))
    if mi:
        p = mi[0]["payload"]
        print("    model=%r response_model=%r tokens=%s/%s/%s status=%r"
              % (p.get("model"), p.get("response_model"), p.get("tokens_prompt"),
                 p.get("tokens_completion"), p.get("tokens_total"), p.get("status")))
    if cr:
        print("    cost_usd=%r" % (cr[0]["payload"].get("cost_usd"),))
    # agent.output.status is the ONLY signal the Status column derives from.
    ao = next((e["payload"] for e in events if e.get("event_type") == "agent.output"), None)
    print("    agent.output.status=%r (-> Status column)  events=%s"
          % (ao.get("status") if ao else None, [e["event_type"] for e in events]))
    print("    extracted=%s" % (extracted.model_dump_json()[:220],))
    _write_corpus(interactions, openai_version=openai_version)
    print("  ->", _write([payload], "industry", "legal_instructor_contract_extract"), "\n")


if __name__ == "__main__":
    generate_instructor_single(Stratix())
