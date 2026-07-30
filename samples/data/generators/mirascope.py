"""ADP-PORT Family-B recorder for the **mirascope** framework adapter.

Records ONE REAL, fully-instrumented ``mirascope`` v2 run and seals TWO artifacts
from that single run:

* ``samples/data/traces/industry/insurance_mirascope_fnol_intake.jsonl`` — the
  sealed sample trace (the record-real-once observer seam: real ``@llm.call``
  under the real ``MirascopeAdapter``, background upload suppressed).
* ``tests/fixtures/recorded/mirascope/fnol_intake.json`` — the recorded-corpus
  fixture: the REAL upstream ``chat.completions`` wire body the model actually
  returned, captured at the transport as the run happened. Replayed offline by
  ``tests/instrument/adapters/frameworks/test_mirascope_recorded.py`` so the
  adapter's real parser is held to a real response shape.

Recording both from the same call is the point: the corpus fixture is provably
the body that produced the shipped trace, not a second, differently-shaped run.

THE LANE (Insurance / FNOL intake) -> ``insurance_mirascope_fnol_intake``
    A NAMED ``fnol_intake_agent`` (mirascope resolves the Agent identity from the
    decorated function's name) turns a policyholder's free-text first-notice-of-loss
    narrative — the kind a call-centre rep pastes in verbatim — into the structured
    FNOL record a claims system can actually open a claim from: policy number,
    claimant, loss date/type/location, vehicles, injuries, police report, and a
    triage severity. This is mirascope's headline strength (a typed ``format=``
    spec on a plain decorated function) applied to a real insurance task.

TRANSPORT — ollama/llama3:8b, local, $0.00
    mirascope v2 ships a first-party ``OllamaProvider`` (its OpenAI-compatible
    ``/v1/`` surface), so this lane is a genuine mirascope run against a real
    local model — no provider spend. ``mode="json"`` is required and honest:
    mirascope's default structured-output mode is forced tool-calling, which
    ``llama3:8b`` does not support (the server rejects it with a real 400), while
    ``json`` mode uses the provider's JSON mode and llama3:8b extracts the FNOL
    fields reliably.

The framework deps (``mirascope``) are imported FUNCTION-LOCALLY so this module
imports in any venv.
"""

from __future__ import annotations

import os
import sys
import json
import datetime as _dt

# --- path bootstrap so this module runs standalone AND when imported by the
#     central _generate_fixtures.main() ---------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))          # samples/data/generators
_DATA = os.path.dirname(_HERE)                              # samples/data
_SAMPLES = os.path.dirname(_DATA)                           # samples
_REPO = os.path.dirname(_SAMPLES)                           # repo root
for _p in (os.path.join(_REPO, "src"), _DATA, _SAMPLES):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# This file is named ``mirascope.py``; if a bare-script launch put its own
# directory on sys.path it would SHADOW the real ``mirascope`` package. Drop our
# own dir so the function-local ``import mirascope`` always resolves to the
# installed framework (a no-op in the integrated ``generators.mirascope`` path).
sys.path[:] = [_q for _q in sys.path if os.path.abspath(_q) != _HERE]

import httpx  # noqa: E402

from layerlens import Stratix  # noqa: E402  (re-exported for the seam signature)
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

# Reuse the shared record-real-once seam (_write / _CAPTURE) from the central
# fixture generator; fall back to a self-contained copy so this module still
# records standalone if _generate_fixtures isn't importable.
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


#: The local model this lane records against. mirascope namespaces the provider
#: into the model id, which is exactly what the adapter has to normalise.
OLLAMA_MODEL_ID = os.environ.get("SAMPLE_MIRASCOPE_MODEL", "ollama/llama3:8b")

#: Where the recorded-corpus fixture (the real upstream body) is sealed.
_CORPUS = os.path.join(_REPO, "tests", "fixtures", "recorded", "mirascope")


# ---------------------------------------------------------------------------
# Upstream capture — the real request happens; the real body is kept
# ---------------------------------------------------------------------------
class _RecordingTransport(httpx.BaseTransport):
    """Performs the REAL request and captures the response body verbatim.

    Mirrors ``tests/fixtures/record_corpus._RecordingTransport``: the original
    stream is consumed by the capture, so a fresh response carrying the already
    decompressed bytes is handed back to the SDK (headers that would claim an
    encoding/length it no longer has are dropped).
    """

    def __init__(self) -> None:
        self._real = httpx.HTTPTransport()
        self.interactions: list = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.read()
        response = self._real.handle_request(request)
        raw = response.read()
        body_json = None
        body_text = None
        try:
            body_json = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            body_text = raw.decode("utf-8", errors="replace")
        self.interactions.append(
            {
                "request": {"method": request.method, "path": request.url.path},
                "response": {
                    "status_code": response.status_code,
                    "json": body_json,
                    "text": body_text,
                    "headers": {
                        k: v for k, v in response.headers.items() if k.lower() == "content-type"
                    },
                },
            }
        )
        passthrough = httpx.Headers(
            [
                (k, v)
                for k, v in response.headers.items()
                if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
            ]
        )
        return httpx.Response(response.status_code, headers=passthrough, content=raw, request=request)


def _seal_corpus(interactions: list, *, scenario: str, model: str) -> str:
    """Write the captured upstream body as a recorded-corpus http fixture."""
    from tests.instrument._recorded import scrub  # type: ignore[import-not-found]
    from importlib.metadata import version

    fixture = scrub(
        {
            "provenance": {
                "provider": "mirascope",
                "sdk_version": version("mirascope"),
                "model": model,
                "scenario": scenario,
                "captured_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
            },
            "transport": "http",
            "interactions": interactions,
        }
    )
    os.makedirs(_CORPUS, exist_ok=True)
    out = os.path.join(_CORPUS, f"{scenario}.json")
    with open(out, "w") as f:
        json.dump(fixture, f, indent=2, sort_keys=False)
        f.write("\n")
    return out


# ---------------------------------------------------------------------------
# The scenario — a real policyholder FNOL narrative
# ---------------------------------------------------------------------------
# Verbatim intake notes of the kind a call-centre rep types while a policyholder
# reports a loss: unstructured, out of order, with the facts a claims system needs
# scattered through the prose. Extracting THIS into an openable claim record is the
# real business task.
FNOL_NARRATIVE = (
    "Caller: Denise Okonkwo, reached at 512-555-0147.\n\n"
    "\"Hi — I need to report an accident. It happened yesterday, the 11th of March, "
    "a little after 5:30 in the evening. I was heading home on Lamar Boulevard, just "
    "past the Barton Springs intersection here in Austin, and I was stopped at the "
    "light when a silver pickup came up behind me and just didn't stop. Hit me hard "
    "enough that I got pushed into the intersection. My car is a 2021 Honda Accord — "
    "the whole back end is crushed, the trunk won't shut and the bumper is basically "
    "on the ground. It's drivable but barely. The other driver stayed, he was "
    "apologetic, said he was looking at his phone. His truck had some front-end "
    "damage too. My neck and shoulder have been stiff since last night so I saw my "
    "doctor this morning, she said it's whiplash and to take it easy for a week. "
    "Nobody else was hurt, my daughter wasn't in the car. APD came out and did a "
    "report, the officer gave me a number, it was APD-2025-0311-4417. My policy is "
    "AUTO-TX-4482910. What happens now?\""
)


def generate_mirascope_single(client: Stratix) -> None:
    """Mirascope SINGLE (typed FNOL intake): a NAMED ``fnol_intake_agent``
    ``@llm.call`` with a typed ``format=`` spec turns a policyholder's free-text
    first-notice-of-loss narrative into the structured FNOL record a claims system
    can open a claim from. Real mirascope v2 over a real local ollama model,
    recorded under the real MirascopeAdapter -> one honest agent node with a real
    ``model.invoke`` (real ``model_id`` / ``provider_id`` / token usage) +
    ``cost.record`` + the typed ``tool.result`` output.

    The REAL upstream wire body is captured at the transport during the same call
    and sealed as the recorded-corpus fixture.
    """
    from pydantic import BaseModel, Field
    import mirascope.llm as llm
    from mirascope.llm.providers.ollama import OllamaProvider
    from mirascope.llm.providers.provider_registry import (
        PROVIDER_REGISTRY,
        provider_singleton,
        reset_provider_registry,
    )
    from openai import OpenAI

    from layerlens.instrument.adapters.frameworks.mirascope import MirascopeAdapter

    class FirstNoticeOfLoss(BaseModel):
        """The structured FNOL record extracted from a policyholder's narrative."""

        policy_number: str = Field(description="The policyholder's stated policy number.")
        claimant_name: str = Field(description="Full name of the person reporting the loss.")
        loss_date: str = Field(description="The date the loss occurred, as stated by the caller.")
        loss_type: str = Field(
            description="The kind of loss, e.g. 'rear-end collision', 'theft', 'hail damage'."
        )
        loss_location: str = Field(description="Where the loss occurred, as stated by the caller.")
        insured_vehicle: str = Field(description="The insured vehicle's year, make and model.")
        damage_description: str = Field(description="The damage to the insured vehicle.")
        injuries_reported: bool = Field(description="True if ANY injury was reported by the caller.")
        injury_description: str = Field(
            description="The injuries reported, or 'none' if no injury was reported."
        )
        police_report_number: str = Field(
            description="The police report number, or 'none' if the caller gave none."
        )
        other_party_involved: bool = Field(description="True if another party was involved in the loss.")
        severity: str = Field(description="Triage severity: exactly one of LOW, MEDIUM, or HIGH.")

    # A REAL mirascope OllamaProvider whose ONLY modification is a transport that
    # records the body on its way back — the request itself is real and hits the
    # real local model.
    recorder = _RecordingTransport()
    provider = OllamaProvider()
    provider.client = OpenAI(
        api_key="ollama",
        base_url=os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1/",
        http_client=httpx.Client(transport=recorder, timeout=120.0),
    )
    saved = dict(PROVIDER_REGISTRY)
    llm.register_provider(provider, scope="ollama/")

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = MirascopeAdapter(client, capture_config=_CAPTURE)
    try:
        adapter.connect()

        # ``mode="json"``: mirascope's default structured-output mode is forced
        # tool-calling, which llama3:8b genuinely does not support (the server
        # returns a real 400). json mode is the honest way to get a typed record
        # out of this model.
        @llm.call(OLLAMA_MODEL_ID, format=llm.format(FirstNoticeOfLoss, mode="json"))
        def fnol_intake_agent(narrative: str):
            return (
                "You are an auto-insurance first-notice-of-loss intake agent. Read the "
                "call-centre intake notes below and extract the FNOL record. Use ONLY "
                "facts the caller actually states — never invent a policy number, a "
                "report number, a date or an injury. Set severity to HIGH if any injury "
                "was reported or the vehicle is undrivable, MEDIUM for significant "
                "damage without injury, LOW for minor cosmetic damage.\n\n"
                f"Intake notes:\n{narrative}"
            )

        response = fnol_intake_agent(FNOL_NARRATIVE)
        record = response.parse()
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
        reset_provider_registry()
        PROVIDER_REGISTRY.update(saved)
        provider_singleton.cache_clear()

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for mirascope fnol-intake")
    payload["tags"] = [
        "layerlens-sample", "industry", "insurance", "fnol-intake", "typed-extraction",
    ]

    events = payload.get("events", [])
    agents = sorted({(e.get("payload") or {}).get("agent_name")
                     for e in events if (e.get("payload") or {}).get("agent_name")})
    frameworks = sorted({(e.get("payload") or {}).get("framework")
                         for e in events if (e.get("payload") or {}).get("framework")})
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    print("  mirascope-fnol-intake (typed FNOL extraction)  agents=%s frameworks=%s "
          "model.invoke=%d cost.record=%d" % (agents, frameworks, len(mi), len(cr)))
    if mi:
        p = mi[0]["payload"]
        print("    model=%s model_id=%s provider=%s tokens=%s/%s response_model=%s"
              % (p.get("model"), p.get("model_id"), p.get("provider"), p.get("tokens_prompt"),
                 p.get("tokens_completion"), p.get("response_model")))
    print("    typed record: policy=%s claimant=%s severity=%s injuries=%s"
          % (record.policy_number, record.claimant_name, record.severity, record.injuries_reported))
    print("  ->", _write([payload], "industry", "insurance_mirascope_fnol_intake"))
    print("  ->", _seal_corpus(recorder.interactions, scenario="fnol_intake",
                               model=OLLAMA_MODEL_ID), "\n")


if __name__ == "__main__":
    _client = Stratix()
    generate_mirascope_single(_client)
