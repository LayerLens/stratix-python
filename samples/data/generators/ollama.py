"""ADP-W2 Family-B recorder for the ``ollama`` provider adapter (record-real-once).

Records REAL local ``ollama.chat`` runs against an on-prem Ollama server and
writes them as sealed real-trace fixtures under
``samples/data/traces/industry/``. Ollama is the platform's privacy-preserving,
zero-API-cost, data-residency provider — the signature use-case is an
**on-premise / air-gapped clinical assistant** that never sends patient data off
the hospital network, so both fixtures are healthcare on-prem scenarios.

Ollama is a **provider** adapter, not an agent framework: it emits genuine
``model.invoke`` / ``cost.record`` events but declares NO agent identity and no
handoff/graph topology. So both traces render the HONEST empty-state — the Agent
column is ``—`` (nothing is invented) and the trace shows the real OTel span
waterfall (``trace.root`` -> ``model.invoke`` [-> ``tool.result``]). The
``cost.record`` carries ``cost_usd = None`` because a local model incurs no API
cost — that is honest, not a gap.

* ``generate_ollama_single`` -> ``healthcare_onprem_clinical_ollama.jsonl``:
  ONE real clinical-decision-support chat turn. A clinician asks the on-prem
  assistant a de-identified, synthetic community-acquired-pneumonia question; the
  local ``llama3:8b`` model answers. One ``model.invoke`` (framework=ollama,
  model=llama3:8b, real prompt/completion token counts, endpoint captured) +
  one ``cost.record`` (cost_usd=None) + a synthesized ``trace.root``. Empty-state
  render, real waterfall.

* ``generate_ollama_multi`` -> ``healthcare_onprem_medsafety_ollama.jsonl``:
  a REAL 2-turn medication-safety **tool-use loop**. Reconciling a new
  prescription (amiodarone) against a synthetic patient's active medication list,
  the assistant (turn 1) names the drug pairs to verify; a REAL deterministic
  ``check_interactions`` tool queries a genuine drug-interaction reference and its
  finding is emitted as a ``tool.result`` event; the assistant (turn 2) grounds
  its final medication-safety assessment on the verified findings. TWO real
  ``model.invoke`` events + one ``tool.result`` (real KB data) + ``cost.record``s
  + ``trace.root``. This is a provider tool-use LOOP, not a multi-agent graph —
  it renders the honest empty-state (Agent ``—``), NOT a DAG, with a real
  multi-step waterfall. (``llama3:8b`` has no native tool-calling, so the loop is
  orchestrated by the recorder around real model turns; the tool genuinely runs.)

Both recorders reuse the record-real-once seam from the sibling
``_generate_fixtures`` module (``_write`` + ``_CAPTURE``): a ``TraceCollector`` is
opened, the real ``ollama.chat`` calls emit into it, the collector is flushed
(which synthesizes the ``trace.root`` on the dangling root span and — because no
agent name was ever declared — deliberately does NOT synthesize an
``agent.identity``, keeping the honest empty-state), and the flushed payload is
observed via ``set_trace_observer`` with ``enqueue_upload`` no-op'd so nothing is
uploaded during generation. The samples upload the captured fixtures themselves.

Requires a reachable local Ollama server (``OLLAMA_HOST``, default
``http://localhost:11434``) with ``OLLAMA_MODEL`` (default ``llama3:8b``) pulled.
A missing/unreachable server raises a precise RuntimeError so ``main`` skips it
rather than fabricating a fixture.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any, Callable

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
# When this file is run directly (``python .../generators/ollama.py``) Python puts
# THIS directory on sys.path[0]; since the module is named ``ollama.py`` that would
# shadow the real ``ollama`` package for the function-local ``import ollama``.
# Drop the self-dir so ``import ollama`` always resolves to site-packages. (When
# imported as ``generators.ollama`` — the production path — the self-dir is not on
# sys.path anyway, so this is a no-op.)
sys.path[:] = [_p for _p in sys.path if os.path.abspath(_p or ".") != _HERE]
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from layerlens import Stratix  # noqa: E402
from layerlens.instrument._collector import TraceCollector, set_trace_observer  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._context import (  # noqa: E402
    _current_collector,
    _current_span_id,
    _push_span,
    _pop_span,
)

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE

SINGLE_STEM = "healthcare_onprem_clinical_ollama"
MULTI_STEM = "healthcare_onprem_medsafety_ollama"

_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")


# --------------------------------------------------------------------------
# Provider capture seam: drive real ``ollama.chat`` calls under a manually
# managed collector + root span, then flush. NO ``@trace`` wrapper (that would
# emit an agent.identity) — a provider trace declares no agent, so flush()'s
# honest-identity synthesis correctly leaves it as "—" while still synthesizing
# the structural ``trace.root`` so the waterfall renders. Mirrors the
# ``_generate_fixtures.capture`` seam but for the agent-free provider path.
# --------------------------------------------------------------------------
def _capture_provider(client: Stratix, driver: Callable[[Any], None]) -> dict:
    import ollama
    from layerlens.instrument.adapters.providers.ollama import (
        instrument_ollama,
        uninstrument_ollama,
    )

    # The client talks to the local server at OLLAMA_HOST (default localhost);
    # surface it so the adapter captures the honest endpoint (data-residency).
    os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    collector = TraceCollector(client, _CAPTURE)
    root_span_id = uuid.uuid4().hex[:16]
    col_token = _current_collector.set(collector)
    span_snapshot = _push_span(root_span_id, "ollama")
    try:
        instrument_ollama(ollama)
        try:
            driver(ollama)
        finally:
            uninstrument_ollama()
    finally:
        _pop_span(span_snapshot)
        _current_collector.reset(col_token)
        collector.flush()  # synthesizes trace.root; NO agent.identity (honest —)
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for ollama (is the server reachable?)")
    return payload


def _reply_text(resp: Any) -> str:
    """Read the assistant text from an ollama ChatResponse object or dict."""
    try:
        return resp["message"]["content"]
    except Exception:
        return resp.message.content


def _summarize(payload: dict, label: str) -> None:
    from collections import Counter

    events = payload.get("events", [])
    counts = dict(Counter(e.get("event_type") for e in events))
    has_agent = any(e.get("event_type") == "agent.identity" for e in events)
    models = sorted(
        {(e.get("payload") or {}).get("model") for e in events
         if e.get("event_type") == "model.invoke"} - {None}
    )
    print(
        "  ollama %s  events=%d models=%s agent_identity=%s counts=%s"
        % (label, len(events), models, has_agent, counts)
    )


# --------------------------------------------------------------------------
# Single: one on-prem clinical-decision-support chat turn (empty-state render)
# --------------------------------------------------------------------------
_CLINICAL_SYSTEM = (
    "You are an on-premise clinical decision-support assistant running entirely "
    "inside the hospital network — no patient data leaves the building. You "
    "support licensed clinicians and are not a substitute for clinical judgement. "
    "Answer concisely (under 150 words)."
)
# De-identified, synthetic case — non-sensitive.
_CLINICAL_CASE = (
    "A 68-year-old outpatient is diagnosed with community-acquired pneumonia "
    "(CURB-65 score 1), has no drug allergies, normal renal and hepatic function, "
    "and has not been hospitalized or on antibiotics in the last 90 days. Suggest "
    "a reasonable first-line oral antibiotic regimen and the two most important "
    "things to monitor or counsel the patient on."
)


def generate_ollama_single(client: Stratix) -> dict:
    """Record one real on-prem clinical chat turn (provider empty-state)."""
    def driver(ollama: Any) -> None:
        resp = ollama.chat(
            model=_OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _CLINICAL_SYSTEM},
                {"role": "user", "content": _CLINICAL_CASE},
            ],
        )
        _reply_text(resp)  # drain/parse the real reply

    payload = _capture_provider(client, driver)
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "healthcare",
        "clinical-decision-support",
        "on-prem",
        "ollama",
    ]
    _summarize(payload, "single (on-prem clinical chat)")
    print("  ->", _write([payload], "industry", SINGLE_STEM), "\n")
    return payload


# --------------------------------------------------------------------------
# Multi: on-prem medication-safety TOOL-USE LOOP (empty-state, real waterfall)
# --------------------------------------------------------------------------
# A synthetic patient's active medication list + a newly proposed prescription.
_PATIENT_ACTIVE_MEDS = ["warfarin", "lisinopril", "atorvastatin"]
_PROPOSED_NEW_DRUG = "amiodarone"  # newly started for new-onset atrial fibrillation

# A REAL, well-established drug-interaction reference (the "hospital formulary"
# the tool queries). These are genuine, widely-documented interactions — the tool
# returns reference data, never model-fabricated content.
_INTERACTION_KB: dict[frozenset[str], dict[str, str]] = {
    frozenset({"warfarin", "amiodarone"}): {
        "severity": "major",
        "mechanism": (
            "Amiodarone inhibits CYP2C9 and P-glycoprotein, reducing warfarin "
            "clearance and markedly raising INR and bleeding risk."
        ),
        "management": (
            "Empirically reduce the warfarin dose ~30-50% when starting "
            "amiodarone and monitor INR closely for several weeks."
        ),
    },
    frozenset({"atorvastatin", "amiodarone"}): {
        "severity": "moderate",
        "mechanism": (
            "Amiodarone inhibits CYP3A4, increasing atorvastatin exposure and the "
            "risk of myopathy/rhabdomyolysis."
        ),
        "management": (
            "Use the lowest effective statin dose and counsel on muscle "
            "pain/weakness; consider a statin less dependent on CYP3A4."
        ),
    },
    frozenset({"lisinopril", "amiodarone"}): {
        "severity": "none",
        "mechanism": "No clinically significant pharmacokinetic interaction documented.",
        "management": "No specific action required beyond routine monitoring.",
    },
}


def _check_interactions(new_drug: str, current_meds: list[str]) -> dict:
    """REAL tool fn: check a new drug against each current med in the reference.

    Deterministic lookup over ``_INTERACTION_KB`` — the authoritative reference,
    independent of the model. Returns the significant findings (severity !=
    none) plus the full pairs checked, so the second model turn is grounded on
    verified data rather than the model's own recall.
    """
    findings = []
    pairs_checked = []
    for med in current_meds:
        pair = frozenset({new_drug.lower(), med.lower()})
        pairs_checked.append("%s + %s" % (new_drug, med))
        entry = _INTERACTION_KB.get(pair)
        if entry and entry["severity"] != "none":
            findings.append({"pair": "%s + %s" % (new_drug, med), **entry})
    return {
        "new_drug": new_drug,
        "pairs_checked": pairs_checked,
        "significant_findings": findings,
        "source": "hospital formulary interaction reference (on-prem)",
    }


def _emit_tool_result(tool_name: str, arguments: dict, result: dict) -> None:
    """Emit a tool.result event for the REAL local tool run, onto a child span."""
    col = _current_collector.get()
    if col is None:
        return
    col.emit(
        "tool.result",
        {
            "provider": "ollama",
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "status": "ok",
        },
        span_id=uuid.uuid4().hex[:16],
        parent_span_id=_current_span_id.get(),
        span_name="tool:%s" % tool_name,
    )


_MEDSAFETY_TURN1_SYSTEM = (
    "You are an on-premise medication-safety assistant inside the hospital "
    "network. A clinician wants to add a new medication to a patient's regimen. "
    "List ONLY the drug pairs that must be checked for interactions against the "
    "formulary reference, one per line as 'newDrug + currentDrug'. Do not assess "
    "the interactions yet — just name the pairs to verify."
)
_MEDSAFETY_TURN2_SYSTEM = (
    "You are an on-premise medication-safety assistant inside the hospital "
    "network. Using ONLY the verified interaction findings from the formulary "
    "reference below (do not rely on memory), give a medication-safety "
    "assessment: an overall risk verdict (SAFE / CAUTION / CONTRAINDICATED), the "
    "key interactions and their management, and what to monitor. Answer concisely "
    "(under 150 words)."
)


def generate_ollama_multi(client: Stratix) -> dict:
    """Record a real 2-turn on-prem medication-safety tool-use loop (empty-state)."""
    def driver(ollama: Any) -> None:
        med_list = ", ".join(_PATIENT_ACTIVE_MEDS)
        # Turn 1 — the assistant names the pairs to verify (real model.invoke).
        turn1_user = (
            "Patient active medications: %s.\nProposed new medication: %s "
            "(for new-onset atrial fibrillation).\nWhich drug pairs should be "
            "checked?" % (med_list, _PROPOSED_NEW_DRUG)
        )
        resp1 = ollama.chat(
            model=_OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _MEDSAFETY_TURN1_SYSTEM},
                {"role": "user", "content": turn1_user},
            ],
        )
        _reply_text(resp1)  # the model's requested pairs (the trigger)

        # REAL tool run: authoritative interaction check (independent of model).
        args = {"new_drug": _PROPOSED_NEW_DRUG, "current_meds": _PATIENT_ACTIVE_MEDS}
        result = _check_interactions(**args)
        _emit_tool_result("check_interactions", args, result)

        # Turn 2 — grounded final assessment on the verified findings.
        turn2_user = (
            "New medication: %s. Patient active medications: %s.\n\n"
            "Verified interaction findings from the formulary reference:\n%s\n\n"
            "Give the medication-safety assessment."
            % (_PROPOSED_NEW_DRUG, med_list, json.dumps(result, indent=2))
        )
        resp2 = ollama.chat(
            model=_OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _MEDSAFETY_TURN2_SYSTEM},
                {"role": "user", "content": turn2_user},
            ],
        )
        _reply_text(resp2)

    payload = _capture_provider(client, driver)
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "healthcare",
        "medication-safety",
        "on-prem",
        "tool-use",
        "ollama",
    ]
    events = payload.get("events", [])
    n_invoke = sum(1 for e in events if e.get("event_type") == "model.invoke")
    n_tool = sum(1 for e in events if e.get("event_type") == "tool.result")
    print(
        "  ollama multi (on-prem med-safety tool-use loop)  "
        "model.invoke=%d tool.result=%d" % (n_invoke, n_tool)
    )
    _summarize(payload, "multi")
    print("  ->", _write([payload], "industry", MULTI_STEM), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_ollama_single(_client)
    generate_ollama_multi(_client)
