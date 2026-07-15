"""ADP-W2 Family-B recorders for the **pydantic_ai** framework adapter.

Records two REAL, fully-instrumented ``pydantic-ai`` runs and writes each sealed
trace to ``samples/data/traces/industry/<stem>.jsonl``. Both fixtures are genuine
runs of the real ``PydanticAIAdapter`` over a real ``pydantic_ai.Agent`` backed by
a real OpenAI model (gpt-4o-mini) — nothing is fabricated. The framework deps
(``pydantic_ai``) are imported FUNCTION-LOCALLY so this module imports in any venv.

Two lanes (Financial-services domain; de-conflicted from the W1 insurance stems):

* ``generate_pydantic_ai_single`` -> ``financialservices_pydantic_extract``
  A single NAMED typed-extraction Agent (``loan-intake-extractor``) with a
  Pydantic ``output_type`` (``LoanApplication``) that extracts structured
  loan-application fields from a free-text applicant message. Showcases
  pydantic-ai's typed-output strength. Renders one honest agent node
  (Agent column = ``loan-intake-extractor``) with real ``model.invoke`` +
  priced ``cost.record`` + typed ``agent.output``.

* ``generate_pydantic_ai_multi`` -> ``financialservices_pydantic_underwriting``
  A single NAMED agent (``credit-underwriting-assistant``) driving a GENUINE
  **tool-use LOOP**: it calls three real ``@agent.tool_plain`` functions
  (``fetch_credit_score`` / ``get_debt_obligations`` / ``lookup_underwriting_policy``)
  in sequence before issuing an APPROVE / REFER / DECLINE recommendation.

  HONESTY / MARKED: pydantic-ai has **no handoff hook** — cross-agent delegation
  in pydantic-ai runs each sub-agent as a *separate* trace, so a single trace can
  never carry ``agent.handoff`` or >=2 agent nodes. Per the ADP-W2 map + the
  honesty rules, the "multi" lane for this adapter is therefore a real
  single-agent tool-use loop (multiple ``tool.call`` / ``tool.result`` events),
  NOT a multi-agent handoff DAG. This is marked in the payload metadata and tags.

Both lanes record the NON-streaming path (``run_sync``): pydantic-ai's streamed
output drops the output content (``StreamedRunResult`` exposes output only via
``await get_output()``, which the adapter cannot read post-yield), so a streamed
run would ship an ``agent.output`` with no output — the non-streaming path is the
honest, complete one.
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
for _p in (os.path.join(_REPO, "src"), _DATA, _SAMPLES):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# This file is named ``pydantic_ai.py``; if a bare-script launch put its own
# directory on sys.path it would SHADOW the real ``pydantic_ai`` package. Drop
# our own dir so the function-local ``import pydantic_ai`` always resolves to the
# installed framework (a no-op in the integrated ``generators.pydantic_ai`` path).
sys.path[:] = [_q for _q in sys.path if os.path.abspath(_q) != _HERE]

from layerlens import Stratix  # noqa: E402  (re-exported for the seam signature)
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


def _capture_pydantic_ai(client: Stratix, agent, run_prompt: str, *, max_tokens: int = 600) -> dict:
    """Run a real pydantic-ai Agent under the PydanticAIAdapter + observer seam
    (no background upload) and return the sealed trace payload."""
    from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = PydanticAIAdapter(client, capture_config=_CAPTURE)
    try:
        adapter.connect(target=agent)
        # run_sync == the NON-streaming path (streamed output drops content).
        agent.run_sync(run_prompt, model_settings={"max_tokens": max_tokens, "temperature": 0.1})
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for pydantic_ai run")
    return payload


# ---------------------------------------------------------------------------
# SINGLE: typed-extraction Agent (financial-services loan intake)
# ---------------------------------------------------------------------------
_LOAN_APPLICATION_MESSAGE = (
    "Hi, my name is Marcus Whitfield. I'd like to apply for a personal loan of "
    "$28,500 to consolidate three high-interest credit-card balances into a single "
    "monthly payment. I've worked full-time as a registered nurse at Mercy General "
    "for the past six years and my gross salary is about $94,000 a year. I'd like to "
    "repay the loan over four years — so a 48-month term. Thanks!"
)


def generate_pydantic_ai_single(client: Stratix) -> None:
    """PydanticAI SINGLE (typed extraction): a NAMED ``loan-intake-extractor``
    Agent with a Pydantic ``output_type`` extracts structured loan-application
    fields from a free-text applicant message. Real OpenAI (gpt-4o-mini),
    recorded under the real PydanticAIAdapter -> one honest agent node with a
    real ``model.invoke`` + priced ``cost.record`` + typed ``agent.output``."""
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent

    class LoanApplication(BaseModel):
        """Structured loan-application record extracted from a free-text message."""

        applicant_name: str = Field(description="Full name of the applicant.")
        loan_amount_usd: float = Field(description="Requested loan principal in USD.")
        loan_purpose: str = Field(description="Stated purpose of the loan.")
        annual_income_usd: float = Field(description="Applicant's stated gross annual income in USD.")
        employment_status: str = Field(description="Employment situation, e.g. 'full-time registered nurse'.")
        requested_term_months: int = Field(description="Requested repayment term in months.")

    agent = Agent(
        "openai:%s" % OPENAI_MODEL,
        name="loan-intake-extractor",
        output_type=LoanApplication,
        system_prompt=(
            "You are loan-intake-extractor, a consumer-lending intake assistant. "
            "Extract the structured loan-application fields from the applicant's "
            "free-text message. Use ONLY information the applicant actually states; "
            "do not invent or assume values."
        ),
    )

    payload = _capture_pydantic_ai(client, agent, _LOAN_APPLICATION_MESSAGE, max_tokens=400)
    payload["tags"] = [
        "layerlens-sample", "industry", "financial-services", "loan-intake", "typed-extraction",
    ]

    events = payload.get("events", [])
    agents = sorted({(e.get("payload") or {}).get("agent_name")
                     for e in events if (e.get("payload") or {}).get("agent_name")})
    frameworks = sorted({(e.get("payload") or {}).get("framework")
                         for e in events if (e.get("payload") or {}).get("framework")})
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    out = next((e["payload"].get("output") for e in events
                if e.get("event_type") == "agent.output" and (e.get("payload") or {}).get("output")), None)
    print("  pydantic_ai-single (typed extraction)  agents=%s frameworks=%s model.invoke=%d cost.record=%d"
          % (agents, frameworks, len(mi), len(cr)))
    print("    typed output=%r" % (str(out)[:120],))
    print("  ->", _write([payload], "industry", "financialservices_pydantic_extract"), "\n")


# ---------------------------------------------------------------------------
# MULTI (tool-use LOOP — pydantic-ai has NO handoff): credit underwriting
# ---------------------------------------------------------------------------
# Real deterministic domain back-ends the tools read from (genuine tool output;
# the recommendation is the model's real reasoning over these facts).
_CREDIT_BUREAU = {
    "APP-70412": {"fico": 706, "delinquencies_24mo": 0, "credit_age_years": 11, "hard_inquiries_6mo": 1},
}
_DEBT_LEDGER = {
    "APP-70412": {"monthly_debt_usd": 1850, "revolving_utilization_pct": 34, "open_tradelines": 7},
}
_UW_POLICY = {
    "conventional_mortgage": {"min_fico": 680, "max_dti_pct": 43, "max_ltv_pct": 80},
    "auto_loan": {"min_fico": 620, "max_dti_pct": 50, "max_ltv_pct": 120},
    "personal_loan": {"min_fico": 660, "max_dti_pct": 40, "max_ltv_pct": None},
}

_UNDERWRITING_TASK = (
    "Underwrite loan application APP-70412: a conventional_mortgage of $420,000 on a "
    "single-family property appraised at $525,000, applicant gross annual income "
    "$138,000. You do NOT have the applicant's credit score, debt obligations, or the "
    "underwriting policy in this prompt — you MUST call fetch_credit_score, "
    "get_debt_obligations, and lookup_underwriting_policy to gather them first. Then "
    "compute the debt-to-income and loan-to-value ratios and give a final "
    "APPROVE / REFER / DECLINE recommendation with a one-line rationale."
)


def generate_pydantic_ai_multi(client: Stratix) -> None:
    """PydanticAI MULTI == genuine single-agent TOOL-USE LOOP (MARKED).

    pydantic-ai has no handoff hook, so a single trace cannot carry
    ``agent.handoff`` or >=2 agent nodes (cross-agent delegation runs each
    sub-agent as a separate trace). Per the ADP-W2 map + honesty rules, this
    lane is a REAL single-agent tool-use loop: a NAMED
    ``credit-underwriting-assistant`` calls three real ``@agent.tool_plain``
    functions in sequence, then issues an APPROVE/REFER/DECLINE recommendation.
    The trace carries multiple real ``model.invoke`` + ``tool.call`` /
    ``tool.result`` events under ONE honest agent node."""
    from pydantic_ai import Agent

    agent = Agent(
        "openai:%s" % OPENAI_MODEL,
        name="credit-underwriting-assistant",
        system_prompt=(
            "You are credit-underwriting-assistant, a consumer-lending underwriter. "
            "For a loan application you gather the applicant's credit and debt data and "
            "the applicable underwriting policy via your tools, then apply the policy "
            "thresholds (min FICO, max DTI, max LTV) to recommend APPROVE, REFER (to a "
            "human underwriter), or DECLINE. Always call the tools before deciding; never "
            "guess the credit score, debts, or policy limits. Be concise."
        ),
    )

    @agent.tool_plain
    def fetch_credit_score(applicant_id: str) -> dict:
        """Fetch the credit-bureau summary (FICO, delinquencies, credit age) for an applicant id."""
        rec = _CREDIT_BUREAU.get((applicant_id or "").strip().upper())
        if rec is None:
            return {"applicant_id": applicant_id, "found": False}
        return {"applicant_id": applicant_id, "found": True, **rec}

    @agent.tool_plain
    def get_debt_obligations(applicant_id: str) -> dict:
        """Fetch the applicant's current monthly debt obligations and revolving utilization."""
        rec = _DEBT_LEDGER.get((applicant_id or "").strip().upper())
        if rec is None:
            return {"applicant_id": applicant_id, "found": False}
        return {"applicant_id": applicant_id, "found": True, **rec}

    @agent.tool_plain
    def lookup_underwriting_policy(loan_type: str) -> dict:
        """Look up the underwriting policy thresholds (min FICO, max DTI %, max LTV %) for a loan type."""
        key = (loan_type or "").strip().lower().replace(" ", "_")
        pol = _UW_POLICY.get(key)
        if pol is None:
            return {"loan_type": loan_type, "found": False}
        return {"loan_type": key, "found": True, **pol}

    payload = _capture_pydantic_ai(client, agent, _UNDERWRITING_TASK, max_tokens=700)
    payload["tags"] = [
        "layerlens-sample", "industry", "financial-services",
        "credit-underwriting", "tool-use", "tool-use-loop",
    ]
    # HONEST provenance: this is a single-agent tool-use loop, NOT a handoff DAG
    # (pydantic-ai cannot emit agent.handoff / >=2 agent nodes in one trace).
    payload["metadata"] = {
        "topology": "single-agent-tool-use-loop",
        "reason": "pydantic-ai has no handoff hook; cross-agent delegation runs each "
                  "sub-agent as a separate trace, so a single trace carries one agent "
                  "node + a real multi-tool loop (no agent.handoff).",
    }

    events = payload.get("events", [])
    agents = sorted({(e.get("payload") or {}).get("agent_name")
                     for e in events if (e.get("payload") or {}).get("agent_name")})
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    tc = [e for e in events if e.get("event_type") == "tool.call"]
    tr = [e for e in events if e.get("event_type") == "tool.result"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    tool_names = sorted({(e.get("payload") or {}).get("tool_name")
                         for e in events if e.get("event_type") in ("tool.call", "tool.result")
                         and (e.get("payload") or {}).get("tool_name")})
    print("  pydantic_ai-multi (tool-use LOOP; NO handoff)  agents=%s model.invoke=%d "
          "tool.call=%d tool.result=%d cost.record=%d" % (agents, len(mi), len(tc), len(tr), len(cr)))
    print("    tools=%s" % (tool_names,))
    print("  ->", _write([payload], "industry", "financialservices_pydantic_underwriting"), "\n")


if __name__ == "__main__":
    _client = Stratix()
    generate_pydantic_ai_single(_client)
    generate_pydantic_ai_multi(_client)
