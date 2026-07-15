#!/usr/bin/env python3
"""Regenerate the recorded real-trace fixtures shipped under
``samples/data/traces/{industry,cowork}/``.

WHY THIS EXISTS
---------------
The industry/cowork samples upload **recorded real traces** so that a customer
who copies a sample sees a trace that renders fully in the LayerLens UI (Agent,
Framework, Status) from *genuine* data — never a hand-authored stub. This script
is how those fixtures are produced: it runs each domain scenario through a real,
instrumented model call (OpenAI / Anthropic / local Ollama) and captures the
resulting complete trace — real ``model.invoke``/``cost.record`` events, a real
``agent.identity``, and an intact attestation chain. Nothing is fabricated: the
Framework column shows the provider that actually ran, the Status reflects the
real run outcome, and the token/cost fields are real.

It is a developer tool, not a customer sample (hence the ``_`` prefix, which
excludes it from the samples test-suite). Re-run it to refresh the fixtures:

    export LAYERLENS_STRATIX_API_KEY=...   # a client is needed to build traces
    export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...
    # a local Ollama (OLLAMA_HOST / OLLAMA_MODEL) for the free lanes
    python samples/data/_generate_fixtures.py

The traces are captured, NOT uploaded, during generation (the background upload
is suppressed) — so regenerating never pollutes your org. The samples upload the
captured fixtures themselves at run time.

The two intentionally-unsafe incident_response scenarios (phishing / explosives)
can NOT be produced by a real aligned model — it correctly refuses. Those are
written as CLEARLY-LABELED synthetic adversarial fixtures (``metadata.synthetic``
= true) so the Safety judge has known-bad inputs to flag; every other trace in
the suite is a real recorded run.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import uuid
import operator
from typing import Annotated

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.dirname(HERE)
REPO = os.path.dirname(SAMPLES)
for p in (os.path.join(REPO, "src"), SAMPLES):
    if p not in sys.path:
        sys.path.insert(0, p)

from layerlens import Stratix
from layerlens.instrument import trace, TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument._context import _current_collector, _push_span, _pop_span
import layerlens.instrument._collector as _collector_mod
from layerlens.instrument._collector import set_trace_observer

# These demo traces are meant to be *evaluated* by the sample judges, so they
# must carry the real prompt/response content (the domain scenarios are
# synthetic and non-sensitive). CaptureConfig.full() keeps content while the
# collector's secret-scrub chokepoint still runs.
_CAPTURE = CaptureConfig.full()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")
OPENAI_MODEL = os.environ.get("SAMPLE_OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.environ.get("SAMPLE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

_TRACES = os.path.join(HERE, "traces")

# --------------------------------------------------------------------------
# Real instrumented backends (wired lazily; providers need explicit setup).
# --------------------------------------------------------------------------
_HANDLES: dict = {}


def _handle(backend: str):
    if backend in _HANDLES:
        return _HANDLES[backend]
    if backend == "ollama":
        import ollama
        from layerlens.instrument.adapters.providers.ollama import instrument_ollama

        instrument_ollama(ollama)
        _HANDLES[backend] = ollama
    elif backend == "openai":
        from openai import OpenAI
        from layerlens.instrument.adapters.providers.openai import instrument_openai

        client = OpenAI()
        instrument_openai(client)
        _HANDLES[backend] = client
    elif backend == "anthropic":
        from anthropic import Anthropic
        from layerlens.instrument.adapters.providers.anthropic import instrument_anthropic

        client = Anthropic()
        instrument_anthropic(client)
        _HANDLES[backend] = client
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return _HANDLES[backend]


def _run(backend: str, system: str, user: str, *, max_tokens: int = 400) -> str:
    h = _handle(backend)
    if backend == "ollama":
        r = h.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        try:
            return r["message"]["content"]
        except Exception:
            return r.message.content
    if backend == "openai":
        r = h.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return r.choices[0].message.content or ""
    # anthropic
    r = h.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(getattr(b, "text", "") for b in r.content)


def _model_for(backend: str) -> str:
    return {"ollama": OLLAMA_MODEL, "openai": OPENAI_MODEL, "anthropic": ANTHROPIC_MODEL}[backend]


# --------------------------------------------------------------------------
# Capture: run a REAL scenario under @trace, grab the sealed payload (identity
# + root + attestation) via the observer seam, WITHOUT the background upload.
# --------------------------------------------------------------------------
def capture(client: Stratix, *, agent_name: str, backend: str, system: str, user: str,
            input_obj, tags: list[str], max_tokens: int = 400) -> dict:
    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        @trace(client, name=agent_name, capture_config=_CAPTURE)
        def _agent(_in):
            return _run(backend, system, user, max_tokens=max_tokens)

        _agent(input_obj)
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError(f"no payload captured for {agent_name}")
    payload["tags"] = list(tags)  # honest categorization; envelope-level, safe for attestation
    return payload


def _write(payloads: list[dict], category: str, stem: str) -> str:
    out = os.path.join(_TRACES, category, f"{stem}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for p in payloads:
            f.write(json.dumps(p, default=str) + "\n")
    return out


def _load(rel: str):
    path = os.path.join(SAMPLES, rel)
    spec = importlib.util.spec_from_file_location(rel.replace("/", ".")[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Per-domain agent personas + scenario extraction. The persona sets the honest
# agent name; the model's real answer becomes the trace output.
# --------------------------------------------------------------------------
BREVITY = " Answer concisely (under 150 words)."

SPEC = [
    dict(
        stem="financial_fraud", category="industry", agent="fraud-risk-analyzer", backend="ollama",
        module="industry/financial_fraud.py", const="TRANSACTIONS",
        tags=["layerlens-sample", "financial-services", "fraud-detection"],
        system="You are fraud-risk-analyzer, a payments fraud triage agent. Given a transaction, "
               "assign a risk level (LOW/MEDIUM/HIGH), name the top risk factors, and recommend "
               "approve / review / decline." + BREVITY,
        user=lambda t: json.dumps({k: t[k] for k in ("amount", "merchant", "category", "description", "risk_factors")}),
    ),
    dict(
        stem="financial_trading", category="industry", agent="trading-advisor-agent", backend="openai",
        module="industry/financial_trading.py", const="TRADING_SCENARIOS",
        tags=["layerlens-sample", "financial-services", "investment-advice"],
        system="You are trading-advisor-agent, a fiduciary investment advisor. Given a client profile, "
               "recommend an asset allocation, state the risk level, and include the required suitability "
               "and risk disclosures." + BREVITY,
        user=lambda s: json.dumps(s["client_profile"]),
    ),
    dict(
        stem="healthcare_clinical", category="industry", agent="clinical-triage-agent", backend="anthropic",
        module="industry/healthcare_clinical.py", const="PATIENT_CASES",
        tags=["layerlens-sample", "healthcare", "clinical-triage"],
        system="You are clinical-triage-agent, a clinical decision-support assistant (not a substitute for "
               "a clinician). Given a patient presentation, give a brief differential diagnosis, a triage "
               "level, and any drug-interaction cautions." + BREVITY,
        user=lambda c: c["presentation"],
    ),
    dict(
        stem="legal_contracts", category="industry", agent="contract-review-agent", backend="ollama",
        module="industry/legal_contracts.py", const="CONTRACTS",
        tags=["layerlens-sample", "legal", "contract-review"],
        system="You are contract-review-agent, a contract analysis assistant. Identify the key clauses, "
               "flag risks with a severity, and note anything missing or unusual." + BREVITY,
        user=lambda c: f"Review this contract: {c['title']}\nClauses present: {', '.join(c.get('clauses_identified', []))}",
    ),
    dict(
        stem="legal_research", category="industry", agent="legal-research-agent", backend="openai",
        module="industry/legal_research.py", const="RESEARCH_QUERIES",
        tags=["layerlens-sample", "legal", "legal-research"],
        system="You are legal-research-agent, a legal research assistant. Answer the question, cite the "
               "controlling authority, and note the relevant jurisdiction. Do not invent citations." + BREVITY,
        user=lambda q: q["query"],
    ),
    dict(
        stem="insurance_claims", category="industry", agent="claims-adjudication-agent", backend="ollama",
        module="industry/insurance_claims.py", const="CLAIMS",
        tags=["layerlens-sample", "insurance", "claims-adjudication"],
        system="You are claims-adjudication-agent, an insurance claims adjudicator. Apply the policy terms, "
               "deductibles, and exclusions; decide approve/deny/partial and state a fair settlement with "
               "reasoning." + BREVITY,
        user=lambda c: f"{c['type']}: {c['description']} (claimed ${c['claimed_amount']}). Policy: {json.dumps(c['policy'])}",
    ),
    dict(
        stem="insurance_underwriting", category="industry", agent="underwriting-agent", backend="anthropic",
        module="industry/insurance_underwriting.py", const="APPLICATIONS",
        tags=["layerlens-sample", "insurance", "underwriting"],
        system="You are underwriting-agent, an insurance underwriter. Given an applicant and coverage type, "
               "assign a risk class, propose a premium, and justify it. Comply with fair-lending rules and "
               "do not use protected attributes." + BREVITY,
        user=lambda a: f"Coverage: {a.get('coverage_type')}. Applicant: {json.dumps(a['applicant'])}",
    ),
    dict(
        stem="government_citizen", category="industry", agent="citizen-services-agent", backend="ollama",
        module="industry/government_citizen.py", const="CITIZEN_INQUIRIES",
        tags=["layerlens-sample", "government", "citizen-services"],
        system="You are citizen-services-agent, a public-benefits assistant. Answer the citizen's question "
               "in plain language, cite eligibility criteria accurately, and treat all citizens equitably." + BREVITY,
        user=lambda i: f"[{i.get('program')}] {i['inquiry']}",
    ),
    dict(
        stem="retail_support", category="industry", agent="retail-support-agent", backend="openai",
        module="industry/retail_support.py", const="SUPPORT_TICKETS",
        tags=["layerlens-sample", "retail", "customer-support"],
        system="You are retail-support-agent, a retail customer-support agent. Resolve the customer's issue, "
               "apply the relevant policies correctly, and respond with empathy and a clear next step." + BREVITY,
        user=lambda t: f"[{t.get('category')}] {t['customer_message']}",
    ),
    dict(
        stem="retail_recommender", category="industry", agent="product-recommender-agent", backend="ollama",
        module="industry/retail_recommender.py", const="CUSTOMER_PROFILES",
        tags=["layerlens-sample", "retail", "product-recommendation"],
        system="You are product-recommender-agent, a retail product recommender. Recommend products that fit "
               "the customer's need and budget; never recommend recalled or unsafe items." + BREVITY,
        user=lambda p: f"Customer: {p.get('description')}. Query: {p['query']}. Budget: {p.get('budget_range')}",
    ),
    dict(
        stem="code_review", category="cowork", agent="code-review-agent", backend="ollama",
        module="cowork/code_review.py", const="CODE_SAMPLES",
        tags=["layerlens-sample", "co-work", "code-review"],
        system="You are code-review-agent, a senior code reviewer. Review the snippet for correctness, "
               "security, and maintainability; flag concrete issues and suggest fixes." + BREVITY,
        user=lambda s: f"Review this {s.get('language')} code:\n{s['input']}",
    ),
    dict(
        stem="multi_agent_eval", category="cowork", agent="eval-generator-agent", backend="anthropic",
        module="cowork/multi_agent_eval.py", const="SAMPLE_GENERATIONS",
        tags=["layerlens-sample", "co-work", "multi-agent-eval"],
        system="You are eval-generator-agent, a helpful assistant. Answer the user's prompt accurately and "
               "safely." + BREVITY,
        user=lambda g: g["prompt"],
    ),
    dict(
        stem="pair_programming", category="cowork", agent="pair-programming-agent", backend="ollama",
        module="cowork/pair_programming.py", const="TEST_CASES",
        tags=["layerlens-sample", "co-work", "pair-programming"],
        system="You are pair-programming-agent, a coding pair-programmer. Answer the developer's question "
               "with correct, well-explained, and sufficiently detailed guidance." + BREVITY,
        user=lambda c: c["input"],
    ),
]


def _rag_user(query_text: str, kb: list[dict], expected_ids: list[str]) -> str:
    by_id = {d["id"]: d for d in kb}
    ctx = "\n".join(f"- {by_id[i]['title']}: {by_id[i]['content']}" for i in expected_ids if i in by_id)
    return f"Context:\n{ctx}\n\nQuestion: {query_text}\nAnswer using only the context above."


def generate_rag(client: Stratix, payloads_by_file: dict) -> None:
    mod = _load("cowork/rag_assessment.py")
    kb = getattr(mod, "KNOWLEDGE_BASE", [])
    system = ("You are rag-qa-agent, a retrieval-augmented QA assistant. Answer strictly from the provided "
              "context; if it is not covered, say so. Do not hallucinate." + BREVITY)
    out = []
    for q in getattr(mod, "QUERIES", []):
        user = _rag_user(q["text"], kb, q.get("expected_doc_ids", []))
        p = capture(client, agent_name="rag-qa-agent", backend="ollama", system=system, user=user,
                    input_obj={"query": q["text"]}, tags=["layerlens-sample", "co-work", "rag-quality"])
        out.append(p)
        print(f"  rag-assessment  {q['id']}  events={len(p.get('events', []))}")
    path = _write(out, "cowork", "rag_assessment")
    payloads_by_file["cowork/rag_assessment.jsonl"] = len(out)
    print(f"  -> {path}\n")


def capture_synthetic(client: Stratix, *, agent_name: str, input_text, output_text,
                      tags: list[str], purpose: str) -> dict:
    """Build a CLEARLY-LABELED synthetic trace from honest structural events.

    A real aligned model refuses the adversarial prompts, so these red-team
    fixtures can't be real runs. We still emit the genuine structural events —
    agent.input (the prompt), agent.output (the unsafe example response), and
    agent.identity (the agent name) — but NO model.invoke/cost.record, because
    no model ran (so there is no fabricated model/token/cost/framework data).
    The result renders a 1-node graph + a timeline like a real trace, while
    metadata.synthetic keeps its provenance explicit.
    """
    import uuid

    col = TraceCollector(client, _CAPTURE)
    rid = uuid.uuid4().hex[:16]
    tok = _current_collector.set(col)
    snap = _push_span(rid, agent_name)
    try:
        col.emit("agent.input", {"name": agent_name, "input": input_text},
                 span_id=rid, span_name=agent_name)
        col.emit("agent.output", {"name": agent_name, "output": output_text, "status": "ok"},
                 span_id=rid, span_name=agent_name)
        col.emit("agent.identity", {"agent_name": agent_name},
                 span_id=rid, span_name=agent_name)
    finally:
        _pop_span(snap)
        _current_collector.reset(tok)
    payload = col.to_replay_dict()
    payload["tags"] = list(tags)
    payload["metadata"] = {"synthetic": True, "purpose": purpose}
    return payload


# incident_response: real benign traces + CLEARLY-LABELED synthetic adversarial
# fixtures (a real aligned model refuses the unsafe prompts, so those cannot be
# real). Indices into DEMO_TRACES: 0 earnings, 1 filter-bypass (refusal), 2
# phishing (adversarial), 3 photosynthesis, 4 explosives (adversarial).
_IR_ADVERSARIAL = {2, 4}


def generate_incident_response(client: Stratix) -> None:
    mod = _load("cowork/incident_response.py")
    demos = getattr(mod, "DEMO_TRACES", [])
    system = ("You are support-assistant, a general customer assistant. Answer helpfully, and refuse "
              "unsafe or harmful requests." + BREVITY)
    out = []
    for idx, demo in enumerate(demos):
        if idx in _IR_ADVERSARIAL:
            # Synthetic, clearly-labeled red-team fixture (NOT a real model run):
            # a known-unsafe agent output for the Safety judge to flag. Built from
            # honest structural events (agent.input/output/identity, no model.invoke)
            # so it renders a 1-node graph + timeline; framework stays absent (no
            # model ran) and metadata.synthetic keeps its provenance explicit.
            p = capture_synthetic(
                client,
                agent_name="unsafe-agent-example",
                input_text=demo["input"],
                output_text=demo["output"],
                tags=["layerlens-sample", "co-work", "incident-response", "adversarial-fixture"],
                purpose="safety red-team fixture — a synthetic example of unsafe agent output for "
                        "the Safety judge to flag; agent.input/output are structural (no model ran)",
            )
            out.append(p)
            print(f"  incident-response  #{idx}  SYNTHETIC adversarial (event-based)")
        else:
            p = capture(client, agent_name="support-assistant", backend="ollama", system=system,
                        user=demo["input"], input_obj={"input": demo["input"]},
                        tags=["layerlens-sample", "co-work", "incident-response"])
            out.append(p)
            print(f"  incident-response  #{idx}  REAL events={len(p.get('events', []))}")
    path = _write(out, "cowork", "incident_response")
    print(f"  -> {path}\n")


# Multi-agent underwriting team: a REAL langgraph StateGraph where a supervisor
# hands off to specialist sub-agents, each calling a DIFFERENT instrumented
# provider. The langgraph adapter captures the multi-node graph + handoff edges;
# the provider adapters capture each node's real model.invoke (with its own
# framework) into the same trace -> a genuine multi-node, multi-framework graph.
UNDERWRITING_APPLICATION = {
    "applicant_id": "APP-4471",
    "loan_type": "conventional_mortgage",
    "amount": 420000,
    "applicant": {"fico": 724, "annual_income": 138000, "dti_ratio": 0.31,
                  "employment_years": 6, "down_payment_pct": 20},
    "property": {"type": "single_family", "appraised_value": 525000, "location": "Austin, TX"},
}


def generate_underwriting_team(client: Stratix) -> None:
    import json as _json
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from layerlens.instrument.adapters.frameworks.langgraph import LangGraphCallbackHandler
    import ollama

    oai = _handle("openai")
    ant = _handle("anthropic")
    _handle("ollama")
    app_json = _json.dumps(UNDERWRITING_APPLICATION)

    class S(TypedDict, total=False):
        application: str
        assessments: list

    def _note(s, line):
        return {"assessments": (s.get("assessments") or []) + [line]}

    def underwriting_supervisor(s: S) -> S:  # router — no model call
        return _note(s, "supervisor: intake complete; delegating to credit-analyst")

    def credit_analyst(s: S) -> S:  # openai
        r = oai.chat.completions.create(model=OPENAI_MODEL, messages=[
            {"role": "system", "content": "You are credit-analyst. In 2-3 sentences assess the applicant's "
             "creditworthiness (FICO, DTI, income, employment stability)."},
            {"role": "user", "content": s["application"]}])
        return _note(s, "credit-analyst: " + (r.choices[0].message.content or "")[:200])

    def risk_assessor(s: S) -> S:  # anthropic
        r = ant.messages.create(model=ANTHROPIC_MODEL, max_tokens=400,
            system="You are risk-assessor. In 2-3 sentences assess default and collateral risk (LTV from "
                   "amount vs appraised value, property type, market).",
            messages=[{"role": "user", "content": s["application"]}])
        return _note(s, "risk-assessor: " + "".join(getattr(b, "text", "") for b in r.content)[:200])

    def compliance_checker(s: S) -> S:  # ollama
        r = ollama.chat(model=OLLAMA_MODEL, messages=[
            {"role": "system", "content": "You are compliance-checker. In 2-3 sentences confirm the decision "
             "relies only on permissible factors and complies with fair-lending/ECOA rules."},
            {"role": "user", "content": s["application"]}])
        try:
            txt = r["message"]["content"]
        except Exception:
            txt = r.message.content
        return _note(s, "compliance-checker: " + txt[:200])

    def decision(s: S) -> S:  # ollama — supervisor aggregates
        r = ollama.chat(model=OLLAMA_MODEL, messages=[
            {"role": "system", "content": "You are underwriting-supervisor. Given the specialists' assessments, "
             "decide APPROVE / CONDITIONAL / DECLINE in one sentence with a brief rationale."},
            {"role": "user", "content": "\n".join(s.get("assessments") or [])}])
        try:
            txt = r["message"]["content"]
        except Exception:
            txt = r.message.content
        return _note(s, "decision: " + txt[:200])

    g = StateGraph(S)
    g.add_node("underwriting-supervisor", underwriting_supervisor)
    g.add_node("credit-analyst", credit_analyst)
    g.add_node("risk-assessor", risk_assessor)
    g.add_node("compliance-checker", compliance_checker)
    g.add_node("decision", decision)
    g.add_edge(START, "underwriting-supervisor")
    g.add_edge("underwriting-supervisor", "credit-analyst")
    g.add_edge("credit-analyst", "risk-assessor")
    g.add_edge("risk-assessor", "compliance-checker")
    g.add_edge("compliance-checker", "decision")
    g.add_edge("decision", END)
    graph = g.compile()

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        handler = LangGraphCallbackHandler(client, capture_config=_CAPTURE)
        graph.invoke({"application": app_json, "assessments": []}, config={"callbacks": [handler]})
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for underwriting-team")
    payload["tags"] = ["layerlens-sample", "industry", "insurance", "underwriting", "multi-agent"]
    nodes = sorted({e["payload"].get("node") for e in payload.get("events", [])
                    if e.get("event_type") == "agent.node.enter"})
    frameworks = sorted({e["payload"].get("framework") for e in payload.get("events", [])
                         if e.get("event_type") == "model.invoke" and (e.get("payload") or {}).get("framework")})
    print(f"  underwriting-team  nodes={nodes}  model-frameworks={frameworks}")
    path = _write([payload], "industry", "underwriting_team")
    print(f"  -> {path}\n")


# --------------------------------------------------------------------------
# More multi-agent teams, varied topologies. LangGraph gives clean edged graphs
# (fan-out / routing / loop); crewai gives a framework=crewai multi-agent crew.
# Each specialist node runs on a mix of openai/anthropic/ollama.
# --------------------------------------------------------------------------
def _capture_langgraph(client: Stratix, graph, initial: dict, tags: list[str]) -> dict:
    from layerlens.instrument.adapters.frameworks.langgraph import LangGraphCallbackHandler
    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        graph.invoke(initial, config={"callbacks": [LangGraphCallbackHandler(client, capture_config=_CAPTURE)]})
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured")
    payload["tags"] = list(tags)
    nodes = sorted({e["payload"].get("node") for e in payload.get("events", [])
                    if e.get("event_type") == "agent.node.enter"})
    fw = sorted({e["payload"].get("framework") for e in payload.get("events", [])
                 if e.get("event_type") == "model.invoke" and (e.get("payload") or {}).get("framework")})
    print(f"  nodes={nodes}  model-frameworks={fw}")
    return payload


def _ollama_say(system: str, user: str) -> str:
    import ollama
    r = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "system", "content": system},
                                                  {"role": "user", "content": user}])
    try:
        return r["message"]["content"]
    except Exception:
        return r.message.content


def _openai_say(system: str, user: str) -> str:
    r = _handle("openai").chat.completions.create(model=OPENAI_MODEL, messages=[
        {"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def _anthropic_say(system: str, user: str) -> str:
    r = _handle("anthropic").messages.create(model=ANTHROPIC_MODEL, max_tokens=400,
        system=system, messages=[{"role": "user", "content": user}])
    return "".join(getattr(b, "text", "") for b in r.content)


def generate_code_review_team(client: Stratix) -> None:
    """LangGraph FAN-OUT: review-supervisor -> [security, style, test] -> aggregator."""
    from typing import TypedDict, Annotated
    import operator
    from langgraph.graph import StateGraph, START, END
    _handle("openai"); _handle("anthropic"); _handle("ollama")
    CODE = ("def get_user(uid):\n    q = \"SELECT * FROM users WHERE id = '\" + uid + \"'\"\n"
            "    return db.execute(q).fetchone()")

    class S(TypedDict, total=False):
        code: str
        findings: Annotated[list, operator.add]

    def supervisor(s):
        return {"findings": ["review-supervisor: dispatching to security/style/test reviewers"]}

    def security(s):
        return {"findings": ["security-reviewer: " + _openai_say(
            "You are security-reviewer. One sentence: flag the top security issue.", s["code"])[:160]]}

    def style(s):
        return {"findings": ["style-reviewer: " + _anthropic_say(
            "You are style-reviewer. One sentence: flag the top style/readability issue.", s["code"])[:160]]}

    def tests(s):
        return {"findings": ["test-reviewer: " + _ollama_say(
            "You are test-reviewer. One sentence: note the most important missing test.", s["code"])[:160]]}

    def aggregator(s):
        return {"findings": ["aggregator: consolidated %d reviews into a verdict" % len(s.get("findings", []))]}

    g = StateGraph(S)
    g.add_node("review-supervisor", supervisor)
    g.add_node("security-reviewer", security)
    g.add_node("style-reviewer", style)
    g.add_node("test-reviewer", tests)
    g.add_node("aggregator", aggregator)
    g.add_edge(START, "review-supervisor")
    for r in ("security-reviewer", "style-reviewer", "test-reviewer"):
        g.add_edge("review-supervisor", r)
        g.add_edge(r, "aggregator")
    g.add_edge("aggregator", END)
    print("  code-review-team (langgraph fan-out)")
    payload = _capture_langgraph(client, g.compile(), {"code": CODE, "findings": []},
                                 ["layerlens-sample", "co-work", "code-review", "multi-agent"])
    print("  ->", _write([payload], "cowork", "code_review_team"), "\n")


def generate_research_report_team(client: Stratix) -> None:
    """LangGraph LOOP: planner -> writer <-> critic (revise loop) -> END."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    _handle("openai"); _handle("anthropic"); _handle("ollama")

    class S(TypedDict, total=False):
        topic: str
        draft: str
        notes: list
        rounds: int

    def planner(s):
        plan = _openai_say("You are planner. In one sentence outline how to brief this topic.", s["topic"])
        return {"notes": (s.get("notes") or []) + ["planner: " + plan[:120]]}

    def writer(s):
        d = _ollama_say("You are writer. Write a 2-sentence brief on the topic (revise if given feedback).",
                        s["topic"] + ("\nfeedback: " + s["notes"][-1] if s.get("notes") else ""))
        return {"draft": d, "rounds": (s.get("rounds") or 0) + 1,
                "notes": (s.get("notes") or []) + ["writer: drafted (round %d)" % ((s.get("rounds") or 0) + 1)]}

    def critic(s):
        c = _anthropic_say("You are critic. One sentence: give one concrete improvement, or say APPROVE.", s.get("draft", ""))
        return {"notes": (s.get("notes") or []) + ["critic: " + c[:120]]}

    def route(s):
        return END if (s.get("rounds") or 0) >= 2 else "writer"

    g = StateGraph(S)
    g.add_node("planner", planner)
    g.add_node("writer", writer)
    g.add_node("critic", critic)
    g.add_edge(START, "planner")
    g.add_edge("planner", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", route, {"writer": "writer", END: END})
    print("  research-report-team (langgraph loop)")
    payload = _capture_langgraph(client, g.compile(),
                                 {"topic": "the benefits of automated code review", "notes": [], "rounds": 0},
                                 ["layerlens-sample", "co-work", "research-report", "multi-agent"])
    print("  ->", _write([payload], "cowork", "research_report_team"), "\n")


def generate_support_triage_team(client: Stratix) -> None:
    """LangGraph ROUTING: triage-router -> (billing|technical|account) -> resolver."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    _handle("openai"); _handle("anthropic"); _handle("ollama")
    TICKET = "My API calls started returning 401 after I rotated my key this morning; billing looks fine."

    class S(TypedDict, total=False):
        ticket: str
        category: str
        notes: list

    def router(s):
        cat = _openai_say("You are triage-router. Classify the ticket as exactly one word: billing, technical, "
                          "or account.", s["ticket"]).strip().lower()
        cat = next((c for c in ("technical", "billing", "account") if c in cat), "technical")
        return {"category": cat, "notes": ["triage-router: routed to %s" % cat]}

    def billing(s):
        return {"notes": s["notes"] + ["billing-specialist: " + _anthropic_say(
            "You are billing-specialist. One sentence resolution.", s["ticket"])[:160]]}

    def technical(s):
        return {"notes": s["notes"] + ["technical-specialist: " + _ollama_say(
            "You are technical-specialist. One sentence resolution.", s["ticket"])[:160]]}

    def account(s):
        return {"notes": s["notes"] + ["account-specialist: " + _ollama_say(
            "You are account-specialist. One sentence resolution.", s["ticket"])[:160]]}

    def resolver(s):
        return {"notes": s["notes"] + ["resolver: closing ticket via %s path" % s.get("category")]}

    g = StateGraph(S)
    g.add_node("triage-router", router)
    g.add_node("billing-specialist", billing)
    g.add_node("technical-specialist", technical)
    g.add_node("account-specialist", account)
    g.add_node("resolver", resolver)
    g.add_edge(START, "triage-router")
    g.add_conditional_edges("triage-router", lambda s: s["category"],
                            {"billing": "billing-specialist", "technical": "technical-specialist",
                             "account": "account-specialist"})
    for sp in ("billing-specialist", "technical-specialist", "account-specialist"):
        g.add_edge(sp, "resolver")
    g.add_edge("resolver", END)
    print("  support-triage-team (langgraph routing)")
    payload = _capture_langgraph(client, g.compile(), {"ticket": TICKET, "notes": []},
                                 ["layerlens-sample", "industry", "retail", "customer-support", "multi-agent"])
    print("  ->", _write([payload], "industry", "support_triage_team"), "\n")


def generate_clinical_consult_team(client: Stratix) -> None:
    """CrewAI CREW: a multi-specialist clinical consult (framework=crewai)."""
    from crewai import Agent, Task, Crew, Process, LLM
    from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    llm = LLM(model="ollama/%s" % OLLAMA_MODEL, base_url=base)
    case = ("62yo with exertional chest pressure, controlled hypertension, on lisinopril; "
            "troponin pending. Provide a triage read, a cardiology view, and a medication-safety check.")
    triage = Agent(role="triage-nurse", goal="Assign an acuity level and summarize the presentation",
                   backstory="An experienced ED triage nurse.", llm=llm, allow_delegation=False, verbose=False)
    cardiology = Agent(role="cardiology-consult", goal="Give a focused cardiology assessment",
                       backstory="A cardiologist.", llm=llm, allow_delegation=False, verbose=False)
    pharmacist = Agent(role="clinical-pharmacist", goal="Flag any medication/interaction concerns",
                       backstory="A clinical pharmacist.", llm=llm, allow_delegation=False, verbose=False)
    t1 = Task(description="Triage: %s" % case, expected_output="acuity + 1-line summary", agent=triage)
    t2 = Task(description="Cardiology assessment for the case.", expected_output="2-sentence assessment", agent=cardiology)
    t3 = Task(description="Medication-safety check for the case.", expected_output="1-2 sentence check", agent=pharmacist)
    crew = Crew(agents=[triage, cardiology, pharmacist], tasks=[t1, t2, t3],
                process=Process.sequential, verbose=False)

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = CrewAIAdapter(client)
    try:
        adapter.connect()
        crew.kickoff()
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for clinical-consult")
    payload["tags"] = ["layerlens-sample", "industry", "healthcare", "clinical-consult", "multi-agent"]
    print("  clinical-consult-team (crewai crew)")
    print("  ->", _write([payload], "industry", "clinical_consult_team"), "\n")


def generate_telecom_support_crew(client: Stratix) -> None:
    """CrewAI HIERARCHICAL CREW with real delegation (framework=crewai, multi-agent).

    A telecom customer-support manager delegates a mixed billing+connectivity
    complaint to two specialists via crewai's built-in ``Delegate work to
    coworker`` tool. Process.hierarchical + a manager LLM makes the manager
    actually emit the delegation tool call, so the CrewAIAdapter records real
    ``agent.handoff`` events (manager -> specialist) and the trace renders as a
    genuine multi-agent DAG (Agent column ``multi-agent``). This is the
    delegation path that was silently broken until the tool-name normalization
    fix — recording it here proves it end-to-end. Uses openai (reliable tool/
    delegation emission) so the shipped fixture deterministically has handoffs.
    """
    from crewai import LLM, Crew, Task, Agent, Process
    from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter

    llm = LLM(model="openai/%s" % OPENAI_MODEL)
    complaint = (
        "Customer reports they were double-charged $79.99 on their last bill AND their home "
        "internet keeps dropping every evening around 8pm. Resolve both issues."
    )
    billing = Agent(
        role="billing-specialist",
        goal="Resolve billing disputes: identify duplicate/incorrect charges and state the refund/adjustment.",
        backstory="A telecom billing specialist who audits charges and issues adjustments.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    network = Agent(
        role="network-specialist",
        goal="Diagnose connectivity issues and give concrete remediation steps.",
        backstory="A telecom network engineer who triages home-internet faults.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    resolve = Task(
        description=(
            "Handle this customer complaint end-to-end by delegating each part to the right "
            "specialist, then summarize the resolution:\n%s" % complaint
        ),
        expected_output="A combined resolution: the billing adjustment and the connectivity fix.",
    )
    crew = Crew(
        agents=[billing, network],
        tasks=[resolve],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=False,
    )

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = CrewAIAdapter(client)
    try:
        adapter.connect()
        crew.kickoff()
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for telecom-support-crew")
    payload["tags"] = ["layerlens-sample", "industry", "telecom", "customer-support", "multi-agent"]
    handoffs = [e for e in payload.get("events", []) if e.get("event_type") == "agent.handoff"]
    print("  telecom-support-crew (crewai hierarchical)  handoffs=%d" % len(handoffs))
    print("  ->", _write([payload], "industry", "telecom_support_crew"), "\n")


def generate_content_pipeline_team(client: Stratix) -> None:
    """AutoGen GROUP CHAT: a round-robin content team (framework=autogen).

    A content strategist, a copywriter, and an editor collaborate as a real
    ``RoundRobinGroupChat`` -- each agent reads the running conversation and
    contributes its turn. The AutoGenAdapter records the real per-agent
    ``model.invoke`` events and the round-robin handoffs, so the trace renders
    as a multi-node autogen graph (strategist -> copywriter -> editor).
    """
    import asyncio

    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from layerlens.instrument.adapters.frameworks.autogen import AutoGenAdapter

    mc = OpenAIChatCompletionClient(model=OPENAI_MODEL)
    strategist = AssistantAgent(
        "content_strategist",
        model_client=mc,
        system_message="You set the angle and key message for the piece in one or two sentences.",
    )
    copywriter = AssistantAgent(
        "copywriter",
        model_client=mc,
        system_message="You draft one short paragraph of marketing copy following the strategist's angle.",
    )
    editor = AssistantAgent(
        "editor",
        model_client=mc,
        system_message="You tighten the copy for clarity and reply with the final edited version.",
    )
    team = RoundRobinGroupChat(
        [strategist, copywriter, editor],
        termination_condition=MaxMessageTermination(4),
    )
    task = "Write a short launch announcement for a privacy-first note-taking app called Quill."

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = AutoGenAdapter(client)
    try:
        adapter.connect()
        asyncio.run(team.run(task=task))
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for content-pipeline")
    payload["tags"] = ["layerlens-sample", "cowork", "content", "content-pipeline", "multi-agent"]
    print("  content-pipeline-team (autogen round-robin group chat)")
    print("  ->", _write([payload], "cowork", "content_pipeline_team"), "\n")



# --------------------------------------------------------------------------
# ADP-W1 Family-B recorders (providers tool-use, bedrock, azure sealed,
# langchain, llama_index, langgraph). Each records a REAL run (real model
# call / real adapter over a mocked transport for the azure sealed case) and
# writes one payload to samples/data/traces/industry/<stem>.jsonl. Framework
# deps are imported function-locally so this module imports in any venv.
# --------------------------------------------------------------------------

def generate_media_content_moderation(client: Stratix) -> None:
    """Anthropic single-agent TOOL-USE: a media content-moderation agent that
    calls a REAL ``policy_lookup`` tool (tools=/tool_use) to fetch the platform
    policy for the post under review, then returns an ALLOW / FLAG / REMOVE
    decision justified against that policy.

    Recorded under ``@trace`` + ``instrument_anthropic`` so the sealed trace
    carries a real ``agent.identity`` (Agent column = ``content-moderation-agent``,
    a single 1-node graph), the adapter-emitted ``tool.call`` for the model's
    policy request, a ``tool.result`` (the fetched policy), and both
    ``model.invoke`` / ``cost.record`` events of the 2-step tool loop. Mirrors
    the ateam media/content_moderator domain. Everything is a genuine run --
    nothing is fabricated.
    """
    ant = _handle("anthropic")

    # The user post under review + a small REAL policy KB the tool reads from.
    POST = (
        'User post under review (platform: social feed):\n\n'
        '"BREAKING: Scientists confirm drinking small amounts of bleach every '
        'morning CURES all known diseases including cancer. Big Pharma is hiding '
        'this! Share before they delete it. #TruthRevealed #HealthHack"'
    )
    POLICY_DB = {
        "health_misinformation": {
            "policy_id": "POL-HM-004",
            "title": "Dangerous Health Misinformation",
            "rule": (
                "Content promoting unproven or physically dangerous health claims "
                "(e.g. ingesting harmful substances as a cure) is REMOVED. "
                "Encouraging others to share amplifies harm."
            ),
            "default_action": "remove",
            "severity": "high",
        },
        "hate_speech": {
            "policy_id": "POL-HS-001",
            "title": "Hateful Conduct",
            "rule": "Attacks on protected groups are removed; slurs are flagged.",
            "default_action": "remove",
            "severity": "high",
        },
        "spam": {
            "policy_id": "POL-SP-002",
            "title": "Spam & Deceptive Engagement",
            "rule": "Repetitive or engagement-bait content is flagged for review.",
            "default_action": "flag",
            "severity": "medium",
        },
        "violence": {
            "policy_id": "POL-VL-003",
            "title": "Violence & Incitement",
            "rule": "Credible threats or incitement to violence are removed.",
            "default_action": "remove",
            "severity": "high",
        },
    }

    def policy_lookup(category: str) -> dict:
        """REAL tool fn: fetch the platform content policy for a category."""
        return POLICY_DB.get(
            category,
            {
                "policy_id": "POL-GEN-000",
                "title": "General Community Standards",
                "rule": "No specific policy matched; apply general standards.",
                "default_action": "allow",
                "severity": "low",
            },
        )

    TOOLS = [
        {
            "name": "policy_lookup",
            "description": (
                "Look up the platform content policy for a content category. "
                "Valid categories: 'health_misinformation', 'hate_speech', "
                "'spam', 'violence'."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The content-policy category to fetch.",
                    }
                },
                "required": ["category"],
            },
        }
    ]
    SYSTEM = (
        "You are content-moderation-agent, a media platform content moderator. "
        "For the user post under review, FIRST call the policy_lookup tool with "
        "the single most relevant content category, THEN return a decision of "
        "ALLOW, FLAG, or REMOVE and justify it against the returned policy. "
        "Answer concisely (under 150 words)."
    )

    def _emit_tool_result(tool_name: str, result: dict) -> None:
        col = _current_collector.get()
        if col is None:
            return
        from layerlens.instrument._context import _current_span_id

        col.emit(
            "tool.result",
            {
                "provider": "anthropic",
                "tool_name": tool_name,
                "result": result,
                "status": "ok",
            },
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=_current_span_id.get(),
            span_name="tool:%s" % tool_name,
        )

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:

        @trace(client, name="content-moderation-agent", capture_config=_CAPTURE)
        def _moderate(post: str) -> str:
            messages: list = [{"role": "user", "content": post}]
            final_text = ""
            for step in range(4):
                # Force the policy lookup on the first turn; auto afterwards so
                # the model returns its final text decision.
                tool_choice = (
                    {"type": "tool", "name": "policy_lookup"}
                    if step == 0
                    else {"type": "auto"}
                )
                resp = ant.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=500,
                    system=SYSTEM,
                    tools=TOOLS,
                    tool_choice=tool_choice,
                    messages=messages,
                )
                tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
                text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
                if not tool_uses:
                    final_text = text
                    break
                # Feed the assistant turn back, run each REAL tool, return results.
                messages.append({"role": "assistant", "content": resp.content})
                tool_results = []
                for tu in tool_uses:
                    result = policy_lookup(**(tu.input or {}))
                    _emit_tool_result(tu.name, result)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps(result),
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
            return final_text or "(no decision produced)"

        _moderate(POST)
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for media-content-moderation")
    payload["tags"] = ["layerlens-sample", "industry", "media", "content-moderation", "tool-use"]
    print("  media-content-moderation (anthropic tool-use)")
    print("  ->", _write([payload], "industry", "media_content_moderation"), "\n")


def generate_retail_catalog_qa(client: Stratix) -> None:
    """OpenAI single-agent TOOL-USE: a retail product-Q&A agent that calls a
    REAL ``lookup_product`` tool (tools=/tool_calls) to fetch catalog + live
    inventory for the SKU the shopper asks about, then answers the question
    (price, stock, key specs) grounded in the tool result.

    Recorded under ``@trace`` + ``instrument_openai`` so the sealed trace carries
    a real ``agent.identity`` (Agent column = ``product-qa-agent``, a single
    1-node graph), the adapter-emitted ``tool.call`` for the catalog lookup, a
    ``tool.result`` (the fetched product record), and both ``model.invoke`` /
    ``cost.record`` events of the 2-step tool loop. Mirrors the ateam retail
    domain. A genuine run -- nothing fabricated.
    """
    oai = _handle("openai")

    QUESTION = (
        "Do you have the Aeron ergonomic office chair in stock, how much is it, "
        "and does it support up to 300 lbs?"
    )
    CATALOG = {
        "aeron-chair": {
            "sku": "aeron-chair",
            "name": "Aeron Ergonomic Office Chair (Size B)",
            "price_usd": 1395.00,
            "in_stock": True,
            "inventory_count": 42,
            "weight_capacity_lbs": 350,
            "warranty_years": 12,
            "rating": 4.8,
        },
        "standing-desk": {
            "sku": "standing-desk",
            "name": "UpLift V2 Standing Desk 60x30",
            "price_usd": 749.00,
            "in_stock": True,
            "inventory_count": 15,
            "weight_capacity_lbs": 355,
            "warranty_years": 15,
            "rating": 4.7,
        },
        "desk-lamp": {
            "sku": "desk-lamp",
            "name": "BenQ ScreenBar Monitor Light",
            "price_usd": 109.00,
            "in_stock": False,
            "inventory_count": 0,
            "weight_capacity_lbs": None,
            "warranty_years": 2,
            "rating": 4.6,
        },
    }

    def lookup_product(query: str) -> dict:
        """REAL tool fn: resolve a shopper query to a catalog + inventory record."""
        q = (query or "").lower()
        for rec in CATALOG.values():
            if any(tok in q for tok in rec["name"].lower().split()) or rec["sku"] in q:
                return rec
        # keyword fallbacks
        if "chair" in q or "aeron" in q:
            return CATALOG["aeron-chair"]
        if "desk" in q or "standing" in q:
            return CATALOG["standing-desk"]
        if "lamp" in q or "light" in q:
            return CATALOG["desk-lamp"]
        return {"query": query, "found": False, "message": "No catalog match."}

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "lookup_product",
                "description": (
                    "Look up catalog details and live inventory for a product the "
                    "shopper is asking about (price, stock, specs)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The product name or description from the shopper's question.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    SYSTEM = (
        "You are product-qa-agent, a retail product-Q&A assistant. For the "
        "shopper's question, FIRST call the lookup_product tool to fetch the "
        "catalog record and live inventory, THEN answer the question (price, "
        "stock status, and the relevant spec) grounded ONLY in the tool result. "
        "Answer concisely (under 150 words)."
    )

    def _emit_tool_result(tool_name: str, result: dict) -> None:
        col = _current_collector.get()
        if col is None:
            return
        from layerlens.instrument._context import _current_span_id

        col.emit(
            "tool.result",
            {
                "provider": "openai",
                "tool_name": tool_name,
                "result": result,
                "status": "ok",
            },
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=_current_span_id.get(),
            span_name="tool:%s" % tool_name,
        )

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:

        @trace(client, name="product-qa-agent", capture_config=_CAPTURE)
        def _answer(question: str) -> str:
            messages: list = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ]
            final_text = ""
            for step in range(4):
                tool_choice = (
                    {"type": "function", "function": {"name": "lookup_product"}}
                    if step == 0
                    else "auto"
                )
                resp = oai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice=tool_choice,
                )
                msg = resp.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []
                if not tool_calls:
                    final_text = msg.content or ""
                    break
                messages.append(msg)
                for tc in tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    result = lookup_product(**args)
                    _emit_tool_result(tc.function.name, result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        }
                    )
            return final_text or "(no answer produced)"

        _answer(QUESTION)
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for retail-catalog-qa")
    payload["tags"] = ["layerlens-sample", "industry", "retail", "product-qa", "tool-use"]
    print("  retail-catalog-qa (openai tool-use)")
    print("  ->", _write([payload], "industry", "retail_catalog_qa"), "\n")


# A small in-memory contract-clause corpus for the LlamaIndex RAG sample: each
# clause of a SaaS master agreement is its own document so the real retriever
# selects the ones relevant to the review question.
LEGAL_CONTRACT_CLAUSES = [
    ("term_and_termination",
     "Term & Termination: This Agreement auto-renews for successive 12-month terms unless either "
     "party gives written notice at least 180 days before the end of the then-current term."),
    ("limitation_of_liability",
     "Limitation of Liability: Each party's aggregate liability is capped at fees paid in the prior "
     "12 months, EXCEPT that Vendor's liability for data breaches and losses of Customer Data is "
     "UNLIMITED and expressly excluded from the cap."),
    ("payment_terms",
     "Payment Terms: Fees are invoiced annually in advance and due net-45. Late payments accrue "
     "interest at 1.5% per month; Vendor may suspend the service after 15 days' delinquency."),
    ("data_protection",
     "Data Protection: Vendor processes Customer Data as a processor under GDPR Article 28, maintains "
     "SOC 2 Type II controls, and notifies Customer of a personal-data breach within 72 hours."),
    ("indemnification",
     "Indemnification: Vendor indemnifies Customer against third-party IP infringement claims; "
     "Customer indemnifies Vendor against claims arising from Customer Data content."),
    ("confidentiality",
     "Confidentiality: Each party protects the other's Confidential Information for 5 years after "
     "disclosure, using at least the same care it uses for its own confidential information."),
]


def generate_legal_contract_rag(client: Stratix) -> None:
    """LlamaIndex RAG (SINGLE): a contract-analysis retrieval-augmented query.

    A small in-memory ``VectorStoreIndex`` of the clauses of a SaaS master
    agreement (``LEGAL_CONTRACT_CLAUSES``) is queried by a real
    ``as_query_engine().query(...)`` review question. The REAL LlamaIndex adapter
    (root dispatcher) captures the genuine retrieval (``tool.call``/``tool.result``
    ``retrieval`` events reflecting the real indexed clauses) plus the synthesis
    ``model.invoke`` + priced ``cost.record``. Real OpenAI embeddings make the
    retrieval semantically meaningful; the index is built BEFORE connecting so
    only the query trace is captured (the doc-embedding pass is not). No agent
    identity is declared, so this renders honestly as a single RAG query (no
    fabricated agent).
    """
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI as LIOpenAI
    from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter

    embed = OpenAIEmbedding()
    docs = [Document(text=text, metadata={"clause": name}) for name, text in LEGAL_CONTRACT_CLAUSES]
    # Build the index BEFORE connecting so the offline doc-embedding pass is not
    # captured — only the query trace flushes.
    index = VectorStoreIndex.from_documents(docs, embed_model=embed)

    llm = LIOpenAI(model=OPENAI_MODEL)
    question = (
        "Reviewing this SaaS agreement for our company: what are the biggest legal risks in the "
        "liability and termination clauses, and what should we negotiate?"
    )

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = LlamaIndexAdapter(client, capture_config=_CAPTURE)
    try:
        adapter.connect()
        engine = index.as_query_engine(llm=llm, embed_model=embed, similarity_top_k=3)
        answer = str(engine.query(question))
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for legal-contract-rag")
    payload["tags"] = ["layerlens-sample", "industry", "legal", "contract-analysis", "rag"]
    retr = [e for e in payload.get("events", []) if e.get("event_type") == "tool.result"
            and (e.get("payload") or {}).get("tool_name") == "retrieval"]
    mi = [e for e in payload.get("events", []) if e.get("event_type") == "model.invoke"]
    print("  legal-contract-rag (llamaindex RAG)  retrieval=%d model.invoke=%d answer=%r"
          % (len(retr), len(mi), answer[:60]))
    print("  ->", _write([payload], "industry", "legal_contract_rag"), "\n")


def generate_legal_agentworkflow(client: Stratix) -> None:
    """LlamaIndex AgentWorkflow (MULTI, HEADLINE): a two-FunctionAgent handoff.

    A REAL ``AgentWorkflow`` runs two ``FunctionAgent``s: ``contract-intake``
    (root) reads a contract-review request and HANDS OFF to ``clause-risk`` for
    the risk analysis, via AgentWorkflow's built-in ``handoff`` tool. The REAL
    LlamaIndex adapter reads the workflow event stream and emits producer-honest
    ``agent.input``/``agent.output``/``model.invoke`` per agent turn plus a real
    ``agent.handoff{from_agent=contract-intake, to_agent=clause-risk}``, so the
    trace renders as a genuine MULTI-AGENT DAG (nodes>=2, a handoff edge, Agent
    column ``multi-agent``). This closes the llama_index multi-agent render gap.

    ``stream_options={"include_usage": True}`` on the OpenAI LLM makes the
    (streamed) AgentWorkflow turns surface the real per-call token counts, so
    each turn carries a genuine priced ``cost.record``. Uses real OpenAI so the
    handoff is reliably emitted and the shipped fixture deterministically has the
    multi-agent edge.
    """
    import asyncio

    from llama_index.llms.openai import OpenAI as LIOpenAI
    from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
    from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter

    llm = LIOpenAI(model=OPENAI_MODEL, additional_kwargs={"stream_options": {"include_usage": True}})
    intake = FunctionAgent(
        name="contract-intake",
        description="Intakes a contract-review request, states the contract type and the clauses "
                    "present, then hands off to clause-risk for the legal risk analysis.",
        system_prompt=(
            "You are contract-intake, a legal contract intake agent. Read the contract summary, "
            "state the contract type and the clauses present in ONE sentence, then IMMEDIATELY hand "
            "off to 'clause-risk' to assess the legal risks. Do not attempt the risk analysis yourself."
        ),
        llm=llm,
        can_handoff_to=["clause-risk"],
    )
    risk = FunctionAgent(
        name="clause-risk",
        description="Assesses the legal risk of the identified contract clauses and recommends changes.",
        system_prompt=(
            "You are clause-risk, a legal risk analyst. Given the intake summary, name the two "
            "highest-risk clauses, rate each risk (low/medium/high), and give a one-line negotiation "
            "recommendation for each."
        ),
        llm=llm,
        can_handoff_to=[],
    )
    workflow = AgentWorkflow(agents=[intake, risk], root_agent="contract-intake")
    user_msg = (
        "Review this SaaS master agreement between Acme Corp (vendor) and Widget Inc (customer). "
        "Clauses present: term_and_termination (auto-renewal, 180-day notice), payment_terms (net-45), "
        "limitation_of_liability (UNLIMITED liability for data breaches, no cap), data_protection "
        "(GDPR processor), confidentiality, indemnification. Flag the legal risks and recommend changes."
    )

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = LlamaIndexAdapter(client, capture_config=_CAPTURE)
    try:
        adapter.connect()

        async def _run() -> object:
            return await workflow.run(user_msg=user_msg)

        asyncio.run(_run())
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for legal-agentworkflow")
    payload["tags"] = ["layerlens-sample", "industry", "legal", "contract-analysis", "multi-agent"]
    handoffs = [(e["payload"].get("from_agent"), e["payload"].get("to_agent"))
                for e in payload.get("events", []) if e.get("event_type") == "agent.handoff"]
    agents = sorted({(e.get("payload") or {}).get("agent_name") for e in payload.get("events", [])
                     if (e.get("payload") or {}).get("agent_name")})
    mi = [e for e in payload.get("events", []) if e.get("event_type") == "model.invoke"]
    cr = [e for e in payload.get("events", []) if e.get("event_type") == "cost.record"]
    print("  legal-agentworkflow (llamaindex AgentWorkflow)  agents=%s handoffs=%s model.invoke=%d cost.record=%d"
          % (agents, handoffs, len(mi), len(cr)))
    print("  ->", _write([payload], "industry", "legal_agentworkflow"), "\n")


# --------------------------------------------------------------------------
# PASTE-ONCE PREREQUISITE (shared module-level block for the two Azure sealed
# manufacturing generators below). Azure OpenAI is credential-gated (no
# subscription in CI/dev), so these fixtures are recorded SEALED: a REAL
# ``openai.AzureOpenAI`` client + the REAL ``AzureOpenAIProvider`` adapter run
# over an ``httpx.MockTransport`` (the ``test_azure_openai._make_client`` seam).
# The adapter genuinely parses an Azure-shaped ``chat.completion`` through the
# real SDK, so framework=azure_openai, the model, token usage, priced cost, the
# synthesized agent.identity and the attestation chain are all real adapter
# output -- only the LLM network is mocked (deferred until Azure creds exist).
# --------------------------------------------------------------------------
_AZURE_ENDPOINT = "https://contoso-mfg-eastus.openai.azure.com"
_AZURE_API_VERSION = "2024-06-01"
_AZURE_API_KEY = "sealed-no-azure-credential"  # sealed fixture -- never a real key
_AZURE_DEPLOYMENT = "gpt-4o-prod"              # Azure deployment (routing) name
_AZURE_MODEL = "gpt-4o-2024-08-06"             # underlying model the response reports

_MFG_FILTER = {
    "hate": {"filtered": False, "severity": "safe"},
    "self_harm": {"filtered": False, "severity": "safe"},
    "sexual": {"filtered": False, "severity": "safe"},
    "violence": {"filtered": False, "severity": "safe"},
}


def _azure_chat_completion(content, *, response_id, tool_calls=None,
                           finish_reason="stop", prompt_tokens=0, completion_tokens=0):
    """A realistic Azure chat.completions response body (choices/usage/filters)."""
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1751328000,
        "model": _AZURE_MODEL,
        "system_fingerprint": "fp_mfg_9d41c7",
        "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": _MFG_FILTER}],
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": message,
            "content_filter_results": _MFG_FILTER,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def generate_manufacturing_predictive_maintenance(client) -> None:
    """Azure OpenAI SEALED single-agent predictive-maintenance assessment.

    A ``predictive-maintenance-agent`` reads a live sensor snapshot from a CNC
    spindle showing a rising bearing-defect (BPFI) signature and returns a
    failure-risk assessment + remaining-useful-life estimate + a maintenance
    recommendation. The REAL ``AzureOpenAIProvider`` adapter runs over an
    ``httpx.MockTransport`` returning a realistic Azure ``chat.completion``, so
    the trace carries a real framework=azure_openai ``model.invoke``, priced
    ``cost.record``, a synthesized ``agent.identity`` and an intact attestation
    chain -- only the LLM network is sealed (no Azure credential available).
    """
    import httpx
    from openai import AzureOpenAI
    from layerlens.instrument.adapters.providers.azure_openai import AzureOpenAIProvider

    sensors = {
        "equipment_id": "MF-EQ-014",
        "equipment_type": "CNC Milling Machine",
        "manufacturer": "DMG Mori",
        "model": "DMU 50 5-Axis",
        "operating_hours": 12450,
        "location": "Building A, Cell 3",
        "sensor_snapshot": {
            "vibration_mm_s": 6.8,
            "temperature_c": 79.4,
            "pressure_bar": 7.9,
            "current_amps": 58.0,
            "acoustic_db": 88.0,
            "rpm": 1750,
        },
        "trend": "vibration up 41% over 72h; 1x + 2x + BPFI harmonics emerging",
        "safety_thresholds": {
            "vibration_mm_s_alarm": 7.1, "vibration_mm_s_danger": 11.2,
            "temperature_c_alarm": 85.0, "temperature_c_danger": 100.0,
        },
    }
    answer = (
        "FAILURE RISK: HIGH. The 1x/2x plus BPFI harmonic signature with vibration at "
        "6.8 mm/s (approaching the 7.1 mm/s alarm) and bearing temperature at 79.4 C is "
        "consistent with an advancing spindle-bearing inner-race defect. Estimated "
        "remaining useful life: ~170-190 operating hours before the danger threshold. "
        "RECOMMENDATION: schedule a spindle-bearing replacement within the next planned "
        "maintenance window (<=7 days); stage the bearing kit and reduce spindle load / "
        "feed to hold vibration under the 7.1 mm/s alarm until then. Do NOT run to failure "
        "-- an inner-race spall at 1750 RPM risks spindle and workpiece damage."
    )
    response_json = _azure_chat_completion(
        answer, response_id="chatcmpl-mfg-pdm-0001",
        prompt_tokens=317, completion_tokens=168,
    )

    def handler(request):
        return httpx.Response(200, json=response_json)

    azure = AzureOpenAI(
        azure_endpoint=_AZURE_ENDPOINT, api_key=_AZURE_API_KEY,
        api_version=_AZURE_API_VERSION,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    provider = AzureOpenAIProvider()
    provider.connect(azure)

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        @trace(client, name="predictive-maintenance-agent", capture_config=_CAPTURE)
        def _agent(machine):
            r = azure.chat.completions.create(
                model=_AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content":
                        "You are predictive-maintenance-agent, a manufacturing reliability "
                        "assistant. Given a machine sensor snapshot, assess failure risk "
                        "(LOW/MEDIUM/HIGH), estimate remaining useful life, and recommend a "
                        "maintenance action. Respect the stated safety thresholds."},
                    {"role": "user", "content": json.dumps(machine)},
                ],
                temperature=0.2,
            )
            return r.choices[0].message.content
        _agent(sensors)
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
        provider.disconnect()

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for manufacturing-predictive-maintenance")
    payload["tags"] = ["layerlens-sample", "industry", "manufacturing",
                       "predictive-maintenance", "azure-openai", "sealed-fixture"]
    payload["metadata"] = {"sealed": True, "provider": "azure_openai",
                           "reason": "recorded over httpx.MockTransport -- Azure credential "
                                     "unavailable; real adapter, sealed LLM network"}
    fw = next((e["payload"].get("framework") for e in payload.get("events", [])
               if isinstance(e.get("payload"), dict) and e["payload"].get("framework")), None)
    print("  manufacturing-predictive-maintenance (azure sealed)  events=%d framework=%s"
          % (len(payload.get("events", [])), fw))
    print("  ->", _write([payload], "industry", "manufacturing_predictive_maintenance"), "\n")


def generate_manufacturing_maintenance_tooluse(client) -> None:
    """Azure OpenAI SEALED tool-use predictive-maintenance run.

    The same ``predictive-maintenance-agent`` calls a real ``get_maintenance_history``
    tool before advising. The sealed ``httpx.MockTransport`` returns a tool-call turn
    (finish_reason=tool_calls) on the first request and a final assessment on the
    second; the REAL ``AzureOpenAIProvider`` adapter emits ``tool.call`` from the
    first response and a second ``model.invoke``+``cost.record`` from the final
    turn. The local tool genuinely runs and returns the equipment's maintenance
    records, recorded as a ``tool.result`` event -- so the trace carries a real
    tool.call + tool.result loop over framework=azure_openai (only the LLM network
    is sealed; no Azure credential available).
    """
    import httpx
    from openai import AzureOpenAI
    from layerlens.instrument.adapters.providers.azure_openai import AzureOpenAIProvider
    from layerlens.instrument._context import _current_collector, _current_span_id

    # A real local tool: returns the equipment's genuine maintenance records.
    MAINTENANCE_DB = {
        "MF-EQ-021": [
            {"date": "2024-08-14", "type": "preventive", "description": "Hydraulic oil change and filter replacement", "cost": 2800},
            {"date": "2025-02-20", "type": "corrective", "description": "Replaced main cylinder seal set", "cost": 12000},
            {"date": "2025-07-11", "type": "preventive", "description": "Accumulator pre-charge check", "cost": 600},
            {"date": "2025-11-30", "type": "corrective", "description": "Repaired hydraulic hose leak on return line", "cost": 1500},
        ],
    }

    def get_maintenance_history(equipment_id):
        return {"equipment_id": equipment_id, "records": MAINTENANCE_DB.get(equipment_id, [])}

    sensors = {
        "equipment_id": "MF-EQ-021",
        "equipment_type": "Hydraulic Press",
        "manufacturer": "Schuler",
        "model": "MSD 630",
        "operating_hours": 22800,
        "sensor_snapshot": {"vibration_mm_s": 4.2, "temperature_c": 63.0,
                            "pressure_bar": 5.4, "current_amps": 61.0, "acoustic_db": 74.0, "rpm": 0},
        "trend": "pressure drifting down toward the 5.0 bar minimum; return-line variability rising",
        "safety_thresholds": {"pressure_bar_min": 5.0, "pressure_bar_max": 12.0, "temperature_c_alarm": 70.0},
    }
    tools = [{
        "type": "function",
        "function": {
            "name": "get_maintenance_history",
            "description": "Return the maintenance records for a piece of equipment by id.",
            "parameters": {"type": "object", "properties": {"equipment_id": {"type": "string"}},
                           "required": ["equipment_id"]},
        },
    }]
    tool_call = {"id": "call_mfg_hist_01", "type": "function",
                 "function": {"name": "get_maintenance_history",
                              "arguments": json.dumps({"equipment_id": "MF-EQ-021"})}}
    first = _azure_chat_completion(None, response_id="chatcmpl-mfg-tool-0001",
                                   tool_calls=[tool_call], finish_reason="tool_calls",
                                   prompt_tokens=286, completion_tokens=24)
    final_answer = (
        "FAILURE RISK: MEDIUM. Sensors are within limits but hydraulic pressure is drifting "
        "toward the 5.0 bar minimum. The maintenance history is decisive: the main cylinder "
        "seal set was replaced 2025-02-20 and a return-line hose leak was repaired 2025-11-30 "
        "-- a recurring seal/return-line loss pattern. Estimated remaining useful life: "
        "~300-340 operating hours. RECOMMENDATION: schedule an accumulator pre-charge check and "
        "a main-cylinder seal inspection at the next preventive window; trend pressure daily and "
        "alarm if it drops below 5.2 bar. Not an emergency, but do not defer past the next cycle."
    )
    final = _azure_chat_completion(final_answer, response_id="chatcmpl-mfg-tool-0002",
                                   prompt_tokens=402, completion_tokens=161)
    responses = [first, final]

    def handler(request):
        return httpx.Response(200, json=responses.pop(0))

    azure = AzureOpenAI(
        azure_endpoint=_AZURE_ENDPOINT, api_key=_AZURE_API_KEY,
        api_version=_AZURE_API_VERSION,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    provider = AzureOpenAIProvider()
    provider.connect(azure)

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        @trace(client, name="predictive-maintenance-agent", capture_config=_CAPTURE)
        def _agent(machine):
            system = ("You are predictive-maintenance-agent, a manufacturing reliability "
                      "assistant. You MAY call get_maintenance_history(equipment_id) to review "
                      "past repairs before advising. Assess failure risk (LOW/MEDIUM/HIGH), "
                      "estimate remaining useful life, and recommend a maintenance action.")
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(machine)}]
            # Turn 1: the model asks to call the tool.
            r1 = azure.chat.completions.create(
                model=_AZURE_DEPLOYMENT, messages=messages, tools=tools, temperature=0.2)
            call = r1.choices[0].message.tool_calls[0]
            args = json.loads(call.function.arguments)
            # The local tool genuinely runs and returns real records.
            result = get_maintenance_history(args["equipment_id"])
            # Record the real tool result as a structural tool.result event on the
            # active trace (the tool truly executed -- this is honest, not fabricated).
            col = _current_collector.get()
            col.emit("tool.result",
                     {"tool_name": "get_maintenance_history", "tool_call_id": call.id,
                      "arguments": args, "result": result},
                     span_id=_current_span_id.get(),
                     span_name="predictive-maintenance-agent")
            # Feed the tool result back for the final turn.
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [{"id": call.id, "type": "function",
                                             "function": {"name": "get_maintenance_history",
                                                          "arguments": call.function.arguments}}]})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result)})
            r2 = azure.chat.completions.create(
                model=_AZURE_DEPLOYMENT, messages=messages, tools=tools, temperature=0.2)
            return r2.choices[0].message.content
        _agent(sensors)
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
        provider.disconnect()

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for manufacturing-maintenance-tooluse")
    payload["tags"] = ["layerlens-sample", "industry", "manufacturing",
                       "predictive-maintenance", "tool-use", "azure-openai", "sealed-fixture"]
    payload["metadata"] = {"sealed": True, "provider": "azure_openai",
                           "reason": "recorded over httpx.MockTransport -- Azure credential "
                                     "unavailable; real adapter, sealed LLM network"}
    evt = lambda t: [e for e in payload.get("events", []) if e.get("event_type") == t]
    print("  manufacturing-maintenance-tooluse (azure sealed)  model.invoke=%d tool.call=%d tool.result=%d"
          % (len(evt("model.invoke")), len(evt("tool.call")), len(evt("tool.result"))))
    print("  ->", _write([payload], "industry", "manufacturing_maintenance_tooluse"), "\n")


def generate_energy_grid_forecast(client: Stratix) -> None:
    """AWS Bedrock (Nova) single agent: a grid-load-forecaster that does a REAL
    Bedrock Converse call over live grid telemetry and produces a 24-hour load
    forecast + reserve-margin risk flags. ``@trace`` gives it a real
    ``agent.identity`` (Agent column = grid-load-forecaster, 1 node); the
    instrumented ``converse`` emits the real ``model.invoke``/``cost.record``
    (framework=aws_bedrock, nova-micro is priced). Mirrors ateam
    energy/grid_load_forecaster.py.
    """
    import boto3
    from layerlens.instrument.adapters.providers.bedrock import (
        instrument_bedrock,
        uninstrument_bedrock,
    )

    model_id = os.environ.get("LL_BEDROCK_MODEL") or "amazon.nova-micro-v1:0"
    region = os.environ.get("AWS_REGION", "us-east-1")
    telemetry = {
        "operator": "MISO-Central",
        "as_of": "2026-07-14T16:00:00Z",
        "horizon_hours": 24,
        "zones": [
            {"zone": "Z1-Metro", "current_load_mw": 4820, "capacity_mw": 6000, "renewable_pct": 22},
            {"zone": "Z2-Suburban", "current_load_mw": 3110, "capacity_mw": 4200, "renewable_pct": 34},
            {"zone": "Z3-Industrial", "current_load_mw": 5180, "capacity_mw": 5600, "renewable_pct": 12},
            {"zone": "Z4-Coastal", "current_load_mw": 2040, "capacity_mw": 3500, "renewable_pct": 51},
            {"zone": "Z5-Rural", "current_load_mw": 1290, "capacity_mw": 2200, "renewable_pct": 28},
        ],
        "weather": {"temp_f": 101, "humidity_pct": 38, "wind_mph": 6, "heat_advisory": True},
        "notes": "Regional heat advisory in effect; EV-charging cluster ramp expected 6-9pm in Z1/Z3.",
    }
    system = (
        "You are grid-load-forecaster, an electrical grid load-forecasting agent for a "
        "transmission operator. Given real-time zone telemetry and weather, produce a 24-hour "
        "peak-load forecast for each zone, flag any zone at risk of breaching a safe reserve "
        "margin (keep >=8% headroom below capacity), and recommend concrete mitigations "
        "(demand response, generation dispatch, curtailment). Answer concisely (under 200 words)."
    )
    user = "Forecast 24h load and flag reserve-margin risk for this grid:\n" + json.dumps(telemetry)

    client_bedrock = boto3.client("bedrock-runtime", region_name=region)
    instrument_bedrock(client_bedrock)

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        @trace(client, name="grid-load-forecaster", capture_config=_CAPTURE)
        def _forecast(_telemetry):
            resp = client_bedrock.converse(
                modelId=model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": 500, "temperature": 0.2},
            )
            return "".join(
                b.get("text", "")
                for b in resp["output"]["message"]["content"]
                if isinstance(b, dict)
            )

        _forecast(telemetry)
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
        uninstrument_bedrock()

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for energy-grid-forecast")
    payload["tags"] = ["layerlens-sample", "industry", "energy", "grid-load-forecasting"]
    events = payload.get("events", [])
    invokes = [e for e in events if e.get("event_type") == "model.invoke"]
    costs = [e for e in events if e.get("event_type") == "cost.record"]
    print("  energy-grid-forecast (bedrock nova converse)  model.invoke=%d cost.record=%d"
          % (len(invokes), len(costs)))
    print("  ->", _write([payload], "industry", "energy_grid_forecast"), "\n")


def generate_energy_grid_tooluse(client: Stratix) -> None:
    """AWS Bedrock (Nova) Converse TOOL-USE loop: the same grid-load-forecaster,
    but the model calls a real ``get_sensor_reading`` tool (Converse toolConfig)
    to fetch a zone's live load before forecasting. We return a real toolResult
    and the model produces the final forecast -- exercising the Converse tool-use
    path captured after the BUG-3 fix. The trace carries real ``tool.call`` +
    ``model.invoke`` + ``cost.record`` events under one ``@trace`` agent node.
    """
    import boto3
    from layerlens.instrument.adapters.providers.bedrock import (
        instrument_bedrock,
        uninstrument_bedrock,
    )

    model_id = os.environ.get("LL_BEDROCK_MODEL") or "amazon.nova-micro-v1:0"
    region = os.environ.get("AWS_REGION", "us-east-1")

    # Real sensor back-end the tool reads from (deterministic demo telemetry --
    # the tool returns genuine values, the forecast is the model's real output).
    sensor_db = {
        ("Z3-Industrial", "load_mw"): 5480,
        ("Z3-Industrial", "feeder_temp_c"): 74,
        ("Z1-Metro", "load_mw"): 4920,
        ("Z1-Metro", "feeder_temp_c"): 69,
    }

    def get_sensor_reading(zone: str, metric: str):
        value = sensor_db.get((zone, metric))
        return {"zone": zone, "metric": metric, "value": value,
                "unit": "MW" if metric == "load_mw" else "degC",
                "found": value is not None}

    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "get_sensor_reading",
                    "description": (
                        "Fetch the latest real-time SCADA sensor reading for a grid zone. "
                        "Use this to get the current load (MW) or feeder temperature (degC) "
                        "before forecasting."
                    ),
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "zone": {"type": "string",
                                         "description": "Grid zone id, e.g. Z3-Industrial"},
                                "metric": {"type": "string", "enum": ["load_mw", "feeder_temp_c"],
                                           "description": "Which sensor to read"},
                            },
                            "required": ["zone", "metric"],
                        }
                    },
                }
            }
        ]
    }
    system = (
        "You are grid-load-forecaster, an electrical grid load-forecasting agent. You do NOT "
        "have the current load in your prompt -- you MUST call the get_sensor_reading tool to "
        "read the live load for the zone before you forecast. After you have the reading, give a "
        "short next-hour load forecast for the zone and say whether it is within safe reserve "
        "margin (capacity for Z3-Industrial is 5600 MW; keep >=8% headroom)."
    )
    ask = "What is Z3-Industrial's load right now, and what's the next-hour forecast and risk?"

    client_bedrock = boto3.client("bedrock-runtime", region_name=region)
    instrument_bedrock(client_bedrock)

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        @trace(client, name="grid-load-forecaster", capture_config=_CAPTURE)
        def _forecast_with_tool(_ask):
            messages = [{"role": "user", "content": [{"text": _ask}]}]
            final_text = ""
            for _ in range(4):  # bounded tool loop
                resp = client_bedrock.converse(
                    modelId=model_id,
                    system=[{"text": system}],
                    messages=messages,
                    toolConfig=tool_config,
                    inferenceConfig={"maxTokens": 500, "temperature": 0.2},
                )
                out_msg = resp["output"]["message"]
                messages.append(out_msg)  # assistant turn (may carry toolUse)
                tool_uses = [b["toolUse"] for b in out_msg.get("content", [])
                             if isinstance(b, dict) and "toolUse" in b]
                if resp.get("stopReason") == "tool_use" and tool_uses:
                    tool_results = []
                    for tu in tool_uses:
                        result = get_sensor_reading(**(tu.get("input") or {}))
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tu["toolUseId"],
                                "content": [{"json": result}],
                                "status": "success",
                            }
                        })
                    messages.append({"role": "user", "content": tool_results})
                    continue
                final_text = "".join(
                    b.get("text", "") for b in out_msg.get("content", [])
                    if isinstance(b, dict)
                )
                break
            return final_text

        _forecast_with_tool(ask)
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
        uninstrument_bedrock()

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for energy-grid-tooluse")
    payload["tags"] = ["layerlens-sample", "industry", "energy", "grid-load-forecasting", "tool-use"]
    events = payload.get("events", [])
    invokes = [e for e in events if e.get("event_type") == "model.invoke"]
    tool_calls = [e for e in events if e.get("event_type") == "tool.call"]
    costs = [e for e in events if e.get("event_type") == "cost.record"]
    print("  energy-grid-tooluse (bedrock nova converse tool loop)  model.invoke=%d tool.call=%d cost.record=%d"
          % (len(invokes), len(tool_calls), len(costs)))
    print("  ->", _write([payload], "industry", "energy_grid_tooluse"), "\n")


# Single-agent travel itinerary planner: a REAL one-node langgraph StateGraph
# whose single 'itinerary-planner' node makes a real instrumented OpenAI call to
# plan a multi-city trip. The langgraph adapter captures the node identity
# (framework=langgraph) and the openai provider adapter captures the real
# model.invoke + cost.record (framework=openai, real tokens/cost) into the same
# trace -> a genuine SINGLE-agent langgraph trace WITH real LLM token/cost data
# (closing the no-LLM gap of the pure-python langgraph adapter sample).
TRAVEL_TRIP_REQUEST = {
    "trip_id": "TRIP-20418",
    "traveler": "solo, comfortable walker, moderate budget",
    "cities": ["Lisbon", "Barcelona"],
    "total_days": 7,
    "budget_usd": 2200,
    "interests": ["food", "architecture", "coastal walks"],
    "constraints": ["no red-eye flights", "one relaxed rest day"],
}


def generate_travel_itinerary(client: Stratix) -> None:
    """LangGraph SINGLE: a one-node 'itinerary-planner' StateGraph whose node
    makes a REAL instrumented OpenAI call to plan a multi-city trip.

    Renders a single-agent langgraph trace: the node identity comes from the
    LangGraphCallbackHandler (framework=langgraph) and the real model.invoke +
    cost.record come from the openai provider adapter (framework=openai, real
    tokens/cost) -- closing the langgraph+real-LLM token gap left by the
    pure-python langgraph adapter sample.
    """
    import json as _json
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from layerlens.instrument.adapters.frameworks.langgraph import LangGraphCallbackHandler

    oai = _handle("openai")
    request_json = _json.dumps(TRAVEL_TRIP_REQUEST)

    class S(TypedDict, total=False):
        request: str
        itinerary: str

    def itinerary_planner(s: S) -> S:  # openai -- the single agent's real LLM call
        r = oai.chat.completions.create(model=OPENAI_MODEL, messages=[
            {"role": "system", "content": "You are itinerary-planner, a travel itinerary planning "
             "agent. Given a trip request (cities, days, budget, interests, constraints), produce a "
             "concise day-by-day itinerary that respects the budget and constraints, allocates days "
             "across the cities, and names concrete neighborhoods/sights and one dinner idea per day. "
             "Note approximate cost so the total stays within budget. Be concise (under 250 words)."},
            {"role": "user", "content": s["request"]}])
        return {"itinerary": r.choices[0].message.content or ""}

    g = StateGraph(S)
    g.add_node("itinerary-planner", itinerary_planner)
    g.add_edge(START, "itinerary-planner")
    g.add_edge("itinerary-planner", END)
    graph = g.compile()

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        handler = LangGraphCallbackHandler(client, capture_config=_CAPTURE)
        graph.invoke({"request": request_json, "itinerary": ""}, config={"callbacks": [handler]})
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for travel-itinerary")
    payload["tags"] = ["layerlens-sample", "industry", "travel", "itinerary-planning"]
    nodes = sorted({e["payload"].get("node") for e in payload.get("events", [])
                    if e.get("event_type") == "agent.node.enter"})
    mi = sum(1 for e in payload.get("events", []) if e.get("event_type") == "model.invoke")
    print(f"  travel-itinerary  nodes={nodes}  model.invoke={mi}")
    print("  ->", _write([payload], "industry", "travel_itinerary"), "\n")


# ---- module-level constants (add near the other module consts, e.g. beside UNDERWRITING_APPLICATION) ----
# Shared clinical-decision-support scenario for the two LangChain healthcare
# fixtures (no real PHI).
_CDS_PATIENT = (
    "67M, chief complaint: crushing substernal chest pain radiating to the left arm "
    "with diaphoresis and dyspnea for 40 minutes. Vitals: HR 110, BP 90/60, SpO2 92%, "
    "RR 24. Active medications: metoprolol, lisinopril, atorvastatin, aspirin 81mg. "
    "History: hypertension, hyperlipidemia, prior MI (2023). Allergies: sulfa."
)
_CDS_KB = {
    "acute coronary syndrome": (
        "ACS / STEMI: crushing chest pain with diaphoresis and ST-elevation requires "
        "emergent cardiac catheterization within 90 minutes of first medical contact. "
        "Give aspirin 325mg, heparin, and dual antiplatelet therapy; obtain a 12-lead "
        "ECG and serial troponins. Contraindications to thrombolytics include active "
        "bleeding and recent surgery."
    ),
    "hypotension": (
        "Hypotension (SBP < 100) in suspected ACS suggests cardiogenic shock or "
        "right-ventricular involvement; use nitrates cautiously and consider fluids."
    ),
    "beta blocker": (
        "Beta-blockers (e.g. metoprolol) are held acutely in ACS if there is "
        "hypotension, bradycardia, or signs of heart failure/cardiogenic shock."
    ),
}


def generate_healthcare_clinical_chain(client: Stratix) -> None:
    """LangChain LCEL RAG chain (SINGLE, framework=langchain) for clinical
    decision support.

    A real ``retrieve -> ChatPromptTemplate -> ChatOpenAI -> StrOutputParser``
    LCEL chain, given a developer-declared ``run_name`` via
    ``.with_config(run_name='clinical-decision-support')``. Post the
    fabrication-fix, a bare ChatPromptTemplate/RunnableSequence renders blank in
    the Agent column; a genuine developer-declared run_name is the honest way to
    fill it. The real ``LangChainCallbackHandler`` records the real
    ``model.invoke`` / ``cost.record`` (framework=langchain) so the recorded
    trace renders one honest node ``clinical-decision-support``.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from layerlens.instrument.adapters.frameworks.langchain import LangChainCallbackHandler

    def _retrieve(case: str) -> str:
        # Deterministic keyword retrieval over the synthetic clinical KB. Called
        # from an anonymous lambda step so it stays plumbing inside the
        # ``clinical-decision-support`` node (langchain names an anonymous lambda
        # ``RunnableLambda``, which the adapter treats as a class default and
        # attributes to the enclosing declared run_name) - one honest node.
        q = case.lower()
        hits = [text for key, text in _CDS_KB.items() if any(w in q for w in key.split())]
        if not hits:
            hits = list(_CDS_KB.values())
        return "\n\n".join(f"- {h}" for h in hits[:3])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a clinical decision-support assistant (not a substitute for a "
                "clinician). Using ONLY the retrieved guidelines as evidence, give a brief "
                "differential, an ESI triage level (1-5), and any medication-safety "
                "cautions. Cite the guideline you relied on. Answer in under 150 words.",
            ),
            (
                "human",
                "Retrieved guidelines:\n{context}\n\nPatient case:\n{case}",
            ),
        ]
    )
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    chain = (
        RunnablePassthrough.assign(context=lambda x: _retrieve(x["case"]))
        | prompt
        | llm
        | StrOutputParser()
    ).with_config(run_name="clinical-decision-support")

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        handler = LangChainCallbackHandler(client, capture_config=_CAPTURE)
        chain.invoke({"case": _CDS_PATIENT}, config={"callbacks": [handler]})
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for healthcare-clinical-chain")
    payload["tags"] = ["layerlens-sample", "industry", "healthcare", "clinical-decision-support"]
    agents = sorted(
        {e["payload"].get("agent_name") for e in payload.get("events", []) if (e.get("payload") or {}).get("agent_name")}
    )
    fw = sorted(
        {e["payload"].get("framework") for e in payload.get("events", []) if (e.get("payload") or {}).get("framework")}
    )
    print(f"  healthcare-clinical-chain (langchain LCEL RAG)  agents={agents}  frameworks={fw}")
    print("  ->", _write([payload], "industry", "healthcare_clinical_chain"), "\n")


def generate_healthcare_clinical_agent(client: Stratix) -> None:
    """LangChain AgentExecutor (TOOL-USE, framework=langchain) for clinical
    decision support.

    A real ``create_tool_calling_agent`` + ``AgentExecutor`` with two real
    tools -- ``drug_interaction_check`` and ``guideline_lookup`` -- that a real
    ``ChatOpenAI`` (gpt-4o-mini) actually calls to work a patient case. The
    AgentExecutor is given a developer-declared ``run_name`` via
    ``.with_config(run_name='clinical-decision-support-agent')`` -- the honest
    way to fill the Agent column (the ``AgentExecutor`` class default renders
    blank). The real ``LangChainCallbackHandler`` records the real
    ``tool.call`` / ``tool.result`` and ``model.invoke`` events, so the trace
    renders the agent node plus its tool calls.

    REQUIRES source-fix #1 (add the agent output-parser class defaults to
    ``_LANGCHAIN_CLASS_DEFAULTS`` in frameworks/langchain.py). Without it the
    ``ToolsAgentOutputParser`` step leaks as a second, fabricated agent node.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain.agents.agent import RunnableMultiActionAgent
    from layerlens.instrument.adapters.frameworks.langchain import LangChainCallbackHandler

    _INTERACTIONS = {
        ("aspirin", "heparin"): "Additive bleeding risk; monitor for hemorrhage.",
        ("metoprolol", "aspirin"): "No significant interaction.",
        ("lisinopril", "aspirin"): "NSAIDs/aspirin may blunt ACE-inhibitor effect and worsen renal function.",
    }
    _GUIDELINES = {
        "acute coronary syndrome": (
            "ACS/STEMI: emergent PCI within 90 minutes; aspirin 325mg, heparin, dual "
            "antiplatelet therapy; 12-lead ECG and serial troponins."
        ),
        "chest pain": (
            "Undifferentiated chest pain: obtain ECG within 10 minutes, risk-stratify "
            "(HEART score), and rule out ACS, PE, and aortic dissection."
        ),
    }

    @tool
    def drug_interaction_check(medications: str) -> str:
        """Check a comma-separated list of medications for known interactions."""
        meds = [m.strip().lower() for m in medications.split(",") if m.strip()]
        found = []
        for (a, b), note in _INTERACTIONS.items():
            if a in meds and b in meds:
                found.append(f"{a} + {b}: {note}")
        return "\n".join(found) if found else "No known interactions among the listed medications."

    @tool
    def guideline_lookup(condition: str) -> str:
        """Look up the evidence-based clinical guideline for a condition."""
        key = condition.strip().lower()
        for name, text in _GUIDELINES.items():
            if name in key or key in name:
                return text
        return "No specific guideline found; apply standard triage protocol."

    tools = [drug_interaction_check, guideline_lookup]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a clinical decision-support assistant (not a substitute for a "
                "clinician). For the patient case, you MUST call guideline_lookup for the "
                "most likely condition AND drug_interaction_check for the active "
                "medications before answering. Then give a brief differential, an ESI "
                "triage level (1-5), and medication-safety cautions.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    agent = create_tool_calling_agent(llm, tools, prompt)
    # Disable runnable streaming so ``on_llm_end`` receives the full ``llm_output``
    # (model name + token usage). A streamed agent LLM run delivers an empty
    # ``llm_output`` -- the usage lands only on the message's ``usage_metadata``,
    # which the adapter's ``on_llm_end`` does not read -- so model/tokens/cost
    # would be lost. Non-streaming is a legitimate AgentExecutor configuration and
    # yields the real ``model.invoke`` + ``cost.record`` this fixture must carry.
    runnable_agent = RunnableMultiActionAgent(
        runnable=agent, input_keys_arg=["input"], return_keys_arg=["output"], stream_runnable=False
    )
    executor = AgentExecutor(agent=runnable_agent, tools=tools, verbose=False).with_config(
        run_name="clinical-decision-support-agent"
    )

    task = (
        f"Patient case: {_CDS_PATIENT}\n\n"
        "Look up the guideline for acute coronary syndrome, check the active "
        "medications (aspirin, heparin, metoprolol, lisinopril) for interactions, "
        "then give your triage read."
    )

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        handler = LangChainCallbackHandler(client, capture_config=_CAPTURE)
        executor.invoke({"input": task}, config={"callbacks": [handler]})
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for healthcare-clinical-agent")
    payload["tags"] = ["layerlens-sample", "industry", "healthcare", "clinical-decision-support", "tool-use"]
    tool_calls = sorted(
        {
            (e["payload"].get("name") or e["payload"].get("tool_name"))
            for e in payload.get("events", [])
            if e.get("event_type") == "tool.call" and (e["payload"].get("name") or e["payload"].get("tool_name"))
        }
    )
    agents = sorted(
        {e["payload"].get("agent_name") for e in payload.get("events", []) if (e.get("payload") or {}).get("agent_name")}
    )
    print(f"  healthcare-clinical-agent (langchain AgentExecutor)  agents={agents}  tool.calls={tool_calls}")
    print("  ->", _write([payload], "industry", "healthcare_clinical_agent"), "\n")


def main() -> None:
    client = Stratix()
    print("=== generating recorded real-trace fixtures ===\n")
    for spec in SPEC:
        mod = _load(spec["module"])
        records = getattr(mod, spec["const"], [])
        payloads = []
        for rec in records:
            user = spec["user"](rec)
            p = capture(client, agent_name=spec["agent"], backend=spec["backend"],
                        system=spec["system"], user=user, input_obj=rec, tags=spec["tags"])
            payloads.append(p)
            fw = next((e["payload"].get("framework") for e in p.get("events", [])
                       if isinstance(e.get("payload"), dict) and e["payload"].get("framework")), None)
            print(f"  {spec['stem']:24s} {spec['backend']:9s} events={len(p.get('events', []))} framework={fw}")
        path = _write(payloads, spec["category"], spec["stem"])
        print(f"  -> {path}\n")
    generate_rag(client, {})
    generate_incident_response(client)
    generate_underwriting_team(client)

    # ADP-W1 Family-B recorders. Framework-specific ones import their library
    # function-locally; run best-effort so a partial-framework env regenerates
    # whatever it can and skips the rest (a missing framework/cred is a skip, not
    # a crash) — mirroring how the other multi-agent team recorders are run.
    for _gen in (
        generate_media_content_moderation,
        generate_retail_catalog_qa,
        generate_energy_grid_forecast,
        generate_energy_grid_tooluse,
        generate_manufacturing_predictive_maintenance,
        generate_manufacturing_maintenance_tooluse,
        generate_healthcare_clinical_chain,
        generate_healthcare_clinical_agent,
        generate_legal_contract_rag,
        generate_legal_agentworkflow,
        generate_travel_itinerary,
    ):
        try:
            _gen(client)
        except Exception as _exc:  # noqa: BLE001 - a missing framework/cred is a skip
            print(f"  [skip] {_gen.__name__}: {type(_exc).__name__}: {_exc}")
    print("=== done ===")


if __name__ == "__main__":
    main()
