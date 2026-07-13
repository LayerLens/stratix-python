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
    print("=== done ===")


if __name__ == "__main__":
    main()
