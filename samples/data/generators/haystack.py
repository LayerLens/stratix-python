"""ADP-W2 Family-B recorders for the **haystack** framework adapter.

Records two REAL, fully-instrumented ``haystack`` (2.x) pipeline runs and writes
each sealed trace to ``samples/data/traces/industry/<stem>.jsonl``. Both fixtures
are genuine runs of the real ``HaystackAdapter`` (global-tracer swap) over a real
``haystack.Pipeline`` whose ``OpenAIGenerator`` makes a real OpenAI call
(gpt-4o-mini) — nothing is fabricated. The framework deps (``haystack``) are
imported FUNCTION-LOCALLY so this module imports in any venv.

Two lanes (Legal domain; de-conflicted from the W1 ``legal_*`` stems):

* ``generate_haystack_single`` -> ``legal_haystack_clause_qa``
  A minimal single-component pipeline: one honest ``llm`` node
  (``OpenAIGenerator``) that interprets a specific contract clause and answers a
  plain-language question about it. Mirrors the ``test_haystack_recorded``
  single-generator shape. Renders one honest component node (``llm``) with a real
  ``model.invoke`` + priced ``cost.record`` + the pipeline ``agent.input`` /
  ``agent.output`` / ``environment.config`` (1 node).

* ``generate_haystack_multi`` -> ``legal_haystack_rag``
  A GENUINE multi-component RAG pipeline: ``retriever`` (BM25 over a contract-
  clause corpus) -> ``prompt_builder`` -> ``llm`` (``OpenAIGenerator``). Three
  honest component nodes + edges, so ``environment.config`` lists >=2 distinct
  producer-declared nodes and the trace renders a real component DAG.

  HONESTY / MARKED: haystack is a **pipeline framework, NOT a multi-agent one**.
  It emits no ``agent.identity`` and no ``agent.handoff`` — the honest multi-node
  topology is a *component* DAG (retriever -> prompt_builder -> llm), not an
  agent handoff/delegation graph. Per the ADP-W2 map + the honesty rules, the
  "multi" lane for this adapter is therefore a real multi-COMPONENT RAG pipeline
  (>=2 honest component nodes), NOT a multi-AGENT graph. This is marked in the
  payload metadata and tags. The Agent column renders honest empty-state (no
  agent identity is fabricated); the graph is the component DAG.

Both lanes record the real ``Pipeline.run`` path (haystack has no token-stream
delta path — the adapter reads the final component output at span end), and the
real ``OpenAIGenerator`` reports the model id echoed in the OpenAI *response*
(``gpt-4o-mini-*``), not the requested id — the tell that the real provider body
flowed through the real pipeline.
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
# This file is named ``haystack.py``; if a bare-script launch put its own
# directory on sys.path it would SHADOW the real ``haystack`` package. Drop our
# own dir so the function-local ``import haystack`` always resolves to the
# installed framework (a no-op in the integrated ``generators.haystack`` path).
sys.path[:] = [_q for _q in sys.path if os.path.abspath(_q) != _HERE]

from layerlens import Stratix  # noqa: E402  (re-exported for the seam signature)
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

# Reuse the shared record-real-once seam (_write / _CAPTURE) from the central
# fixture generator; fall back to a self-contained copy if it isn't importable.
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


OPENAI_MODEL = os.environ.get("SAMPLE_OPENAI_MODEL", "gpt-4o-mini")


def _capture_haystack(client: Stratix, pipeline, run_inputs: dict, *, tags: list) -> tuple:
    """Run a real ``haystack.Pipeline`` under the real ``HaystackAdapter`` +
    observer seam (no background upload) and return ``(sealed_payload, result)``.

    The adapter swaps the global ``haystack.tracing`` tracer, so each
    ``Pipeline.run`` opens/closes its own collector and the observer captures the
    sealed trace when ``_on_pipeline_end`` flushes the run.
    """
    from layerlens.instrument.adapters.frameworks.haystack import HaystackAdapter

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = HaystackAdapter(client, capture_config=_CAPTURE)
    try:
        adapter.connect()
        result = pipeline.run(run_inputs)
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for haystack pipeline run")
    payload["tags"] = list(tags)
    return payload, result


def _summarize(payload: dict) -> dict:
    events = payload.get("events", [])
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    env = next((e for e in events if e.get("event_type") == "environment.config"), None)
    components = sorted(
        {c.get("name") for c in ((env or {}).get("payload") or {}).get("components", []) if c.get("name")}
    )
    frameworks = sorted(
        {(e.get("payload") or {}).get("framework") for e in events if (e.get("payload") or {}).get("framework")}
    )
    models = sorted({(e.get("payload") or {}).get("model") for e in mi if (e.get("payload") or {}).get("model")})
    return {
        "n_events": len(events),
        "model.invoke": len(mi),
        "cost.record": len(cr),
        "components": components,
        "frameworks": frameworks,
        "models": models,
    }


# ---------------------------------------------------------------------------
# SINGLE: single-component clause-Q&A (one honest ``llm`` node)
# ---------------------------------------------------------------------------
_CONTRACT_CLAUSE = (
    "SECTION 9. LIMITATION OF LIABILITY. Except for the parties' indemnification "
    "obligations under Section 8 and either party's breach of its confidentiality "
    "obligations under Section 11, in no event shall either party's aggregate "
    "liability arising out of or related to this Agreement exceed the total fees "
    "paid or payable by Customer to Vendor under this Agreement during the twelve "
    "(12) months immediately preceding the event giving rise to the claim. In no "
    "event shall either party be liable for any indirect, incidental, "
    "consequential, special, or punitive damages, or for lost profits or loss of "
    "data, even if advised of the possibility of such damages."
)
_CLAUSE_QUESTION = (
    "In plain language: does this limitation-of-liability clause cap the Vendor's "
    "liability for a data breach caused by the Vendor, and are there any carve-outs "
    "that would let the cap be exceeded?"
)


def generate_haystack_single(client: Stratix) -> None:
    """Haystack SINGLE (single-component clause Q&A): a minimal pipeline with one
    honest ``llm`` node (``OpenAIGenerator``) that interprets a specific contract
    clause and answers a plain-language question about it. Real OpenAI
    (gpt-4o-mini), recorded under the real ``HaystackAdapter`` -> one honest
    component node with a real ``model.invoke`` + priced ``cost.record``."""
    from haystack import Pipeline
    from haystack.components.generators.openai import OpenAIGenerator

    prompt = (
        "You are a contract-clause assistant for an in-house legal team. Read the "
        "contract clause below and answer the question concisely and accurately, "
        "grounding your answer ONLY in the clause text. Do not invent terms.\n\n"
        "CLAUSE:\n%s\n\nQUESTION: %s\n\nAnswer (under 150 words):" % (_CONTRACT_CLAUSE, _CLAUSE_QUESTION)
    )

    pipe = Pipeline()
    pipe.add_component(
        "llm",
        OpenAIGenerator(model=OPENAI_MODEL, generation_kwargs={"max_tokens": 300, "temperature": 0.2}),
    )

    payload, result = _capture_haystack(
        client,
        pipe,
        {"llm": {"prompt": prompt}},
        tags=["layerlens-sample", "industry", "legal", "contract-clause-qa", "haystack"],
    )

    s = _summarize(payload)
    reply = ""
    try:
        reply = result["llm"]["replies"][0]
    except Exception:
        pass
    print("  haystack-single (clause Q&A, single llm node)  components=%s frameworks=%s models=%s "
          "model.invoke=%d cost.record=%d" % (s["components"], s["frameworks"], s["models"],
                                              s["model.invoke"], s["cost.record"]))
    print("    reply=%r" % (str(reply)[:120],))
    print("  ->", _write([payload], "industry", "legal_haystack_clause_qa"), "\n")


# ---------------------------------------------------------------------------
# MULTI (multi-COMPONENT RAG — haystack is NOT multi-agent): contract-clause RAG
# ---------------------------------------------------------------------------
# A small, real contract-clause corpus the BM25 retriever ranks over. Genuine
# tool output; the answer is the model's real reasoning over the retrieved text.
_CONTRACT_CLAUSE_CORPUS = [
    "SECTION 8. INDEMNIFICATION. Vendor shall defend, indemnify, and hold harmless "
    "Customer from and against any third-party claims, damages, and reasonable "
    "attorneys' fees arising out of (a) Vendor's breach of its data-security "
    "obligations, or (b) a security incident or data breach caused by Vendor's "
    "software or Vendor's gross negligence. This indemnity is not subject to the "
    "limitation of liability in Section 9.",
    "SECTION 9. LIMITATION OF LIABILITY. Except for the indemnification obligations "
    "under Section 8 and breaches of confidentiality under Section 11, each party's "
    "aggregate liability shall not exceed the fees paid in the twelve (12) months "
    "preceding the claim, and neither party shall be liable for indirect or "
    "consequential damages.",
    "SECTION 11. CONFIDENTIALITY. Each party shall protect the other's Confidential "
    "Information using at least reasonable care and shall not disclose it except to "
    "personnel with a need to know who are bound by confidentiality obligations.",
    "SECTION 13. TERM AND TERMINATION. This Agreement renews for successive one-year "
    "terms unless either party gives sixty (60) days' written notice of non-renewal. "
    "Customer may terminate for convenience on thirty (30) days' notice.",
    "SECTION 15. GOVERNING LAW. This Agreement is governed by the laws of the State "
    "of Delaware, without regard to its conflict-of-laws principles, and the parties "
    "consent to the exclusive jurisdiction of the state and federal courts located "
    "in Wilmington, Delaware.",
    "SECTION 7. DATA SECURITY. Vendor shall maintain administrative, physical, and "
    "technical safeguards for the protection of Customer Data that meet or exceed "
    "SOC 2 Type II controls, and shall notify Customer of any confirmed data breach "
    "within seventy-two (72) hours.",
]
_RAG_QUESTION = (
    "If Vendor's software causes a customer data breach, what does the "
    "indemnification clause require of the Vendor, and does the limitation-of-"
    "liability cap apply to that indemnification obligation?"
)
_RAG_TEMPLATE = (
    "You are a contract-analysis assistant. Using ONLY the contract excerpts below, "
    "answer the question. Cite the relevant section numbers. If the excerpts do not "
    "address the question, say so — do not invent terms.\n\n"
    "EXCERPTS:\n"
    "{% for doc in documents %}- {{ doc.content }}\n{% endfor %}\n"
    "QUESTION: {{ question }}\n\nAnswer (under 180 words):"
)


def generate_haystack_multi(client: Stratix) -> None:
    """Haystack MULTI == genuine multi-COMPONENT RAG pipeline (MARKED).

    haystack is a pipeline framework, not a multi-agent one: it emits no
    ``agent.identity`` / ``agent.handoff``. Per the ADP-W2 map + honesty rules,
    this "multi" lane is a REAL multi-component RAG DAG — ``retriever`` (BM25 over
    a contract-clause corpus) -> ``prompt_builder`` -> ``llm`` (``OpenAIGenerator``)
    — so ``environment.config`` lists >=2 distinct producer-declared component
    nodes and the trace renders a real component DAG (NOT an agent handoff graph).
    The trace carries the retriever ``tool.call`` / ``tool.result``, the real
    ``model.invoke`` + priced ``cost.record``, and the pipeline lifecycle."""
    from haystack import Document, Pipeline
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators.openai import OpenAIGenerator

    store = InMemoryDocumentStore()
    store.write_documents([Document(content=c) for c in _CONTRACT_CLAUSE_CORPUS])

    pipe = Pipeline()
    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=3))
    pipe.add_component("prompt_builder", PromptBuilder(template=_RAG_TEMPLATE, required_variables=["question"]))
    pipe.add_component(
        "llm",
        OpenAIGenerator(model=OPENAI_MODEL, generation_kwargs={"max_tokens": 350, "temperature": 0.2}),
    )
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    payload, result = _capture_haystack(
        client,
        pipe,
        {"retriever": {"query": _RAG_QUESTION}, "prompt_builder": {"question": _RAG_QUESTION}},
        tags=["layerlens-sample", "industry", "legal", "contract-clause-rag", "haystack", "multi-component"],
    )
    # HONEST provenance: this is a multi-COMPONENT pipeline DAG, NOT a multi-agent
    # handoff graph (haystack emits no agent.identity / agent.handoff).
    payload["metadata"] = {
        "topology": "haystack-pipeline-component-dag",
        "reason": "haystack is a pipeline framework, not multi-agent: the honest multi-node "
                  "topology is a component DAG (retriever -> prompt_builder -> llm). It emits "
                  "no agent.identity / agent.handoff, so the trace carries >=2 honest component "
                  "nodes (environment.config) rather than agent nodes/handoff edges.",
    }

    s = _summarize(payload)
    reply = ""
    try:
        reply = result["llm"]["replies"][0]
    except Exception:
        pass
    print("  haystack-multi (multi-component RAG DAG; NO agents/handoff)  components=%s frameworks=%s "
          "models=%s model.invoke=%d cost.record=%d" % (s["components"], s["frameworks"], s["models"],
                                                        s["model.invoke"], s["cost.record"]))
    print("    reply=%r" % (str(reply)[:120],))
    print("  ->", _write([payload], "industry", "legal_haystack_rag"), "\n")


if __name__ == "__main__":
    _client = Stratix()
    generate_haystack_single(_client)
    generate_haystack_multi(_client)
