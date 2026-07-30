"""ADP-W2 Family-B recorders for the **embedding / vector_store** adapters.

Records two REAL, fully-instrumented retrieval traces and writes each sealed
trace to ``samples/data/traces/industry/<stem>.jsonl``. Both fixtures are genuine
runs of the real ``VectorStoreAdapter`` (single) and the real ``EmbeddingAdapter``
+ ``VectorStoreAdapter`` (multi) over a real in-process Chroma collection whose
vectors are REAL OpenAI ``text-embedding-3-small`` embeddings of an insurance
policy-document corpus — nothing is fabricated. The framework deps (``openai``,
``chromadb``) are imported FUNCTION-LOCALLY so this module imports in any venv.

HONESTY / EMPTY-STATE (MARKED): the embedding + vector_store adapters are
**metadata-only, NON-agentic** cross-cutting instrumentation. They emit ONLY
``embedding.create`` / ``retrieval.query`` events — no ``agent.identity`` /
``agent.handoff`` / ``model.invoke`` / ``cost.record``. So the honest render is an
**empty-state** (Agent column = "—") + an event waterfall, NOT an agent DAG. The
recorded traces carry no fabricated agent; the atlas graph engine derives zero
nodes (mirrors the ``embedding-s1`` / ``vector_store-s1`` graph-honesty fixtures).
Neither adapter emits ``cost_usd`` (HELD design gap: embeddings carry real
``total_tokens`` but no priced cost.record), so these fixtures honestly carry NO
cost — the normal recorded path, not a fabricated cost.

Two lanes (Insurance domain; de-conflicted from the W1 ``insurance_*`` stems):

* ``generate_embedding_single`` -> ``insurance_policy_retrieval``
  A single semantic retrieval over the policy-document corpus using the real
  ``VectorStoreAdapter`` over Chroma. The corpus + query are embedded with REAL
  OpenAI embeddings at index/setup time (untraced), then ONE traced
  ``collection.query`` emits a single ``retrieval.query`` event with the REAL
  match count + REAL cosine-distance summary the Chroma engine computed. Renders
  honest empty-state + a 2-event waterfall (retrieval.query + trace.root),
  mirroring ``vector_store-s1``.

* ``generate_embedding_multi`` -> ``insurance_policy_rag``
  A genuine embed->retrieve RAG **loop** (the retrieval half of RAG): for each
  policyholder question the real ``EmbeddingAdapter`` embeds the query
  (``embedding.create``, real OpenAI, 1536-D, real ``total_tokens``) and the real
  ``VectorStoreAdapter`` retrieves the nearest policy clauses (``retrieval.query``,
  real Chroma distances). MARKED ``genuinely_multi_agent=false``: this is a
  retrieval loop across two cross-cutting adapters, NOT a multi-agent graph — the
  honest render is still empty-state + a waterfall of the real embed/retrieve
  events, no fabricated agents.

The strong tells the real path flowed through: ``dimensions`` == the real 1536-D
vector the OpenAI SDK returned, ``total_tokens`` == the real ``usage.total_tokens``
per query, and the distance summary == the real cosine distances the Chroma HNSW
index computed over the OpenAI-embedded corpus.
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

from layerlens import Stratix  # noqa: E402  (re-exported for the seam signature)
from layerlens.instrument import trace_context  # noqa: E402
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


EMBED_MODEL = os.environ.get("SAMPLE_EMBED_MODEL", "text-embedding-3-small")


# ---------------------------------------------------------------------------
# Insurance policy-document corpus (real personal-auto + home policy language).
# Genuine clause text; the vectors added to Chroma are REAL OpenAI embeddings.
# ---------------------------------------------------------------------------
_POLICY_CORPUS = [
    ("collision",
     "COLLISION COVERAGE. We will pay for direct and accidental loss to your covered auto "
     "caused by collision with another vehicle or object, or by upset of your covered auto, "
     "less the applicable collision deductible shown in the Declarations."),
    ("comprehensive",
     "OTHER THAN COLLISION (COMPREHENSIVE) COVERAGE. We will pay for direct and accidental "
     "loss to your covered auto not caused by collision, including loss caused by fire, theft "
     "or larceny, falling objects, hail, flood, vandalism, or contact with a bird or animal, "
     "less the applicable comprehensive deductible."),
    ("deductible",
     "DEDUCTIBLE. The deductible is the amount you pay out of pocket for each covered loss "
     "before we pay. This policy carries a 500 dollar collision deductible and a 250 dollar "
     "comprehensive deductible. A separate deductible applies to each covered auto."),
    ("rental_reimbursement",
     "RENTAL REIMBURSEMENT. If your covered auto is withdrawn from use for more than 24 hours "
     "due to a covered loss, we will reimburse rental-car expenses up to 40 dollars per day "
     "and 1,200 dollars per loss while repairs are being completed."),
    ("liability",
     "LIABILITY COVERAGE. We will pay damages for bodily injury or property damage for which "
     "any insured becomes legally responsible because of an auto accident, up to the limits of "
     "liability shown in the Declarations, and will settle or defend any resulting claim."),
    ("uninsured_motorist",
     "UNINSURED / UNDERINSURED MOTORIST COVERAGE. We will pay compensatory damages an insured "
     "is legally entitled to recover from the owner or operator of an uninsured or underinsured "
     "motor vehicle because of bodily injury sustained in an accident."),
    ("exclusions",
     "GENERAL EXCLUSIONS. This policy does not cover loss caused by wear and tear, freezing, "
     "mechanical or electrical breakdown, road damage to tires, intentional damage, racing or "
     "speed contests, or use of a vehicle to carry persons or property for a fee."),
    ("glass",
     "GLASS COVERAGE. Loss to window glass is covered under Comprehensive. If you carry "
     "full-glass coverage, the comprehensive deductible is waived for the repair or replacement "
     "of a damaged windshield or other safety glass."),
    ("towing",
     "TOWING AND LABOR. We will pay towing and labor costs incurred each time your covered auto "
     "is disabled, up to 75 dollars per disablement, provided the labor is performed at the "
     "place of disablement."),
    ("home_water",
     "HOMEOWNERS WATER DAMAGE. We insure for accidental discharge or overflow of water from a "
     "plumbing, heating, or air-conditioning system. We do not insure for loss caused by "
     "continuous or repeated seepage over a period of weeks, or by flood or surface water."),
    ("home_theft",
     "HOMEOWNERS THEFT. We insure for loss by theft of covered personal property, subject to a "
     "1,000 dollar special limit for jewelry, watches, and furs, and a 2,500 dollar special "
     "limit for firearms taken by theft."),
    ("claim_filing",
     "DUTIES AFTER A LOSS. In the event of a covered loss you must promptly notify us, protect "
     "the property from further damage, cooperate in the investigation, and submit a signed, "
     "sworn proof of loss within 60 days of our request."),
]


def _embed_texts(oc, texts):
    """REAL OpenAI embeddings for *texts* -> list of vectors (untraced setup)."""
    resp = oc.embeddings.create(model=EMBED_MODEL, input=list(texts))
    return [d.embedding for d in resp.data]


def _build_index(oc):
    """Build a REAL in-process Chroma collection over the policy corpus, whose
    vectors are REAL OpenAI embeddings. Runs at setup (untraced)."""
    import uuid

    import chromadb

    cc = chromadb.EphemeralClient()
    # Unique name: chromadb shares process-global state across EphemeralClient
    # instances, so a fixed name collides when both lanes build in one process.
    coll = cc.create_collection(
        name=f"insurance_policy_kb_{uuid.uuid4().hex[:8]}", metadata={"hnsw:space": "cosine"}
    )
    vecs = _embed_texts(oc, [c[1] for c in _POLICY_CORPUS])
    coll.add(
        ids=[c[0] for c in _POLICY_CORPUS],
        embeddings=vecs,
        documents=[c[1] for c in _POLICY_CORPUS],
    )
    return coll


def _capture(client: Stratix, run_fn, *, tags: list, metadata: dict | None = None) -> dict:
    """Run *run_fn* inside a real ``trace_context`` (which establishes the
    collector + span and flushes on exit) under the observer seam WITHOUT the
    background upload, and return the sealed trace payload.

    The embedding / vector_store adapters emit into ``_current_collector``, so
    inside ``trace_context`` their events land in one trace; on flush the
    collector synthesizes ONE content-free ``trace.root`` (no agent is
    fabricated) and the observer captures the sealed, attested payload."""
    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        with trace_context(client, capture_config=_CAPTURE):
            run_fn()
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for embedding/vector_store run")
    payload["tags"] = list(tags)
    if metadata:
        payload["metadata"] = metadata
    return payload


def _summarize(payload: dict) -> dict:
    events = payload.get("events", [])
    emb = [e for e in events if e.get("event_type") == "embedding.create"]
    ret = [e for e in events if e.get("event_type") == "retrieval.query"]
    agents = [e for e in events if e.get("event_type") in ("agent.identity", "agent.handoff")]
    return {
        "n_events": len(events),
        "embedding.create": len(emb),
        "retrieval.query": len(ret),
        "agent_nodes": len(agents),  # MUST be 0 (honest empty-state)
        "dims": sorted({(e.get("payload") or {}).get("dimensions") for e in emb if (e.get("payload") or {}).get("dimensions")}),
        "providers": sorted({(e.get("payload") or {}).get("provider") for e in events if (e.get("payload") or {}).get("provider")}),
    }


# ---------------------------------------------------------------------------
# SINGLE: policy-document semantic retrieval (VectorStoreAdapter only)
# ---------------------------------------------------------------------------
_SINGLE_QUESTION = (
    "A deer ran into the side of my car on a rural road and dented the door. Is that "
    "damage covered, and what deductible would apply?"
)


def generate_embedding_single(client: Stratix) -> None:
    """Embedding SINGLE == a single semantic retrieval over an insurance
    policy-document corpus via the real ``VectorStoreAdapter`` over Chroma. The
    corpus + query are embedded with REAL OpenAI embeddings (untraced setup);
    ONE traced ``collection.query`` emits a single ``retrieval.query`` event with
    the REAL match count + REAL cosine distances. Honest empty-state render
    (Agent = "—") + a 2-event waterfall — mirrors ``vector_store-s1``."""
    import openai

    from layerlens.instrument.adapters.frameworks.vector_store import VectorStoreAdapter

    oc = openai.OpenAI()
    coll = _build_index(oc)
    # Embed the query at setup (untraced) so the SINGLE lane captures ONLY the
    # retrieval surface (retrieval.query), exactly like the vector_store-s1 lane.
    qvec = _embed_texts(oc, [_SINGLE_QUESTION])[0]

    adapter = VectorStoreAdapter(None, capture_config=_CAPTURE)
    adapter.connect(coll)  # auto-detects Chroma, wraps collection.query

    def _run():
        coll.query(query_embeddings=[qvec], n_results=4)

    try:
        payload = _capture(
            client,
            _run,
            tags=["layerlens-sample", "industry", "insurance", "policy-retrieval",
                  "vector_store", "empty-state"],
            metadata={
                "topology": "vector-store-retrieval-empty-state",
                "reason": "vector_store is a metadata-only, non-agentic retrieval adapter: it "
                          "emits retrieval.query only (no agent.identity/handoff/model.invoke). "
                          "The honest render is empty-state (Agent = '—') + a waterfall, NOT an "
                          "agent DAG. genuinely_multi_agent=false.",
                "genuinely_multi_agent": False,
            },
        )
    finally:
        adapter.disconnect()

    s = _summarize(payload)
    assert s["retrieval.query"] == 1, s
    assert s["agent_nodes"] == 0, "empty-state fixture must fabricate NO agent node"
    print("  embedding-single (policy semantic retrieval; empty-state)  events=%d retrieval.query=%d "
          "providers=%s agent_nodes=%d" % (s["n_events"], s["retrieval.query"], s["providers"], s["agent_nodes"]))
    print("  ->", _write([payload], "industry", "insurance_policy_retrieval"), "\n")


# ---------------------------------------------------------------------------
# MULTI: embed->retrieve RAG loop (EmbeddingAdapter + VectorStoreAdapter)
# ---------------------------------------------------------------------------
_MULTI_QUESTIONS = [
    "A rock cracked my windshield on the highway. Is the glass covered and do I pay a deductible?",
    "I backed into a pole in a parking lot and dented the bumper. What deductible applies?",
    "My covered car was stolen from my driveway overnight. Which coverage handles theft?",
    "A pipe burst and water flooded my basement. Does my homeowners policy cover that?",
]


def generate_embedding_multi(client: Stratix) -> None:
    """Embedding MULTI == a genuine embed->retrieve RAG loop (MARKED
    ``genuinely_multi_agent=false``). For each policyholder question the real
    ``EmbeddingAdapter`` embeds the query (``embedding.create``, real OpenAI,
    1536-D) and the real ``VectorStoreAdapter`` retrieves the nearest policy
    clauses (``retrieval.query``, real Chroma cosine distances). This is a
    retrieval loop across two cross-cutting adapters, NOT a multi-agent graph:
    the honest render is empty-state + a waterfall of the real embed/retrieve
    events, no fabricated agents."""
    import openai

    from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter
    from layerlens.instrument.adapters.frameworks.vector_store import VectorStoreAdapter

    oc = openai.OpenAI()
    coll = _build_index(oc)  # untraced: build the KB before wrapping

    emb_adapter = EmbeddingAdapter(None, capture_config=_CAPTURE)
    emb_adapter.connect(oc)  # wraps oc.embeddings.create
    vs_adapter = VectorStoreAdapter(None, capture_config=_CAPTURE)
    vs_adapter.connect(coll)  # wraps coll.query

    def _run():
        for q in _MULTI_QUESTIONS:
            # embed the query (traced) -> retrieve nearest policy clauses (traced)
            r = oc.embeddings.create(model=EMBED_MODEL, input=q)
            qvec = r.data[0].embedding
            coll.query(query_embeddings=[qvec], n_results=3)

    try:
        payload = _capture(
            client,
            _run,
            tags=["layerlens-sample", "industry", "insurance", "policy-rag",
                  "embedding", "vector_store", "empty-state", "retrieval-loop"],
            metadata={
                "topology": "embed-retrieve-rag-loop-empty-state",
                "reason": "embedding + vector_store are metadata-only, non-agentic cross-cutting "
                          "adapters. This 'multi' lane is a genuine embed->retrieve RAG retrieval "
                          "loop (embedding.create -> retrieval.query per question), NOT a "
                          "multi-agent graph. The honest render is empty-state (Agent = '—') + a "
                          "waterfall of the real embed/retrieve events; no agent is fabricated.",
                "genuinely_multi_agent": False,
            },
        )
    finally:
        vs_adapter.disconnect()
        emb_adapter.disconnect()

    s = _summarize(payload)
    n = len(_MULTI_QUESTIONS)
    assert s["embedding.create"] == n and s["retrieval.query"] == n, s
    assert s["agent_nodes"] == 0, "empty-state fixture must fabricate NO agent node"
    assert s["dims"] == [1536], s  # real OpenAI text-embedding-3-small vector length
    print("  embedding-multi (embed->retrieve RAG loop; empty-state, NOT multi-agent)  events=%d "
          "embedding.create=%d retrieval.query=%d dims=%s providers=%s agent_nodes=%d"
          % (s["n_events"], s["embedding.create"], s["retrieval.query"], s["dims"], s["providers"], s["agent_nodes"]))
    print("  ->", _write([payload], "industry", "insurance_policy_rag"), "\n")


if __name__ == "__main__":
    _client = Stratix()
    generate_embedding_single(_client)
    generate_embedding_multi(_client)
