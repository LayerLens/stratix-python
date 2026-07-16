"""ADP-PORT Family-B recorder for the ``openinference`` INGESTION adapter
(record-real-once).

WHAT MAKES THIS A REAL RUN
--------------------------
``openinference`` is an *ingestion* adapter: it patches nothing and makes no LLM
call of its own — it CONSUMES OpenTelemetry spans that an OpenInference
auto-instrumentor already produced. So the honest "real run" for it is not "call
a model", it is: **drive a real workload under real OpenInference/OTel
instrumentation, export the real spans, and feed those real spans to the
adapter.** That is exactly what this module does — nothing is hand-authored:

* The LLM span is produced by the REAL ``openinference-instrumentation-openai``
  ``OpenAIInstrumentor``, auto-patching a REAL ``openai.OpenAI`` client that makes
  a REAL ``chat.completions.create`` call to OpenAI ``gpt-4o-mini``. The span's
  ``llm.model_name`` (the resolved dated id), ``llm.token_count.*``,
  ``llm.provider`` and input/output values are whatever that real call produced.
* The AGENT / TOOL / RETRIEVER spans are produced by the REAL OpenInference
  ``OITracer`` wrapping this app's own steps, with every attribute built by the
  OpenInference library's OWN helpers (``get_span_kind_attributes`` /
  ``get_retriever_attributes`` / ``get_tool_attributes`` / ``oi.Document``) — i.e.
  app-level instrumentation exactly as an OpenInference-instrumented app writes
  it, NOT string literals typed by hand. Their timings, statuses and payloads are
  the real executed steps'.
* Both artifacts below come from ONE run of that single workload.

TWO SEALED ARTIFACTS, ONE RUN
-----------------------------
1. ``samples/data/traces/industry/retail_openinference_support.jsonl`` — the
   sealed LayerLens trace. Captured through the adapter's LIVE production wiring
   (``provider.add_span_processor(adapter.span_processor())``), observed via the
   ``_generate_fixtures`` capture seam (``set_trace_observer`` + a no-op
   ``enqueue_upload``) so regeneration NEVER uploads or pollutes an org.
2. ``tests/fixtures/recorded/openinference/default.json`` — the same real spans
   sealed as OTLP/JSON, encoded by the REAL OpenTelemetry OTLP encoder
   (``opentelemetry.exporter.otlp.proto.common``) with ids hex-transcoded per the
   OTLP/JSON spec. This is the adapter's genuine *upstream input*, which
   ``tests/instrument/adapters/frameworks/test_openinference_recorded.py`` replays
   — honoring the corpus rule: record UPSTREAM of the parser, assert DOWNSTREAM.

THE SCENARIO (Retail)
---------------------
A footwear e-commerce support assistant answers a real warranty-vs-return-window
question: a customer's boots split at the seam ~4 months after delivery. The
30-day return window has expired, but the 12-month manufacturing-defect warranty
still applies and makes return shipping free — so the answer is only correct if
retrieval surfaces the right policy. A genuine RAG business task, not a toy.

HONEST RENDER NOTE (no fabricated agent)
----------------------------------------
The Agent column renders honest EMPTY-STATE. An OpenInference AGENT span declares
its identity only as a span NAME, and ``_identity.py`` deliberately forbids a span
name as an Agent-column source, so the adapter keeps the name in ``agent_id`` and
never writes ``agent_name``. No agent is invented. Framework renders
``openinference``; Status comes from the real span statuses; the model/token
fields are the real call's, and ``cost.record`` is really derived from those real
tokens (gpt-4o-mini is priced) — never a fabricated 0.0.
"""

from __future__ import annotations

import os
import sys
import json
import base64
from datetime import datetime, timezone

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config + model name).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)  # samples/data
_SAMPLES = os.path.dirname(_DATA)  # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# This module is named ``openinference.py`` (to match the adapter), and the REAL
# ``openinference`` distribution is a NAMESPACE package — a regular module on
# sys.path SHADOWS it. When this file is run directly Python puts its own
# directory at sys.path[0], which would make the function-local ``import
# openinference.instrumentation`` resolve to *this file* and fail. Drop this
# module's own directory from the path so the framework import always resolves to
# the installed package (a no-op when imported as ``generators.openinference``).
sys.path[:] = [_p for _p in sys.path if os.path.abspath(_p or ".") != _HERE]

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE
OPENAI_MODEL = _gf.OPENAI_MODEL

# tests/fixtures/recorded/openinference/default.json — the recorded-corpus twin.
_CORPUS = os.path.join(_REPO, "tests", "fixtures", "recorded", "openinference", "default.json")


# --------------------------------------------------------------------------
# The REAL retail knowledge base + order book the instrumented steps read from.
# Small, realistic, non-sensitive store policy — the retrieval below is a real
# lexical search over it, and the model answers strictly from what it returns.
# --------------------------------------------------------------------------
_POLICY_KB = [
    (
        "POL-RET-01",
        "Returns & Refunds",
        "Unworn items in original packaging may be returned within 30 days of "
        "delivery for a full refund to the original payment method. Items marked "
        "final sale are not returnable.",
    ),
    (
        "POL-SHIP-02",
        "Shipping",
        "Standard shipping is 3-5 business days. Orders over $75 ship free. "
        "Expedited shipping is $12.95 and delivers in 1-2 business days.",
    ),
    (
        "POL-RET-03",
        "Return Shipping Costs",
        "Return shipping is free for items that arrive damaged, incorrect, or "
        "that fail under warranty. For all other returns a $6.99 prepaid label "
        "fee is deducted from the refund.",
    ),
    (
        "POL-WAR-04",
        "Footwear Defect Warranty",
        "Manufacturing defects such as split seams, delamination, or sole "
        "separation are covered for 12 months from the delivery date. Covered "
        "defects are replaced or refunded regardless of the 30-day return "
        "window, and are not treated as ordinary wear and tear.",
    ),
    (
        "POL-WAR-05",
        "Warranty Exclusions",
        "The defect warranty does not cover ordinary wear of outsoles or "
        "footbeds, damage from improper care, or items purchased final sale.",
    ),
]

_ORDER_BOOK = {
    "SO-884213": {
        "order_id": "SO-884213",
        "sku": "SUMMIT-TRAIL-9M",
        "item": "Summit Trail Waterproof Hiking Boot, M9",
        "price_usd": 189.00,
        "ordered_on": "2026-03-14",
        "delivered_on": "2026-03-19",
        "final_sale": False,
        "channel": "web",
    }
}

_QUESTION = (
    "I ordered the Summit Trail boots back in March — about four months ago — and "
    "the seam along the left heel has completely split open. They were not final "
    "sale. Can I still get a refund, and do I have to pay for the return shipping?"
)


def _lookup_order(order_id: str) -> dict:
    """REAL tool fn: fetch the order record (item, dates, final-sale flag)."""
    rec = _ORDER_BOOK.get((order_id or "").strip().upper())
    if rec is None:
        return {"order_id": order_id, "found": False, "message": "No such order."}
    return dict(rec, found=True)


def _search_policies(query: str, top_k: int = 3) -> list:
    """REAL retrieval: lexical overlap scoring over the store policy corpus.

    Returns ``[(score, doc_id, title, body)]`` best-first. A real (if simple)
    scorer over a real corpus — the model only sees what this actually returns.
    """
    tokens = [t for t in query.lower().replace("?", " ").replace(",", " ").split() if len(t) > 3]
    scored = []
    for doc_id, title, body in _POLICY_KB:
        haystack = (title + " " + body).lower()
        score = sum(haystack.count(tok) for tok in tokens)
        if score:
            scored.append((score, doc_id, title, body))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored[:top_k]


# --------------------------------------------------------------------------
# OTLP/JSON sealing — the REAL OTel encoder, ids hex-transcoded per the spec.
# --------------------------------------------------------------------------
def _b64_to_hex(value: str) -> str:
    """protobuf's MessageToDict base64s ``bytes``; OTLP/JSON mandates lower-hex."""
    return base64.b64decode(value).hex()


def _otlp_json(spans) -> dict:
    """Encode real ReadableSpans to OTLP/JSON with the REAL OTel encoder.

    ``MessageToDict`` renders protobuf ``bytes`` as base64, but the OTLP/JSON
    spec (and every real collector export, incl. atlas's own otlp-fixtures)
    carries trace/span ids as lower-case hex — so the id fields are transcoded.
    Everything else is the encoder's own output, untouched.
    """
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
    from google.protobuf.json_format import MessageToDict

    doc = MessageToDict(encode_spans(spans))
    for rs in doc.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                for key in ("traceId", "spanId", "parentSpanId"):
                    if sp.get(key):
                        sp[key] = _b64_to_hex(sp[key])
    return doc


def _iter_otlp_spans(doc: dict):
    """Yield each span dict out of an OTLP/JSON document (the adapter's input)."""
    for rs in doc.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                yield sp


def _write_corpus(doc: dict, model: str, instrumentor_version: str, scenario: str = "default") -> str:
    """Seal the real spans as the recorded-corpus fixture (transport=object)."""
    fixture = {
        "provenance": {
            "provider": "openinference",
            "sdk_version": instrumentor_version,
            "model": model,
            "scenario": scenario,
            "captured_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        },
        "transport": "object",
        "sdk": "openinference",
        "note": (
            "REAL OpenInference/OTel spans exported from one real instrumented "
            "retail-support run (openinference-instrumentation-openai "
            "auto-instrumenting real openai gpt-4o-mini calls, plus OITracer "
            "app-level AGENT/TOOL/RETRIEVER spans), encoded to OTLP/JSON by the "
            "real OTel OTLP encoder. This is the adapter's upstream INPUT."
        ),
        "otlp": doc,
    }
    path = os.path.join(os.path.dirname(_CORPUS), "%s.json" % scenario)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2, sort_keys=False)
        f.write("\n")
    return path


# --------------------------------------------------------------------------
# The real instrumented run
# --------------------------------------------------------------------------
def generate_openinference_single(client: Stratix) -> dict:
    """Record ONE real OpenInference-instrumented retail-support RAG run.

    Drives the workload under a real OTel ``TracerProvider`` carrying BOTH the
    adapter's live ``span_processor()`` (-> the sealed LayerLens trace) and an
    ``InMemorySpanExporter`` (-> the sealed OTLP/JSON span corpus), so the two
    shipped artifacts are two views of the SAME real spans.
    """
    from importlib.metadata import version

    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from openinference.instrumentation.openai import OpenAIInstrumentor
    import openinference.instrumentation as oi
    from openinference.semconv.trace import OpenInferenceSpanKindValues
    from openai import OpenAI

    from layerlens.instrument.adapters.frameworks.openinference import OpenInferenceAdapter

    exporter = InMemorySpanExporter()
    provider = SDKTracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig_upload = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None

    instrumentor = OpenAIInstrumentor()
    adapter = OpenInferenceAdapter(client, capture_config=_CAPTURE)
    adapter.connect()
    # The adapter's LIVE production wiring: it is an OTel SpanProcessor.
    provider.add_span_processor(adapter.span_processor())

    answer = ""
    try:
        # The REAL OpenInference auto-instrumentor for the openai SDK.
        instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
        tracer = oi.OITracer(provider.get_tracer("retail.support.assistant"), config=oi.TraceConfig())
        openai_client = OpenAI()

        agent_attrs = dict(
            oi.get_span_kind_attributes(OpenInferenceSpanKindValues.AGENT),
            **oi.get_input_attributes(_QUESTION),
        )
        with tracer.start_as_current_span("retail_support_agent", attributes=agent_attrs) as agent_span:
            # 1) REAL tool step — look the order up in the order book.
            with tracer.start_as_current_span(
                "order_lookup",
                attributes=oi.get_span_kind_attributes(OpenInferenceSpanKindValues.TOOL),
            ) as tool_span:
                tool_span.set_attributes(
                    oi.get_tool_attributes(
                        name="order_lookup",
                        description=(
                            "Fetch a customer order record (item, order/delivery dates, final-sale flag) by order id."
                        ),
                        parameters={"order_id": "SO-884213"},
                    )
                )
                order = _lookup_order("SO-884213")
                tool_span.set_attributes(oi.get_output_attributes(oi.safe_json_dumps(order)))

            # 2) REAL retrieval step over the real policy corpus.
            with tracer.start_as_current_span(
                "policy_retriever",
                attributes=oi.get_span_kind_attributes(OpenInferenceSpanKindValues.RETRIEVER),
            ) as retriever_span:
                retriever_span.set_attributes(oi.get_input_attributes(_QUESTION))
                hits = _search_policies(_QUESTION)
                retriever_span.set_attributes(
                    oi.get_retriever_attributes(
                        documents=[
                            oi.Document(
                                id=doc_id,
                                content=body,
                                score=float(score),
                                metadata={"title": title},
                            )
                            for score, doc_id, title, body in hits
                        ]
                    )
                )

            # 3) REAL LLM step — auto-instrumented by OpenInference.
            excerpts = "\n".join("[%s] %s: %s" % (doc_id, title, body) for _s, doc_id, title, body in hits)
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the customer-support assistant for an online "
                            "footwear retailer. Answer the customer using ONLY the "
                            "store policy excerpts provided, and cite the policy ids "
                            "you rely on. State clearly whether a refund is available "
                            "and who pays return shipping. Answer concisely (under "
                            "150 words)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Order record:\n%s\n\nStore policy excerpts:\n%s\n\n"
                            "Customer message:\n%s" % (oi.safe_json_dumps(order), excerpts, _QUESTION)
                        ),
                    },
                ],
            )
            answer = response.choices[0].message.content or ""
            agent_span.set_attributes(oi.get_output_attributes(answer))
    finally:
        try:
            instrumentor.uninstrument()
        except Exception:
            pass
        try:
            # Seals + flushes every open collector (adapter._on_disconnect).
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig_upload

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for openinference retail-support run")
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "retail",
        "customer-support",
        "rag",
        "openinference",
    ]

    # Seal the SAME real spans as the recorded-corpus OTLP/JSON input.
    spans = exporter.get_finished_spans()
    doc = _otlp_json(spans)
    corpus_path = _write_corpus(
        doc,
        model=OPENAI_MODEL,
        instrumentor_version="openinference-instrumentation-openai %s"
        % version("openinference-instrumentation-openai"),
    )

    events = payload.get("events", [])
    kinds = sorted({e.get("event_type") for e in events})
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    print(
        "  openinference single (retail_support_agent RAG, real OI spans)  "
        "spans=%d events=%d types=%s model.invoke=%d cost.record=%d"
        % (len(spans), len(events), kinds, len(mi), len(cr))
    )
    if mi:
        p = mi[0]["payload"]
        print(
            "    real model=%s tokens=%s/%s provider=%s"
            % (p.get("model"), p.get("prompt_tokens"), p.get("completion_tokens"), p.get("provider"))
        )
    if cr:
        print("    real cost_usd=%s" % cr[0]["payload"].get("cost_usd"))
    print("  ->", corpus_path)
    print("  ->", _write([payload], "industry", "retail_openinference_support"), "\n")
    return payload


def generate_openinference_multi(client: Stratix) -> dict:
    """Record ONE real MULTI-AGENT OpenInference-instrumented support run.

    A genuine delegation: a ``support-triage-supervisor`` routes the customer's
    warranty-vs-return question to two REAL specialist sub-agents — a
    ``warranty-specialist`` and a ``returns-specialist`` — each of which does its
    own real retrieval over the real policy corpus and makes its own real
    ``gpt-4o-mini`` call, then the supervisor synthesizes their findings in a
    final real call. Three real AGENT spans (supervisor + two children), wired by
    the OTel span hierarchy, so the adapter emits three agent.input/output pairs
    with distinct ``agent_id``s and parent edges — a real multi-agent DAG, not an
    invented one. Every AGENT/RETRIEVER span is written by the OpenInference
    library's own helpers; every LLM span is the auto-instrumentor's; nothing is
    hand-authored. Both shipped artifacts are two views of the SAME real spans.
    """
    from importlib.metadata import version

    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from openinference.instrumentation.openai import OpenAIInstrumentor
    import openinference.instrumentation as oi
    from openinference.semconv.trace import OpenInferenceSpanKindValues
    from openai import OpenAI

    from layerlens.instrument.adapters.frameworks.openinference import OpenInferenceAdapter

    exporter = InMemorySpanExporter()
    provider = SDKTracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig_upload = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None

    instrumentor = OpenAIInstrumentor()
    adapter = OpenInferenceAdapter(client, capture_config=_CAPTURE)
    adapter.connect()
    provider.add_span_processor(adapter.span_processor())

    def _specialist(tracer, openai_client, *, name: str, brief: str, query: str) -> str:
        """A REAL specialist sub-agent: its own AGENT span, real retrieval, real LLM."""
        agent_attrs = dict(
            oi.get_span_kind_attributes(OpenInferenceSpanKindValues.AGENT),
            **oi.get_input_attributes(query),
        )
        with tracer.start_as_current_span(name, attributes=agent_attrs) as span:
            with tracer.start_as_current_span(
                "%s_policy_lookup" % name.split("-")[0],
                attributes=oi.get_span_kind_attributes(OpenInferenceSpanKindValues.RETRIEVER),
            ) as retr:
                retr.set_attributes(oi.get_input_attributes(query))
                hits = _search_policies(query)
                retr.set_attributes(
                    oi.get_retriever_attributes(
                        documents=[
                            oi.Document(id=d, content=b, score=float(s), metadata={"title": t}) for s, d, t, b in hits
                        ]
                    )
                )
            excerpts = "\n".join("[%s] %s: %s" % (d, t, b) for _s, d, t, b in hits)
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": brief},
                    {
                        "role": "user",
                        "content": "Order:\n%s\n\nPolicies:\n%s\n\nCustomer:\n%s"
                        % (oi.safe_json_dumps(_lookup_order("SO-884213")), excerpts, _QUESTION),
                    },
                ],
            )
            finding = resp.choices[0].message.content or ""
            span.set_attributes(oi.get_output_attributes(finding))
            return finding

    answer = ""
    try:
        instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
        tracer = oi.OITracer(provider.get_tracer("retail.support.triage"), config=oi.TraceConfig())
        openai_client = OpenAI()

        sup_attrs = dict(
            oi.get_span_kind_attributes(OpenInferenceSpanKindValues.AGENT),
            **oi.get_input_attributes(_QUESTION),
        )
        with tracer.start_as_current_span("support-triage-supervisor", attributes=sup_attrs) as sup_span:
            warranty = _specialist(
                tracer,
                openai_client,
                name="warranty-specialist",
                brief=(
                    "You are the warranty specialist for a footwear retailer. Using ONLY the "
                    "policy excerpts, decide whether the described damage is covered by the "
                    "manufacturing-defect warranty and what that means for a refund and return "
                    "shipping. Cite policy ids. Under 80 words."
                ),
                query="split seam manufacturing defect warranty coverage refund return shipping",
            )
            returns = _specialist(
                tracer,
                openai_client,
                name="returns-specialist",
                brief=(
                    "You are the returns specialist for a footwear retailer. Using ONLY the "
                    "policy excerpts, state whether the 30-day return window still applies to "
                    "this ~4-month-old order and who pays return shipping for a non-final-sale "
                    "item. Cite policy ids. Under 80 words."
                ),
                query="30 day return window refund return shipping cost final sale",
            )
            # The supervisor SYNTHESIZES the two specialists' real findings.
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the support triage supervisor. Combine the warranty and "
                            "returns specialists' findings into one clear answer for the "
                            "customer: is a refund available, and who pays return shipping? "
                            "Cite the policy ids the specialists relied on. Under 120 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Warranty specialist:\n%s\n\nReturns specialist:\n%s\n\nCustomer:\n%s"
                        % (warranty, returns, _QUESTION),
                    },
                ],
            )
            answer = resp.choices[0].message.content or ""
            sup_span.set_attributes(oi.get_output_attributes(answer))
    finally:
        try:
            instrumentor.uninstrument()
        except Exception:
            pass
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig_upload

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for openinference multi-agent run")
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "retail",
        "customer-support",
        "multi-agent",
        "openinference",
    ]

    spans = exporter.get_finished_spans()
    doc = _otlp_json(spans)
    corpus_path = _write_corpus(
        doc,
        model=OPENAI_MODEL,
        instrumentor_version="openinference-instrumentation-openai %s"
        % version("openinference-instrumentation-openai"),
        scenario="team",
    )

    events = payload.get("events", [])
    agents = sorted({e["payload"].get("agent_id") for e in events if e.get("event_type") == "agent.input"})
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    print(
        "  openinference multi (triage supervisor + 2 specialists, real OI spans)  "
        "spans=%d events=%d agents=%s model.invoke=%d" % (len(spans), len(events), agents, len(mi))
    )
    print("  ->", corpus_path)
    print("  ->", _write([payload], "industry", "retail_openinference_support_team"), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_openinference_single(_client)
    generate_openinference_multi(_client)
