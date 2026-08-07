"""OpenInference adapter — ingests OpenInference-instrumented OpenTelemetry spans.

OpenInference is the Arize set of OpenTelemetry semantic conventions +
auto-instrumentation libraries for LLM/agent apps. Unlike every other framework
adapter, this one wraps nothing and patches ZERO symbols: it *consumes* spans
another library already produced, either live (plug :meth:`span_processor` into
a ``TracerProvider``) or offline (feed :meth:`ingest_span` / :meth:`ingest_spans`
exported span dicts / OTLP JSON). That widens coverage to every
OpenInference-instrumented app with no per-framework work.

Span kind -> event mapping (see :func:`span_to_events`)::

    LLM       -> model.invoke        (+ cost.record when the model is priced)
    EMBEDDING -> embedding.create
    TOOL      -> tool.call
    RERANKER  -> tool.call           (subtype reranker)
    RETRIEVER -> retrieval.query
    AGENT     -> agent.input + agent.output   (a PAIR from one span)
    CHAIN     -> agent.input + agent.output
    GUARDRAIL -> policy.violation when triggered, else tool.call (subtype guardrail)
    EVALUATOR -> evaluation.result
    unknown   -> agent.interaction   (nothing is ever dropped)

This mapping is a CROSS-LANGUAGE CONTRACT: atlas re-implements it in Go at
``apps/otlp-ingest/ingest/openinference.go``. Event types and payload field names
here must stay identical to that file — change them together or an OpenInference
trace renders differently depending on whether it arrived via the SDK or OTLP.

The contract is PINNED, not merely documented: ``tests/instrument/adapters/frameworks/
test_openinference_conformance.py`` feeds a shared span corpus through this module and
asserts the events match a committed oracle generated from the REAL Go bridge, and
atlas's ``TestOpenInferenceConformanceOracleIsCurrent`` fails if the Go side drifts
from that same oracle. Drift on either side is caught. Change the mapping here and
you must regenerate the oracle — see ``tests/.../oi_conformance/README.md``.

KNOWN DIVERGENCES (deliberate, encoded as named exceptions in the conformance lane):

* ``_common`` stamps a content-free ``status`` (OK/ERROR/UNSET) the Go mirror does
  not emit. REQUIRED here because this side runs the collector-tier redaction
  backstop, which strips ``error`` from model.invoke / agent.output under the default
  ``capture_content=False`` — leaving a failed LLM call indistinguishable from a
  successful one without it (LAY-3620). The Go mirror stamps only ``error`` and needs
  the same ``status`` field to re-converge.
* D1 — the duration field NAME: ``latency_ms`` here (the SDK canon, see
  ``tests/instrument/_event_schema.py``) vs ``duration_ms`` in Go (the OTLP path's
  platform-wide convention — convert.go/merge.go/writer.go all use it). Renaming
  either side alone would break it against its OWN siblings, so it is documented and
  LEFT. The lane exempts the name and still pins the value.

KNOWN DIVERGENCES FROM THE ATEAM REFERENCE (LAY-3622 D3, adjudicated 2026-08-03).

The Go oracle above is the parity target. ateam's Python OpenInference adapter
(``stratix/sdk/python/adapters/openinference/events.py``) is a SECOND implementation
of the same mapping and the ticket asked for its "golden output" — no such golden
exists (see the report). A differential of both ``span_to_events`` over the shared
24-span corpus found **0 event-type mismatches** and byte-identical flat-token canon:
dispatch parity is genuinely met. Ten payload/field-level divergences remain, each
adjudicated below. Numbered ``AT-n`` deliberately: ``D1``-``D4`` are already taken by
this module and its conformance lane for the GO-facing divergences, and reusing them
would corrupt both schemes.

SDK IS MORE HONEST — KEEP, do not "fix" toward ateam:

* AT-1 — error set-condition. ateam sets ``error`` whenever ``status_message`` is
  truthy, testing the message BEFORE the status and never consulting it when a message
  exists, so an OK span carrying "cache miss; refetched" is labelled failed. Here
  ``_set_error`` returns early unless the status IS ``ERROR``.
* AT-2 — an empty ``tool.parameters`` must not shadow a populated ``input.value``.
  ateam uses an ``is not None`` test, so ``tool.parameters=""`` wins and the real tool
  input is LOST. Here ``_first_non_empty`` skips present-but-empty.
* AT-3 — embedding count. ateam counts only a list-valued ``embedding.embeddings`` and
  misses the FLATTENED indexed form real instrumentors emit (inconsistently: its own
  retrieval-doc counter does handle flattened indices, so this is an oversight, not a
  policy). Here ``_embedding_count`` falls back to the flattened form and OMITS the key
  when there is nothing to count.
* AT-4 — content gating. ateam's ``_common`` takes no ``capture_content`` parameter and
  ships ``metadata`` / ``invocation_parameters`` / ``tool_description`` unconditionally
  — a real privacy leak (end-user PII and tool schemas under privacy mode). All three
  go through ``_set_if_capturing`` here.
* AT-5 — ``_as_int`` rejects a bool. ateam's has no guard, so a
  ``llm.token_count.prompt=True`` attribute becomes ``prompt_tokens=1`` — a fabricated
  measurement that then gets PRICED. Here a bool yields None, i.e. omit the key.
* AT-6 — error-message privacy. ateam ships the producer's raw ``status_message``
  ungated (observed leaking an API key and an end-user email under the privacy-default
  config). Here ``capture_content=False`` substitutes the content-free ``_ERROR_SIGNAL``,
  which is exactly why the ``status`` divergence above is required.

DECIDED, with reasons:

* AT-7 — ``duration_ns``: ateam emits it on all 26 events alongside ``latency_ms``; we
  emit neither name twice. KEEP OURS. ``tests/instrument/_event_schema.py`` declares
  ``latency_ms`` canonical and FAILS any payload carrying ``duration_ns`` unless the
  adapter is in its exception list (openinference is not), the Go mirror emits
  ``duration_ms`` so adding ``duration_ns`` would break the conformance key-set
  equality, and it is the same measurement twice.
* AT-8 — ``status``: we emit it on all 26 events, ateam on none. KEEP OURS — and note
  WHY ateam does not need it: only because it leaks the raw ``error`` ungated (AT-6).
  Under our privacy-default config the backstop strips ``error``, leaving ``status`` as
  the ONLY surviving failure signal.
* AT-9 — ``tenant_id``: ateam's ``_common`` stamps the client's own org id onto every
  payload. KEEP OUR OMISSION — the upload envelope already carries org scoping in the
  request PATH (``/organizations/{org}/projects/{project}/traces/upload``) and the
  ``Trace`` model carries ``organization_id``, so a per-payload copy is redundant and a
  cross-tenant MISLABELLING hazard if the two ever disagree. (ateam's own OTLP envelope
  path publishes with ``tenant_id=org_id`` anyway, stamping tenancy twice.)
* AT-10 — ``cost.record`` is the only event-type-level divergence: ateam emits none, the
  Go bridge emits none, and the oracle contains zero. KEEP IT — cost is real value — but
  note it is emitted from ``ingest_span``, OUTSIDE the pinned ``span_to_events``
  boundary. That is exactly how the LAY-3622 fabricated-cost defect shipped unnoticed,
  so it is pinned separately by ``test_openinference_ingest_contract.py`` and by the
  invariant lane in ``tests/instrument/test_cost_chokepoint.py``.

* AT-11 — ``environment.config`` (L4a) is the SECOND SDK-only event type, added with
  the OTLP envelope decoder. Neither ateam's adapter nor the Go bridge emits one:
  ateam's bridge merges Resource attributes into each span but never lifts them into
  an event, and atlas does not need to because it already stores the same data as
  trace-level fields (``CanonicalTrace.ServiceName`` / ``.Environment`` /
  ``.ResourceAttrs``). Emitting it would therefore duplicate atlas's own storage
  purely to keep an oracle aligned. It is emitted from :meth:`ingest_resource_group`
  — the ENVELOPE level, once per Resource block per source trace — and NOT from
  ``span_to_events``, which is the pinned boundary the oracle compares positionally.
  Unlike AT-10 it was pinned by its own tests from the start.

NOT EXERCISED BY THE PINNED CORPUS: AT-5 (no corpus span carries a boolean token
attribute) and AT-9 (materialises only when the client exposes an org id). AT-5 now has
a direct unit test; AT-9 is documentation-only by nature.

Usage::

    adapter = instrument_openinference(client)
    provider.add_span_processor(adapter.span_processor())  # live
    adapter.ingest_spans(exported_spans)  # offline
    adapter.flush()
"""

from __future__ import annotations

import base64
import logging
import binascii
import contextlib
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone

from ..._collector import TraceCollector
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

#: The adapter ingests plain span dicts with neither OpenTelemetry nor the
#: OpenInference semconv package installed, so the import is probed for the
#: version surface only and never gates ingestion (``_check_dependency`` is
#: deliberately never called for this adapter).
try:
    import opentelemetry  # noqa: F401  # pyright: ignore[reportMissingImports]

    _HAS_OTEL = True
except ImportError:  # pragma: no cover - exercised by the no-otel import lane
    _HAS_OTEL = False

FRAMEWORK = "openinference"

# --- OpenInference semantic-convention attribute keys ---------------------
# Mirrored as string literals so the ``openinference-semantic-conventions``
# package is never a runtime dependency.
SPAN_KIND_KEY = "openinference.span.kind"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
LLM_MODEL_NAME = "llm.model_name"
LLM_PROVIDER = "llm.provider"
LLM_SYSTEM = "llm.system"
LLM_TOKEN_PROMPT = "llm.token_count.prompt"
LLM_TOKEN_COMPLETION = "llm.token_count.completion"
LLM_TOKEN_TOTAL = "llm.token_count.total"
LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
TOOL_NAME = "tool.name"
TOOL_DESCRIPTION = "tool.description"
TOOL_PARAMETERS = "tool.parameters"
RETRIEVAL_DOCUMENTS = "retrieval.documents"
EMBEDDING_MODEL_NAME = "embedding.model_name"
EMBEDDING_EMBEDDINGS = "embedding.embeddings"
RERANKER_MODEL_NAME = "reranker.model_name"
RERANKER_QUERY = "reranker.query"
RERANKER_TOP_K = "reranker.top_k"
METADATA = "metadata"
SESSION_ID = "session.id"
USER_ID = "user.id"

# Canonical OpenInference span-kind values.
SPAN_KIND_LLM = "LLM"
SPAN_KIND_EMBEDDING = "EMBEDDING"
SPAN_KIND_TOOL = "TOOL"
SPAN_KIND_RETRIEVER = "RETRIEVER"
SPAN_KIND_RERANKER = "RERANKER"
SPAN_KIND_AGENT = "AGENT"
SPAN_KIND_CHAIN = "CHAIN"
SPAN_KIND_GUARDRAIL = "GUARDRAIL"
SPAN_KIND_EVALUATOR = "EVALUATOR"

#: Free-text status message a span carries when it errored. The literal is the
#: content-FREE fallback: it states THAT the span failed without echoing the
#: producer's message, so the failure stays visible under ``capture_content=False``.
_ERROR_SIGNAL = "span status ERROR"

# --- OTel status codes ---------------------------------------------------
# An exported span states its status as an int enum (OTLP protobuf/JSON), a
# ``STATUS_CODE_*`` name, or the bare name. All three must resolve to the same
# canonical string, because "did this span error" decides whether a GUARDRAIL
# span is a policy.violation or an ordinary tool.call.
_STATUS_BY_CODE = {0: "UNSET", 1: "OK", 2: "ERROR"}


def _safe_str(value: Any, limit: int = 2000) -> str:
    """Render *value* as a string capped at *limit*, declaring any truncation."""
    if value is None:
        return ""
    try:
        rendered = value if isinstance(value, str) else str(value)
    except Exception:  # pragma: no cover - defensive against odd __str__
        return "<unrenderable>"
    if len(rendered) <= limit:
        return rendered
    return rendered[:limit] + f"...[truncated {len(rendered) - limit} chars]"


def _as_int(value: Any) -> Optional[int]:
    """Coerce *value* to int, or None. None means OMIT the key — never zero-fill."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _set_if_capturing(payload: Dict[str, Any], key: str, value: Any, *, capture_content: bool) -> None:
    """Set ``payload[key] = value`` only when content capture is enabled.

    The module-level twin of ``FrameworkAdapter._set_if_capturing`` — identical
    semantics — so the normalisers below stay pure functions (directly unit
    testable, and structurally aligned with the Go mirror) while still routing
    every content field through ONE gate.
    """
    if capture_content and value is not None:
        payload[key] = value


def _first_non_empty(attrs: Dict[str, Any], *keys: str) -> Any:
    """The first attribute among *keys* that carries an actual value, else None.

    Mirrors the Go bridge's ``oiString`` (openinference.go): a present-but-EMPTY
    string is NOT a value — it falls through to the next key. An ``is not None``
    test would let ``tool.parameters=""`` shadow a populated ``input.value`` and
    silently lose the real payload (D3). A non-string value is returned as-is;
    :func:`_safe_str` renders it, matching the Go stringify.
    """
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value:
                return value
            continue
        return value
    return None


def _flattened_index_count(attrs: Dict[str, Any], prefix_key: str) -> int:
    """Count DISTINCT integer indices among ``<prefix_key>.{i}.*`` flattened keys.

    OTel attributes are scalars (or scalar arrays) — they CANNOT hold a nested
    object — so a real OpenInference instrumentor flattens collections into
    indexed keys. Only integer heads count: ``<prefix>.foo.bar`` is some other
    attribute, not a collection member, and counting it would inflate the total.
    """
    indices: set[int] = set()
    prefix = prefix_key + "."
    for key in attrs:
        if key.startswith(prefix):
            head = key[len(prefix) :].split(".", 1)[0]
            idx = _as_int(head)
            if idx is not None:
                indices.add(idx)
    return len(indices)


def _duration_ns(record: Dict[str, Any]) -> Optional[int]:
    start = record.get("start_ns")
    end = record.get("end_ns")
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        return end - start
    return None


def _common(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    """Correlation + timing + status fields shared by every emitted event."""
    payload: Dict[str, Any] = {
        "framework": FRAMEWORK,
        "run_id": record.get("trace_id") or record.get("span_id") or "unknown",
        "span_id": record.get("span_id"),
        "trace_id": record.get("trace_id"),
        "span_kind": record.get("span_kind"),
        "span_name": record.get("name"),
    }
    # The envelope timestamp_ns the collector stamps is INGEST time, so on an
    # offline replay of historical spans this payload ``timestamp`` (epoch
    # SECONDS of the real span start) is the only carrier of the span's true
    # time. Prefer the span's start; fall back to its end.
    ts_ns = record.get("start_ns")
    if not isinstance(ts_ns, int):
        ts_ns = record.get("end_ns")
    if isinstance(ts_ns, int):
        payload["timestamp"] = ts_ns / 1_000_000_000
    parent = record.get("parent_span_id")
    if parent:
        payload["parent_span_id"] = parent
    attrs = record.get("attributes") or {}
    for src, dst in ((SESSION_ID, "session_id"), (USER_ID, "user_id")):
        if attrs.get(src):
            payload[dst] = _safe_str(attrs[src], limit=200)
    # ``metadata`` is an arbitrary producer-supplied blob (routinely user/session
    # /prompt context), so it is content — ateam leaves it ungated on every event.
    if attrs.get(METADATA) is not None:
        _set_if_capturing(payload, "metadata", _safe_str(attrs[METADATA], limit=1000), capture_content=capture_content)
    # latency_ms is the canonical duration field; ateam's redundant companion
    # ``duration_ns`` carries the same measurement and is schema drift here.
    # Omitted entirely (not clamped) unless BOTH bounds are ints and end >= start
    # — a negative/partial duration is not a real measurement.
    dur = _duration_ns(record)
    if dur is not None:
        payload["latency_ms"] = dur / 1_000_000
    # The span's canonical OK/ERROR/UNSET status is STRUCTURE, not content, and is
    # the ONLY failure signal that survives the collector-tier backstop: ``error``
    # is listed in _CONTENT_KEYS for model.invoke and agent.output, so under the
    # DEFAULT config (capture_content=False) the honest _ERROR_SIGNAL below is
    # stripped before upload — leaving a failed LLM call indistinguishable from a
    # successful one. No _CONTENT_KEYS entry strips ``status``, so it carries the
    # failure across redaction (LAY-3620, redact-without-going-blind). Omitted, never
    # defaulted to "OK", when the span declares no status — an unknown status is not
    # a passing one.
    status = record.get("status")
    if status:
        payload["status"] = status
    _set_error(payload, record, capture_content=capture_content)
    return payload


def _set_error(record_payload: Dict[str, Any], record: Dict[str, Any], *, capture_content: bool) -> None:
    """Stamp the honest error signal, gating only the producer's free text.

    The SET-CONDITION is the span STATUS, never the message (D2, Go parity —
    ``openinference.go`` ``spanStatusIsError``). OTel lets a SUCCESSFUL span carry
    a status description ("cache miss; refetched"); keying off a truthy message
    would label that healthy span as failed — a fabricated failure. A span is
    errored iff it says it is.

    ``status_message`` is a producer free-text string that can carry prompt or
    user data, so it is content and rides behind the gate. That the span FAILED
    is structure, not content, so under ``capture_content=False`` an errored span
    still reports the content-free :data:`_ERROR_SIGNAL` — redaction strips the
    text without blinding the failure.
    """
    if record.get("status") != "ERROR":
        return
    message = record.get("status_message")
    if message and capture_content:
        record_payload["error"] = _safe_str(message, limit=400)
    else:
        record_payload["error"] = _ERROR_SIGNAL


def _retrieval_doc_count(attrs: Dict[str, Any]) -> int:
    """Count retrieved documents: a list-valued attr, else flattened
    ``retrieval.documents.{i}.document.*`` index keys.

    An honest MEASURED zero when the span carries no document attrs — the count
    is a real observation, not a placeholder. The document CONTENT is never
    emitted (only the count), even under ``capture_content=True``.
    """
    docs = attrs.get(RETRIEVAL_DOCUMENTS)
    if isinstance(docs, (list, tuple)):
        return len(docs)
    return _flattened_index_count(attrs, RETRIEVAL_DOCUMENTS)


def _embedding_count(attrs: Dict[str, Any]) -> Optional[int]:
    """Count embeddings: a list-valued attr, else the FLATTENED indexed form.

    Real OpenInference instrumentors emit ``embedding.embeddings.{i}.embedding.vector``
    — OTel attributes cannot carry a list of nested objects — so counting only the
    list form leaves ``embedding_count`` absent on virtually every real span (D4).
    This mirrors the proven :func:`_retrieval_doc_count` approach.

    Returns None (=> OMIT the key) when the span carries no embeddings attribute at
    all: unlike a retriever, where "no document keys" is a real measured zero, an
    embedding span with no embeddings attribute is one whose count is UNKNOWN — the
    instrumentor simply may not record it. Reporting 0 there would assert something
    the span does not support.
    """
    embeddings = attrs.get(EMBEDDING_EMBEDDINGS)
    if isinstance(embeddings, (list, tuple)):
        return len(embeddings)
    count = _flattened_index_count(attrs, EMBEDDING_EMBEDDINGS)
    return count or None


def normalize_llm_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    model_name = _safe_str(attrs.get(LLM_MODEL_NAME) or "unknown", limit=200)
    # ``model_name`` is required by a STRICT consumer (ateam's SCHEMA_REGISTRY);
    # on THIS platform it is advisory — atlas does no per-event validation, and 97%
    # of our own committed model.invoke events omit it. See
    # ``layerlens.instrument._ingest_contract`` for the one declared contract and
    # why it is split that way (LAY-3622 F1). Emitting it anyway is what makes this
    # adapter portable to the strict bar. ``model`` carries the SAME string (display
    # readers read ``model``). The "unknown" literal is an explicit declared-unknown
    # — the model is never guessed or inferred from elsewhere.
    payload["model_name"] = model_name
    payload["model"] = model_name
    payload["provider"] = _safe_str(attrs.get(LLM_PROVIDER) or attrs.get(LLM_SYSTEM) or "unknown", limit=200)
    # Token canon: flat prompt_tokens/completion_tokens (+ total_tokens) with
    # input_tokens/output_tokens dual-written. The tokens_prompt/tokens_completion
    # names ``_normalize_tokens`` produces are NOT used here — they would break the
    # Go mirror's field names.
    pt = _as_int(attrs.get(LLM_TOKEN_PROMPT))
    ct = _as_int(attrs.get(LLM_TOKEN_COMPLETION))
    tt = _as_int(attrs.get(LLM_TOKEN_TOTAL))
    if pt is not None:
        payload["prompt_tokens"] = pt
        payload["input_tokens"] = pt
    if ct is not None:
        payload["completion_tokens"] = ct
        payload["output_tokens"] = ct
    if tt is not None:
        payload["total_tokens"] = tt
    # llm.invocation_parameters is the raw request-params JSON, which routinely
    # carries tools / tool_choice / response_format — tool JSON-Schemas with
    # natural-language descriptions, which this repo classifies as content (#17).
    if attrs.get(LLM_INVOCATION_PARAMETERS) is not None:
        _set_if_capturing(
            payload,
            "invocation_parameters",
            _safe_str(attrs[LLM_INVOCATION_PARAMETERS], limit=1000),
            capture_content=capture_content,
        )
    if attrs.get(INPUT_VALUE) is not None:
        _set_if_capturing(payload, "prompt", _safe_str(attrs[INPUT_VALUE]), capture_content=capture_content)
    if attrs.get(OUTPUT_VALUE) is not None:
        _set_if_capturing(payload, "output", _safe_str(attrs[OUTPUT_VALUE]), capture_content=capture_content)
    return payload


def normalize_embedding_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    model_name = _safe_str(attrs.get(EMBEDDING_MODEL_NAME) or "unknown", limit=200)
    payload["model_name"] = model_name
    payload["model"] = model_name
    # No llm.system fallback here (unlike an LLM span) — mirrors the Go contract.
    payload["provider"] = _safe_str(attrs.get(LLM_PROVIDER) or "unknown", limit=200)
    count = _embedding_count(attrs)
    if count is not None:
        payload["embedding_count"] = count
    pt = _as_int(attrs.get(LLM_TOKEN_PROMPT))
    if pt is not None:
        payload["input_tokens"] = pt
        payload["prompt_tokens"] = pt
    if attrs.get(INPUT_VALUE) is not None:
        _set_if_capturing(payload, "input", _safe_str(attrs[INPUT_VALUE]), capture_content=capture_content)
    return payload


def normalize_tool_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    payload["tool_name"] = _safe_str(attrs.get(TOOL_NAME) or record.get("name") or "unknown", limit=200)
    # A tool description is free-text natural language the caller authored —
    # content by the same rule that strips tool JSON-Schemas from params (#17).
    if attrs.get(TOOL_DESCRIPTION):
        _set_if_capturing(
            payload,
            "tool_description",
            _safe_str(attrs[TOOL_DESCRIPTION], limit=400),
            capture_content=capture_content,
        )
    # First-NON-EMPTY, not first-present (D3, Go parity): a present-but-empty
    # tool.parameters must fall through to input.value rather than shadow it —
    # an ``is not None`` test emits input='' and LOSES the real tool input.
    params = _first_non_empty(attrs, TOOL_PARAMETERS, INPUT_VALUE)
    if params is not None:
        _set_if_capturing(payload, "input", _safe_str(params), capture_content=capture_content)
    if attrs.get(OUTPUT_VALUE) is not None:
        _set_if_capturing(payload, "output", _safe_str(attrs[OUTPUT_VALUE]), capture_content=capture_content)
    return payload


def normalize_reranker_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    payload["subtype"] = "reranker"
    payload["tool_name"] = _safe_str(attrs.get(RERANKER_MODEL_NAME) or record.get("name") or "reranker", limit=200)
    top_k = _as_int(attrs.get(RERANKER_TOP_K))
    if top_k is not None:
        payload["top_k"] = top_k
    # A reranker deliberately emits no ``output``: its output.value is the
    # reranked document list (corpus text), which is not modelled.
    if attrs.get(RERANKER_QUERY) is not None:
        _set_if_capturing(payload, "input", _safe_str(attrs[RERANKER_QUERY]), capture_content=capture_content)
    return payload


def normalize_retriever_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    payload["document_count"] = _retrieval_doc_count(attrs)
    if attrs.get(INPUT_VALUE) is not None:
        _set_if_capturing(payload, "query", _safe_str(attrs[INPUT_VALUE]), capture_content=capture_content)
    return payload


def normalize_evaluator_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    """Normalise an EVALUATOR span to the honest name + raw output ONLY.

    No score / label / dimension / is_passing / threshold is emitted: an
    OpenInference EVALUATOR span carries no normative score attribute, so there
    is nothing honest to put in one. Emitting a default 0.0/1.0 would be a
    fabricated grade, so the event stays deliberately non-conformant with the
    typed evaluation model instead.
    """
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    payload["evaluator_name"] = _safe_str(record.get("name") or "evaluator", limit=200)
    if attrs.get(OUTPUT_VALUE) is not None:
        _set_if_capturing(payload, "result", _safe_str(attrs[OUTPUT_VALUE]), capture_content=capture_content)
    return payload


def _agent_input_output(record: Dict[str, Any], *, capture_content: bool) -> List[Tuple[str, Dict[str, Any]]]:
    """An AGENT/CHAIN span carries both input and output; emit a pair.

    ``agent_id`` is the span name — the only identity an OpenInference AGENT span
    declares. It is deliberately NOT also written to ``agent_name``: ``_identity.py``
    forbids a span name as an Agent-column source, and ``agent_name`` is exactly
    the key it reads. Keeping the name in ``agent_id`` (which no identity tier
    reads) preserves the wire contract while the Agent column stays an honest "—".

    ``input_text``/``output_text`` are required by a STRICT consumer (ateam's
    ``SCHEMA_REGISTRY``) and ADVISORY on this platform — see
    ``layerlens.instrument._ingest_contract`` (LAY-3622 F1). Under
    ``capture_content=False`` they are set present-but-EMPTY rather than omitted, so
    the turn survives a strict consumer instead of being rejected; recording the
    turn with empty text is the honest privacy outcome, and the un-required
    ``input``/``output`` duplicates are omitted.

    This invariant now holds END-TO-END (LAY-3622 F2). It did not before: both names
    are in ``_CONTENT_KEYS``, so the collector-tier backstop deleted them and the
    empty-string mitigation never reached the wire. ``_is_content_free``
    (``_capture_config.py``) now keeps a content key whose value is already empty —
    privacy-neutral by construction, and a POPULATED value is still deleted.
    """
    attrs = record.get("attributes") or {}
    # E2b: the nameless fallback is LOWER-cased to match the Go mirror
    # (openinference.go: firstNonEmpty(spanName, strings.ToLower(kind), "agent")).
    # agent_id is a graph NODE id, so an upper-cased fallback rendered the same
    # nameless AGENT span as a differently-named node depending on arrival path.
    agent_id = _safe_str(record.get("name") or (record.get("span_kind") or "agent").lower(), limit=200)
    operation = (record.get("span_kind") or "AGENT").lower()
    in_text = attrs.get(INPUT_VALUE) if capture_content else None
    out_text = attrs.get(OUTPUT_VALUE) if capture_content else None

    in_payload = _common(record, capture_content=capture_content)
    in_payload["operation"] = operation
    in_payload["agent_id"] = agent_id
    in_payload["input_text"] = _safe_str(in_text) if in_text is not None else ""
    if attrs.get(INPUT_VALUE) is not None:
        _set_if_capturing(in_payload, "input", _safe_str(attrs[INPUT_VALUE]), capture_content=capture_content)

    out_payload = _common(record, capture_content=capture_content)
    out_payload["operation"] = operation
    out_payload["agent_id"] = agent_id
    out_payload["output_text"] = _safe_str(out_text) if out_text is not None else ""
    if attrs.get(OUTPUT_VALUE) is not None:
        _set_if_capturing(out_payload, "output", _safe_str(attrs[OUTPUT_VALUE]), capture_content=capture_content)
    # The output happened when the span ENDED — stamping it at span start
    # reverses the turn's chronology in the replay timeline.
    end_ns = record.get("end_ns")
    if isinstance(end_ns, int):
        out_payload["timestamp"] = end_ns / 1_000_000_000
    return [("agent.input", in_payload), ("agent.output", out_payload)]


def _guardrail_events(record: Dict[str, Any], *, capture_content: bool) -> List[Tuple[str, Dict[str, Any]]]:
    """A GUARDRAIL span: a violation when it triggered, else an ordinary tool.call.

    "Triggered" is inferred solely from the span status being ERROR — OpenInference
    declares no ``guardrail.triggered`` attribute. A guardrail that blocks without
    setting an error status is therefore recorded as a PASSING tool.call; no
    violation is ever manufactured for a check that looks clean.
    """
    attrs = record.get("attributes") or {}
    triggered = record.get("status") == "ERROR"
    payload = _common(record, capture_content=capture_content)
    payload["subtype"] = "guardrail"
    guardrail_name = _safe_str(record.get("name") or "guardrail", limit=200)
    payload["guardrail_name"] = guardrail_name
    if attrs.get(OUTPUT_VALUE) is not None:
        _set_if_capturing(payload, "output", _safe_str(attrs[OUTPUT_VALUE]), capture_content=capture_content)
    if triggered:
        # A strict consumer requires policy_id + violation_type on policy.violation
        # (advisory here — see ``_ingest_contract``); without them THAT consumer
        # rejects the event and the violation is LOST. Both are DERIVED from the
        # guardrail's own declared identity — the guardrail IS the policy that fired
        # — rather than invented or dropped.
        payload["policy_id"] = guardrail_name
        payload["violation_type"] = "guardrail"
        return [("policy.violation", payload)]
    payload["tool_name"] = guardrail_name
    return [("tool.call", payload)]


def normalize_interaction_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    if attrs.get(INPUT_VALUE) is not None:
        _set_if_capturing(payload, "input", _safe_str(attrs[INPUT_VALUE]), capture_content=capture_content)
    if attrs.get(OUTPUT_VALUE) is not None:
        _set_if_capturing(payload, "output", _safe_str(attrs[OUTPUT_VALUE]), capture_content=capture_content)
    return payload


def span_to_events(record: Dict[str, Any], *, capture_content: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
    """Map one OpenInference SpanRecord to ``(event_type, payload)`` pairs.

    An unknown — or entirely absent — span kind maps to ``agent.interaction``
    carrying the full correlation skeleton, so no span is ever silently dropped.

    Accepts a fully-extracted SpanRecord or a raw span dict whose kind is only in
    ``attributes["openinference.span.kind"]``, so callers can normalise
    OpenInference spans without the adapter's extraction step.
    """
    # E2a: TRIM before matching, as the Go mirror does
    # (openinference.go: strings.ToUpper(strings.TrimSpace(...))). Without the
    # strip, a span whose kind is " LLM " fell through to agent.interaction here
    # while Go typed it as model.invoke — the same span rendering differently
    # depending on whether it arrived via the SDK or via OTLP.
    kind = str(record.get("span_kind") or "").strip().upper()
    if not kind:
        raw = (record.get("attributes") or {}).get(SPAN_KIND_KEY)
        if raw is not None:
            kind = str(getattr(raw, "value", raw)).upper()

    if kind == SPAN_KIND_LLM:
        return [("model.invoke", normalize_llm_span(record, capture_content=capture_content))]
    if kind == SPAN_KIND_EMBEDDING:
        return [("embedding.create", normalize_embedding_span(record, capture_content=capture_content))]
    if kind == SPAN_KIND_TOOL:
        return [("tool.call", normalize_tool_span(record, capture_content=capture_content))]
    if kind == SPAN_KIND_RERANKER:
        return [("tool.call", normalize_reranker_span(record, capture_content=capture_content))]
    if kind == SPAN_KIND_RETRIEVER:
        return [("retrieval.query", normalize_retriever_span(record, capture_content=capture_content))]
    if kind == SPAN_KIND_EVALUATOR:
        return [("evaluation.result", normalize_evaluator_span(record, capture_content=capture_content))]
    if kind in (SPAN_KIND_AGENT, SPAN_KIND_CHAIN):
        return _agent_input_output(record, capture_content=capture_content)
    if kind == SPAN_KIND_GUARDRAIL:
        return _guardrail_events(record, capture_content=capture_content)
    return [("agent.interaction", normalize_interaction_span(record, capture_content=capture_content))]


# --- Version / availability probes ---------------------------------------


def _detect_otel_version() -> Optional[str]:
    with contextlib.suppress(Exception):
        from importlib.metadata import version

        return version("opentelemetry-api")
    return None


def _detect_openinference_version() -> Optional[str]:
    with contextlib.suppress(Exception):
        from importlib.metadata import version

        return version("openinference-semantic-conventions")
    return None


def _detect_framework_version() -> Optional[str]:
    oi = _detect_openinference_version()
    if oi:
        return f"openinference-semconv {oi}"
    otel = _detect_otel_version()
    if otel:
        return f"opentelemetry-api {otel}"
    return None


def _is_available() -> bool:
    """True when OpenTelemetry or the OpenInference semconv package is installed.

    Informational only: the adapter ingests plain span dicts without either, so
    absence is never fatal and never blocks ``connect()``.
    """
    return _detect_otel_version() is not None or _detect_openinference_version() is not None


def _get_span_kind(attributes: Dict[str, Any]) -> str:
    """The OpenInference span kind, upper-cased; "" when absent."""
    raw = attributes.get(SPAN_KIND_KEY)
    if raw is None:
        return ""
    # OpenInference stores the kind as either the enum or its ``.value``.
    return str(getattr(raw, "value", raw)).strip().upper()


# --- Span extraction ------------------------------------------------------


def _scale_epoch_to_ns(num: float) -> int:
    """Scale an UNDECLARED epoch numeric to nanoseconds by magnitude detection.

    Only for timestamp fields whose unit the producer never stated (a bare
    ``start_time`` on a hand-rolled export). Plausible epochs land in distinct
    bands: seconds ~1.7e9, ms ~1.7e12, µs ~1.7e15, ns ~1.7e18. Callers that DO
    know the unit (``startTimeUnixNano``, an OTel ``ReadableSpan.start_time``)
    must pass ``declared_ns=True`` to :func:`_as_ns` instead — magnitude guessing
    misreads a small relative-ns fixture timestamp as a seconds epoch, and a
    declared unit is always more trustworthy than an inferred one.
    """
    magnitude = abs(num)
    if magnitude >= 1e17:  # already nanoseconds
        return int(num)
    if magnitude >= 1e14:  # microseconds
        return int(num * 1_000)
    if magnitude >= 1e11:  # milliseconds
        return int(num * 1_000_000)
    if magnitude >= 1e8:  # seconds
        return int(num * 1_000_000_000)
    return int(num)


def _as_ns(value: Any, *, declared_ns: bool = False) -> Optional[int]:
    """Coerce a timestamp to integer nanoseconds.

    Accepts int/float epochs, numeric strings, and the ISO-8601 strings a
    JSON-exported OTLP dump carries. ``declared_ns`` marks a source that already
    states nanoseconds, which is then trusted verbatim rather than magnitude-guessed.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if declared_ns else _scale_epoch_to_ns(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            num = float(stripped) if "." in stripped or "e" in stripped.lower() else int(stripped)
        except ValueError:
            return _iso_to_ns(stripped)
        return int(num) if declared_ns else _scale_epoch_to_ns(num)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iso_to_ns(value: str) -> Optional[int]:
    """Parse an ISO-8601 instant to epoch nanoseconds, or None.

    ``datetime.fromisoformat`` rejects 9-digit (nanosecond) fractional seconds
    before Python 3.11 — exactly what an OTLP JSON dump can carry — so the
    fraction is truncated to microseconds before parsing rather than silently
    returning None on the older interpreters this package still supports.
    """
    text = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        trimmed = _trim_fractional_seconds(text)
        if trimmed is None:
            return None
        try:
            dt = datetime.fromisoformat(trimmed)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _trim_fractional_seconds(text: str) -> Optional[str]:
    """Truncate a sub-second fraction to 6 digits so ``fromisoformat`` accepts it."""
    dot = text.find(".")
    if dot == -1:
        return None
    end = dot + 1
    while end < len(text) and text[end].isdigit():
        end += 1
    digits = text[dot + 1 : end]
    if len(digits) <= 6:
        return None
    return text[: dot + 1] + digits[:6] + text[end:]


def _otlp_value(value: Any) -> Any:
    """Unwrap an OTLP AnyValue dict (``{"stringValue": ...}``) to a Python scalar.

    ``intValue`` is deliberately left as proto3-JSON produced it — a STRING, since
    the JSON mapping encodes int64 as a string. Numeric attributes are normalised
    downstream by :func:`_as_int`, so token counts agree with the Go bridge; see
    the KNOWN DIVERGENCES block for the residual (a non-numeric attribute holding
    an int stays a string here where Go, reading protobuf, has an int64).
    """
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            return value[key]
    if "arrayValue" in value:
        av = value["arrayValue"]
        if isinstance(av, dict):
            arr = av.get("values", [])
            if isinstance(arr, (list, tuple)):
                return [_otlp_value(v) for v in arr]
    if "kvlistValue" in value:
        # Go's bridge maps a KvlistValue to a real map (otlp-ingest
        # convert.go:340-341); leaving the raw OTLP wrapper here shipped an
        # internal wire structure into the event payload and diverged from the
        # oracle's reference implementation.
        kv = value["kvlistValue"]
        items = kv.get("values", []) if isinstance(kv, dict) else []
        if isinstance(items, (list, tuple)):
            return {
                str(item.get("key")): _otlp_value(item.get("value"))
                for item in items
                if isinstance(item, dict) and "key" in item
            }
    return value


def _coerce_status(status: Any) -> Tuple[Optional[str], Optional[str]]:
    """Normalise any span status shape to ``(code, message)``.

    Handles the bare name ("ERROR"), an OTLP int enum (2), a ``STATUS_CODE_ERROR``
    name, an enum object exposing ``.name``/``.value``, and the dict forms of each.
    Every shape must resolve identically: the code decides whether a GUARDRAIL span
    is a policy.violation or a clean tool.call, so a status that fails to resolve
    silently downgrades a real violation.
    """
    if status is None:
        return None, None
    if isinstance(status, dict):
        # ``code`` may legitimately be 0 (UNSET), so test presence, not truthiness.
        code: Any = status.get("code")
        if code is None:
            code = status.get("status_code")
        message = status.get("message") or status.get("description")
        return _status_code_name(code), message
    return _status_code_name(status), None


def _status_code_name(code: Any) -> Optional[str]:
    """Canonical UNSET / OK / ERROR for any OTel status-code representation."""
    if code is None:
        return None
    if isinstance(code, bool):
        return None
    if isinstance(code, int):
        return _STATUS_BY_CODE.get(code)
    # An enum (OTel StatusCode) exposes .name; its .value is the int above.
    name = getattr(code, "name", None)
    if isinstance(name, str):
        return _normalize_status_name(name)
    value = getattr(code, "value", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return _STATUS_BY_CODE.get(value)
    if isinstance(code, str):
        return _normalize_status_name(code)
    return None


def _normalize_status_name(name: str) -> Optional[str]:
    text = name.strip().upper()
    if not text:
        return None
    # OTLP JSON spells the enum as STATUS_CODE_ERROR; OTel Python as ERROR.
    if text.startswith("STATUS_CODE_"):
        text = text[len("STATUS_CODE_") :]
    if text.isdigit():
        return _STATUS_BY_CODE.get(int(text))
    return text


def _coerce_id(value: Any, *, width: int) -> Optional[str]:
    """Render a trace/span id as lower-case hex, zero-padded to *width*.

    An int id MUST be padded: an unpadded ``format(v, "x")`` drops leading zeros,
    so the same span ingested live (OTel object) and offline (exported dict with
    int ids) would yield different id strings and fail to correlate — and a short
    span id is not the 16-hex shape the rest of the SDK assumes.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return format(value, f"0{width}x")
    if isinstance(value, bytes):
        return value.hex()
    text = str(value)
    return text or None


def _decode_otlp_id(value: Any, *, width: int) -> Optional[str]:
    """Normalise an OTLP envelope id (hex string OR base64 bytes) to padded hex.

    proto3-JSON encodes a ``bytes`` field as base64, so a spec-compliant OTLP/HTTP
    JSON export carries ``traceId``/``spanId`` base64-encoded — while many tools
    (and every fixture in this repo) emit plain hex. Guessing wrong silently breaks
    id correlation, so a clean 16/32-char hex string is treated as hex and anything
    else is attempted as base64 before falling back to :func:`_coerce_id`.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if len(text) in (16, 32) and all(c in "0123456789abcdefABCDEF" for c in text):
            return text.lower()
        try:
            decoded = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError):
            return _coerce_id(text, width=width)
        if decoded:
            return decoded.hex()
        return _coerce_id(text, width=width)
    return _coerce_id(value, width=width)


def _attrs_from_otlp_list(attr_list: Any) -> Dict[str, Any]:
    """Flatten an OTLP ``[{key, value}]`` attribute list into a dict."""
    out: Dict[str, Any] = {}
    if isinstance(attr_list, (list, tuple)):
        for item in attr_list:
            if isinstance(item, dict) and "key" in item:
                out[str(item["key"])] = _otlp_value(item.get("value"))
    return out


#: OTel semantic-convention RESOURCE attributes that describe the environment, and
#: the ``environment.config`` payload key each maps to (LAY-3622 L4a).
#:
#: Curated on purpose. A Resource block can carry arbitrary vendor attributes, and
#: some deployments put credentials or customer identifiers there, so dumping the
#: whole block into an event would turn an environment record into an exfiltration
#: path. Only these well-known, non-secret keys are lifted.
_ENVIRONMENT_RESOURCE_KEYS: Tuple[Tuple[str, str], ...] = (
    ("service.name", "service_name"),
    ("service.version", "service_version"),
    ("service.namespace", "service_namespace"),
    ("deployment.environment", "environment"),
    # OTel renamed it in semconv 1.27; exporters emit either spelling.
    ("deployment.environment.name", "environment"),
    ("cloud.provider", "cloud_provider"),
    ("cloud.region", "region"),
    ("cloud.platform", "cloud_platform"),
    ("telemetry.sdk.name", "telemetry_sdk"),
    ("telemetry.sdk.language", "telemetry_language"),
)


def environment_config_from_resource(resource_attrs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build an ``environment.config`` payload from OTLP Resource attributes.

    Every OTLP export carries a Resource block, which is exactly the
    ``EnvironmentInfo{type, region, attributes}`` material the L4a capture layer
    describes — and until the envelope decoder landed the SDK could not even SEE
    it, so an OpenInference trace lost ``service.name`` entirely.

    Returns ``None`` when the block carries none of the known keys: an
    ``environment.config`` with nothing in it is noise, and inventing a default
    environment would be a fabricated measurement.
    """
    payload: Dict[str, Any] = {}
    for source, dest in _ENVIRONMENT_RESOURCE_KEYS:
        if dest in payload:
            continue  # first spelling wins (deployment.environment before .name)
        value = resource_attrs.get(source)
        if value is not None and value != "":
            payload[dest] = _safe_str(value, limit=200)
    if not payload:
        return None
    payload["framework"] = FRAMEWORK
    return payload


def otlp_json_to_resource_groups(
    request: Dict[str, Any],
) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Flatten an OTLP/JSON export into ``[(resource_attrs, [span_record, ...])]``.

    Preserves the resource grouping that :func:`otlp_json_to_span_records`
    discards, so a caller can emit one ``environment.config`` per Resource block
    instead of once per span (the same environment repeated N times).
    """
    groups: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    resource_spans = request.get("resourceSpans") or request.get("resource_spans") or []
    if not isinstance(resource_spans, (list, tuple)):
        return groups
    for rs in resource_spans:
        if not isinstance(rs, dict):
            continue
        resource = rs.get("resource")
        resource_attrs = _attrs_from_otlp_list(resource.get("attributes")) if isinstance(resource, dict) else {}
        records: List[Dict[str, Any]] = []
        scope_spans = (
            rs.get("scopeSpans")
            or rs.get("scope_spans")
            # OTLP <=0.19 spelling; still emitted by older collectors.
            or rs.get("instrumentationLibrarySpans")
            or []
        )
        if isinstance(scope_spans, (list, tuple)):
            for ss in scope_spans:
                if not isinstance(ss, dict):
                    continue
                spans = ss.get("spans") or []
                if not isinstance(spans, (list, tuple)):
                    continue
                for span in spans:
                    if isinstance(span, dict):
                        records.append(_span_record(span, resource_attrs))
        groups.append((resource_attrs, records))
    return groups


def _span_record(span: Dict[str, Any], resource_attrs: Dict[str, Any]) -> Dict[str, Any]:
    """One OTLP/JSON span -> a SpanRecord, with resource attributes merged in
    (**span wins on conflict**)."""
    attrs = {**resource_attrs, **_attrs_from_otlp_list(span.get("attributes"))}
    status_code, status_msg = _coerce_status(span.get("status"))
    return {
        "span_kind": _get_span_kind(attrs),
        "name": span.get("name"),
        "attributes": attrs,
        "trace_id": _decode_otlp_id(span.get("traceId") or span.get("trace_id"), width=32),
        "span_id": _decode_otlp_id(span.get("spanId") or span.get("span_id"), width=16),
        "parent_span_id": _decode_otlp_id(span.get("parentSpanId") or span.get("parent_span_id"), width=16),
        "start_ns": _first_ns(span, ("startTimeUnixNano", "start_time_unix_nano"), ()),
        "end_ns": _first_ns(span, ("endTimeUnixNano", "end_time_unix_nano"), ()),
        "status": status_code,
        "status_message": status_msg,
    }


def otlp_json_to_span_records(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten an OTLP/HTTP JSON ``ExportTraceServiceRequest`` into span records
    consumable by :meth:`OpenInferenceAdapter.ingest_span`.

    ``_extract_record`` understands per-SPAN OTLP spellings but not the envelope:
    the caller had to walk ``resourceSpans -> scopeSpans -> spans`` itself. This is
    that walk, shipped.

    Resource-level attributes are merged into every span (**span attributes win on
    conflict**) so resource-scoped ``service.name`` / ``deployment.environment`` /
    ``session.id`` / tenancy tags survive — the same precedence ateam's bridge uses.
    A malformed member is skipped rather than aborting the export.

    Resource GROUPING is discarded here; use :func:`otlp_json_to_resource_groups`
    when you need it (e.g. to emit one ``environment.config`` per Resource block).
    """
    return [record for _attrs, records in otlp_json_to_resource_groups(request) for record in records]


def otlp_protobuf_to_resource_groups(
    data: bytes,
) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Protobuf twin of :func:`otlp_json_to_resource_groups`.

    Requires ``opentelemetry-proto``, which the SDK does NOT declare as a
    dependency (the ``openinference`` extra stays empty — the semconv keys are
    string literals). The import is function-local and raises a clean
    ``ImportError`` so a caller can fall back to the JSON path.
    """
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (  # type: ignore[import-not-found]
        ExportTraceServiceRequest,
    )

    request = ExportTraceServiceRequest()
    request.ParseFromString(data)
    groups: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    for rs in request.resource_spans:
        resource_attrs = _attrs_from_proto(rs.resource.attributes) if rs.HasField("resource") else {}
        records: List[Dict[str, Any]] = []
        for ss in rs.scope_spans:
            for span in ss.spans:
                attrs = {**resource_attrs, **_attrs_from_proto(span.attributes)}
                code = span.status.code if span.HasField("status") else 0
                records.append(
                    {
                        "span_kind": _get_span_kind(attrs),
                        "name": span.name,
                        "attributes": attrs,
                        # protobuf ids are raw bytes; .hex() needs no base64 guess.
                        "trace_id": span.trace_id.hex() if span.trace_id else None,
                        "span_id": span.span_id.hex() if span.span_id else None,
                        "parent_span_id": span.parent_span_id.hex() if span.parent_span_id else None,
                        "start_ns": int(span.start_time_unix_nano) or None,
                        "end_ns": int(span.end_time_unix_nano) or None,
                        "status": _status_code_name(int(code)),
                        "status_message": span.status.message or None,
                    }
                )
        groups.append((resource_attrs, records))
    return groups


def otlp_protobuf_to_span_records(data: bytes) -> List[Dict[str, Any]]:
    """Parse a binary OTLP ``ExportTraceServiceRequest`` into span records.

    Resource grouping is discarded; use :func:`otlp_protobuf_to_resource_groups`
    when you need it. Raises ``ImportError`` without ``opentelemetry-proto``.
    """
    return [record for _attrs, records in otlp_protobuf_to_resource_groups(data) for record in records]


def _attrs_from_proto(kvs: Any) -> Dict[str, Any]:
    """Flatten a protobuf ``repeated KeyValue`` into a dict."""
    return {str(kv.key): _proto_anyvalue(kv.value) for kv in kvs}


def _proto_anyvalue(value: Any) -> Any:
    """Unwrap a protobuf ``AnyValue``. Mirrors the Go bridge's conversion
    (otlp-ingest ``convert.go``): int64 stays an int here, because protobuf carries
    it natively — unlike the JSON path, where proto3 encodes it as a string."""
    which = value.WhichOneof("value")
    if which is None:
        return None
    if which == "array_value":
        return [_proto_anyvalue(v) for v in value.array_value.values]
    if which == "kvlist_value":
        return {str(kv.key): _proto_anyvalue(kv.value) for kv in value.kvlist_value.values}
    if which == "bytes_value":
        return value.bytes_value.hex()
    return getattr(value, which)


class OpenInferenceAdapter(FrameworkAdapter):
    """Ingests OpenInference-instrumented OpenTelemetry spans into LayerLens traces.

    Patches nothing. Spans arrive either from :meth:`span_processor` (live) or
    :meth:`ingest_span` / :meth:`ingest_spans` (offline), and are grouped into one
    :class:`TraceCollector` per SOURCE OTel trace id.
    """

    name = "openinference"
    #: Named for the install hint only — never passed to ``_check_dependency``:
    #: the adapter ingests plain span dicts with no OpenTelemetry installed, so a
    #: missing dependency must never raise.
    package = "openinference"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        # One collector per SOURCE trace id: a TraceCollector owns exactly one
        # trace_id, so feeding spans from N OTel traces through one collector
        # would merge N distinct traces into one.
        self._collectors: Dict[str, TraceCollector] = {}
        self._spans_ingested = 0

    # --- Lifecycle ----------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:  # noqa: ARG002
        """Record the environment. Nothing is hooked — spans are pushed/pulled in."""
        self._metadata["otel_available"] = _is_available()
        self._metadata["framework_version"] = _detect_framework_version()
        self._metadata["spans_ingested"] = self._spans_ingested

    def _on_disconnect(self) -> None:
        # A live OTel trace has no end signal, so disconnect (via the processor's
        # shutdown) is the last chance to seal what was captured. Without this
        # every live-captured span is lost at process exit.
        self.flush()

    # --- Public ingestion API ----------------------------------------

    def ingest_span(self, span: Any) -> int:
        """Convert one OpenInference span to events and emit them.

        *span* may be an OpenTelemetry ``ReadableSpan`` or a plain exported dict.
        Returns the number of events emitted.
        """
        record = self._extract_record(span)
        if record is None:
            return 0
        pairs = span_to_events(record, capture_content=self._config.capture_content)
        collector = self._collector_for(record)
        span_id = record.get("span_id") or self._new_span_id()
        for event_type, payload in pairs:
            collector.emit(
                event_type,
                payload,
                span_id=span_id,
                parent_span_id=record.get("parent_span_id"),
                span_name=record.get("name"),
            )
            if event_type == "model.invoke":
                self._emit_cost_record(collector, payload, span_id, record)
        with self._lock:
            self._spans_ingested += 1
            self._metadata["spans_ingested"] = self._spans_ingested
        return len(pairs)

    def ingest_spans(self, spans: Any) -> int:
        """Ingest an iterable of spans. Returns the total number of events emitted."""
        total = 0
        for span in spans:
            total += self.ingest_span(span)
        return total

    def ingest_resource_group(self, resource_attrs: Dict[str, Any], records: List[Dict[str, Any]]) -> int:
        """Ingest one OTLP Resource block: its environment, then its spans.

        Emits at most ONE ``environment.config`` per (Resource block, source trace)
        — the environment describes the resource, so repeating it per span would
        restate the same fact N times. Returns the total events emitted.

        L4a (LAY-3622): every OTLP export carries a Resource block
        (``service.name`` / ``deployment.environment`` / ``cloud.region``), which is
        exactly the environment material the l4a capture layer describes. Before the
        envelope decoder existed the SDK could not see it at all, so an
        OpenInference trace lost its service identity entirely.

        Deliberately emitted HERE and not from :func:`span_to_events`: that function
        is the pinned Python<->Go boundary, and the Go bridge emits no
        ``environment.config`` because atlas already captures the same Resource data
        as trace-level fields (``CanonicalTrace.ServiceName`` / ``.Environment`` /
        ``.ResourceAttrs``). Adding it there would duplicate atlas's own storage
        purely to keep an oracle aligned, and would desynchronise the positional
        26-event comparison. It is therefore an SDK-only event type, like
        ``cost.record`` — and with the Cluster A lesson applied, it is pinned by its
        own test from the start rather than left outside every oracle.
        """
        total = 0
        environment = environment_config_from_resource(resource_attrs)
        seen_traces: set = set()
        for record in records:
            if environment is not None:
                key = str(record.get("trace_id") or record.get("span_id") or "unknown")
                if key not in seen_traces:
                    seen_traces.add(key)
                    collector = self._collector_for(record)
                    before = len(collector.events)
                    collector.emit(
                        "environment.config",
                        dict(environment),
                        span_id=str(record.get("span_id") or key),
                        span_name="otlp.resource",
                    )
                    # CaptureConfig gates l4a, so emit() may legitimately drop it.
                    total += len(collector.events) - before
            total += self.ingest_span(record)
        return total

    def flush(self) -> int:
        """Seal and upload every open collector. Returns how many traces flushed."""
        with self._lock:
            collectors = list(self._collectors.values())
            self._collectors.clear()
        for collector in collectors:
            collector.flush()
        return len(collectors)

    def span_processor(self) -> "_OpenInferenceSpanProcessor":
        """An OTel-compatible SpanProcessor for live capture::

        provider.add_span_processor(adapter.span_processor())
        """
        return _OpenInferenceSpanProcessor(self)

    # --- Internals ----------------------------------------------------

    def _emit_cost_record(
        self,
        collector: TraceCollector,
        payload: Dict[str, Any],
        span_id: str,
        record: Dict[str, Any],
    ) -> None:
        """Emit a cost.record for a priceable LLM span — and nothing otherwise.

        OpenInference carries no price attribute, but a span that declares BOTH a
        real model and token counts supports the same honest derivation the
        provider path already does.

        NO event is emitted whenever ``_price_cost_record`` leaves ``cost_usd``
        unset — an omitted cost, never a fabricated 0.0. That covers two distinct
        cases:

        * the model is the "unknown" sentinel or absent from the pricing table
          (no rate exists), and
        * the model IS priced but the span's token shape cannot be priced — it
          declares only ``llm.token_count.total``, which OpenInference explicitly
          allows, while the pricing formula reads prompt / cached / cache-write /
          completion and never the total (LAY-3622 / A4b).

        The second case used to ship ``cost_usd: 0.0``: the formula summed four
        zeroes, and ``0.0 is not None``, so the guard below passed and a real
        billed call reached the customer as free. Nothing is lost by omitting it —
        ``model.invoke`` still carries the span's honest ``total_tokens``; only the
        unknowable price is withheld.
        """
        cost_payload: Dict[str, Any] = {
            "framework": FRAMEWORK,
            "model": payload.get("model"),
            "provider": payload.get("provider"),
        }
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in payload:
                cost_payload[key] = payload[key]
        if not any(k in cost_payload for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            return
        self._price_cost_record(cost_payload)
        if cost_payload.get("cost_usd") is None:
            return
        collector.emit(
            "cost.record",
            cost_payload,
            span_id=span_id,
            parent_span_id=record.get("parent_span_id"),
            span_name=record.get("name"),
        )

    def _collector_for(self, record: Dict[str, Any]) -> TraceCollector:
        """The collector for this span's SOURCE trace, creating one if needed."""
        key = str(record.get("trace_id") or record.get("span_id") or "unknown")
        with self._lock:
            collector = self._collectors.get(key)
            if collector is None or collector.sealed:
                if collector is not None:
                    # A span arrived after its trace was flushed; a sealed
                    # collector drops silently, so start a new trace and say so.
                    log.info(
                        "layerlens: openinference trace %s continued after flush; starting a new trace",
                        key,
                    )
                collector = TraceCollector(self._client, self._config)
                self._collectors[key] = collector
            return collector

    def _extract_record(self, span: Any) -> Optional[Dict[str, Any]]:
        """Normalize an OTel ReadableSpan or a dict into a SpanRecord."""
        if isinstance(span, dict):
            return self._record_from_dict(span)
        return self._record_from_otel(span)

    def _record_from_dict(self, span: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            attrs = self._coerce_attributes(span.get("attributes"))
            # An already-extracted SpanRecord states its kind directly and its times
            # in declared nanoseconds; a raw exported span states neither. Both still
            # go through id + timestamp coercion — returning a pre-extracted record
            # unnormalized silently drops its timestamp/duration/latency.
            pre_extracted = "span_kind" in span and "attributes" in span
            if pre_extracted:
                kind = str(span.get("span_kind") or "").strip().upper()
            else:
                # E2a: strip on BOTH branches. A dict carrying ``span_kind`` but no
                # ``attributes`` (a shape the public API explicitly accepts) took
                # this branch unstripped, so a padded " AGENT " survived into the
                # record — and ``_agent_input_output`` then derived ``agent_id``
                # (a graph NODE id) as " agent ". Dispatch was unaffected because
                # ``span_to_events`` re-strips, which is exactly why no test caught
                # it: the residue showed up only in the rendered payload.
                kind = _get_span_kind(attrs) or str(span.get("span_kind") or "").strip().upper()
            status_code, status_msg = self._coerce_status(span.get("status"))
            if status_msg is None:
                # A PRE-EXTRACTED record states its status as a bare code string
                # ("ERROR") with the message alongside in ``status_message``.
                # ``_coerce_status`` can only recover a message from the dict form,
                # so re-coercing such a record silently dropped it — and the error
                # text degraded to the generic "span status ERROR" backstop. That
                # is exactly the shape the OTLP flatteners (and ateam's bridge)
                # return, so the message must be honoured when already present.
                status_msg = span.get("status_message")
            return {
                "span_kind": kind,
                "name": span.get("name"),
                "attributes": attrs,
                "trace_id": _coerce_id(span.get("trace_id") or span.get("traceId"), width=32),
                "span_id": _coerce_id(span.get("span_id") or span.get("spanId"), width=16),
                "parent_span_id": _coerce_id(span.get("parent_span_id") or span.get("parentSpanId"), width=16),
                "start_ns": _first_ns(span, ("start_ns", "startTimeUnixNano"), ("start_time",)),
                "end_ns": _first_ns(span, ("end_ns", "endTimeUnixNano"), ("end_time",)),
                "status": status_code,
                "status_message": status_msg,
            }
        except Exception:
            # Degrade rather than propagate, exactly as _record_from_otel does. One
            # malformed dict used to raise out of ingest_spans, aborting every
            # REMAINING span of the batch and stranding the already-ingested ones in
            # an unflushed collector. That is unacceptable now a whole OTLP export
            # feeds through a single call. Logged, never silently swallowed.
            log.info("layerlens: openinference could not extract a span record from a dict", exc_info=True)
            return None

    def _record_from_otel(self, span: Any) -> Optional[Dict[str, Any]]:
        try:
            attrs = self._coerce_attributes(getattr(span, "attributes", None))
            ctx = getattr(span, "context", None) or getattr(span, "get_span_context", None)
            if callable(ctx):
                ctx = ctx()
            trace_id = getattr(ctx, "trace_id", None) if ctx is not None else None
            span_id = getattr(ctx, "span_id", None) if ctx is not None else None
            parent = getattr(span, "parent", None)
            parent_id = getattr(parent, "span_id", None) if parent is not None else None
            status_obj = getattr(span, "status", None)
            status_code = None
            status_msg = None
            if status_obj is not None:
                status_code = _status_code_name(getattr(status_obj, "status_code", None))
                status_msg = getattr(status_obj, "description", None)
            return {
                "span_kind": _get_span_kind(attrs),
                "name": getattr(span, "name", None),
                "attributes": attrs,
                "trace_id": _coerce_id(trace_id, width=32),
                "span_id": _coerce_id(span_id, width=16),
                "parent_span_id": _coerce_id(parent_id, width=16),
                # ReadableSpan start_time/end_time are documented epoch nanoseconds.
                "start_ns": _as_ns(getattr(span, "start_time", None), declared_ns=True),
                "end_ns": _as_ns(getattr(span, "end_time", None), declared_ns=True),
                "status": status_code,
                "status_message": status_msg,
            }
        except Exception:
            # Degrade rather than propagate: a malformed span object must not
            # break the host's OTel pipeline. Logged, never silently swallowed.
            log.info("layerlens: openinference could not extract a span record", exc_info=True)
            return None

    @staticmethod
    def _coerce_attributes(attrs: Any) -> Dict[str, Any]:
        """Accept a dict, an OTel BoundedAttributes mapping, or the OTLP list form."""
        if attrs is None:
            return {}
        if isinstance(attrs, dict):
            return dict(attrs)
        if isinstance(attrs, (list, tuple)):
            out: Dict[str, Any] = {}
            for item in attrs:
                if isinstance(item, dict) and "key" in item:
                    out[str(item["key"])] = _otlp_value(item.get("value"))
            return out
        items = getattr(attrs, "items", None)
        if callable(items):
            try:
                return {str(k): v for k, v in items()}
            except Exception:  # pragma: no cover - defensive
                return {}
        return {}

    @staticmethod
    def _coerce_status(status: Any) -> Tuple[Optional[str], Optional[str]]:
        return _coerce_status(status)


def _first_ns(span: Dict[str, Any], declared_keys: Tuple[str, ...], undeclared_keys: Tuple[str, ...]) -> Optional[int]:
    """First present timestamp among *declared_keys* (trusted as ns), else
    *undeclared_keys* (magnitude-detected / ISO-parsed)."""
    for key in declared_keys:
        if span.get(key) is not None:
            ns = _as_ns(span[key], declared_ns=True)
            if ns is not None:
                return ns
    for key in undeclared_keys:
        if span.get(key) is not None:
            ns = _as_ns(span[key])
            if ns is not None:
                return ns
    return None


class _OpenInferenceSpanProcessor:
    """Duck-typed OpenTelemetry ``SpanProcessor`` forwarding ended spans to the adapter.

    Implements the SpanProcessor surface without importing ``opentelemetry``, so it
    attaches to a real ``TracerProvider`` when OTel is present and constructs
    harmlessly when it is not.
    """

    def __init__(self, adapter: OpenInferenceAdapter) -> None:
        self._adapter = adapter

    def on_start(self, span: Any, parent_context: Any = None) -> None:  # noqa: ARG002
        # Deliberate no-op: a started span has no output and no status. Only a
        # complete record is emitted, on end.
        return None

    def on_ending(self, span: Any) -> None:  # noqa: ARG002
        # opentelemetry-sdk >= 1.29 calls this (as the private ``_on_ending``
        # from the multi-span-processor) just before ``on_end`` for last-chance
        # span mutation. There is nothing to mutate, but a duck-typed processor
        # missing the attribute raises AttributeError on EVERY span end, so both
        # names must exist. Older SDKs simply never call them.
        return None

    _on_ending = on_ending

    def on_end(self, span: Any) -> None:
        try:
            self._adapter.ingest_span(span)
        except Exception:
            # The host's OTel pipeline must never see an exception from us. This
            # is the one place a blanket except is correct — and it always logs.
            log.info("layerlens: openinference span_processor on_end failed", exc_info=True)

    def shutdown(self) -> None:
        self._adapter.disconnect()

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002
        self._adapter.flush()
        return True


def instrument_openinference(client: Any, *, capture_config: Optional[CaptureConfig] = None) -> OpenInferenceAdapter:
    """Build and connect an :class:`OpenInferenceAdapter`.

    Wraps nothing, so the returned adapter is the handle the caller wires up::

        adapter = instrument_openinference(client)
        provider.add_span_processor(adapter.span_processor())
    """
    from .._registry import register

    adapter = OpenInferenceAdapter(client, capture_config=capture_config)
    adapter.connect()
    register("openinference", adapter)
    return adapter


class OpenInferenceOTLPBridge:
    """Routes OTLP trace exports through an :class:`OpenInferenceAdapter`.

    Mirrors ateam's ``OpenInferenceOTLPBridge`` (``stratix/observability/
    openinference_bridge.py:231-253``) so the two implementations stay diffable.

    KNOWN LIMITATION (deliberate, ateam parity): spans are **not** de-duplicated by
    ``span_id``, and there is no ``max_spans`` cap. OTLP exporters retry on failure,
    so a redelivered export re-ingests its spans and double-counts them. ateam
    carries dedup and a cap only on its SERVER-side entry point
    (``otlp_request_to_ingest_events``, which also owns org scoping and OTLP
    ``partial_success`` accounting).

    NOTHING catches it downstream of this bridge, and an earlier version of this
    docstring implied otherwise (corrected under LAY-3622 F5). atlas-app's
    ``apps/otlp-ingest`` DOES de-duplicate a re-sent ``span_id`` — ``ingest/merge.go``
    and ``ingest/writer.go``: "A re-sent span_id is deduped (idempotent)" — but that
    is a separate service guarding the OTLP *endpoint*, and this bridge never reaches
    it. It converts spans into canonical events and uploads them through the traces
    API (``apps/backend/api/v1/organizations/traces/traces_create.go``), which carries
    no span-level dedup or upsert at all.

    So the double-count is end-to-end on THIS path: a caller needing at-most-once
    must de-duplicate before handing the export over, because there is no second line
    of defence behind it.
    """

    def __init__(self, adapter: OpenInferenceAdapter) -> None:
        self._adapter = adapter

    @property
    def adapter(self) -> OpenInferenceAdapter:
        return self._adapter

    def ingest_otlp_json(self, request: Dict[str, Any]) -> int:
        """Ingest an OTLP/HTTP JSON export. Returns LayerLens events emitted.

        Walks Resource GROUPS rather than a flat span list so each block's
        environment is recorded once (L4a) — see
        :meth:`OpenInferenceAdapter.ingest_resource_group`.
        """
        total = 0
        for resource_attrs, records in otlp_json_to_resource_groups(request):
            total += self._adapter.ingest_resource_group(resource_attrs, records)
        return total

    def ingest_otlp_protobuf(self, data: bytes) -> int:
        """Ingest a binary OTLP protobuf export. Returns LayerLens events emitted.

        Raises ``ImportError`` when ``opentelemetry-proto`` is absent, so a caller
        can fall back to :meth:`ingest_otlp_json`.
        """
        total = 0
        for resource_attrs, records in otlp_protobuf_to_resource_groups(data):
            total += self._adapter.ingest_resource_group(resource_attrs, records)
        return total


__all__ = [
    "OpenInferenceAdapter",
    "OpenInferenceOTLPBridge",
    "instrument_openinference",
    "environment_config_from_resource",
    "otlp_json_to_resource_groups",
    "otlp_json_to_span_records",
    "otlp_protobuf_to_resource_groups",
    "otlp_protobuf_to_span_records",
    "span_to_events",
    "normalize_llm_span",
    "normalize_embedding_span",
    "normalize_tool_span",
    "normalize_reranker_span",
    "normalize_retriever_span",
    "normalize_evaluator_span",
    "normalize_interaction_span",
]
