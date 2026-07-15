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

KNOWN DIVERGENCE (tracked): ``_common`` stamps a content-free ``status``
(OK/ERROR/UNSET) that the Go mirror does not yet emit. It is REQUIRED here because
this side runs the collector-tier redaction backstop, which strips ``error`` from
model.invoke / agent.output under the default ``capture_content=False`` — leaving a
failed LLM call indistinguishable from a successful one without it (LAY-3620). The
Go mirror stamps only ``error`` (openinference.go:83-91) and needs the same
``status`` field to re-converge.

Usage::

    adapter = instrument_openinference(client)
    provider.add_span_processor(adapter.span_processor())   # live
    adapter.ingest_spans(exported_spans)                    # offline
    adapter.flush()
"""

from __future__ import annotations

import logging
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


def _set_if_capturing(
    payload: Dict[str, Any], key: str, value: Any, *, capture_content: bool
) -> None:
    """Set ``payload[key] = value`` only when content capture is enabled.

    The module-level twin of ``FrameworkAdapter._set_if_capturing`` — identical
    semantics — so the normalisers below stay pure functions (directly unit
    testable, and structurally aligned with the Go mirror) while still routing
    every content field through ONE gate.
    """
    if capture_content and value is not None:
        payload[key] = value


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
        _set_if_capturing(
            payload, "metadata", _safe_str(attrs[METADATA], limit=1000), capture_content=capture_content
        )
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

    ``status_message`` is a producer free-text string that can carry prompt or
    user data, so it is content and rides behind the gate. That the span FAILED
    is structure, not content, so under ``capture_content=False`` an errored span
    still reports the content-free :data:`_ERROR_SIGNAL` — redaction strips the
    text without blinding the failure.
    """
    message = record.get("status_message")
    errored = record.get("status") == "ERROR"
    if message and capture_content:
        record_payload["error"] = _safe_str(message, limit=400)
    elif message or errored:
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
    indices: set[int] = set()
    prefix = RETRIEVAL_DOCUMENTS + "."
    for key in attrs:
        if key.startswith(prefix):
            head = key[len(prefix) :].split(".", 1)[0]
            idx = _as_int(head)
            if idx is not None:
                indices.add(idx)
    return len(indices)


def normalize_llm_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    model_name = _safe_str(attrs.get(LLM_MODEL_NAME) or "unknown", limit=200)
    # ``model_name`` is required at ingest and ``model`` carries the SAME string
    # (display readers read ``model``). The "unknown" literal is an explicit
    # declared-unknown — the model is never guessed or inferred from elsewhere.
    payload["model_name"] = model_name
    payload["model"] = model_name
    payload["provider"] = _safe_str(
        attrs.get(LLM_PROVIDER) or attrs.get(LLM_SYSTEM) or "unknown", limit=200
    )
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
        _set_if_capturing(
            payload, "prompt", _safe_str(attrs[INPUT_VALUE]), capture_content=capture_content
        )
    if attrs.get(OUTPUT_VALUE) is not None:
        _set_if_capturing(
            payload, "output", _safe_str(attrs[OUTPUT_VALUE]), capture_content=capture_content
        )
    return payload


def normalize_embedding_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    model_name = _safe_str(attrs.get(EMBEDDING_MODEL_NAME) or "unknown", limit=200)
    payload["model_name"] = model_name
    payload["model"] = model_name
    # No llm.system fallback here (unlike an LLM span) — mirrors the Go contract.
    payload["provider"] = _safe_str(attrs.get(LLM_PROVIDER) or "unknown", limit=200)
    embeddings = attrs.get(EMBEDDING_EMBEDDINGS)
    if isinstance(embeddings, (list, tuple)):
        payload["embedding_count"] = len(embeddings)
    pt = _as_int(attrs.get(LLM_TOKEN_PROMPT))
    if pt is not None:
        payload["input_tokens"] = pt
        payload["prompt_tokens"] = pt
    if attrs.get(INPUT_VALUE) is not None:
        _set_if_capturing(
            payload, "input", _safe_str(attrs[INPUT_VALUE]), capture_content=capture_content
        )
    return payload


def normalize_tool_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    payload["tool_name"] = _safe_str(
        attrs.get(TOOL_NAME) or record.get("name") or "unknown", limit=200
    )
    # A tool description is free-text natural language the caller authored —
    # content by the same rule that strips tool JSON-Schemas from params (#17).
    if attrs.get(TOOL_DESCRIPTION):
        _set_if_capturing(
            payload,
            "tool_description",
            _safe_str(attrs[TOOL_DESCRIPTION], limit=400),
            capture_content=capture_content,
        )
    params = (
        attrs.get(TOOL_PARAMETERS) if attrs.get(TOOL_PARAMETERS) is not None else attrs.get(INPUT_VALUE)
    )
    if params is not None:
        _set_if_capturing(payload, "input", _safe_str(params), capture_content=capture_content)
    if attrs.get(OUTPUT_VALUE) is not None:
        _set_if_capturing(
            payload, "output", _safe_str(attrs[OUTPUT_VALUE]), capture_content=capture_content
        )
    return payload


def normalize_reranker_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    payload["subtype"] = "reranker"
    payload["tool_name"] = _safe_str(
        attrs.get(RERANKER_MODEL_NAME) or record.get("name") or "reranker", limit=200
    )
    top_k = _as_int(attrs.get(RERANKER_TOP_K))
    if top_k is not None:
        payload["top_k"] = top_k
    # A reranker deliberately emits no ``output``: its output.value is the
    # reranked document list (corpus text), which is not modelled.
    if attrs.get(RERANKER_QUERY) is not None:
        _set_if_capturing(
            payload, "input", _safe_str(attrs[RERANKER_QUERY]), capture_content=capture_content
        )
    return payload


def normalize_retriever_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    payload["document_count"] = _retrieval_doc_count(attrs)
    if attrs.get(INPUT_VALUE) is not None:
        _set_if_capturing(
            payload, "query", _safe_str(attrs[INPUT_VALUE]), capture_content=capture_content
        )
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
        _set_if_capturing(
            payload, "result", _safe_str(attrs[OUTPUT_VALUE]), capture_content=capture_content
        )
    return payload


def _agent_input_output(
    record: Dict[str, Any], *, capture_content: bool
) -> List[Tuple[str, Dict[str, Any]]]:
    """An AGENT/CHAIN span carries both input and output; emit a pair.

    ``agent_id`` is the span name — the only identity an OpenInference AGENT span
    declares. It is deliberately NOT also written to ``agent_name``: ``_identity.py``
    forbids a span name as an Agent-column source, and ``agent_name`` is exactly
    the key it reads. Keeping the name in ``agent_id`` (which no identity tier
    reads) preserves the wire contract while the Agent column stays an honest "—".

    ``input_text``/``output_text`` are ingest-REQUIRED: omitting them rejects the
    event and loses the agent turn entirely, so under ``capture_content=False``
    they are present-but-EMPTY. Recording the turn with empty text is the honest
    privacy outcome; the un-required ``input``/``output`` duplicates are omitted.
    """
    attrs = record.get("attributes") or {}
    agent_id = _safe_str(record.get("name") or (record.get("span_kind") or "agent"), limit=200)
    operation = (record.get("span_kind") or "AGENT").lower()
    in_text = attrs.get(INPUT_VALUE) if capture_content else None
    out_text = attrs.get(OUTPUT_VALUE) if capture_content else None

    in_payload = _common(record, capture_content=capture_content)
    in_payload["operation"] = operation
    in_payload["agent_id"] = agent_id
    in_payload["input_text"] = _safe_str(in_text) if in_text is not None else ""
    if attrs.get(INPUT_VALUE) is not None:
        _set_if_capturing(
            in_payload, "input", _safe_str(attrs[INPUT_VALUE]), capture_content=capture_content
        )

    out_payload = _common(record, capture_content=capture_content)
    out_payload["operation"] = operation
    out_payload["agent_id"] = agent_id
    out_payload["output_text"] = _safe_str(out_text) if out_text is not None else ""
    if attrs.get(OUTPUT_VALUE) is not None:
        _set_if_capturing(
            out_payload, "output", _safe_str(attrs[OUTPUT_VALUE]), capture_content=capture_content
        )
    # The output happened when the span ENDED — stamping it at span start
    # reverses the turn's chronology in the replay timeline.
    end_ns = record.get("end_ns")
    if isinstance(end_ns, int):
        out_payload["timestamp"] = end_ns / 1_000_000_000
    return [("agent.input", in_payload), ("agent.output", out_payload)]


def _guardrail_events(
    record: Dict[str, Any], *, capture_content: bool
) -> List[Tuple[str, Dict[str, Any]]]:
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
        _set_if_capturing(
            payload, "output", _safe_str(attrs[OUTPUT_VALUE]), capture_content=capture_content
        )
    if triggered:
        # policy.violation is schema-required to carry policy_id + violation_type;
        # without them the event is rejected and the violation is LOST. Both are
        # DERIVED from the guardrail's own declared identity — the guardrail IS
        # the policy that fired — rather than invented or dropped.
        payload["policy_id"] = guardrail_name
        payload["violation_type"] = "guardrail"
        return [("policy.violation", payload)]
    payload["tool_name"] = guardrail_name
    return [("tool.call", payload)]


def normalize_interaction_span(record: Dict[str, Any], *, capture_content: bool) -> Dict[str, Any]:
    attrs = record.get("attributes") or {}
    payload = _common(record, capture_content=capture_content)
    if attrs.get(INPUT_VALUE) is not None:
        _set_if_capturing(
            payload, "input", _safe_str(attrs[INPUT_VALUE]), capture_content=capture_content
        )
    if attrs.get(OUTPUT_VALUE) is not None:
        _set_if_capturing(
            payload, "output", _safe_str(attrs[OUTPUT_VALUE]), capture_content=capture_content
        )
    return payload


def span_to_events(
    record: Dict[str, Any], *, capture_content: bool = True
) -> List[Tuple[str, Dict[str, Any]]]:
    """Map one OpenInference SpanRecord to ``(event_type, payload)`` pairs.

    An unknown — or entirely absent — span kind maps to ``agent.interaction``
    carrying the full correlation skeleton, so no span is ever silently dropped.

    Accepts a fully-extracted SpanRecord or a raw span dict whose kind is only in
    ``attributes["openinference.span.kind"]``, so callers can normalise
    OpenInference spans without the adapter's extraction step.
    """
    kind = str(record.get("span_kind") or "").upper()
    if not kind:
        raw = (record.get("attributes") or {}).get(SPAN_KIND_KEY)
        if raw is not None:
            kind = str(getattr(raw, "value", raw)).upper()

    if kind == SPAN_KIND_LLM:
        return [("model.invoke", normalize_llm_span(record, capture_content=capture_content))]
    if kind == SPAN_KIND_EMBEDDING:
        return [
            ("embedding.create", normalize_embedding_span(record, capture_content=capture_content))
        ]
    if kind == SPAN_KIND_TOOL:
        return [("tool.call", normalize_tool_span(record, capture_content=capture_content))]
    if kind == SPAN_KIND_RERANKER:
        return [("tool.call", normalize_reranker_span(record, capture_content=capture_content))]
    if kind == SPAN_KIND_RETRIEVER:
        return [
            ("retrieval.query", normalize_retriever_span(record, capture_content=capture_content))
        ]
    if kind == SPAN_KIND_EVALUATOR:
        return [
            ("evaluation.result", normalize_evaluator_span(record, capture_content=capture_content))
        ]
    if kind in (SPAN_KIND_AGENT, SPAN_KIND_CHAIN):
        return _agent_input_output(record, capture_content=capture_content)
    if kind == SPAN_KIND_GUARDRAIL:
        return _guardrail_events(record, capture_content=capture_content)
    return [
        ("agent.interaction", normalize_interaction_span(record, capture_content=capture_content))
    ]


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
    return str(getattr(raw, "value", raw)).upper()


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
    """Unwrap an OTLP AnyValue dict (``{"stringValue": ...}``) to a Python scalar."""
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
        """Emit a cost.record for a priced LLM span — and nothing otherwise.

        OpenInference carries no price attribute, but a span that declares BOTH a
        real model and token counts supports the same honest derivation the
        provider path already does. When the model is the "unknown" sentinel or is
        absent from the pricing table, ``_price_cost_record`` leaves ``cost_usd``
        unset and NO event is emitted — an omitted cost, never a fabricated 0.0.
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
        attrs = self._coerce_attributes(span.get("attributes"))
        # An already-extracted SpanRecord states its kind directly and its times
        # in declared nanoseconds; a raw exported span states neither. Both still
        # go through id + timestamp coercion — returning a pre-extracted record
        # unnormalized silently drops its timestamp/duration/latency.
        pre_extracted = "span_kind" in span and "attributes" in span
        if pre_extracted:
            kind = str(span.get("span_kind") or "").upper()
        else:
            kind = _get_span_kind(attrs) or str(span.get("span_kind") or "").upper()
        status_code, status_msg = self._coerce_status(span.get("status"))
        return {
            "span_kind": kind,
            "name": span.get("name"),
            "attributes": attrs,
            "trace_id": _coerce_id(span.get("trace_id") or span.get("traceId"), width=32),
            "span_id": _coerce_id(span.get("span_id") or span.get("spanId"), width=16),
            "parent_span_id": _coerce_id(
                span.get("parent_span_id") or span.get("parentSpanId"), width=16
            ),
            "start_ns": _first_ns(span, ("start_ns", "startTimeUnixNano"), ("start_time",)),
            "end_ns": _first_ns(span, ("end_ns", "endTimeUnixNano"), ("end_time",)),
            "status": status_code,
            "status_message": status_msg,
        }

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


def _first_ns(
    span: Dict[str, Any], declared_keys: Tuple[str, ...], undeclared_keys: Tuple[str, ...]
) -> Optional[int]:
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


def instrument_openinference(
    client: Any, *, capture_config: Optional[CaptureConfig] = None
) -> OpenInferenceAdapter:
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


__all__ = [
    "OpenInferenceAdapter",
    "instrument_openinference",
    "span_to_events",
    "normalize_llm_span",
    "normalize_embedding_span",
    "normalize_tool_span",
    "normalize_reranker_span",
    "normalize_retriever_span",
    "normalize_evaluator_span",
    "normalize_interaction_span",
]
