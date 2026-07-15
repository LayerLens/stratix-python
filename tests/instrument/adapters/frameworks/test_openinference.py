"""Tests for the OpenInference ingestion adapter.

The adapter wraps nothing, so "real behaviour" here means REAL OpenTelemetry:
a real ``TracerProvider`` with the adapter's own ``span_processor()`` attached,
real ``tracer.start_as_current_span`` calls carrying real OpenInference semconv
attribute keys, and the real ``BatchSpanProcessor`` end/shutdown sequence. The
spans that reach the adapter are genuine ``ReadableSpan`` objects — nothing about
the OTel pipeline is mocked. The only mock is the LayerLens upload client.

The span-kind -> event mapping asserted here is a CROSS-LANGUAGE CONTRACT that
atlas mirrors in Go (apps/otlp-ingest/ingest/openinference.go); the exact event
types and payload field names are asserted deliberately so a divergence fails.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.trace import Status, StatusCode  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.openinference import (  # noqa: E402
    OpenInferenceAdapter,
    span_to_events,
    instrument_openinference,
)

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(kind: str, name: str = "span", **over: Any) -> Dict[str, Any]:
    """A SpanRecord with real ns timestamps, overridable per lane."""
    rec: Dict[str, Any] = {
        "span_kind": kind,
        "name": name,
        "attributes": {},
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
        "span_id": "00f067aa0ba902b7",
        "parent_span_id": None,
        "start_ns": 1_700_000_000_000_000_000,
        "end_ns": 1_700_000_000_500_000_000,
        "status": "OK",
        "status_message": None,
    }
    rec.update(over)
    return rec


def _types(pairs: List[Any]) -> List[str]:
    return [t for t, _ in pairs]


# ---------------------------------------------------------------------------
# The span-kind -> event dispatch (the Go-mirrored contract)
# ---------------------------------------------------------------------------


class TestSpanKindDispatch:
    """Every span kind maps to the exact event type the Go mirror produces."""

    @pytest.mark.parametrize(
        "kind,expected",
        [
            ("LLM", ["model.invoke"]),
            ("EMBEDDING", ["embedding.create"]),
            ("TOOL", ["tool.call"]),
            ("RERANKER", ["tool.call"]),
            ("RETRIEVER", ["retrieval.query"]),
            ("EVALUATOR", ["evaluation.result"]),
            ("AGENT", ["agent.input", "agent.output"]),
            ("CHAIN", ["agent.input", "agent.output"]),
        ],
    )
    def test_kind_maps_to_event_types(self, kind: str, expected: List[str]) -> None:
        assert _types(span_to_events(_record(kind))) == expected

    def test_guardrail_triggered_is_a_violation(self) -> None:
        rec = _record("GUARDRAIL", "pii_filter", status="ERROR")
        assert _types(span_to_events(rec)) == ["policy.violation"]

    def test_guardrail_passed_is_a_tool_call_not_a_violation(self) -> None:
        # A clean check must never manufacture a violation.
        rec = _record("GUARDRAIL", "pii_filter", status="OK")
        pairs = span_to_events(rec)
        assert _types(pairs) == ["tool.call"]
        assert "policy_id" not in pairs[0][1]
        assert "violation_type" not in pairs[0][1]

    def test_unknown_kind_falls_back_to_interaction(self) -> None:
        assert _types(span_to_events(_record("SOMETHING_NEW"))) == ["agent.interaction"]

    def test_absent_kind_is_never_dropped(self) -> None:
        # The anti-drop rule: no span is ever silently discarded.
        rec = _record("", "mystery")
        pairs = span_to_events(rec)
        assert _types(pairs) == ["agent.interaction"]
        # ...and it still carries the full correlation skeleton.
        p = pairs[0][1]
        assert p["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
        assert p["span_id"] == "00f067aa0ba902b7"
        assert p["span_name"] == "mystery"
        assert p["latency_ms"] == 500.0

    def test_kind_read_from_attributes_when_record_kind_absent(self) -> None:
        rec = _record("", attributes={"openinference.span.kind": "LLM"})
        assert _types(span_to_events(rec)) == ["model.invoke"]

    def test_kind_read_from_an_enum_valued_attribute(self) -> None:
        class _Kind:
            value = "RETRIEVER"

        rec = _record("", attributes={"openinference.span.kind": _Kind()})
        assert _types(span_to_events(rec)) == ["retrieval.query"]


# ---------------------------------------------------------------------------
# Payload field names (the wire contract atlas' Go mirror must agree with)
# ---------------------------------------------------------------------------


class TestPayloadContract:
    def test_llm_span_fields_and_token_dual_write(self) -> None:
        rec = _record(
            "LLM",
            "openai.chat",
            attributes={
                "llm.model_name": "gpt-4o",
                "llm.provider": "openai",
                "llm.token_count.prompt": 100,
                "llm.token_count.completion": 50,
                "llm.token_count.total": 150,
            },
        )
        p = span_to_events(rec)[0][1]
        assert p["framework"] == "openinference"
        assert p["model_name"] == "gpt-4o"
        assert p["model"] == "gpt-4o"  # dual-write: display readers read `model`
        assert p["provider"] == "openai"
        # Dual-written token vocabulary — NOT the tokens_prompt/tokens_completion
        # names _normalize_tokens produces (they would break the Go mirror).
        assert p["prompt_tokens"] == 100 and p["input_tokens"] == 100
        assert p["completion_tokens"] == 50 and p["output_tokens"] == 50
        assert p["total_tokens"] == 150
        assert "tokens_prompt" not in p and "tokens_completion" not in p

    def test_llm_provider_falls_back_to_llm_system(self) -> None:
        rec = _record("LLM", attributes={"llm.model_name": "m", "llm.system": "anthropic"})
        assert span_to_events(rec)[0][1]["provider"] == "anthropic"

    def test_embedding_provider_has_no_llm_system_fallback(self) -> None:
        # Deliberate divergence from the LLM span — mirrors the Go contract.
        rec = _record("EMBEDDING", attributes={"embedding.model_name": "m", "llm.system": "openai"})
        assert span_to_events(rec)[0][1]["provider"] == "unknown"

    def test_embedding_count_from_a_list_attribute(self) -> None:
        rec = _record(
            "EMBEDDING",
            attributes={"embedding.model_name": "m", "embedding.embeddings": [[0.1], [0.2], [0.3]]},
        )
        assert span_to_events(rec)[0][1]["embedding_count"] == 3

    def test_reranker_fields(self) -> None:
        rec = _record(
            "RERANKER",
            "rerank",
            attributes={"reranker.model_name": "rerank-v3", "reranker.top_k": 5},
        )
        p = span_to_events(rec)[0][1]
        assert p["subtype"] == "reranker"
        assert p["tool_name"] == "rerank-v3"
        assert p["top_k"] == 5
        # A reranker deliberately emits no output / tool_description.
        assert "output" not in p and "tool_description" not in p

    def test_retriever_counts_flattened_document_keys(self) -> None:
        rec = _record(
            "RETRIEVER",
            attributes={
                "retrieval.documents.0.document.content": "a",
                "retrieval.documents.0.document.score": 0.9,
                "retrieval.documents.1.document.content": "b",
                "retrieval.documents.2.document.content": "c",
            },
        )
        p = span_to_events(rec)[0][1]
        assert p["document_count"] == 3
        # The corpus text itself is NEVER emitted — only the count.
        assert "documents" not in p and "content" not in p
        assert not any("retrieval.documents" in str(v) for v in p.values())

    def test_retriever_document_count_is_an_honest_measured_zero(self) -> None:
        p = span_to_events(_record("RETRIEVER"))[0][1]
        assert p["document_count"] == 0

    def test_agent_pair_shares_agent_id_and_output_is_stamped_at_span_end(self) -> None:
        rec = _record("AGENT", "research_agent")
        pairs = span_to_events(rec)
        (in_t, in_p), (out_t, out_p) = pairs
        assert (in_t, out_t) == ("agent.input", "agent.output")
        assert in_p["agent_id"] == out_p["agent_id"] == "research_agent"
        assert in_p["operation"] == out_p["operation"] == "agent"
        # The output happened when the span ENDED; stamping it at start reverses
        # the turn's chronology.
        assert in_p["timestamp"] == 1_700_000_000.0
        assert out_p["timestamp"] == 1_700_000_000.5
        assert out_p["timestamp"] > in_p["timestamp"]

    def test_chain_operation_is_chain(self) -> None:
        pairs = span_to_events(_record("CHAIN", "my_chain"))
        assert pairs[0][1]["operation"] == "chain"

    def test_triggered_guardrail_derives_policy_id_from_its_own_identity(self) -> None:
        rec = _record("GUARDRAIL", "pii_filter", status="ERROR")
        p = span_to_events(rec)[0][1]
        assert p["subtype"] == "guardrail"
        assert p["guardrail_name"] == "pii_filter"
        assert p["policy_id"] == "pii_filter"  # derived, not invented
        assert p["violation_type"] == "guardrail"

    def test_timestamp_and_latency_from_span_bounds(self) -> None:
        p = span_to_events(_record("LLM"))[0][1]
        assert p["timestamp"] == 1_700_000_000.0
        assert p["latency_ms"] == 500.0
        # latency_ms is the canonical duration field; duration_ns is schema drift.
        assert "duration_ns" not in p

    def test_session_and_user_ids_carried(self) -> None:
        rec = _record("LLM", attributes={"session.id": "sess-1", "user.id": "user-1"})
        p = span_to_events(rec)[0][1]
        assert p["session_id"] == "sess-1"
        assert p["user_id"] == "user-1"

    def test_run_id_falls_back_trace_then_span(self) -> None:
        assert span_to_events(_record("LLM"))[0][1]["run_id"] == "0af7651916cd43dd8448eb211c80319c"
        rec = _record("LLM", trace_id=None)
        assert span_to_events(rec)[0][1]["run_id"] == "00f067aa0ba902b7"
        rec = _record("LLM", trace_id=None, span_id=None)
        assert span_to_events(rec)[0][1]["run_id"] == "unknown"

    def test_long_content_is_truncated_with_a_declared_marker(self) -> None:
        rec = _record("LLM", attributes={"input.value": "x" * 2500})
        prompt = span_to_events(rec)[0][1]["prompt"]
        assert prompt.startswith("x" * 2000)
        assert prompt.endswith("...[truncated 500 chars]")


# ---------------------------------------------------------------------------
# Honesty skips — omit, never zero-fill or guess
# ---------------------------------------------------------------------------


class TestHonestySkips:
    def test_tokens_are_omitted_not_zero_filled(self) -> None:
        p = span_to_events(_record("LLM", attributes={"llm.model_name": "gpt-4o"}))[0][1]
        for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens"):
            assert key not in p, f"{key} must be OMITTED when absent, never zero-filled"

    def test_non_coercible_tokens_are_omitted(self) -> None:
        rec = _record("LLM", attributes={"llm.model_name": "m", "llm.token_count.prompt": "abc"})
        assert "prompt_tokens" not in span_to_events(rec)[0][1]

    def test_model_degrades_to_an_explicit_unknown_never_a_guess(self) -> None:
        # An explicit declared-unknown, not a model inferred from the span name.
        rec = _record("LLM", "openai.chat.completions.create")
        p = span_to_events(rec)[0][1]
        assert p["model_name"] == "unknown" and p["model"] == "unknown"
        assert p["provider"] == "unknown"

    def test_evaluator_never_fabricates_a_score(self) -> None:
        rec = _record("EVALUATOR", "hallucination_eval", attributes={"output.value": "PASS"})
        p = span_to_events(rec)[0][1]
        assert p["evaluator_name"] == "hallucination_eval"
        # OpenInference carries no normative score attribute; inventing one would
        # be a fabricated grade.
        for key in ("score", "label", "dimension", "is_passing", "threshold", "grader_id"):
            assert key not in p, f"{key} must never be fabricated on an EVALUATOR span"

    def test_negative_duration_is_dropped_not_clamped(self) -> None:
        rec = _record("LLM", start_ns=1_700_000_001_000_000_000, end_ns=1_700_000_000_000_000_000)
        assert "latency_ms" not in span_to_events(rec)[0][1]

    def test_timestamp_omitted_when_the_span_has_no_bounds(self) -> None:
        p = span_to_events(_record("LLM", start_ns=None, end_ns=None))[0][1]
        assert "timestamp" not in p

    def test_timestamp_falls_back_to_end_when_start_is_absent(self) -> None:
        p = span_to_events(_record("LLM", start_ns=None))[0][1]
        assert p["timestamp"] == 1_700_000_000.5

    def test_error_absent_for_a_clean_span(self) -> None:
        assert "error" not in span_to_events(_record("LLM", status="OK"))[0][1]
        assert "error" not in span_to_events(_record("LLM", status="UNSET"))[0][1]

    def test_names_degrade_to_declared_sentinels(self) -> None:
        assert span_to_events(_record("TOOL", ""))[0][1]["tool_name"] == "unknown"
        assert span_to_events(_record("RERANKER", ""))[0][1]["tool_name"] == "reranker"
        assert span_to_events(_record("EVALUATOR", ""))[0][1]["evaluator_name"] == "evaluator"
        assert span_to_events(_record("GUARDRAIL", "", status="OK"))[0][1]["guardrail_name"] == "guardrail"


# ---------------------------------------------------------------------------
# Identity — the span name must NOT reach the Agent column
# ---------------------------------------------------------------------------


class TestIdentityIsNotFabricated:
    def test_agent_id_carries_the_span_name_but_agent_name_never_does(self) -> None:
        # _identity.py forbids a span name as an Agent-column source and reads
        # `agent_name`; `agent_id` is read by no identity tier. Writing the span
        # name to agent_name would silently fabricate the Agent column.
        pairs = span_to_events(_record("AGENT", "research_agent"))
        for _, p in pairs:
            assert p["agent_id"] == "research_agent"
            assert "agent_name" not in p, "span name must never reach agent_name"
            assert "crew_name" not in p and "node" not in p and "from_agent" not in p

    def test_a_real_trace_renders_an_honest_blank_agent(self, mock_client: Mock) -> None:
        from layerlens.instrument._identity import honest_agent_identity

        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter.ingest_span(_record("AGENT", "research_agent"))
        adapter.flush()

        events = uploaded["events"]
        assert find_events(events, "agent.input"), "the agent turn must be recorded"
        # No identity tier fires -> the Agent column is an honest "—".
        assert honest_agent_identity(events) is None
        assert not find_events(events, "agent.identity")


# ---------------------------------------------------------------------------
# Real OpenTelemetry — real TracerProvider, real ReadableSpan, real processors
# ---------------------------------------------------------------------------


class TestRealOpenTelemetry:
    def test_live_spans_through_a_real_tracer_provider(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_Noop()))
        provider.add_span_processor(adapter.span_processor())
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span(
            "openai.chat",
            attributes={
                "openinference.span.kind": "LLM",
                "llm.model_name": "gpt-4o",
                "llm.provider": "openai",
                "llm.token_count.prompt": 12,
                "llm.token_count.completion": 4,
            },
        ):
            pass
        provider.shutdown()
        adapter.flush()

        ev = find_event(uploaded["events"], "model.invoke")
        assert ev["payload"]["model"] == "gpt-4o"
        assert ev["payload"]["prompt_tokens"] == 12
        # A real OTel span id is 16 hex chars and passes through as the envelope
        # span id, preserving the source tree for free.
        assert len(ev["span_id"]) == 16
        assert ev["span_name"] == "openai.chat"

    def test_real_batch_span_processor_export_thread(self, mock_client: Mock) -> None:
        # BatchSpanProcessor calls on_end on the SDK's own EXPORT THREAD, where
        # ContextVars do not propagate — self._emit()/_current_collector would
        # silently drop every span. The explicitly-held collector must not.
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(_NoopExporter()))
        provider.add_span_processor(adapter.span_processor())
        tracer = provider.get_tracer(__name__)
        with tracer.start_as_current_span(
            "retrieve", attributes={"openinference.span.kind": "RETRIEVER"}
        ):
            pass
        provider.shutdown()
        adapter.flush()

        assert find_event(uploaded["events"], "retrieval.query")

    def test_real_span_tree_parent_child_is_preserved(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        provider = TracerProvider()
        provider.add_span_processor(adapter.span_processor())
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span("agent", attributes={"openinference.span.kind": "AGENT"}):
            with tracer.start_as_current_span(
                "openai.chat",
                attributes={"openinference.span.kind": "LLM", "llm.model_name": "gpt-4o"},
            ):
                pass
        provider.shutdown()
        adapter.flush()

        events = uploaded["events"]
        llm = find_event(events, "model.invoke")
        agent_in = find_event(events, "agent.input")
        # The real OTel parent link survives into the LayerLens span tree.
        assert llm["parent_span_id"] == agent_in["span_id"]

    def test_real_error_status_marks_the_span(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        provider = TracerProvider()
        provider.add_span_processor(adapter.span_processor())
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span(
            "guard", attributes={"openinference.span.kind": "GUARDRAIL"}
        ) as span:
            span.set_status(Status(StatusCode.ERROR, "blocked: policy hit"))
        provider.shutdown()
        adapter.flush()

        # A REAL OTel StatusCode.ERROR enum must resolve to a violation.
        ev = find_event(uploaded["events"], "policy.violation")
        assert ev["payload"]["error"] == "blocked: policy hit"
        assert ev["payload"]["violation_type"] == "guardrail"

    def test_two_source_traces_stay_two_traces(self, mock_client: Mock) -> None:
        # One collector owns ONE trace_id: spans from N OTel traces fed through
        # one collector would merge N traces into one.
        flushed: List[Dict[str, Any]] = []

        def _capture(path: str) -> None:
            import json

            with open(path) as f:
                flushed.append(json.load(f)[0])

        mock_client.traces.upload.side_effect = _capture
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        provider = TracerProvider()
        provider.add_span_processor(adapter.span_processor())
        tracer = provider.get_tracer(__name__)

        for name in ("first", "second"):
            with tracer.start_as_current_span(
                name, attributes={"openinference.span.kind": "AGENT"}
            ):
                pass
        provider.shutdown()
        adapter.flush()

        assert len(flushed) == 2, "two source OTel traces must stay two LayerLens traces"
        assert len({t["trace_id"] for t in flushed}) == 2

    def test_on_ending_alias_exists_for_sdk_1_29_plus(self, mock_client: Mock) -> None:
        # opentelemetry-sdk >= 1.29's multi-span-processor calls the private
        # `_on_ending`; a duck-typed processor missing it raises AttributeError
        # on EVERY span end.
        proc = OpenInferenceAdapter(mock_client).span_processor()
        assert hasattr(proc, "on_ending") and hasattr(proc, "_on_ending")
        assert proc._on_ending(object()) is None

    def test_on_start_is_a_deliberate_noop(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        proc = adapter.span_processor()
        # A started span has no output and no status — nothing may be emitted.
        assert proc.on_start(object(), None) is None
        assert uploaded["events"] == []

    def test_shutdown_flushes_so_live_spans_are_never_lost(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        proc = adapter.span_processor()
        adapter.ingest_span(_record("LLM", attributes={"llm.model_name": "gpt-4o"}))
        assert uploaded["events"] == [], "nothing is uploaded before a flush"
        proc.shutdown()  # -> adapter.disconnect() -> flush
        assert find_event(uploaded["events"], "model.invoke")

    def test_on_end_never_breaks_the_host_pipeline(self, mock_client: Mock) -> None:
        adapter = OpenInferenceAdapter(mock_client)
        adapter.connect()
        proc = adapter.span_processor()

        # A dict-shaped span whose access raises reaches _record_from_dict, which
        # has no guard of its own — so this genuinely escapes ingest_span and
        # on_end's blanket except is the ONLY thing protecting the host's OTel
        # pipeline. (An object-shaped span is swallowed earlier by
        # _record_from_otel, which would make this assertion vacuous.)
        class _HostileDict(Dict[str, Any]):
            def get(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("host span exploded")

        hostile = _HostileDict()
        with pytest.raises(RuntimeError):
            adapter.ingest_span(hostile)  # proves the guard below is load-bearing

        proc.on_end(hostile)  # must NOT raise


class _Noop:
    def export(self, spans: Any) -> Any:
        return None

    def shutdown(self) -> None:
        return None


class _NoopExporter:
    def export(self, spans: Any) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# ---------------------------------------------------------------------------
# Span extraction / normalization
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_otlp_json_span_dict(self, mock_client: Mock) -> None:
        adapter = OpenInferenceAdapter(mock_client)
        span = {
            "name": "openai.chat",
            "traceId": "0af7651916cd43dd8448eb211c80319c",
            "spanId": "00f067aa0ba902b7",
            "startTimeUnixNano": "1700000000000000000",
            "endTimeUnixNano": "1700000000500000000",
            "status": {"code": 2, "message": "boom"},
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
                {"key": "llm.model_name", "value": {"stringValue": "gpt-4o"}},
                {"key": "llm.token_count.prompt", "value": {"intValue": "42"}},
            ],
        }
        rec = adapter._extract_record(span)
        assert rec is not None
        assert rec["span_kind"] == "LLM"
        assert rec["attributes"]["llm.model_name"] == "gpt-4o"
        assert rec["start_ns"] == 1_700_000_000_000_000_000
        # OTLP int status 2 == STATUS_CODE_ERROR must resolve to ERROR.
        assert rec["status"] == "ERROR"
        assert rec["status_message"] == "boom"

    def test_int_ids_are_zero_padded_so_live_and_offline_agree(self, mock_client: Mock) -> None:
        adapter = OpenInferenceAdapter(mock_client)
        rec = adapter._extract_record(
            {
                "name": "s",
                "trace_id": 0x0AF7651916CD43DD8448EB211C80319C,
                "span_id": 0x00F067AA0BA902B7,
                "attributes": {"openinference.span.kind": "LLM"},
            }
        )
        assert rec is not None
        # Unpadded hex would drop the leading zeros and break correlation with
        # the same span ingested live.
        assert rec["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
        assert rec["span_id"] == "00f067aa0ba902b7"
        assert len(rec["span_id"]) == 16

    def test_pre_extracted_record_still_gets_normalized(self, mock_client: Mock) -> None:
        # A dict carrying BOTH span_kind and attributes looks like an
        # already-extracted SpanRecord, but a hand-rolled exporter plausibly emits
        # one while still spelling its times/ids the raw way. Returning it verbatim
        # (no id coercion, no timestamp normalization) silently loses timestamp /
        # latency and leaves ids in a shape that will not correlate — so the keys
        # used here are deliberately the UNnormalized ones.
        adapter = OpenInferenceAdapter(mock_client)
        rec = adapter._extract_record(
            {
                "span_kind": "llm",
                "attributes": {"llm.model_name": "gpt-4o"},
                "name": "s",
                "trace_id": 0x0AF7651916CD43DD8448EB211C80319C,
                "span_id": 0x00F067AA0BA902B7,
                "startTimeUnixNano": "1700000000000000000",
                "endTimeUnixNano": "1700000000500000000",
                "status": {"code": 2, "message": "boom"},
            }
        )
        assert rec is not None
        assert rec["span_kind"] == "LLM"
        assert rec["start_ns"] == 1_700_000_000_000_000_000
        assert rec["end_ns"] == 1_700_000_000_500_000_000
        assert rec["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
        assert rec["span_id"] == "00f067aa0ba902b7"
        assert rec["status"] == "ERROR"
        # ...and it still produces a fully-timed event.
        p = span_to_events(rec)[0][1]
        assert p["timestamp"] == 1_700_000_000.0
        assert p["latency_ms"] == 500.0

    def test_iso8601_times_are_parsed(self, mock_client: Mock) -> None:
        adapter = OpenInferenceAdapter(mock_client)
        rec = adapter._extract_record(
            {
                "name": "s",
                "attributes": {"openinference.span.kind": "LLM"},
                "start_time": "2023-11-14T22:13:20+00:00",
            }
        )
        assert rec is not None
        assert rec["start_ns"] == 1_700_000_000_000_000_000

    def test_iso8601_with_nanosecond_precision(self, mock_client: Mock) -> None:
        # An OTLP JSON dump can carry 9-digit fractional seconds, which
        # datetime.fromisoformat rejects before 3.11.
        adapter = OpenInferenceAdapter(mock_client)
        rec = adapter._extract_record(
            {
                "name": "s",
                "attributes": {"openinference.span.kind": "LLM"},
                "start_time": "2023-11-14T22:13:20.123456789Z",
            }
        )
        assert rec is not None
        assert rec["start_ns"] == 1_700_000_000_123_456_000

    def test_declared_nanos_are_trusted_not_magnitude_guessed(self, mock_client: Mock) -> None:
        # A synthetic/fixture span 1s into a relative timeline has magnitude 1e9,
        # which magnitude-detection misreads as a SECONDS epoch (-> 1e18).
        adapter = OpenInferenceAdapter(mock_client)
        rec = adapter._extract_record(
            {
                "name": "s",
                "attributes": {"openinference.span.kind": "LLM"},
                "startTimeUnixNano": 1_000_000_000,
                "endTimeUnixNano": 2_000_000_000,
            }
        )
        assert rec is not None
        assert rec["start_ns"] == 1_000_000_000
        assert rec["end_ns"] == 2_000_000_000

    def test_a_malformed_span_yields_no_events_and_no_count(self, mock_client: Mock) -> None:
        adapter = OpenInferenceAdapter(mock_client)
        adapter.connect()

        class _Exploding:
            @property
            def attributes(self) -> Any:
                raise RuntimeError("nope")

        assert adapter.ingest_span(_Exploding()) == 0
        assert adapter._spans_ingested == 0  # the failure is not counted as a success

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ({"stringValue": "hi"}, "hi"),
            ({"intValue": "42"}, "42"),
            ({"doubleValue": 1.5}, 1.5),
            ({"boolValue": True}, True),
            ({"arrayValue": {"values": [{"stringValue": "a"}, {"stringValue": "b"}]}}, ["a", "b"]),
            ("plain", "plain"),
        ],
    )
    def test_otlp_value_unwrapping(self, raw: Any, expected: Any) -> None:
        from layerlens.instrument.adapters.frameworks.openinference import _otlp_value

        assert _otlp_value(raw) == expected


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_connect_records_environment_and_hooks_nothing(self, mock_client: Mock) -> None:
        adapter = OpenInferenceAdapter(mock_client)
        adapter.connect()
        info = adapter.adapter_info()
        assert info.name == "openinference"
        assert info.adapter_type == "framework"
        assert info.connected is True
        assert info.metadata["otel_available"] is True
        assert info.metadata["spans_ingested"] == 0

    def test_connect_never_raises_when_otel_is_missing(self, mock_client: Mock, monkeypatch) -> None:
        # The adapter ingests plain span dicts with nothing installed; a missing
        # dependency must degrade, never raise.
        import layerlens.instrument.adapters.frameworks.openinference as mod

        monkeypatch.setattr(mod, "_detect_otel_version", lambda: None)
        monkeypatch.setattr(mod, "_detect_openinference_version", lambda: None)
        adapter = OpenInferenceAdapter(mock_client)
        adapter.connect()  # must not raise
        assert adapter.adapter_info().metadata["otel_available"] is False
        # ...and it still ingests.
        assert adapter.ingest_span(_record("LLM", attributes={"llm.model_name": "m"})) == 1

    def test_ingest_spans_sums_events(self, mock_client: Mock) -> None:
        adapter = OpenInferenceAdapter(mock_client)
        adapter.connect()
        # LLM=1, AGENT=2 (a pair), RETRIEVER=1
        n = adapter.ingest_spans([_record("LLM"), _record("AGENT"), _record("RETRIEVER")])
        assert n == 4
        assert adapter._spans_ingested == 3

    def test_instrument_openinference_connects_and_registers(self, mock_client: Mock) -> None:
        from layerlens.instrument.adapters._registry import get, disconnect_all

        try:
            adapter = instrument_openinference(mock_client)
            assert isinstance(adapter, OpenInferenceAdapter)
            assert adapter.is_connected
            assert get("openinference") is adapter
        finally:
            disconnect_all()

    def test_disconnect_flushes_pending_traces(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter.ingest_span(_record("LLM", attributes={"llm.model_name": "gpt-4o"}))
        adapter.disconnect()
        assert find_event(uploaded["events"], "model.invoke")
        assert not adapter.is_connected
