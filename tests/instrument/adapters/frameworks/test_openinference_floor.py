"""Offline redaction + error + attestation + cost floor for the OpenInference adapter.

Runs in plain CI with no credentials and no network. Every span here is a REAL
``ReadableSpan`` produced by a real ``TracerProvider`` (or a real OTLP JSON export
of one), not a hand-built stub — the OpenInference attribute keys are the ones the
real instrumentors emit.

* Redaction   — ``capture_content=False`` keeps the full span topology (ids, kinds,
                counts, model, tokens, latency) while a SENTINEL sweep over the
                whole serialized trace proves no prompt / output / metadata /
                invocation-params / tool-description text survives. A
                ``capture_content=True`` VACUITY CONTROL proves the same path DOES
                carry that content otherwise, so the sweep can fail.
* Error       — a REAL ``opentelemetry.trace.Status(StatusCode.ERROR)`` set on a real
                span (and the real OTLP int-enum wire form of it) surfaces as the
                honest error signal, and a triggered GUARDRAIL becomes a
                policy.violation rather than a clean tool.call.
* Attestation — offline ``verify_chain`` over the collected payload, plus a TAMPER
                control that must FAIL.
* Cost        — a priced model + real token counts yields an honestly-derived
                cost.record; an unpriced/unknown model yields NO cost.record at all
                (the honest-omission proof) — never a fabricated 0.0.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.trace import Status, StatusCode  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402

from layerlens.attestation import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.openinference import (  # noqa: E402
    OpenInferenceAdapter,
)

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

# Sentinels: each is a distinct, greppable string standing in for one class of
# user content that a real OpenInference span carries.
S_PROMPT = "SENTINEL_PROMPT_alpha_ssn_123_45_6789"
S_OUTPUT = "SENTINEL_OUTPUT_bravo_diagnosis_text"
S_METADATA = "SENTINEL_METADATA_charlie_user_context"
S_PARAMS = "SENTINEL_INVOCATION_PARAMS_delta_tool_schema"
S_TOOLDESC = "SENTINEL_TOOL_DESCRIPTION_echo_free_text"
S_TOOLIN = "SENTINEL_TOOL_INPUT_foxtrot_args"
S_QUERY = "SENTINEL_RETRIEVER_QUERY_golf_question"
S_EVAL = "SENTINEL_EVAL_RESULT_hotel_reasoning"
S_GUARD = "SENTINEL_GUARDRAIL_OUTPUT_india_blocked_reason"
S_RERANK = "SENTINEL_RERANKER_QUERY_juliett"
S_EMBED = "SENTINEL_EMBEDDING_INPUT_kilo"
S_AGENT_IN = "SENTINEL_AGENT_INPUT_lima"
S_AGENT_OUT = "SENTINEL_AGENT_OUTPUT_mike"
S_ERRMSG = "SENTINEL_STATUS_MESSAGE_november_leaked_prompt"

ALL_SENTINELS = [
    S_PROMPT,
    S_OUTPUT,
    S_METADATA,
    S_PARAMS,
    S_TOOLDESC,
    S_TOOLIN,
    S_QUERY,
    S_EVAL,
    S_GUARD,
    S_RERANK,
    S_EMBED,
    S_AGENT_IN,
    S_AGENT_OUT,
    S_ERRMSG,
]


def _drive_every_span_kind(adapter: OpenInferenceAdapter) -> None:
    """Emit one REAL OTel span per OpenInference kind, each loaded with content."""
    provider = TracerProvider()
    provider.add_span_processor(adapter.span_processor())
    tracer = provider.get_tracer(__name__)

    with tracer.start_as_current_span(
        "assistant",
        attributes={
            "openinference.span.kind": "AGENT",
            "input.value": S_AGENT_IN,
            "output.value": S_AGENT_OUT,
            "metadata": S_METADATA,
        },
    ):
        with tracer.start_as_current_span(
            "openai.chat",
            attributes={
                "openinference.span.kind": "LLM",
                "llm.model_name": "gpt-4o",
                "llm.provider": "openai",
                "llm.token_count.prompt": 100,
                "llm.token_count.completion": 50,
                "llm.token_count.total": 150,
                "llm.invocation_parameters": S_PARAMS,
                "input.value": S_PROMPT,
                "output.value": S_OUTPUT,
                "session.id": "sess-42",
                "user.id": "user-7",
            },
        ):
            pass
        with tracer.start_as_current_span(
            "lookup_customer",
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.name": "lookup_customer",
                "tool.description": S_TOOLDESC,
                "tool.parameters": S_TOOLIN,
                "output.value": S_OUTPUT,
            },
        ):
            pass
        with tracer.start_as_current_span(
            "search",
            attributes={
                "openinference.span.kind": "RETRIEVER",
                "input.value": S_QUERY,
                "retrieval.documents.0.document.content": "corpus text 0",
                "retrieval.documents.1.document.content": "corpus text 1",
            },
        ):
            pass
        with tracer.start_as_current_span(
            "embed",
            attributes={
                "openinference.span.kind": "EMBEDDING",
                "embedding.model_name": "text-embedding-3-small",
                "llm.token_count.prompt": 8,
                "input.value": S_EMBED,
            },
        ):
            pass
        with tracer.start_as_current_span(
            "rerank",
            attributes={
                "openinference.span.kind": "RERANKER",
                "reranker.model_name": "rerank-v3",
                "reranker.top_k": 3,
                "reranker.query": S_RERANK,
            },
        ):
            pass
        with tracer.start_as_current_span(
            "hallucination_eval",
            attributes={"openinference.span.kind": "EVALUATOR", "output.value": S_EVAL},
        ):
            pass
        with tracer.start_as_current_span(
            "pii_guard", attributes={"openinference.span.kind": "GUARDRAIL", "output.value": S_GUARD}
        ) as guard:
            guard.set_status(Status(StatusCode.ERROR, S_ERRMSG))
    provider.shutdown()
    adapter.flush()


class TestRedaction:
    def test_no_content_strips_every_sentinel_but_keeps_topology(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.standard())
        assert adapter._config.capture_content is False
        adapter.connect()
        _drive_every_span_kind(adapter)

        events = uploaded["events"]
        assert events, "no-content mode must still record the trace"

        # SENTINEL SWEEP over the WHOLE serialized trace — payloads, envelopes,
        # span names, attestation, everything that would be uploaded.
        blob = json.dumps(uploaded)
        for sentinel in ALL_SENTINELS:
            assert sentinel not in blob, f"{sentinel} leaked under capture_content=False"

        # ...and the STRUCTURE survives: every span kind still produced its event.
        for event_type in (
            "model.invoke",
            "embedding.create",
            "tool.call",
            "retrieval.query",
            "evaluation.result",
            "agent.input",
            "agent.output",
            "policy.violation",
        ):
            assert find_events(events, event_type), f"{event_type} lost under no-content"

        llm = find_event(events, "model.invoke")["payload"]
        assert llm["model"] == "gpt-4o"  # model is metadata, not content
        assert llm["prompt_tokens"] == 100 and llm["completion_tokens"] == 50
        assert llm["latency_ms"] >= 0
        assert llm["session_id"] == "sess-42" and llm["user_id"] == "user-7"
        assert "prompt" not in llm and "output" not in llm
        assert "invocation_parameters" not in llm

        tool = find_event(events, "tool.call")["payload"]
        assert tool["tool_name"] == "lookup_customer"  # the NAME is metadata
        assert "tool_description" not in tool  # the free-text description is not
        assert "input" not in tool and "output" not in tool

        retr = find_event(events, "retrieval.query")["payload"]
        assert retr["document_count"] == 2  # the count survives; the corpus never ships
        assert "query" not in retr

        ev = find_event(events, "evaluation.result")["payload"]
        assert ev["evaluator_name"] == "hallucination_eval"
        assert "result" not in ev

        # An agent turn is still RECORDED under privacy mode — the event exists and
        # its topology survives; only the turn's words are gone. ateam keeps
        # input_text/output_text present-but-empty because ITS ingest requires the
        # key; LayerLens has no such requirement (tests/instrument/_event_schema.py
        # never asks for them), so the collector backstop drops them outright, which
        # is strictly the more private of the two.
        agent_in = find_event(events, "agent.input")["payload"]
        agent_out = find_event(events, "agent.output")["payload"]
        assert not agent_in.get("input_text") and not agent_out.get("output_text")
        assert agent_in["agent_id"] == "assistant"  # topology survives
        assert "input" not in agent_in and "output" not in agent_out

        # The violation still fires and still names its policy.
        viol = find_event(events, "policy.violation")["payload"]
        assert viol["policy_id"] == "pii_guard"
        assert viol["violation_type"] == "guardrail"
        # The failure stays VISIBLE — only the producer's free text is stripped.
        assert viol["error"] == "span status ERROR"

    def test_vacuity_control_content_does_flow_when_capturing(self, mock_client: Mock) -> None:
        # Proves the sweep above can fail: the identical path DOES carry every
        # sentinel under capture_content=True.
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        assert adapter._config.capture_content is True
        adapter.connect()
        _drive_every_span_kind(adapter)

        blob = json.dumps(uploaded)
        for sentinel in ALL_SENTINELS:
            assert sentinel in blob, (
                f"{sentinel} absent even with capture_content=True — the redaction "
                "test would pass vacuously"
            )

    def test_metadata_is_gated(self, mock_client: Mock) -> None:
        # metadata is an arbitrary producer blob (routinely user/session context),
        # and no _CONTENT_KEYS entry covers it — the emit-site gate is the only
        # thing standing between it and the wire.
        for capture, expected in ((False, False), (True, True)):
            uploaded = capture_framework_trace(mock_client)
            cfg = CaptureConfig.full() if capture else CaptureConfig.standard()
            adapter = OpenInferenceAdapter(mock_client, capture_config=cfg)
            adapter.connect()
            adapter.ingest_span(
                {
                    "name": "s",
                    "attributes": {"openinference.span.kind": "LLM", "metadata": S_METADATA},
                    "trace_id": "t1",
                    "span_id": "s1",
                }
            )
            adapter.flush()
            payload = find_event(uploaded["events"], "model.invoke")["payload"]
            assert ("metadata" in payload) is expected


class TestError:
    def test_real_otel_error_status_enum_surfaces_the_message(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        provider = TracerProvider()
        provider.add_span_processor(adapter.span_processor())
        tracer = provider.get_tracer(__name__)
        with tracer.start_as_current_span(
            "openai.chat",
            attributes={"openinference.span.kind": "LLM", "llm.model_name": "gpt-4o"},
        ) as span:
            # The shape a real instrumented LLM failure carries.
            span.set_status(Status(StatusCode.ERROR, "RateLimitError: 429 Too Many Requests"))
        provider.shutdown()
        adapter.flush()

        payload = find_event(uploaded["events"], "model.invoke")["payload"]
        assert payload["error"] == "RateLimitError: 429 Too Many Requests"

    def _drive_llm(self, mock_client: Mock, config: CaptureConfig, status: Status | None) -> Any:
        """Drive ONE real LLM span through a real TracerProvider; return its model.invoke."""
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=config)
        adapter.connect()
        provider = TracerProvider()
        provider.add_span_processor(adapter.span_processor())
        tracer = provider.get_tracer(__name__)
        with tracer.start_as_current_span(
            "openai.chat",
            attributes={"openinference.span.kind": "LLM", "llm.model_name": "gpt-4o"},
        ) as span:
            if status is not None:
                span.set_status(status)
        provider.shutdown()
        adapter.flush()
        return find_event(uploaded["events"], "model.invoke")["payload"]

    def test_errored_model_invoke_is_distinguishable_from_a_clean_one_by_DEFAULT(
        self, mock_client: Mock
    ) -> None:
        """A failed LLM call must not look identical to a successful one under the DEFAULT config.

        Every other test in this class runs ``CaptureConfig.full()``, which is NOT
        the default — ``FrameworkAdapter.__init__`` defaults to ``standard()``
        (capture_content=False). Under it the collector-tier backstop STRIPS
        ``error`` (it is in ``_CONTENT_KEYS["model.invoke"]``), so the honest
        content-free ``_ERROR_SIGNAL`` the adapter stamps never reaches the wire.
        A content-free ``status`` is the only field that survives redaction
        (no ``_CONTENT_KEYS`` entry strips it), and it is what keeps the failure
        visible (LAY-3620, redact-without-going-blind).
        """
        errored = self._drive_llm(
            mock_client, CaptureConfig.standard(), Status(StatusCode.ERROR, "RateLimitError: 429")
        )
        clean = self._drive_llm(mock_client, CaptureConfig.standard(), Status(StatusCode.OK))

        assert errored["status"] == "ERROR"
        assert clean["status"] == "OK"
        # The whole point: the two must not be indistinguishable.
        volatile = {"span_id", "trace_id", "run_id", "timestamp", "latency_ms"}
        assert {k: v for k, v in errored.items() if k not in volatile} != {
            k: v for k, v in clean.items() if k not in volatile
        }, "an errored model.invoke is byte-for-byte identical to a successful one"
        # Redaction still holds: the producer's free text never survives.
        assert "RateLimitError" not in json.dumps(errored)

    def test_errored_agent_output_keeps_a_failure_signal_under_the_DEFAULT_config(
        self, mock_client: Mock
    ) -> None:
        """``agent.output`` strips ``error`` too — its paired agent.input does not.

        That asymmetry (agent.input KEEPS error='span status ERROR' while its own
        agent.output LOSES it) is what proves the blinding was accidental. Both
        halves of the turn must report the failure.
        """
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.standard())
        adapter.connect()
        provider = TracerProvider()
        provider.add_span_processor(adapter.span_processor())
        tracer = provider.get_tracer(__name__)
        with tracer.start_as_current_span(
            "assistant", attributes={"openinference.span.kind": "AGENT"}
        ) as span:
            span.set_status(Status(StatusCode.ERROR, "ToolExecutionError: boom"))
        provider.shutdown()
        adapter.flush()

        events = uploaded["events"]
        assert find_event(events, "agent.input")["payload"]["status"] == "ERROR"
        assert find_event(events, "agent.output")["payload"]["status"] == "ERROR"
        assert "ToolExecutionError" not in json.dumps(events)

    def test_status_is_never_fabricated_when_the_span_declares_none(
        self, mock_client: Mock
    ) -> None:
        # An honest omission: a span that never set a status must not be reported
        # as OK. OTel's own default for an unset status is UNSET, which is what a
        # real TracerProvider stamps — never a guessed "OK".
        payload = self._drive_llm(mock_client, CaptureConfig.standard(), None)
        assert payload.get("status") in (None, "UNSET")
        assert payload.get("status") != "OK"

    def test_real_recorded_exception_span_is_an_error(self, mock_client: Mock) -> None:
        # A REAL SDK exception escaping a real instrumented span: OTel records the
        # exception and marks the span ERROR via its own machinery.
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        provider = TracerProvider()
        provider.add_span_processor(adapter.span_processor())
        tracer = provider.get_tracer(__name__)

        class RateLimitError(Exception):
            """The shape an openai SDK rate-limit raises through an instrumented call."""

        with pytest.raises(RateLimitError):
            with tracer.start_as_current_span(
                "openai.chat",
                attributes={"openinference.span.kind": "LLM", "llm.model_name": "gpt-4o"},
            ):
                raise RateLimitError("429 rate limit exceeded")
        provider.shutdown()
        adapter.flush()

        payload = find_event(uploaded["events"], "model.invoke")["payload"]
        # OTel's own record_exception path sets StatusCode.ERROR with the message.
        assert "error" in payload
        assert "429 rate limit exceeded" in payload["error"]

    def test_otlp_int_status_enum_triggers_the_violation(self, mock_client: Mock) -> None:
        # The OTLP JSON wire form of an errored span states its status as the int
        # enum 2. A status that fails to resolve silently downgrades a REAL
        # triggered guardrail to a clean tool.call — a lost violation.
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter.ingest_span(
            {
                "name": "pii_guard",
                "traceId": "0af7651916cd43dd8448eb211c80319c",
                "spanId": "00f067aa0ba902b7",
                "status": {"code": 2, "message": "blocked"},
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "GUARDRAIL"}}
                ],
            }
        )
        adapter.flush()

        events = uploaded["events"]
        assert find_events(events, "policy.violation"), (
            "an OTLP int status 2 (STATUS_CODE_ERROR) must trigger the violation"
        )
        assert not find_events(events, "tool.call")
        assert find_event(events, "policy.violation")["payload"]["error"] == "blocked"

    @pytest.mark.parametrize(
        "status", [{"code": 2}, {"code": "STATUS_CODE_ERROR"}, {"code": "ERROR"}, "ERROR", 2]
    )
    def test_every_error_status_spelling_resolves(self, mock_client: Mock, status: Any) -> None:
        adapter = OpenInferenceAdapter(mock_client)
        rec = adapter._extract_record(
            {"name": "g", "attributes": {"openinference.span.kind": "GUARDRAIL"}, "status": status}
        )
        assert rec is not None and rec["status"] == "ERROR", f"{status!r} must resolve to ERROR"

    def test_unset_status_zero_is_not_an_error(self, mock_client: Mock) -> None:
        # OTLP code 0 is STATUS_CODE_UNSET — a valid enum value, not "absent".
        adapter = OpenInferenceAdapter(mock_client)
        rec = adapter._extract_record(
            {"name": "g", "attributes": {"openinference.span.kind": "GUARDRAIL"}, "status": {"code": 0}}
        )
        assert rec is not None and rec["status"] == "UNSET"


class TestAttestation:
    def test_chain_verifies_over_a_real_span_trace(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        _drive_every_span_kind(adapter)

        events = uploaded["events"]
        assert events, "real spans must flush a non-empty trace"
        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(
                hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash")
            )
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured"
        assert len(envelopes) == len(events)
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

    def test_tamper_control_must_fail(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        _drive_every_span_kind(adapter)

        raw = ((uploaded["attestation"] or {}).get("chain") or {}).get("events") or []
        assert len(raw) > 1, "need >1 event to break a link"
        envelopes = [
            AttestationEnvelope(
                hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash")
            )
            for e in raw
        ]
        tampered = list(envelopes)
        tampered[1] = AttestationEnvelope(
            hash="0" * 64, scope=tampered[1].scope, previous_hash=tampered[1].previous_hash
        )
        broken = verify_chain(tampered)
        assert not broken.valid, "verify_chain failed to detect a broken link"
        # The break surfaces at the FOLLOWING link, whose previous_hash no longer
        # matches the rewritten event's hash.
        assert broken.break_index == 2


class TestCost:
    def test_priced_model_yields_an_honestly_derived_cost(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter.ingest_span(
            {
                "name": "openai.chat",
                "trace_id": "t1",
                "span_id": "s1",
                "attributes": {
                    "openinference.span.kind": "LLM",
                    "llm.model_name": "gpt-4o",
                    "llm.token_count.prompt": 1000,
                    "llm.token_count.completion": 500,
                    "llm.token_count.total": 1500,
                },
            }
        )
        adapter.flush()

        cost = find_event(uploaded["events"], "cost.record")["payload"]
        assert cost["model"] == "gpt-4o"
        assert cost["prompt_tokens"] == 1000 and cost["completion_tokens"] == 500
        # Derived from the real pricing table, not invented.
        from layerlens.instrument.adapters.providers.pricing import PRICING, calculate_cost
        from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage

        expected = calculate_cost(
            "gpt-4o",
            NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
            PRICING,
        )
        assert cost["cost_usd"] == expected
        assert cost["cost_usd"] > 0

    def test_unknown_model_emits_no_cost_record_at_all(self, mock_client: Mock) -> None:
        # HONEST OMISSION: an OpenInference span with no declared model degrades
        # the model to the "unknown" sentinel, which prices to nothing. No
        # cost.record may be emitted — never a fabricated 0.0.
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter.ingest_span(
            {
                "name": "some.llm.call",
                "trace_id": "t1",
                "span_id": "s1",
                "attributes": {
                    "openinference.span.kind": "LLM",
                    "llm.token_count.prompt": 1000,
                    "llm.token_count.completion": 500,
                },
            }
        )
        adapter.flush()

        events = uploaded["events"]
        assert find_event(events, "model.invoke")["payload"]["model"] == "unknown"
        assert not find_events(events, "cost.record"), (
            "an unpriceable model must yield NO cost.record, not a fabricated zero"
        )

    def test_unpriced_model_emits_no_cost_record(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter.ingest_span(
            {
                "name": "ollama.chat",
                "trace_id": "t1",
                "span_id": "s1",
                "attributes": {
                    "openinference.span.kind": "LLM",
                    "llm.model_name": "some-local-llama-build",
                    "llm.token_count.prompt": 10,
                    "llm.token_count.completion": 5,
                },
            }
        )
        adapter.flush()
        assert not find_events(uploaded["events"], "cost.record")

    def test_no_tokens_means_no_cost_record(self, mock_client: Mock) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter.ingest_span(
            {
                "name": "openai.chat",
                "trace_id": "t1",
                "span_id": "s1",
                "attributes": {"openinference.span.kind": "LLM", "llm.model_name": "gpt-4o"},
            }
        )
        adapter.flush()
        events = uploaded["events"]
        assert find_events(events, "model.invoke")
        assert not find_events(events, "cost.record")

    def test_embedding_span_carries_tokens_without_a_fabricated_cost(self, mock_client: Mock) -> None:
        # Only LLM spans price; an embedding span keeps its honest token counts.
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenInferenceAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter.ingest_span(
            {
                "name": "embed",
                "trace_id": "t1",
                "span_id": "s1",
                "attributes": {
                    "openinference.span.kind": "EMBEDDING",
                    "embedding.model_name": "text-embedding-3-small",
                    "llm.token_count.prompt": 8,
                },
            }
        )
        adapter.flush()
        emb = find_event(uploaded["events"], "embedding.create")["payload"]
        assert emb["input_tokens"] == 8 and emb["prompt_tokens"] == 8
        assert "cost_usd" not in emb
