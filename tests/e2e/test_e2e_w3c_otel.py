"""End-to-end: W3C Trace Context propagation + OTel GenAI semconv.

Exercises:
- inject_headers / extract_headers round-trip inside a real trace context
- new_traceparent inside and outside a trace
- gen_ai_attributes embedded in real model.invoke events via the OpenAI
  provider (with a mocked OpenAI client so we don't hit the network)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from layerlens.instrument import (
    span,
    trace,
    trace_context,
    inject_headers,
    extract_headers,
    new_traceparent,
)
from layerlens.instrument._w3c import _shorten_trace_id, _parse_traceparent
from layerlens.instrument.adapters.providers.openai import instrument_openai

from .conftest import first_event


class TestPropagationRoundTrip:
    def test_inject_inside_extract_outside(self, client_and_uploads):
        client, _ = client_and_uploads

        with trace_context(client) as parent:
            headers = inject_headers({})

        parsed = extract_headers(headers)
        # Our 16-hex trace_id round-trips through the 32-hex W3C wire form.
        assert parsed["trace_id"] == parent.trace_id

    def test_nested_spans_get_their_own_span_id(self, client_and_uploads):
        client, _ = client_and_uploads

        with trace_context(client):
            outer_tp = inject_headers({})["traceparent"]
            with span("child-span"):
                inner_tp = inject_headers({})["traceparent"]

        outer = _parse_traceparent(outer_tp)
        inner = _parse_traceparent(inner_tp)
        assert outer["trace_id"] == inner["trace_id"]  # same trace
        # Different span_ids -> different parent_span_id positions
        assert outer["parent_span_id"] != inner["parent_span_id"]

    def test_new_traceparent_inside_trace(self, client_and_uploads):
        client, _ = client_and_uploads
        with trace_context(client) as parent:
            tp = new_traceparent()
        parsed = _parse_traceparent(tp)
        assert parsed is not None
        assert _shorten_trace_id(parsed["trace_id"]) == parent.trace_id

    def test_new_traceparent_outside_trace_still_valid(self):
        # No active context — function should still produce a well-formed header
        tp = new_traceparent()
        parsed = _parse_traceparent(tp)
        assert parsed is not None
        assert len(parsed["trace_id"]) == 32

    def test_extract_rejects_malformed_header(self):
        assert extract_headers({"traceparent": "not-a-traceparent"}) == {}


class TestOTelGenAiAttributesInRealProviderCall:
    """Wire instrument_openai to a mock OpenAI client and verify the
    model.invoke event payload has the expected ``otel_gen_ai`` block."""

    def _fake_chat_response(self):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(role="assistant", content="answer", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8, total_tokens=20),
            model="gpt-4o-2024-11-20",
            id="chatcmpl-test-1",
            system_fingerprint="fp_test",
            service_tier="default",
        )

    def test_model_invoke_has_otel_gen_ai_block(self, client_and_uploads):
        client, uploads = client_and_uploads

        # Build a minimal fake OpenAI client with the shape our provider expects
        fake_create = Mock(return_value=self._fake_chat_response())
        fake_openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

        provider = instrument_openai(fake_openai)

        @trace(client)
        def ask(question: str) -> str:
            resp = fake_openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": question}],
                temperature=0.5,
            )
            return resp.choices[0].message.content

        try:
            out = ask("hello?")
        finally:
            provider.disconnect()

        assert out == "answer"
        invoke = first_event(uploads, "model.invoke")
        otel = invoke["payload"].get("otel_gen_ai") or {}
        assert otel["gen_ai.system"] == "openai"
        assert otel["gen_ai.operation.name"] == "chat"
        assert otel["gen_ai.request.model"] == "gpt-4o"
        assert otel["gen_ai.request.temperature"] == 0.5
        assert otel["gen_ai.response.model"] == "gpt-4o-2024-11-20"
        assert otel["gen_ai.response.id"] == "chatcmpl-test-1"
        assert otel["gen_ai.response.finish_reasons"] == ["stop"]
        assert otel["gen_ai.usage.input_tokens"] == 12
        assert otel["gen_ai.usage.output_tokens"] == 8


class TestNoOpOutsideTrace:
    def test_inject_outside_trace_is_passthrough(self):
        headers = {"x-existing": "v"}
        result = inject_headers(headers)
        assert result is headers
        assert result == {"x-existing": "v"}  # no traceparent added
