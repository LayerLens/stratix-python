"""Tests for W3C Trace Context propagation + OTel GenAI semantic conventions."""

from __future__ import annotations

from unittest.mock import Mock

from layerlens.instrument import (
    trace,
    trace_context,
    inject_headers,
    extract_headers,
    new_traceparent,
)
from layerlens.instrument._w3c import (
    _expand_trace_id,
    _shorten_trace_id,
    gen_ai_attributes,
    _build_traceparent,
    _parse_traceparent,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestTraceparentFormat:
    def test_build_traceparent_shape(self):
        result = _build_traceparent("abc123", "def456")
        parts = result.split("-")
        assert len(parts) == 4
        assert parts[0] == "00"
        assert len(parts[1]) == 32
        assert len(parts[2]) == 16
        assert parts[3] == "01"

    def test_round_trip(self):
        tp = _build_traceparent("0123456789abcdef", "fedcba9876543210")
        parsed = _parse_traceparent(tp)
        assert parsed is not None
        assert parsed["trace_id"].endswith("0123456789abcdef")
        assert parsed["parent_span_id"] == "fedcba9876543210"
        assert parsed["trace_flags"] == "01"

    def test_parse_rejects_short_value(self):
        assert _parse_traceparent("not-enough") is None

    def test_parse_rejects_wrong_lengths(self):
        assert _parse_traceparent("00-tooshort-tooshort-01") is None

    def test_shorten_trace_id(self):
        # 32-char W3C id is shortened to its trailing 16 chars
        assert _shorten_trace_id("0" * 16 + "1234567890abcdef") == "1234567890abcdef"

    def test_expand_short_id_yields_32_chars(self):
        result = _expand_trace_id("abc123")
        assert len(result) == 32


# ---------------------------------------------------------------------------
# inject / extract
# ---------------------------------------------------------------------------


class TestInjectHeaders:
    def test_no_op_outside_trace(self):
        # No active collector / span -> headers unchanged
        headers: dict = {}
        result = inject_headers(headers)
        assert result == {}
        assert "traceparent" not in result

    def test_injects_inside_trace(self):
        client = Mock()
        client.traces = Mock()
        client.traces.upload = Mock()

        @trace(client)
        def f():
            return inject_headers({})

        headers = f()
        assert "traceparent" in headers
        parsed = _parse_traceparent(headers["traceparent"])
        assert parsed is not None
        assert parsed["version"] == "00"

    def test_inject_preserves_existing_keys(self):
        client = Mock()
        client.traces = Mock()
        client.traces.upload = Mock()

        @trace(client)
        def f():
            return inject_headers({"x-request-id": "rid-123"})

        headers = f()
        assert headers["x-request-id"] == "rid-123"
        assert "traceparent" in headers


class TestExtractHeaders:
    def test_returns_empty_when_no_traceparent(self):
        assert extract_headers({"x-request-id": "rid"}) == {}

    def test_parses_well_formed_header(self):
        tp = _build_traceparent("trace1234567890a", "span1234567890ab")
        result = extract_headers({"traceparent": tp})
        assert result["trace_id"] == "trace1234567890a"
        assert result["parent_span_id"] == "span1234567890ab"
        assert result["trace_flags"] == "01"
        assert "raw_trace_id" in result

    def test_includes_tracestate_when_present(self):
        tp = _build_traceparent("abc", "def")
        result = extract_headers({"traceparent": tp, "tracestate": "vendor=value"})
        assert result["tracestate"] == "vendor=value"

    def test_case_insensitive_header_name(self):
        tp = _build_traceparent("abc", "def")
        result = extract_headers({"Traceparent": tp})
        assert "trace_id" in result

    def test_rejects_malformed(self):
        assert extract_headers({"traceparent": "junk"}) == {}


class TestRoundTrip:
    def test_inject_then_extract_recovers_ids(self):
        client = Mock()
        client.traces = Mock()
        client.traces.upload = Mock()

        @trace(client)
        def f():
            return inject_headers({})

        headers = f()
        parsed = extract_headers(headers)
        assert "trace_id" in parsed
        assert "parent_span_id" in parsed

    def test_cross_process_propagation_shares_trace_id(self):
        """trace_context(from_context=...) is the upstream API; W3C headers
        are the wire format. Confirm the IDs match."""
        client = Mock()
        client.traces = Mock()
        client.traces.upload = Mock()

        with trace_context(client) as parent:
            headers = inject_headers({})

        extracted = extract_headers(headers)
        # The shortened trace id round-trips back to our internal form.
        assert extracted["trace_id"] == parent.trace_id


class TestNewTraceparent:
    def test_outside_trace_still_returns_valid_header(self):
        tp = new_traceparent()
        parsed = _parse_traceparent(tp)
        assert parsed is not None
        assert len(parsed["trace_id"]) == 32
        assert len(parsed["parent_span_id"]) == 16

    def test_inside_trace_uses_active_context(self):
        client = Mock()
        client.traces = Mock()
        client.traces.upload = Mock()

        with trace_context(client) as parent:
            tp = new_traceparent()

        parsed = _parse_traceparent(tp)
        assert parsed is not None
        assert _shorten_trace_id(parsed["trace_id"]) == parent.trace_id


# ---------------------------------------------------------------------------
# OTel GenAI semantic conventions
# ---------------------------------------------------------------------------


class TestGenAiAttributes:
    def test_basic_chat_attributes(self):
        attrs = gen_ai_attributes(
            provider="openai",
            operation="chat",
            parameters={"model": "gpt-4o", "temperature": 0.7, "max_tokens": 100},
            response_meta={"response_model": "gpt-4o-2024-11-20", "response_id": "abc"},
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
        assert attrs["gen_ai.system"] == "openai"
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.request.model"] == "gpt-4o"
        assert attrs["gen_ai.request.temperature"] == 0.7
        assert attrs["gen_ai.request.max_tokens"] == 100
        assert attrs["gen_ai.response.model"] == "gpt-4o-2024-11-20"
        assert attrs["gen_ai.response.id"] == "abc"
        assert attrs["gen_ai.usage.input_tokens"] == 10
        assert attrs["gen_ai.usage.output_tokens"] == 20

    def test_provider_mapping_to_otel_system(self):
        for provider, expected in [
            ("anthropic", "anthropic"),
            ("azure_openai", "az.ai.openai"),
            ("google_vertex", "gcp.vertex_ai"),
            ("bedrock", "aws.bedrock"),
            ("ollama", "ollama"),
            ("unknown_provider", "unknown_provider"),
        ]:
            attrs = gen_ai_attributes(provider=provider, operation="chat", parameters={}, response_meta={})
            assert attrs["gen_ai.system"] == expected

    def test_drops_missing_values(self):
        attrs = gen_ai_attributes(provider="openai", operation="chat", parameters={}, response_meta={})
        # Required keys present:
        assert "gen_ai.system" in attrs
        assert "gen_ai.operation.name" in attrs
        # No request/response keys when nothing supplied:
        for key in attrs:
            assert key in ("gen_ai.system", "gen_ai.operation.name")

    def test_finish_reason_becomes_list(self):
        attrs = gen_ai_attributes(
            provider="openai",
            operation="chat",
            parameters={},
            response_meta={"finish_reason": "stop"},
        )
        assert attrs["gen_ai.response.finish_reasons"] == ["stop"]

    def test_anthropic_token_aliases(self):
        # Anthropic uses input_tokens / output_tokens
        attrs = gen_ai_attributes(
            provider="anthropic",
            operation="chat",
            parameters={},
            response_meta={},
            usage={"input_tokens": 5, "output_tokens": 7},
        )
        assert attrs["gen_ai.usage.input_tokens"] == 5
        assert attrs["gen_ai.usage.output_tokens"] == 7

    def test_unmapped_param_is_dropped(self):
        attrs = gen_ai_attributes(
            provider="openai",
            operation="chat",
            parameters={"model": "gpt-4o", "custom_internal_flag": True},
            response_meta={},
        )
        assert "gen_ai.request.model" in attrs
        # `custom_internal_flag` has no mapping -> not in attrs
        for key in attrs:
            assert "custom_internal_flag" not in key
