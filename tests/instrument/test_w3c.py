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

    def test_anthropic_stop_reason_becomes_finish_reasons(self):
        # Anthropic stores ``stop_reason`` rather than ``finish_reason``; the
        # mapper should still emit ``gen_ai.response.finish_reasons``.
        attrs = gen_ai_attributes(
            provider="anthropic",
            operation="chat",
            parameters={},
            response_meta={"stop_reason": "end_turn"},
        )
        assert attrs["gen_ai.response.finish_reasons"] == ["end_turn"]

    def test_anthropic_cache_tokens_mapped(self):
        attrs = gen_ai_attributes(
            provider="anthropic",
            operation="chat",
            parameters={},
            response_meta={},
            usage={
                "input_tokens": 50,
                "output_tokens": 20,
                "cache_read_input_tokens": 120,
                "cache_creation_input_tokens": 300,
            },
        )
        assert attrs["gen_ai.usage.cache_read_input_tokens"] == 120
        assert attrs["gen_ai.usage.cache_creation_input_tokens"] == 300

    def test_openai_cached_tokens_alias(self):
        # OpenAI exposes cached prompt tokens under ``cached_tokens``; the
        # mapper should normalise to ``gen_ai.usage.cache_read_input_tokens``.
        attrs = gen_ai_attributes(
            provider="openai",
            operation="chat",
            parameters={},
            response_meta={},
            usage={"prompt_tokens": 100, "completion_tokens": 30, "cached_tokens": 64},
        )
        assert attrs["gen_ai.usage.cache_read_input_tokens"] == 64
        # Anthropic-only field is absent.
        assert "gen_ai.usage.cache_creation_input_tokens" not in attrs

    def test_reasoning_tokens_mapped_openai(self):
        attrs = gen_ai_attributes(
            provider="openai",
            operation="chat",
            parameters={},
            response_meta={},
            usage={"prompt_tokens": 10, "completion_tokens": 50, "reasoning_tokens": 1024},
        )
        assert attrs["gen_ai.usage.reasoning_tokens"] == 1024

    def test_reasoning_tokens_mapped_anthropic_thinking_alias(self):
        # Anthropic's extended-thinking budget surfaces as ``thinking_tokens``;
        # OTel uses the unified ``reasoning_tokens`` attribute.
        attrs = gen_ai_attributes(
            provider="anthropic",
            operation="chat",
            parameters={},
            response_meta={},
            usage={"input_tokens": 12, "output_tokens": 80, "thinking_tokens": 2048},
        )
        assert attrs["gen_ai.usage.reasoning_tokens"] == 2048

    def test_openai_system_fingerprint_and_service_tier(self):
        attrs = gen_ai_attributes(
            provider="openai",
            operation="chat",
            parameters={},
            response_meta={
                "system_fingerprint": "fp_abc123",
                "service_tier": "scale",
            },
        )
        assert attrs["gen_ai.openai.response.system_fingerprint"] == "fp_abc123"
        assert attrs["gen_ai.openai.response.service_tier"] == "scale"

    def test_openai_namespaced_attrs_not_emitted_for_other_providers(self):
        # ``system_fingerprint`` / ``service_tier`` are OpenAI-specific; even
        # if a non-OpenAI provider somehow surfaced them in meta, they must
        # not be emitted under the OpenAI namespace.
        attrs = gen_ai_attributes(
            provider="anthropic",
            operation="chat",
            parameters={},
            response_meta={"system_fingerprint": "fp_should_be_ignored"},
        )
        for key in attrs:
            assert "gen_ai.openai." not in key

    def test_azure_openai_inherits_openai_response_namespace(self):
        # Azure OpenAI is the same vendor; namespaced attrs should still apply.
        attrs = gen_ai_attributes(
            provider="azure_openai",
            operation="chat",
            parameters={},
            response_meta={"system_fingerprint": "fp_azure"},
        )
        assert attrs["gen_ai.openai.response.system_fingerprint"] == "fp_azure"

    # ------------------------------------------------------------------
    # TEL-026 / LAY-2879: ``gen_ai.openai.request.seed``
    # ------------------------------------------------------------------

    def test_openai_seed_emitted_under_vendor_namespace(self):
        attrs = gen_ai_attributes(
            provider="openai",
            operation="chat",
            parameters={"model": "gpt-4o", "seed": 42},
            response_meta={},
        )
        # Vendor-namespaced per TEL-026 acceptance criteria.
        assert attrs["gen_ai.openai.request.seed"] == 42
        # Generic version retained as alias so generic OTel backends still see it.
        assert attrs["gen_ai.request.seed"] == 42

    def test_seed_not_vendor_namespaced_for_non_openai(self):
        attrs = gen_ai_attributes(
            provider="anthropic",
            operation="chat",
            parameters={"seed": 42},
            response_meta={},
        )
        # Anthropic doesn't have a seed concept; even if a caller passes it, it
        # must not be emitted under the OpenAI vendor namespace.
        for key in attrs:
            assert "gen_ai.openai." not in key

    # ------------------------------------------------------------------
    # TEL-028 / LAY-2881: ``gen_ai.anthropic.cache_*_input_tokens``
    # ------------------------------------------------------------------

    def test_anthropic_cache_tokens_emitted_under_vendor_namespace(self):
        attrs = gen_ai_attributes(
            provider="anthropic",
            operation="chat",
            parameters={},
            response_meta={},
            usage={
                "input_tokens": 50,
                "output_tokens": 20,
                "cache_read_input_tokens": 120,
                "cache_creation_input_tokens": 300,
            },
        )
        # Vendor-namespaced per TEL-028 acceptance criteria.
        assert attrs["gen_ai.anthropic.cache_read_input_tokens"] == 120
        assert attrs["gen_ai.anthropic.cache_creation_input_tokens"] == 300
        # Un-namespaced alias retained.
        assert attrs["gen_ai.usage.cache_read_input_tokens"] == 120
        assert attrs["gen_ai.usage.cache_creation_input_tokens"] == 300

    def test_anthropic_cache_namespace_not_emitted_for_openai(self):
        # OpenAI also exposes cached prompt tokens, but the Anthropic-namespaced
        # attributes must only fire on Anthropic spans per TEL-028.
        attrs = gen_ai_attributes(
            provider="openai",
            operation="chat",
            parameters={},
            response_meta={},
            usage={"prompt_tokens": 100, "completion_tokens": 30, "cached_tokens": 64},
        )
        for key in attrs:
            assert "gen_ai.anthropic." not in key
