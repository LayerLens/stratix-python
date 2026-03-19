"""Tests for all 12 source formatters."""

import pytest

from layerlens.instrument.simulators.sources import get_source_formatter, list_sources
from layerlens.instrument.simulators.sources.base import BaseSourceFormatter
from layerlens.instrument.simulators.span_model import (
    SimulatedSpan,
    SpanKind,
    SpanType,
    TokenUsage,
)


def _make_llm_span(**kwargs) -> SimulatedSpan:
    defaults = dict(
        span_id="abc123def456",
        span_type=SpanType.LLM,
        name="chat gpt-4o",
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_001_000_000_000,
        kind=SpanKind.CLIENT,
        provider="openai",
        model="gpt-4o",
        operation="chat",
        token_usage=TokenUsage(prompt_tokens=250, completion_tokens=180),
        finish_reasons=["stop"],
        response_id="chatcmpl-abc123",
        temperature=0.7,
    )
    defaults.update(kwargs)
    return SimulatedSpan(**defaults)


def _make_tool_span(**kwargs) -> SimulatedSpan:
    defaults = dict(
        span_id="tool123",
        span_type=SpanType.TOOL,
        name="tool Get_Order",
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_000_500_000_000,
        kind=SpanKind.INTERNAL,
        tool_name="Get_Order",
        tool_call_id="call_xyz",
    )
    defaults.update(kwargs)
    return SimulatedSpan(**defaults)


def _make_agent_span(**kwargs) -> SimulatedSpan:
    defaults = dict(
        span_id="agent123",
        span_type=SpanType.AGENT,
        name="agent Test_Agent",
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_005_000_000_000,
        kind=SpanKind.SERVER,
        agent_name="Test_Agent",
    )
    defaults.update(kwargs)
    return SimulatedSpan(**defaults)


def _make_eval_span(**kwargs) -> SimulatedSpan:
    defaults = dict(
        span_id="eval123",
        span_type=SpanType.EVALUATION,
        name="evaluation accuracy",
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_000_500_000_000,
        kind=SpanKind.INTERNAL,
        eval_dimension="factual_accuracy",
        eval_score=0.92,
        eval_label="pass",
    )
    defaults.update(kwargs)
    return SimulatedSpan(**defaults)


class TestSourceRegistry:
    def test_list_sources_has_12(self):
        sources = list_sources()
        assert len(sources) == 12

    def test_all_sources_retrievable(self):
        for name in list_sources():
            formatter = get_source_formatter(name)
            assert isinstance(formatter, BaseSourceFormatter)

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            get_source_formatter("nonexistent")


class TestGenericOTel:
    def test_llm_span_attributes(self):
        formatter = get_source_formatter("generic_otel")
        span = _make_llm_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        assert span.attributes["gen_ai.system"] == "openai"
        assert span.attributes["gen_ai.request.model"] == "gpt-4o"
        assert span.attributes["gen_ai.usage.input_tokens"] == 250
        assert span.attributes["gen_ai.usage.output_tokens"] == 180
        assert span.attributes["gen_ai.response.finish_reasons"] == ["stop"]

    def test_tool_span_attributes(self):
        formatter = get_source_formatter("generic_otel")
        span = _make_tool_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        assert span.attributes["gen_ai.tool.name"] == "Get_Order"

    def test_resource_attributes(self):
        formatter = get_source_formatter("generic_otel")
        attrs = formatter.get_resource_attributes()
        assert "service.name" in attrs
        assert "telemetry.sdk.name" in attrs

    def test_scope(self):
        formatter = get_source_formatter("generic_otel")
        name, version = formatter.get_scope()
        assert "genai" in name


class TestAgentForceOTLP:
    def test_salesforce_attributes(self):
        formatter = get_source_formatter("agentforce_otlp")
        span = _make_llm_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        assert "sf.org_id" in span.attributes
        assert "sf.agent_id" in span.attributes
        assert span.attributes["sf.llm.api_type"] == "chat_completion"

    def test_agent_sf_attributes(self):
        formatter = get_source_formatter("agentforce_otlp")
        span = _make_agent_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        assert span.attributes["sf.agent.type"] == "copilot"

    def test_tool_sf_attributes(self):
        formatter = get_source_formatter("agentforce_otlp")
        span = _make_tool_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        assert span.attributes["sf.action.type"] == "flow"


class TestAgentForceSOQL:
    def test_dmo_type_mapping(self):
        formatter = get_source_formatter("agentforce_soql")
        profile = formatter.get_default_profile()

        agent = _make_agent_span()
        formatter.enrich_span(agent, profile)
        assert agent.attributes["sf.dmo.type"] == "BotSession"

        llm = _make_llm_span()
        formatter.enrich_span(llm, profile)
        assert llm.attributes["sf.dmo.type"] == "GenAiInteraction"

        tool = _make_tool_span()
        formatter.enrich_span(tool, profile)
        assert tool.attributes["sf.dmo.type"] == "BotSessionAction"


class TestOpenAISource:
    def test_fidelity(self):
        formatter = get_source_formatter("openai")
        span = _make_llm_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        attrs = span.attributes
        assert "gen_ai.system" in attrs and attrs["gen_ai.system"] == "openai"
        assert "gen_ai.openai.response.system_fingerprint" in attrs
        assert "gen_ai.openai.response.service_tier" in attrs
        # Must NOT have Anthropic attributes
        assert "gen_ai.usage.cache_creation_input_tokens" not in attrs

    def test_profile(self):
        formatter = get_source_formatter("openai")
        profile = formatter.get_default_profile()
        assert profile.provider_name == "openai"
        assert "gpt-4o" in profile.models


class TestAnthropicSource:
    def test_fidelity(self):
        formatter = get_source_formatter("anthropic")
        span = _make_llm_span(provider="anthropic", model="claude-sonnet-4-20250514")
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        attrs = span.attributes
        assert attrs["gen_ai.system"] == "anthropic"
        assert "gen_ai.usage.cache_creation_input_tokens" in attrs
        # Anthropic uses "end_turn" instead of "stop"
        assert attrs["gen_ai.response.finish_reasons"] == ["end_turn"]
        # Must NOT have OpenAI attributes
        assert "gen_ai.openai.response.system_fingerprint" not in attrs

    def test_cache_tokens(self):
        formatter = get_source_formatter("anthropic")
        span = _make_llm_span(
            token_usage=TokenUsage(
                prompt_tokens=250, completion_tokens=180, cached_tokens=100
            )
        )
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        assert span.attributes["gen_ai.usage.cache_read_input_tokens"] == 100


class TestAzureOpenAISource:
    def test_azure_attributes(self):
        formatter = get_source_formatter("azure_openai")
        span = _make_llm_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        attrs = span.attributes
        assert "gen_ai.azure.deployment" in attrs
        assert "gen_ai.azure.api_version" in attrs
        assert attrs["az.namespace"] == "Microsoft.CognitiveServices"

    def test_resource_has_cloud_provider(self):
        formatter = get_source_formatter("azure_openai")
        attrs = formatter.get_resource_attributes()
        assert attrs["cloud.provider"] == "azure"


class TestBedrockSource:
    def test_bedrock_attributes(self):
        formatter = get_source_formatter("bedrock")
        span = _make_llm_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        attrs = span.attributes
        assert attrs["gen_ai.system"] == "aws.bedrock"
        assert attrs["cloud.provider"] == "aws"
        assert "aws.bedrock.family" in attrs

    def test_resource_has_cloud_platform(self):
        formatter = get_source_formatter("bedrock")
        attrs = formatter.get_resource_attributes()
        assert attrs["cloud.platform"] == "aws_bedrock"


class TestGoogleVertexSource:
    def test_vertex_attributes(self):
        formatter = get_source_formatter("google_vertex")
        span = _make_llm_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        attrs = span.attributes
        assert attrs["gen_ai.system"] == "vertex_ai"
        assert "gen_ai.google.safety_ratings" in attrs
        # Vertex uses STOP enum
        assert attrs["gen_ai.response.finish_reasons"] == ["STOP"]


class TestOllamaSource:
    def test_ollama_attributes(self):
        formatter = get_source_formatter("ollama")
        span = _make_llm_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        attrs = span.attributes
        assert attrs["gen_ai.system"] == "ollama"
        assert attrs["gen_ai.usage.cost"] == 0.0
        assert "gen_ai.ollama.prompt_eval_count" in attrs
        assert attrs["server.address"] == "localhost:11434"


class TestLiteLLMSource:
    def test_litellm_attributes(self):
        formatter = get_source_formatter("litellm")
        span = _make_llm_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        attrs = span.attributes
        assert attrs["gen_ai.system"] == "litellm"
        assert "litellm.routed_model" in attrs
        assert attrs["litellm.routed_model"].startswith("openai/")


class TestLangfuseSource:
    def test_langfuse_attributes(self):
        formatter = get_source_formatter("langfuse")
        span = _make_llm_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        attrs = span.attributes
        assert attrs["langfuse.observation_type"] == "generation"
        assert "langfuse.project_id" in attrs

    def test_langfuse_agent(self):
        formatter = get_source_formatter("langfuse")
        span = _make_agent_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        assert span.attributes["langfuse.observation_type"] == "span"
        assert span.attributes["langfuse.trace.name"] == "Test_Agent"


class TestJSONLSource:
    def test_jsonl_attributes(self):
        formatter = get_source_formatter("jsonl")
        span = _make_llm_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        attrs = span.attributes
        assert attrs["stratix.import.format"] == "jsonl"
        assert attrs["stratix.event_type"] == "model.invoke"


class TestAllSourcesCommon:
    @pytest.mark.parametrize("source_name", list_sources())
    def test_enrich_llm_span(self, source_name):
        formatter = get_source_formatter(source_name)
        span = _make_llm_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        # All sources must set gen_ai.system
        assert "gen_ai.system" in span.attributes
        # All sources must set gen_ai.request.model
        assert "gen_ai.request.model" in span.attributes

    @pytest.mark.parametrize("source_name", list_sources())
    def test_enrich_tool_span(self, source_name):
        formatter = get_source_formatter(source_name)
        span = _make_tool_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        assert "gen_ai.tool.name" in span.attributes

    @pytest.mark.parametrize("source_name", list_sources())
    def test_enrich_agent_span(self, source_name):
        formatter = get_source_formatter(source_name)
        span = _make_agent_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        assert "gen_ai.agent.name" in span.attributes

    @pytest.mark.parametrize("source_name", list_sources())
    def test_enrich_eval_span(self, source_name):
        formatter = get_source_formatter(source_name)
        span = _make_eval_span()
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        assert "gen_ai.evaluation.score.value" in span.attributes

    @pytest.mark.parametrize("source_name", list_sources())
    def test_resource_attributes(self, source_name):
        formatter = get_source_formatter(source_name)
        attrs = formatter.get_resource_attributes()
        assert isinstance(attrs, dict)
        assert "service.name" in attrs

    @pytest.mark.parametrize("source_name", list_sources())
    def test_scope(self, source_name):
        formatter = get_source_formatter(source_name)
        name, version = formatter.get_scope()
        assert isinstance(name, str) and len(name) > 0
        assert isinstance(version, str) and len(version) > 0

    @pytest.mark.parametrize("source_name", list_sources())
    def test_default_profile(self, source_name):
        formatter = get_source_formatter(source_name)
        profile = formatter.get_default_profile()
        assert profile.provider_name
        assert profile.default_model

    @pytest.mark.parametrize("source_name", list_sources())
    def test_error_attributes(self, source_name):
        formatter = get_source_formatter(source_name)
        span = _make_llm_span(error_type="rate_limit", http_status_code=429)
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        assert span.attributes["error.type"] == "rate_limit"
        assert span.attributes["http.response.status_code"] == 429

    @pytest.mark.parametrize("source_name", list_sources())
    def test_streaming_attributes(self, source_name):
        formatter = get_source_formatter(source_name)
        span = _make_llm_span(is_streaming=True, ttft_ms=120.0, tpot_ms=35.0)
        profile = formatter.get_default_profile()
        formatter.enrich_span(span, profile)
        assert span.attributes["gen_ai.is_streaming"] is True
