"""Tests for OTel GenAI Semantic Convention attribute emission in the exporter."""

import pytest

from layerlens.instrument.exporters._otel import (
    OTelExporter,
    _get_genai_span_name,
    _SPAN_KIND_MAP,
)


class TestGenAiSpanNaming:
    """Tests for OTel GenAI span naming convention."""

    def test_model_invoke_span_name(self):
        """Test span name for model.invoke: '{operation} {model}'."""
        payload = {
            "model": {"name": "gpt-4o", "provider": "openai"},
            "operation": "chat",
        }
        name = _get_genai_span_name("model.invoke", payload)
        assert name == "chat gpt-4o"

    def test_model_invoke_default_operation(self):
        """Test that default operation is 'chat'."""
        payload = {
            "model": {"name": "claude-3-opus"},
        }
        name = _get_genai_span_name("model.invoke", payload)
        assert name == "chat claude-3-opus"

    def test_model_invoke_no_model_name(self):
        """Test span name when model name is empty."""
        payload = {
            "model": {},
        }
        name = _get_genai_span_name("model.invoke", payload)
        assert name == "chat"

    def test_model_invoke_string_model(self):
        """Test span name when model is a string instead of dict."""
        payload = {
            "model": "gpt-4o",
        }
        name = _get_genai_span_name("model.invoke", payload)
        assert name == "chat gpt-4o"

    def test_model_invoke_embedding_operation(self):
        """Test span name for embedding operation."""
        payload = {
            "model": {"name": "text-embedding-3-small"},
            "operation": "embedding",
        }
        name = _get_genai_span_name("model.invoke", payload)
        assert name == "embedding text-embedding-3-small"

    def test_evaluation_span_name(self):
        """Test span name for evaluation.result."""
        payload = {
            "evaluation": {"dimension": "factual_accuracy"},
        }
        name = _get_genai_span_name("evaluation.result", payload)
        assert name == "evaluation factual_accuracy"

    def test_agent_input_with_agent_id(self):
        """Test span name for agent.input with agent_id."""
        payload = {
            "agent_id": "customer_service",
        }
        name = _get_genai_span_name("agent.input", payload)
        assert name == "agent customer_service"

    def test_agent_input_no_agent_id(self):
        """Test span name for agent.input without agent_id."""
        payload = {}
        name = _get_genai_span_name("agent.input", payload)
        assert name == "stratix.agent.input"

    def test_unknown_event_type(self):
        """Test span name for unknown event type."""
        payload = {}
        name = _get_genai_span_name("custom.event", payload)
        assert name == "stratix.custom.event"


class TestSpanKindMapping:
    """Tests for SpanKind mapping per event type."""

    def test_model_invoke_is_client(self):
        """Test that model.invoke maps to CLIENT."""
        assert _SPAN_KIND_MAP["model.invoke"] == "CLIENT"

    def test_tool_call_is_internal(self):
        """Test that tool.call maps to INTERNAL."""
        assert _SPAN_KIND_MAP["tool.call"] == "INTERNAL"

    def test_agent_input_is_server(self):
        """Test that agent.input maps to SERVER."""
        assert _SPAN_KIND_MAP["agent.input"] == "SERVER"

    def test_agent_output_is_server(self):
        """Test that agent.output maps to SERVER."""
        assert _SPAN_KIND_MAP["agent.output"] == "SERVER"

    def test_evaluation_result_is_internal(self):
        """Test that evaluation.result maps to INTERNAL."""
        assert _SPAN_KIND_MAP["evaluation.result"] == "INTERNAL"


class TestOTelExporterInit:
    """Tests for OTelExporter initialization with gen_ai support."""

    def test_emit_genai_default_true(self):
        """Test that emit_genai_attributes defaults to True."""
        exporter = OTelExporter(endpoint="localhost:4317")
        assert exporter._emit_genai is True

    def test_emit_genai_can_be_disabled(self):
        """Test that gen_ai attribute emission can be disabled."""
        exporter = OTelExporter(
            endpoint="localhost:4317",
            emit_genai_attributes=False,
        )
        assert exporter._emit_genai is False

    def test_capture_content_false_by_default(self, monkeypatch):
        """Test that content capture is off by default."""
        monkeypatch.delenv("OTEL_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
        exporter = OTelExporter(endpoint="localhost:4317")
        assert exporter._capture_content is False

    def test_capture_content_enabled_by_env(self, monkeypatch):
        """Test that content capture is enabled by env var."""
        monkeypatch.setenv("OTEL_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
        exporter = OTelExporter(endpoint="localhost:4317")
        assert exporter._capture_content is True

    def test_capture_content_case_insensitive(self, monkeypatch):
        """Test that env var check is case-insensitive."""
        monkeypatch.setenv("OTEL_GENAI_CAPTURE_MESSAGE_CONTENT", "TRUE")
        exporter = OTelExporter(endpoint="localhost:4317")
        assert exporter._capture_content is True


class TestCaptureConfigOtelBridge:
    """Tests for CaptureConfig.otel_capture_content bridge."""

    def test_otel_capture_requires_both_config_and_env(self, monkeypatch):
        """otel_capture_content is True only when both config flag and env var are set."""
        from layerlens.instrument.adapters._capture import CaptureConfig

        monkeypatch.setenv("OTEL_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
        config = CaptureConfig(capture_content=True)
        assert config.otel_capture_content is True

    def test_otel_capture_false_without_env(self, monkeypatch):
        """otel_capture_content is False when env var is unset."""
        from layerlens.instrument.adapters._capture import CaptureConfig

        monkeypatch.delenv("OTEL_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
        config = CaptureConfig(capture_content=True)
        assert config.otel_capture_content is False

    def test_otel_capture_false_without_config_flag(self, monkeypatch):
        """otel_capture_content is False when config flag is disabled."""
        from layerlens.instrument.adapters._capture import CaptureConfig

        monkeypatch.setenv("OTEL_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
        config = CaptureConfig(capture_content=False)
        assert config.otel_capture_content is False


class TestOTelExporterGenAiAttributes:
    """Tests for gen_ai.* attribute emission logic (unit tests without OTel SDK)."""

    def test_add_genai_attributes_model_invoke(self):
        """Test that _add_genai_attributes sets gen_ai.* for model.invoke."""
        exporter = OTelExporter(endpoint="localhost:4317")

        # Create a mock span to capture set_attribute calls
        attributes = {}

        class MockSpan:
            def set_attribute(self, key, value):
                attributes[key] = value

        payload = {
            "model": {
                "name": "gpt-4o",
                "provider": "openai",
                "parameters": {"temperature": 0.7, "max_tokens": 1000},
            },
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "operation": "chat",
            "metadata": {
                "finish_reason": "stop",
                "response_id": "chatcmpl-abc",
                "response_model": "gpt-4o-2024-05-13",
            },
        }

        exporter._add_genai_attributes(MockSpan(), "model.invoke", payload, {})

        assert attributes["gen_ai.provider.name"] == "openai"
        assert attributes["gen_ai.operation.name"] == "chat"
        assert attributes["gen_ai.request.model"] == "gpt-4o"
        assert attributes["gen_ai.response.model"] == "gpt-4o-2024-05-13"
        assert attributes["gen_ai.usage.input_tokens"] == 100
        assert attributes["gen_ai.usage.output_tokens"] == 50
        assert attributes["gen_ai.request.temperature"] == 0.7
        assert attributes["gen_ai.request.max_tokens"] == 1000
        assert attributes["gen_ai.response.finish_reasons"] == ["stop"]
        assert attributes["gen_ai.response.id"] == "chatcmpl-abc"

    def test_add_genai_attributes_evaluation(self):
        """Test that _add_genai_attributes sets gen_ai.evaluation.* for evaluation.result."""
        exporter = OTelExporter(endpoint="localhost:4317")

        attributes = {}

        class MockSpan:
            def set_attribute(self, key, value):
                attributes[key] = value

        payload = {
            "evaluation": {
                "score": 0.85,
                "dimension": "factual_accuracy",
                "label": "pass",
                "grader_id": "factual_judge_v2",
            },
            "is_passing": True,
        }

        exporter._add_genai_attributes(MockSpan(), "evaluation.result", payload, {})

        assert attributes["gen_ai.evaluation.score.value"] == 0.85
        assert attributes["gen_ai.evaluation.name"] == "factual_accuracy"
        assert attributes["gen_ai.evaluation.score.label"] == "pass"
        # STRATIX extensions (moved from gen_ai.* to stratix.* namespace)
        assert attributes["stratix.evaluation.grader_id"] == "factual_judge_v2"
        assert attributes["stratix.evaluation.is_passing"] is True

    def test_add_genai_attributes_tool_call(self):
        """Test that _add_genai_attributes sets gen_ai.tool.* for tool.call events."""
        exporter = OTelExporter(endpoint="localhost:4317")

        attributes = {}

        class MockSpan:
            def set_attribute(self, key, value):
                attributes[key] = value

        payload = {
            "tool": {
                "name": "get_weather",
                "description": "Get current weather for a location",
            },
            "invocation": {
                "call_id": "call_abc123",
            },
        }

        exporter._add_genai_attributes(MockSpan(), "tool.call", payload, {})

        assert attributes["gen_ai.tool.name"] == "get_weather"
        assert attributes["gen_ai.tool.description"] == "Get current weather for a location"
        assert attributes["gen_ai.tool.call.id"] == "call_abc123"

    def test_add_genai_attributes_agent_span(self):
        """Test that _add_genai_attributes sets gen_ai.agent.name for agent events."""
        exporter = OTelExporter(endpoint="localhost:4317")

        attributes = {}

        class MockSpan:
            def set_attribute(self, key, value):
                attributes[key] = value

        identity = {"agent_id": "customer_service_agent"}

        exporter._add_genai_attributes(MockSpan(), "agent.input", {}, identity)

        assert attributes["gen_ai.agent.name"] == "customer_service_agent"

    def test_add_genai_attributes_agent_description(self):
        """Test that gen_ai.agent.description is set when present."""
        exporter = OTelExporter(endpoint="localhost:4317")

        attributes = {}

        class MockSpan:
            def set_attribute(self, key, value):
                attributes[key] = value

        identity = {
            "agent_id": "order_agent",
            "agent_description": "Handles order inquiries and tracking",
        }

        exporter._add_genai_attributes(MockSpan(), "agent.input", {}, identity)

        assert attributes["gen_ai.agent.name"] == "order_agent"
        assert attributes["gen_ai.agent.description"] == "Handles order inquiries and tracking"

    def test_add_genai_attributes_seed_parameter(self):
        """Test that gen_ai.request.seed is extracted from model parameters."""
        exporter = OTelExporter(endpoint="localhost:4317")

        attributes = {}

        class MockSpan:
            def set_attribute(self, key, value):
                attributes[key] = value

        payload = {
            "model": {
                "name": "gpt-4o",
                "provider": "openai",
                "parameters": {"temperature": 0.7, "seed": 42},
            },
        }

        exporter._add_genai_attributes(MockSpan(), "model.invoke", payload, {})

        assert attributes["gen_ai.request.temperature"] == 0.7
        assert attributes["gen_ai.request.seed"] == 42

    def test_provider_specific_openai(self):
        """Test OpenAI-specific attributes."""
        exporter = OTelExporter(endpoint="localhost:4317")

        attributes = {}

        class MockSpan:
            def set_attribute(self, key, value):
                attributes[key] = value

        metadata = {
            "system_fingerprint": "fp_abc123",
            "service_tier": "default",
            "seed": 42,
        }

        exporter._add_provider_specific_attributes(MockSpan(), "openai", metadata)

        assert attributes["gen_ai.openai.response.system_fingerprint"] == "fp_abc123"
        assert attributes["gen_ai.openai.response.service_tier"] == "default"
        assert attributes["gen_ai.openai.request.seed"] == 42

    def test_provider_specific_anthropic(self):
        """Test Anthropic-specific attributes."""
        exporter = OTelExporter(endpoint="localhost:4317")

        attributes = {}

        class MockSpan:
            def set_attribute(self, key, value):
                attributes[key] = value

        metadata = {
            "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 1000,
        }

        exporter._add_provider_specific_attributes(MockSpan(), "anthropic", metadata)

        assert attributes["gen_ai.usage.cache_creation_input_tokens"] == 500
        assert attributes["gen_ai.usage.cache_read_input_tokens"] == 1000

    def test_provider_specific_bedrock(self):
        """Test Bedrock-specific attributes."""
        exporter = OTelExporter(endpoint="localhost:4317")

        attributes = {}

        class MockSpan:
            def set_attribute(self, key, value):
                attributes[key] = value

        metadata = {
            "guardrail_id": "gr-12345",
            "knowledge_base_id": "kb-67890",
            "agent_id": "ag-abc",
        }

        exporter._add_provider_specific_attributes(MockSpan(), "bedrock", metadata)

        assert attributes["aws.bedrock.guardrail.id"] == "gr-12345"
        assert attributes["aws.bedrock.knowledge_base.id"] == "kb-67890"
        assert attributes["aws.bedrock.agent.id"] == "ag-abc"

    def test_genai_disabled(self):
        """Test that gen_ai.* attributes are not emitted when disabled."""
        exporter = OTelExporter(
            endpoint="localhost:4317",
            emit_genai_attributes=False,
        )
        assert exporter._emit_genai is False
