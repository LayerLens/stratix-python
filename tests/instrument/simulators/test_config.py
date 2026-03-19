"""Tests for SimulatorConfig and related configuration models."""

import json
import os
import tempfile

import pytest

from layerlens.instrument.simulators.config import (
    ContentConfig,
    ContentTier,
    ConversationConfig,
    ErrorConfig,
    OutputFormat,
    ScenarioName,
    SimulatorConfig,
    SourceFormat,
    StreamingConfig,
)


class TestSourceFormat:
    def test_all_12_sources(self):
        sources = list(SourceFormat)
        assert len(sources) == 12

    def test_source_values(self):
        assert SourceFormat.GENERIC_OTEL.value == "generic_otel"
        assert SourceFormat.AGENTFORCE_OTLP.value == "agentforce_otlp"
        assert SourceFormat.OPENAI.value == "openai"
        assert SourceFormat.ANTHROPIC.value == "anthropic"
        assert SourceFormat.BEDROCK.value == "bedrock"

    def test_source_from_string(self):
        assert SourceFormat("openai") == SourceFormat.OPENAI


class TestOutputFormat:
    def test_all_3_formats(self):
        assert len(list(OutputFormat)) == 3

    def test_format_values(self):
        assert OutputFormat.OTLP_JSON.value == "otlp_json"
        assert OutputFormat.LANGFUSE_JSON.value == "langfuse_json"
        assert OutputFormat.STRATIX_NATIVE.value == "stratix_native"


class TestScenarioName:
    def test_all_5_scenarios(self):
        assert len(list(ScenarioName)) == 5

    def test_scenario_values(self):
        assert ScenarioName.CUSTOMER_SERVICE.value == "customer_service"
        assert ScenarioName.SALES.value == "sales"
        assert ScenarioName.IT_HELPDESK.value == "it_helpdesk"


class TestConversationConfig:
    def test_defaults(self):
        config = ConversationConfig()
        assert config.enabled is False
        assert config.turns_min == 2
        assert config.turns_max == 5

    def test_max_gte_min_validation(self):
        with pytest.raises(ValueError, match="turns_max"):
            ConversationConfig(turns_min=5, turns_max=2)

    def test_valid_range(self):
        config = ConversationConfig(enabled=True, turns_min=3, turns_max=8)
        assert config.turns_min == 3
        assert config.turns_max == 8


class TestStreamingConfig:
    def test_defaults(self):
        config = StreamingConfig()
        assert config.enabled is False
        assert config.ttft_ms_min == 50.0
        assert config.ttft_ms_max == 500.0

    def test_custom_values(self):
        config = StreamingConfig(enabled=True, ttft_ms_min=100.0, ttft_ms_max=200.0)
        assert config.ttft_ms_min == 100.0


class TestErrorConfig:
    def test_defaults(self):
        config = ErrorConfig()
        assert config.enabled is False
        assert config.rate_limit_probability == 0.05

    def test_probability_bounds(self):
        with pytest.raises(ValueError):
            ErrorConfig(rate_limit_probability=1.5)
        with pytest.raises(ValueError):
            ErrorConfig(rate_limit_probability=-0.1)


class TestSimulatorConfig:
    def test_defaults(self):
        config = SimulatorConfig()
        assert config.source_format == SourceFormat.GENERIC_OTEL
        assert config.output_format == OutputFormat.OTLP_JSON
        assert config.scenario == ScenarioName.CUSTOMER_SERVICE
        assert config.count == 1
        assert config.seed is None
        assert config.include_content is False
        assert config.dry_run is False

    def test_minimal_preset(self):
        config = SimulatorConfig.minimal()
        assert config.count == 1
        assert config.errors.enabled is False
        assert config.streaming.enabled is False
        assert config.conversation.enabled is False
        assert config.content.tier == ContentTier.TEMPLATE

    def test_standard_preset(self):
        config = SimulatorConfig.standard()
        assert config.count == 10
        assert config.conversation.enabled is True
        assert config.errors.enabled is True
        assert config.errors.rate_limit_probability == 0.05

    def test_full_preset(self):
        config = SimulatorConfig.full()
        assert config.count == 100
        assert config.include_content is True
        assert config.conversation.enabled is True
        assert config.errors.enabled is True
        assert config.streaming.enabled is True

    def test_count_must_be_positive(self):
        with pytest.raises(ValueError):
            SimulatorConfig(count=0)

    def test_serialization_roundtrip(self):
        config = SimulatorConfig.full()
        data = config.model_dump(mode="json")
        restored = SimulatorConfig(**data)
        assert restored.count == config.count
        assert restored.errors.enabled == config.errors.enabled

    def test_from_yaml(self):
        yaml_content = """
simulator:
  source_format: openai
  output_format: otlp_json
  scenario: customer_service
  seed: 42
  count: 25
  include_content: true
  conversation:
    enabled: true
    turns_range: [2, 5]
  streaming:
    enabled: true
    ttft_ms_range: [50.0, 500.0]
  errors:
    enabled: true
    rate_limit_probability: 0.05
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            config = SimulatorConfig.from_yaml(f.name)

        os.unlink(f.name)
        assert config.source_format == SourceFormat.OPENAI
        assert config.count == 25
        assert config.seed == 42
        assert config.conversation.enabled is True
        assert config.conversation.turns_min == 2
        assert config.conversation.turns_max == 5

    def test_to_yaml(self):
        config = SimulatorConfig.minimal()
        yaml_str = config.to_yaml()
        assert "simulator:" in yaml_str
        assert "source_format:" in yaml_str

    def test_to_yaml_file(self):
        config = SimulatorConfig.minimal()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            path = f.name

        config.to_yaml(path)
        restored = SimulatorConfig.from_yaml(path)
        os.unlink(path)
        assert restored.count == config.count
