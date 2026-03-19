"""Tests for CLI module."""

import json
import os
import tempfile

import pytest
from click.testing import CliRunner

from layerlens.instrument.simulators.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestListSources:
    def test_lists_sources(self, runner):
        result = runner.invoke(cli, ["list-sources"])
        assert result.exit_code == 0
        assert "generic_otel" in result.output
        assert "openai" in result.output
        assert "agentforce_otlp" in result.output

    def test_shows_all_12(self, runner):
        result = runner.invoke(cli, ["list-sources"])
        assert result.exit_code == 0
        # Count source names in output
        sources = [
            "generic_otel", "agentforce_otlp", "agentforce_soql",
            "openai", "anthropic", "azure_openai", "bedrock",
            "google_vertex", "ollama", "litellm", "langfuse", "jsonl",
        ]
        for source in sources:
            assert source in result.output


class TestListScenarios:
    def test_lists_scenarios(self, runner):
        result = runner.invoke(cli, ["list-scenarios"])
        assert result.exit_code == 0
        assert "customer_service" in result.output
        assert "sales" in result.output


class TestGenerate:
    def test_basic_generate(self, runner):
        result = runner.invoke(cli, [
            "generate", "--source", "openai", "--count", "1", "--seed", "42",
        ])
        assert result.exit_code == 0

    def test_generate_with_preset(self, runner):
        result = runner.invoke(cli, [
            "generate", "--source", "openai", "--preset", "minimal",
        ])
        assert result.exit_code == 0

    def test_generate_dry_run(self, runner):
        result = runner.invoke(cli, [
            "generate", "--source", "openai", "--dry-run",
        ])
        assert result.exit_code == 0

    def test_generate_to_file(self, runner):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = runner.invoke(cli, [
                "generate", "--source", "openai", "--count", "1",
                "--seed", "42", "-o", path,
            ])
            assert result.exit_code == 0
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, list)
        finally:
            os.unlink(path)

    def test_generate_with_errors(self, runner):
        result = runner.invoke(cli, [
            "generate", "--source", "openai", "--count", "3",
            "--seed", "42", "--errors",
        ])
        assert result.exit_code == 0

    def test_generate_with_streaming(self, runner):
        result = runner.invoke(cli, [
            "generate", "--source", "openai", "--count", "2",
            "--seed", "42", "--streaming",
        ])
        assert result.exit_code == 0

    def test_generate_from_yaml(self, runner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("source_format: openai\ncount: 1\nseed: 42\n")
            path = f.name
        try:
            result = runner.invoke(cli, [
                "generate", "--source", "openai", "--config", path,
            ])
            assert result.exit_code == 0
        finally:
            os.unlink(path)


class TestValidate:
    def test_validate_single_source(self, runner):
        result = runner.invoke(cli, [
            "validate", "--source", "openai", "--count", "1", "--seed", "42",
        ])
        assert result.exit_code == 0
