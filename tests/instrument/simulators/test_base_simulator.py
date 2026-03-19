"""Tests for BaseSimulator and TraceSimulator."""

import json
import tempfile
import os

import pytest

from layerlens.instrument.simulators.base import SimulatorResult, TraceSimulator
from layerlens.instrument.simulators.config import (
    ContentTier,
    OutputFormat,
    ScenarioName,
    SimulatorConfig,
    SourceFormat,
)
from layerlens.instrument.simulators.span_model import SpanType


class TestSimulatorResult:
    def test_defaults(self):
        result = SimulatorResult(run_id="run_test")
        assert result.trace_count == 0
        assert result.span_count == 0
        assert result.validation_status == "pending"

    def test_serialization(self):
        result = SimulatorResult(
            run_id="run_abc",
            trace_count=10,
            span_count=50,
            total_tokens=5000,
            duration_ms=1234.5,
        )
        data = result.model_dump(mode="json")
        assert data["run_id"] == "run_abc"
        assert data["trace_count"] == 10


class TestTraceSimulator:
    def test_initialize(self):
        config = SimulatorConfig.minimal()
        sim = TraceSimulator(config)
        sim.initialize()
        assert sim._initialized is True

    def test_generate_single(self):
        config = SimulatorConfig(
            source_format=SourceFormat.GENERIC_OTEL,
            scenario=ScenarioName.CUSTOMER_SERVICE,
            count=1,
            seed=42,
        )
        sim = TraceSimulator(config)
        traces = sim.generate()
        assert len(traces) == 1
        trace = traces[0]
        assert trace.scenario == "customer_service"
        assert trace.span_count >= 4  # agent + llm + tools + llm + eval

    def test_generate_multiple(self):
        config = SimulatorConfig(count=5, seed=42)
        sim = TraceSimulator(config)
        traces = sim.generate()
        assert len(traces) == 5

    def test_generate_count_override(self):
        config = SimulatorConfig(count=10, seed=42)
        sim = TraceSimulator(config)
        traces = sim.generate(count=3)
        assert len(traces) == 3

    def test_deterministic_generation(self):
        config = SimulatorConfig(count=3, seed=42)
        sim1 = TraceSimulator(config)
        sim2 = TraceSimulator(config)
        t1 = sim1.generate()
        t2 = sim2.generate()
        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert a.trace_id == b.trace_id
            assert a.span_count == b.span_count

    def test_trace_has_agent_span(self):
        config = SimulatorConfig(count=1, seed=42)
        sim = TraceSimulator(config)
        traces = sim.generate()
        trace = traces[0]
        agent_spans = [s for s in trace.spans if s.span_type == SpanType.AGENT]
        assert len(agent_spans) == 1

    def test_trace_has_llm_spans(self):
        config = SimulatorConfig(count=1, seed=42)
        sim = TraceSimulator(config)
        traces = sim.generate()
        trace = traces[0]
        assert len(trace.llm_spans) >= 2  # Planning + response

    def test_trace_has_evaluation_span(self):
        config = SimulatorConfig(count=1, seed=42)
        sim = TraceSimulator(config)
        traces = sim.generate()
        trace = traces[0]
        eval_spans = [s for s in trace.spans if s.span_type == SpanType.EVALUATION]
        assert len(eval_spans) >= 1

    def test_topic_cycling(self):
        config = SimulatorConfig(count=10, seed=42)
        sim = TraceSimulator(config)
        traces = sim.generate()
        topics = [t.topic for t in traces]
        # Should cycle through 5 topics
        assert len(set(topics)) >= 2

    def test_include_content(self):
        config = SimulatorConfig(count=1, seed=42, include_content=True)
        sim = TraceSimulator(config)
        traces = sim.generate()
        trace = traces[0]
        llm_spans = trace.llm_spans
        # First LLM span should have input messages when content is included
        assert len(llm_spans[0].input_messages) >= 1

    def test_exclude_content(self):
        config = SimulatorConfig(count=1, seed=42, include_content=False)
        sim = TraceSimulator(config)
        traces = sim.generate()
        trace = traces[0]
        llm_spans = trace.llm_spans
        assert len(llm_spans[0].input_messages) == 0

    def test_format_output_otlp(self):
        config = SimulatorConfig(count=1, seed=42)
        sim = TraceSimulator(config)
        traces = sim.generate()
        formatted = sim.format_output(traces)
        assert isinstance(formatted, list)
        assert len(formatted) == 1
        # Default output is OTLP JSON
        assert "resourceSpans" in formatted[0]

    def test_generate_and_format(self):
        config = SimulatorConfig(count=3, seed=42)
        sim = TraceSimulator(config)
        formatted, result = sim.generate_and_format()
        assert len(formatted) == 3
        assert result.trace_count == 3
        assert result.span_count > 0
        assert result.total_tokens > 0
        assert result.duration_ms >= 0.0  # may be 0.0 on fast machines
        assert result.run_id.startswith("run_")

    def test_shutdown(self):
        config = SimulatorConfig.minimal()
        sim = TraceSimulator(config)
        sim.initialize()
        assert sim._initialized is True
        sim.shutdown()
        assert sim._initialized is False

    @pytest.mark.parametrize("scenario", list(ScenarioName))
    def test_all_scenarios_generate(self, scenario):
        config = SimulatorConfig(scenario=scenario, count=1, seed=42)
        sim = TraceSimulator(config)
        traces = sim.generate()
        assert len(traces) == 1
        assert traces[0].scenario == scenario.value

    def test_provider_model_mapping(self):
        sim = TraceSimulator(SimulatorConfig(source_format=SourceFormat.ANTHROPIC))
        provider, model = sim._get_provider_model()
        assert provider == "anthropic"
        assert "claude" in model

    def test_provider_model_bedrock(self):
        sim = TraceSimulator(SimulatorConfig(source_format=SourceFormat.BEDROCK))
        provider, model = sim._get_provider_model()
        assert provider == "bedrock"
        assert "anthropic" in model

    def test_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.json")
            config = SimulatorConfig(count=2, seed=42, output_path=path)
            sim = TraceSimulator(config)
            sim.generate_and_format()
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 2

    def test_dry_run_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.json")
            config = SimulatorConfig(count=1, seed=42, output_path=path, dry_run=True)
            sim = TraceSimulator(config)
            sim.generate_and_format()
            assert not os.path.exists(path)

    def test_default_config(self):
        sim = TraceSimulator()
        traces = sim.generate()
        assert len(traces) == 1
