"""Pytest fixtures for testing with the simulator SDK.

Provides ready-to-use fixtures for TraceSimulator, TraceBuilder,
SimulatorConfig presets, sample traces, and parameterised source formatters.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from ..base import TraceSimulator
from ..config import SimulatorConfig
from ..span_model import SimulatedTrace, SpanType
from ..sources import get_source_formatter
from ..trace_builder import TraceBuilder


# --------------------------------------------------------------------------- #
# Core simulator instances
# --------------------------------------------------------------------------- #

@pytest.fixture
def simulator() -> Generator[TraceSimulator, None, None]:
    """Minimal ``TraceSimulator`` instance (initialised, ready to generate)."""
    config = SimulatorConfig.minimal()
    sim = TraceSimulator(config)
    sim.initialize()
    yield sim
    sim.shutdown()


@pytest.fixture
def trace_builder() -> TraceBuilder:
    """``TraceBuilder`` with ``seed=42`` for deterministic trace construction."""
    return TraceBuilder(seed=42)


# --------------------------------------------------------------------------- #
# Config presets
# --------------------------------------------------------------------------- #

@pytest.fixture
def minimal_config() -> SimulatorConfig:
    """``SimulatorConfig.minimal()`` -- 1 trace, template content, no errors."""
    return SimulatorConfig.minimal()


@pytest.fixture
def standard_config() -> SimulatorConfig:
    """``SimulatorConfig.standard()`` -- 10 traces, conversations, 5% errors."""
    return SimulatorConfig.standard()


@pytest.fixture
def full_config() -> SimulatorConfig:
    """``SimulatorConfig.full()`` -- 100 traces, all features enabled."""
    return SimulatorConfig.full()


# --------------------------------------------------------------------------- #
# Sample trace
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_trace() -> SimulatedTrace:
    """A pre-built sample trace containing all four span types.

    Structure::

        agent Service_Agent (root)
          |-- chat gpt-4o  (LLM)
          |-- tool Get_Order_Details  (Tool)
          |-- chat gpt-4o  (LLM, second call)
          |-- evaluation factual_accuracy  (Evaluation)
    """
    trace = (
        TraceBuilder(seed=42)
        .with_scenario("customer_service", topic="order_inquiry")
        .with_source("openai")
        .add_agent_span("Service_Agent")
        .add_llm_span(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=250,
            completion_tokens=180,
            temperature=0.7,
        )
        .add_tool_span(name="Get_Order_Details", latency_ms=350.0)
        .add_llm_span(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=400,
            completion_tokens=220,
            temperature=0.7,
        )
        .add_evaluation_span(dimension="factual_accuracy", score=0.92)
        .build()
    )

    # Sanity checks so fixture consumers can rely on structure
    assert trace.span_count == 5
    assert trace.root_span is not None
    assert trace.root_span.span_type == SpanType.AGENT
    assert len(trace.llm_spans) == 2
    assert len(trace.tool_spans) == 1

    return trace


# --------------------------------------------------------------------------- #
# Parameterised source formatters
# --------------------------------------------------------------------------- #

@pytest.fixture(
    params=["generic_otel", "openai", "anthropic", "agentforce_otlp"],
    ids=["generic_otel", "openai", "anthropic", "agentforce_otlp"],
)
def source_formatter(request: pytest.FixtureRequest):
    """Parameterised fixture yielding common source formatters.

    Tests using this fixture will run once per source format:
    ``generic_otel``, ``openai``, ``anthropic``, ``agentforce_otlp``.
    """
    return get_source_formatter(request.param)
