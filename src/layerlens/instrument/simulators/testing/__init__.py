"""Testing utilities for the simulator SDK.

Provides assertion helpers, Hypothesis property-based strategies,
round-trip validation pipelines, and pytest fixtures for testing
simulator-generated traces and OTLP output.

Quick start (pytest)::

    from layerlens.instrument.simulators.testing import (
        assert_valid_otlp_trace,
        assert_span_tree,
        validate_round_trip,
    )

Quick start (hypothesis)::

    from layerlens.instrument.simulators.testing import (
        simulated_trace,
        token_usage,
    )
"""

# Assertion helpers
from .assertions import (
    assert_deterministic,
    assert_genai_attributes,
    assert_round_trip,
    assert_span_tree,
    assert_token_counts,
    assert_valid_otlp_trace,
)

# Pytest fixtures (imported so pytest auto-discovers them via conftest or plugin)
from .fixtures import (
    full_config,
    minimal_config,
    sample_trace,
    simulator,
    source_formatter,
    standard_config,
    trace_builder,
)

# Hypothesis strategies (lazy — only available if hypothesis is installed)
try:
    from .hypothesis_strategies import (
        error_config,
        simulated_span,
        simulated_trace,
        simulator_config,
        token_usage,
    )
except ImportError:
    pass

# Round-trip validation
from .round_trip import RoundTripResult, validate_all_sources, validate_round_trip

__all__ = [
    # Assertions
    "assert_valid_otlp_trace",
    "assert_genai_attributes",
    "assert_span_tree",
    "assert_token_counts",
    "assert_deterministic",
    "assert_round_trip",
    # Hypothesis strategies
    "token_usage",
    "simulated_span",
    "simulated_trace",
    "simulator_config",
    "error_config",
    # Round-trip
    "RoundTripResult",
    "validate_round_trip",
    "validate_all_sources",
    # Fixtures
    "simulator",
    "trace_builder",
    "minimal_config",
    "standard_config",
    "full_config",
    "sample_trace",
    "source_formatter",
]
