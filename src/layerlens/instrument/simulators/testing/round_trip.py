"""Round-trip validation pipeline for the simulator SDK.

Generates traces through the full pipeline (TraceBuilder -> Source enrichment
-> Output formatting -> structural validation) and reports results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import ScenarioName, SimulatorConfig, SourceFormat
from ..outputs import get_output_formatter
from ..sources import get_source_formatter, list_sources
from ..trace_builder import TraceBuilder
from .assertions import (
    assert_span_tree,
    assert_token_counts,
    assert_valid_otlp_trace,
)


@dataclass
class RoundTripResult:
    """Result of a round-trip validation run."""

    source: str
    output_format: str
    traces_generated: int
    traces_validated: int
    passed: bool
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.source} -> {self.output_format}: "
            f"{self.traces_validated}/{self.traces_generated} validated"
            + (f" ({len(self.errors)} errors)" if self.errors else "")
        )


def _build_sample_trace(
    scenario: str = "customer_service",
    seed: int = 42,
) -> "SimulatedTrace":  # noqa: F821 — forward reference resolved at runtime
    """Build a representative trace with all span types."""
    from ..span_model import SimulatedTrace  # local to avoid circular

    builder = TraceBuilder(seed=seed)
    trace = (
        builder
        .with_scenario(scenario, topic="order_inquiry")
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
    return trace


def validate_round_trip(
    source_format: str,
    output_format: str = "otlp_json",
    scenario: str = "customer_service",
    count: int = 1,
    seed: int = 42,
) -> RoundTripResult:
    """Generate -> enrich -> format -> validate pipeline.

    Parameters
    ----------
    source_format:
        Name of the source formatter (e.g. ``"openai"``, ``"anthropic"``).
    output_format:
        Name of the output formatter (default ``"otlp_json"``).
    scenario:
        Scenario name passed to the TraceBuilder.
    count:
        Number of traces to generate and validate.
    seed:
        Base seed for deterministic generation.

    Returns
    -------
    RoundTripResult
        Summary with pass/fail status and any errors.
    """
    errors: list[str] = []
    validated = 0

    try:
        source_fmt = get_source_formatter(source_format)
    except ValueError as exc:
        return RoundTripResult(
            source=source_format,
            output_format=output_format,
            traces_generated=0,
            traces_validated=0,
            passed=False,
            errors=[f"Source formatter error: {exc}"],
        )

    try:
        output_fmt = get_output_formatter(output_format)
    except ValueError as exc:
        return RoundTripResult(
            source=source_format,
            output_format=output_format,
            traces_generated=0,
            traces_validated=0,
            passed=False,
            errors=[f"Output formatter error: {exc}"],
        )

    for i in range(count):
        trace_seed = seed + i
        try:
            # 1) Build trace
            trace = _build_sample_trace(scenario=scenario, seed=trace_seed)

            # 2) Enrich with source
            profile = source_fmt.get_default_profile()
            trace.resource_attributes = source_fmt.get_resource_attributes()
            scope_name, scope_version = source_fmt.get_scope()
            trace.scope_name = scope_name
            trace.scope_version = scope_version
            for span in trace.spans:
                source_fmt.enrich_span(span, profile, include_content=False)

            # 3) Validate token counts on internal model
            assert_token_counts(trace)

            # 4) Format to wire
            output = output_fmt.format_trace(trace)

            # 5) Validate OTLP structure
            if output_format == "otlp_json":
                assert_valid_otlp_trace(output)
                assert_span_tree(output)

            validated += 1

        except Exception as exc:
            errors.append(f"Trace {i} (seed={trace_seed}): {exc}")

    return RoundTripResult(
        source=source_format,
        output_format=output_format,
        traces_generated=count,
        traces_validated=validated,
        passed=len(errors) == 0,
        errors=errors,
    )


def validate_all_sources(
    count: int = 1,
    seed: int = 42,
) -> list[RoundTripResult]:
    """Run round-trip validation for all registered source formatters.

    Returns a list of ``RoundTripResult`` objects, one per source.
    """
    results: list[RoundTripResult] = []
    for source_name in list_sources():
        result = validate_round_trip(
            source_format=source_name,
            output_format="otlp_json",
            scenario="customer_service",
            count=count,
            seed=seed,
        )
        results.append(result)
    return results
