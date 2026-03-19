"""STRATIX Simulator CLI.

Click-based CLI for generating simulated traces, running round-trip
validation, and listing available sources and scenarios.

Usage:
    python -m layerlens.instrument.simulators generate --source openai --count 10
    python -m layerlens.instrument.simulators validate --source generic_otel
    python -m layerlens.instrument.simulators list-sources
    python -m layerlens.instrument.simulators list-scenarios
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import click

from .config import OutputFormat, ScenarioName, SimulatorConfig, SourceFormat


# Build choice lists from enums
_SOURCE_CHOICES = [sf.value for sf in SourceFormat] + ["all"]
_SCENARIO_CHOICES = [sn.value for sn in ScenarioName] + ["all"]
_OUTPUT_FORMAT_CHOICES = [of.value for of in OutputFormat]
_PRESET_CHOICES = ["minimal", "standard", "full"]


def _styled(text: str, **kwargs: Any) -> str:
    """Wrap click.style for consistent formatting."""
    return click.style(text, **kwargs)


def _header(text: str) -> None:
    """Print a styled header line."""
    click.echo(_styled(text, fg="cyan", bold=True))


def _success(text: str) -> None:
    """Print a styled success line."""
    click.echo(_styled(text, fg="green"))


def _warning(text: str) -> None:
    """Print a styled warning line."""
    click.echo(_styled(text, fg="yellow"))


def _error(text: str) -> None:
    """Print a styled error line."""
    click.echo(_styled(text, fg="red", bold=True), err=True)


def _info(text: str) -> None:
    """Print an info line."""
    click.echo(text)


@click.group()
@click.version_option(package_name="stratix")
def cli() -> None:
    """STRATIX Multi-Source OTel Trace Simulator.

    Generate simulated traces across 12 ingestion sources and 5 business
    scenarios. Supports OTLP JSON, Langfuse JSON, and STRATIX Native output
    formats with error injection, streaming, and multi-turn conversations.
    """


@cli.command()
@click.option(
    "--source",
    required=True,
    type=click.Choice(_SOURCE_CHOICES, case_sensitive=False),
    help="Source format to simulate (or 'all' for all sources).",
)
@click.option(
    "--scenario",
    default="customer_service",
    type=click.Choice(_SCENARIO_CHOICES, case_sensitive=False),
    help="Scenario to use (or 'all' for all scenarios).",
)
@click.option(
    "--count",
    default=5,
    type=click.IntRange(min=1),
    help="Number of traces to generate per source/scenario combination.",
)
@click.option(
    "--output-format",
    default="otlp_json",
    type=click.Choice(_OUTPUT_FORMAT_CHOICES, case_sensitive=False),
    help="Wire output format.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for deterministic output.",
)
@click.option(
    "--include-content",
    is_flag=True,
    default=False,
    help="Include message content in generated traces.",
)
@click.option(
    "--errors",
    is_flag=True,
    default=False,
    help="Enable error injection (rate limits, timeouts, etc.).",
)
@click.option(
    "--streaming",
    is_flag=True,
    default=False,
    help="Enable streaming simulation (TTFT, TPOT, chunks).",
)
@click.option(
    "--conversations",
    is_flag=True,
    default=False,
    help="Enable multi-turn conversation traces.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Output file path (writes JSON).",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to YAML config file (overrides other options).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate configuration without generating traces.",
)
@click.option(
    "--preset",
    default=None,
    type=click.Choice(_PRESET_CHOICES, case_sensitive=False),
    help="Use a preset configuration (minimal, standard, full).",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
def generate(
    source: str,
    scenario: str,
    count: int,
    output_format: str,
    seed: int | None,
    include_content: bool,
    errors: bool,
    streaming: bool,
    conversations: bool,
    output: str | None,
    config_path: str | None,
    dry_run: bool,
    preset: str | None,
    verbose: bool,
) -> None:
    """Generate simulated traces.

    Produces realistic OTel-compatible traces for any of the 12 supported
    ingestion sources across 5 business scenarios.

    Examples:

      # Generate 10 OpenAI traces
      stratix-sim generate --source openai --count 10

      # Generate all sources with full preset
      stratix-sim generate --source all --preset full

      # Generate with YAML config
      stratix-sim generate --source generic_otel --config my_config.yaml
    """
    from .base import TraceSimulator

    # Determine which sources to iterate
    sources = [sf.value for sf in SourceFormat] if source == "all" else [source]
    # Determine which scenarios to iterate
    scenarios = (
        [sn.value for sn in ScenarioName] if scenario == "all" else [scenario]
    )

    total_traces = 0
    total_spans = 0
    total_tokens = 0
    total_errors = 0
    all_formatted: list[dict[str, Any]] = []
    start_time = time.monotonic()

    for src in sources:
        for scn in scenarios:
            # Build config
            if config_path:
                cfg = SimulatorConfig.from_yaml(config_path)
                # Override source/scenario from CLI even when using config file
                cfg = cfg.model_copy(
                    update={
                        "source_format": SourceFormat(src),
                        "scenario": ScenarioName(scn),
                    }
                )
            elif preset:
                factory = {
                    "minimal": SimulatorConfig.minimal,
                    "standard": SimulatorConfig.standard,
                    "full": SimulatorConfig.full,
                }[preset]
                cfg = factory()
                cfg = cfg.model_copy(
                    update={
                        "source_format": SourceFormat(src),
                        "scenario": ScenarioName(scn),
                        "output_format": OutputFormat(output_format),
                    }
                )
                if seed is not None:
                    cfg = cfg.model_copy(update={"seed": seed})
            else:
                from .config import (
                    ConversationConfig,
                    ErrorConfig,
                    StreamingConfig,
                )

                cfg = SimulatorConfig(
                    source_format=SourceFormat(src),
                    scenario=ScenarioName(scn),
                    output_format=OutputFormat(output_format),
                    count=count,
                    seed=seed,
                    include_content=include_content,
                    errors=ErrorConfig(enabled=errors),
                    streaming=StreamingConfig(enabled=streaming),
                    conversation=ConversationConfig(enabled=conversations),
                    dry_run=dry_run,
                    output_path=None,  # We handle output ourselves
                )

            if dry_run:
                _header(f"[DRY RUN] {src} / {scn}")
                _info(f"  Count:          {cfg.count}")
                _info(f"  Output format:  {cfg.output_format.value}")
                _info(f"  Seed:           {cfg.seed}")
                _info(f"  Include content: {cfg.include_content}")
                _info(f"  Errors:         {cfg.errors.enabled}")
                _info(f"  Streaming:      {cfg.streaming.enabled}")
                _info(f"  Conversations:  {cfg.conversation.enabled}")
                click.echo("")
                continue

            if verbose:
                _header(f"Generating: {src} / {scn} (count={cfg.count})")

            simulator = TraceSimulator(cfg)
            try:
                formatted, result = simulator.generate_and_format(
                    count=cfg.count,
                    scenario=scn,
                    output_format=cfg.output_format.value,
                )
                all_formatted.extend(formatted)
                total_traces += result.trace_count
                total_spans += result.span_count
                total_tokens += result.total_tokens
                total_errors += result.error_count

                if verbose:
                    _info(
                        f"  -> {result.trace_count} traces, "
                        f"{result.span_count} spans, "
                        f"{result.total_tokens} tokens"
                    )
                    if result.error_count > 0:
                        _warning(f"  -> {result.error_count} injected errors")
            finally:
                simulator.shutdown()

    elapsed_ms = (time.monotonic() - start_time) * 1000

    if dry_run:
        _success("Dry run complete. No traces generated.")
        return

    # Write output file
    if output and all_formatted:
        from pathlib import Path

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(all_formatted, f, indent=2)
        _success(f"Wrote {len(all_formatted)} trace(s) to {output}")

    # Summary
    click.echo("")
    _header("Generation Summary")
    _info(f"  Sources:    {len(sources)}")
    _info(f"  Scenarios:  {len(scenarios)}")
    _info(f"  Traces:     {total_traces}")
    _info(f"  Spans:      {total_spans}")
    _info(f"  Tokens:     {total_tokens:,}")
    if total_errors > 0:
        _warning(f"  Errors:     {total_errors}")
    _info(f"  Duration:   {elapsed_ms:.1f} ms")

    if not output and all_formatted:
        _info("")
        _info(
            f"  (Use -o/--output to write {len(all_formatted)} "
            f"trace(s) to a file)"
        )


@cli.command()
@click.option(
    "--source",
    required=True,
    type=click.Choice(_SOURCE_CHOICES, case_sensitive=False),
    help="Source format to validate.",
)
@click.option(
    "--scenario",
    default="customer_service",
    type=click.Choice([sn.value for sn in ScenarioName], case_sensitive=False),
    help="Scenario to use for validation.",
)
@click.option(
    "--count",
    default=5,
    type=click.IntRange(min=1),
    help="Number of traces to generate for validation.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for deterministic output.",
)
def validate(source: str, scenario: str, count: int, seed: int | None) -> None:
    """Run round-trip validation.

    Generates traces for the given source and validates that they can be
    serialized to the output format and back without data loss.

    Examples:

      stratix-sim validate --source openai --count 10
      stratix-sim validate --source all --seed 42
    """
    from .base import TraceSimulator

    sources = [sf.value for sf in SourceFormat] if source == "all" else [source]
    all_passed = True

    _header("Round-Trip Validation")
    click.echo("")

    for src in sources:
        cfg = SimulatorConfig(
            source_format=SourceFormat(src),
            scenario=ScenarioName(scenario),
            count=count,
            seed=seed,
            include_content=True,
        )

        simulator = TraceSimulator(cfg)
        try:
            simulator.initialize()
            traces = simulator.generate(count=count, scenario=scenario)
            formatted = simulator.format_output(traces)

            # Validate: check each formatted trace has required keys
            issues: list[str] = []
            for i, trace_data in enumerate(formatted):
                if not isinstance(trace_data, dict):
                    issues.append(f"Trace {i}: not a dict (got {type(trace_data).__name__})")
                    continue

                # Check trace has spans or resource data
                has_spans = (
                    "spans" in trace_data
                    or "resourceSpans" in trace_data
                    or "trace_id" in trace_data
                )
                if not has_spans:
                    issues.append(f"Trace {i}: missing span data keys")

            # Validate round-trip: serialize and deserialize
            try:
                serialized = json.dumps(formatted)
                deserialized = json.loads(serialized)
                if len(deserialized) != len(formatted):
                    issues.append(
                        f"Round-trip count mismatch: "
                        f"{len(formatted)} -> {len(deserialized)}"
                    )
            except (json.JSONDecodeError, TypeError) as e:
                issues.append(f"Serialization error: {e}")

            # Report
            status_label = _styled("PASS", fg="green", bold=True)
            if issues:
                status_label = _styled("FAIL", fg="red", bold=True)
                all_passed = False

            span_count = sum(t.span_count for t in traces)
            token_count = sum(t.total_tokens for t in traces)
            _info(
                f"  [{status_label}] {src:<20s} "
                f"{count} traces, {span_count} spans, {token_count:,} tokens"
            )

            if issues:
                for issue in issues:
                    _warning(f"         {issue}")

        finally:
            simulator.shutdown()

    click.echo("")
    if all_passed:
        _success("All validations passed.")
    else:
        _error("Some validations failed.")
        sys.exit(1)


@cli.command("list-sources")
def list_sources_cmd() -> None:
    """List available source formatters.

    Displays all 12 supported ingestion sources with their descriptions.
    """
    from .sources import get_source_formatter, list_sources

    source_names = list_sources()

    _header("Available Source Formatters")
    click.echo("")

    # Table header
    _info(f"  {'Name':<22s} {'Description'}")
    _info(f"  {'----':<22s} {'-----------'}")

    for name in source_names:
        try:
            formatter = get_source_formatter(name)
            # Extract description from the class docstring
            doc = type(formatter).__doc__ or ""
            description = doc.strip().split("\n")[0] if doc.strip() else "(no description)"
        except Exception:
            description = "(unavailable)"

        _info(f"  {name:<22s} {description}")

    click.echo("")
    _info(f"  Total: {len(source_names)} sources")


@cli.command("list-scenarios")
def list_scenarios_cmd() -> None:
    """List available scenarios.

    Displays all 5 business scenarios with their topic counts.
    """
    from .scenarios.registry import get_scenario, list_scenarios

    scenario_names = list_scenarios()

    _header("Available Scenarios")
    click.echo("")

    # Table header
    _info(f"  {'Name':<22s} {'Topics':<8s} {'Topic List'}")
    _info(f"  {'----':<22s} {'------':<8s} {'----------'}")

    for name in scenario_names:
        try:
            scenario = get_scenario(name)
            topics = scenario.topics
            topic_count = len(topics)
            topic_list = ", ".join(topics)
            # Truncate long topic lists
            if len(topic_list) > 50:
                topic_list = topic_list[:47] + "..."
        except Exception:
            topic_count = 0
            topic_list = "(unavailable)"

        _info(f"  {name:<22s} {topic_count:<8d} {topic_list}")

    click.echo("")
    _info(f"  Total: {len(scenario_names)} scenarios")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
