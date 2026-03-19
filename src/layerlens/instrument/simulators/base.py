"""Base simulator classes.

BaseSimulator ABC mirrors BaseAdapter lifecycle pattern.
TraceSimulator is the main orchestrator implementing the 3-layer architecture:
Scenario → Source → Output.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from .clock import DeterministicClock
from .config import (
    ContentTier,
    OutputFormat,
    ScenarioName,
    SimulatorConfig,
    SourceFormat,
)
from .content.template_provider import TemplateContentProvider
from .identifiers import IDGenerator
from .span_model import SimulatedTrace
from .trace_builder import TraceBuilder

logger = logging.getLogger(__name__)


class SimulatorResult(BaseModel):
    """Result of a simulation run."""

    run_id: str
    trace_count: int = 0
    span_count: int = 0
    total_tokens: int = 0
    error_count: int = 0
    validation_status: str = "pending"
    validation_details: list[dict[str, Any]] = Field(default_factory=list)
    duration_ms: float = 0.0
    config_summary: dict[str, Any] = Field(default_factory=dict)


class BaseSimulator(ABC):
    """Abstract base class for simulators.

    Mirrors BaseAdapter lifecycle: initialize → generate → shutdown.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the simulator (load resources, validate config)."""

    @abstractmethod
    def generate(
        self,
        count: int = 1,
        scenario: str | None = None,
    ) -> list[SimulatedTrace]:
        """Generate simulated traces."""

    @abstractmethod
    def format_output(
        self,
        traces: list[SimulatedTrace],
        output_format: str = "otlp_json",
    ) -> list[dict[str, Any]]:
        """Format traces into wire format."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources."""

    def generate_and_format(
        self,
        count: int | None = None,
        scenario: str | None = None,
        output_format: str | None = None,
    ) -> tuple[list[dict[str, Any]], SimulatorResult]:
        """Generate and format traces in one call."""
        traces = self.generate(count=count or 1, scenario=scenario)
        formatted = self.format_output(traces, output_format=output_format or "otlp_json")
        result = SimulatorResult(
            run_id="",
            trace_count=len(traces),
            span_count=sum(t.span_count for t in traces),
            total_tokens=sum(t.total_tokens for t in traces),
        )
        return formatted, result


class TraceSimulator(BaseSimulator):
    """Main simulator orchestrator.

    Implements the 3-layer architecture:
    SimulatorConfig → Scenario → ContentProvider → TraceBuilder
    → SourceFormatter → OutputFormatter → wire format
    """

    def __init__(self, config: SimulatorConfig | None = None):
        self._config = config or SimulatorConfig.minimal()
        self._clock: DeterministicClock | None = None
        self._ids: IDGenerator | None = None
        self._content_provider: TemplateContentProvider | None = None
        self._initialized = False
        self._source_formatter: Any = None
        self._output_formatter: Any = None

    @property
    def config(self) -> SimulatorConfig:
        return self._config

    def initialize(self, api_key: str | None = None) -> None:
        """Initialize clock, ID generator, and content provider."""
        self._clock = DeterministicClock(seed=self._config.seed)
        self._ids = IDGenerator(seed=self._config.seed)

        # Initialize content provider based on tier
        if self._config.content.tier == ContentTier.TEMPLATE:
            self._content_provider = TemplateContentProvider(seed=self._config.seed)
        elif self._config.content.tier == ContentTier.SEED:
            from .content.seed_provider import SeedContentProvider
            seed_path = self._config.content.seed_data_path
            if seed_path:
                self._content_provider = SeedContentProvider(
                    seed_data_path=seed_path, seed=self._config.seed,
                )
            else:
                logger.warning("Seed tier selected but no seed_data_path configured, falling back to template")
                self._content_provider = TemplateContentProvider(seed=self._config.seed)
        elif self._config.content.tier == ContentTier.LLM:
            from .content.llm_provider import LLMContentProvider
            self._content_provider = LLMContentProvider(
                model=self._config.content.llm_model,
                base_url=self._config.content.llm_base_url,
                cache_enabled=self._config.content.llm_cache_enabled,
                cache_path=self._config.content.llm_cache_path,
                api_key=api_key,
                seed=self._config.seed,
            )
        else:
            self._content_provider = TemplateContentProvider(seed=self._config.seed)

        # Source and output formatters loaded lazily from registries
        self._load_source_formatter()
        self._load_output_formatter()
        self._initialized = True

    def _load_source_formatter(self) -> None:
        """Load source formatter from registry (lazy import to avoid circular deps)."""
        try:
            from .sources import get_source_formatter

            self._source_formatter = get_source_formatter(self._config.source_format.value)
        except (ImportError, ValueError) as e:
            logger.warning("Failed to load source formatter %s: %s", self._config.source_format.value, e)
            self._source_formatter = None

    def _load_output_formatter(self) -> None:
        """Load output formatter from registry (lazy import)."""
        try:
            from .outputs import get_output_formatter

            self._output_formatter = get_output_formatter(self._config.output_format.value)
        except (ImportError, ValueError) as e:
            logger.warning("Failed to load output formatter %s: %s", self._config.output_format.value, e)
            self._output_formatter = None

    def generate(
        self,
        count: int | None = None,
        scenario: str | None = None,
    ) -> list[SimulatedTrace]:
        """Generate simulated traces."""
        if not self._initialized:
            self.initialize()

        num = count if count is not None else self._config.count
        scenario_name = scenario or self._config.scenario.value

        # Handle conversation mode: generate multi-turn conversation traces
        if self._config.conversation.enabled:
            return self._generate_conversation_traces(scenario_name, num)

        traces: list[SimulatedTrace] = []
        for i in range(num):
            trace = self._generate_single_trace(scenario_name, i)
            traces.append(trace)

        return traces

    def _generate_conversation_traces(
        self, scenario: str, count: int
    ) -> list[SimulatedTrace]:
        """Generate traces using ConversationBuilder for multi-turn mode."""
        from .conversation import ConversationBuilder

        if self._content_provider is None or self._clock is None:
            raise RuntimeError("Simulator not initialized. Call initialize() first.")

        provider, model = self._get_provider_model()
        topics = self._content_provider.get_topics(scenario)
        all_traces: list[SimulatedTrace] = []

        conv_builder = ConversationBuilder(
            config=self._config.conversation,
            seed=self._config.seed,
        )

        for i in range(count):
            topic = topics[i % len(topics)] if topics else "General"
            conv_traces = conv_builder.build_conversation(
                scenario=scenario,
                topic=topic,
                provider=provider,
                model=model,
                content_provider=self._content_provider,
                include_content=self._config.include_content,
            )
            # Apply source enrichment, error injection, and streaming to each trace
            for trace in conv_traces:
                if self._source_formatter:
                    trace = self._enrich_trace(trace)
                trace = self._apply_errors(trace, i)
                trace = self._apply_streaming(trace, i)
                all_traces.append(trace)

        return all_traces

    def _generate_single_trace(self, scenario: str, index: int) -> SimulatedTrace:
        """Generate a single trace using TraceBuilder + ContentProvider."""
        if self._content_provider is None or self._clock is None or self._ids is None:
            raise RuntimeError("Simulator not initialized. Call initialize() first.")

        # Select topic
        topics = self._content_provider.get_topics(scenario)
        topic = topics[index % len(topics)] if topics else "General"

        # Select agent name
        agent_names = self._content_provider.get_agent_names(scenario)
        agent_name = agent_names[0] if agent_names else f"{scenario}_Agent"

        # Select tools
        tool_names = self._content_provider.get_tool_names(scenario, topic)

        # Build provider/model from source format
        provider, model = self._get_provider_model()

        # Build trace
        seed = (
            (self._config.seed + index) if self._config.seed is not None else None
        )
        builder = TraceBuilder(seed=seed)
        builder.with_scenario(scenario, topic=topic)
        builder.with_source(self._config.source_format.value)

        # Agent span (root)
        builder.add_agent_span(agent_name)

        # First LLM call (planning)
        prompt_tokens = self._clock.randint(150, 500)
        completion_tokens = self._clock.randint(100, 400)
        input_msgs: list[dict[str, Any]] = []
        output_msg: dict[str, Any] | None = None

        if self._config.include_content:
            system_prompt = self._content_provider.get_system_prompt(scenario, agent_name)
            user_msg = self._content_provider.get_user_message(scenario, topic, turn=1)
            agent_resp = self._content_provider.get_agent_response(scenario, topic, turn=1)
            input_msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            output_msg = {"role": "assistant", "content": agent_resp}

        builder.add_llm_span(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            temperature=0.7,
            input_messages=input_msgs,
            output_message=output_msg,
        )

        # Tool calls
        for tool_name in tool_names[:2]:  # Max 2 tools per trace
            tool_input = (
                self._content_provider.get_tool_input(tool_name, topic)
                if self._config.include_content
                else None
            )
            tool_output = (
                self._content_provider.get_tool_output(tool_name, topic)
                if self._config.include_content
                else None
            )
            builder.add_tool_span(
                name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
            )

        # Second LLM call (response generation)
        builder.add_llm_span(
            provider=provider,
            model=model,
            prompt_tokens=self._clock.randint(300, 800),
            completion_tokens=self._clock.randint(150, 500),
            temperature=0.7,
        )

        # Evaluation span
        eval_score = self._clock.uniform(0.7, 1.0)
        builder.add_evaluation_span(
            dimension="factual_accuracy",
            score=round(eval_score, 2),
        )

        # Apply source enrichment
        trace = builder.build()
        if self._source_formatter:
            trace = self._enrich_trace(trace)

        # Apply error injection and streaming
        trace = self._apply_errors(trace, index)
        trace = self._apply_streaming(trace, index)

        return trace

    def _apply_errors(self, trace: SimulatedTrace, index: int) -> SimulatedTrace:
        """Apply error injection if configured."""
        if not self._config.errors.enabled:
            return trace
        from .errors import inject_errors

        error_seed = (self._config.seed + index + 10000) if self._config.seed is not None else None
        return inject_errors(trace, self._config.errors, seed=error_seed)

    def _apply_streaming(self, trace: SimulatedTrace, index: int) -> SimulatedTrace:
        """Apply streaming behavior if configured."""
        if not self._config.streaming.enabled:
            return trace
        from .streaming import StreamingBehavior

        stream_seed = (self._config.seed + index + 20000) if self._config.seed is not None else None
        behavior = StreamingBehavior(self._config.streaming, seed=stream_seed)
        return behavior.apply(trace)

    def _enrich_trace(self, trace: SimulatedTrace) -> SimulatedTrace:
        """Apply source formatter enrichment to all spans."""
        if not self._source_formatter:
            return trace

        profile = self._source_formatter.get_default_profile()
        trace.resource_attributes = self._source_formatter.get_resource_attributes()
        scope_name, scope_version = self._source_formatter.get_scope()
        trace.scope_name = scope_name
        trace.scope_version = scope_version

        for span in trace.spans:
            self._source_formatter.enrich_span(
                span, profile, include_content=self._config.include_content
            )

        return trace

    def _get_provider_model(self) -> tuple[str, str]:
        """Map source format to default provider and model."""
        provider_models: dict[str, tuple[str, str]] = {
            SourceFormat.GENERIC_OTEL.value: ("openai", "gpt-4o"),
            SourceFormat.AGENTFORCE_OTLP.value: ("openai", "gpt-4o"),
            SourceFormat.AGENTFORCE_SOQL.value: ("openai", "gpt-4o"),
            SourceFormat.OPENAI.value: ("openai", "gpt-4o"),
            SourceFormat.ANTHROPIC.value: ("anthropic", "claude-sonnet-4-20250514"),
            SourceFormat.AZURE_OPENAI.value: ("azure_openai", "gpt-4o"),
            SourceFormat.BEDROCK.value: ("bedrock", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
            SourceFormat.GOOGLE_VERTEX.value: ("google_vertex", "gemini-1.5-pro"),
            SourceFormat.OLLAMA.value: ("ollama", "llama3.1:70b"),
            SourceFormat.LITELLM.value: ("litellm", "gpt-4o"),
            SourceFormat.LANGFUSE.value: ("openai", "gpt-4o"),
            SourceFormat.JSONL.value: ("openai", "gpt-4o"),
        }
        return provider_models.get(
            self._config.source_format.value, ("openai", "gpt-4o")
        )

    def format_output(
        self,
        traces: list[SimulatedTrace],
        output_format: str | None = None,
    ) -> list[dict[str, Any]]:
        """Format traces to wire format using output formatter."""
        fmt = output_format or self._config.output_format.value

        # If a different format was requested at call time, load the appropriate formatter
        if fmt != self._config.output_format.value:
            try:
                from .outputs import get_output_formatter
                formatter = get_output_formatter(fmt)
                return formatter.format_batch(traces)
            except (ImportError, ValueError) as e:
                logger.warning("Failed to load output formatter %s: %s", fmt, e)

        if self._output_formatter:
            return self._output_formatter.format_batch(traces)

        # Fallback: return as dicts
        return [t.model_dump(mode="json") for t in traces]

    def shutdown(self) -> None:
        """Release resources."""
        self._clock = None
        self._ids = None
        self._content_provider = None
        self._source_formatter = None
        self._output_formatter = None
        self._initialized = False

    def generate_and_format(
        self,
        count: int | None = None,
        scenario: str | None = None,
        output_format: str | None = None,
    ) -> tuple[list[dict[str, Any]], SimulatorResult]:
        """Generate, enrich, format, and return traces + result."""
        import time

        start = time.monotonic()
        if not self._initialized:
            self.initialize()

        traces = self.generate(
            count=count if count is not None else self._config.count,
            scenario=scenario,
        )
        formatted = self.format_output(traces, output_format=output_format)
        elapsed = (time.monotonic() - start) * 1000

        error_count = sum(
            1
            for t in traces
            for s in t.spans
            if s.error_type is not None
        )

        run_id = self._ids.run_id() if self._ids else "run_unknown"
        result = SimulatorResult(
            run_id=run_id,
            trace_count=len(traces),
            span_count=sum(t.span_count for t in traces),
            total_tokens=sum(t.total_tokens for t in traces),
            error_count=error_count,
            duration_ms=elapsed,
            config_summary=self._config.model_dump(mode="json"),
        )

        # Write output file if configured
        if self._config.output_path and not self._config.dry_run:
            self._write_output(formatted, self._config.output_path)

        return formatted, result

    def _write_output(self, data: list[dict[str, Any]], path: str) -> None:
        """Write formatted output to file."""
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
