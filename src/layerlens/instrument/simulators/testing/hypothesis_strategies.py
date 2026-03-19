"""Hypothesis strategies for property-based testing of the simulator SDK.

Provides composable strategies for generating random but valid instances
of the simulator's Pydantic models: TokenUsage, SimulatedSpan,
SimulatedTrace, SimulatorConfig, and ErrorConfig.
"""

from __future__ import annotations

import string

import hypothesis.strategies as st
from hypothesis import assume

from ..config import (
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
from ..span_model import (
    SPAN_TYPE_TO_KIND,
    SimulatedSpan,
    SimulatedTrace,
    SpanKind,
    SpanStatus,
    SpanType,
    TokenUsage,
)


# --------------------------------------------------------------------------- #
# Primitive helpers
# --------------------------------------------------------------------------- #

def _hex_string(n_bytes: int) -> st.SearchStrategy[str]:
    """Strategy for hex strings of *n_bytes* bytes (2*n_bytes chars)."""
    return st.binary(min_size=n_bytes, max_size=n_bytes).map(lambda b: b.hex())


def _model_name() -> st.SearchStrategy[str]:
    """Strategy for plausible model names."""
    return st.sampled_from([
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "claude-sonnet-4-20250514",
        "claude-3-haiku-20240307",
        "gemini-1.5-pro",
        "llama3.1:70b",
        "mistral-large-latest",
    ])


def _provider_name() -> st.SearchStrategy[str]:
    """Strategy for provider names."""
    return st.sampled_from([
        "openai",
        "anthropic",
        "azure_openai",
        "bedrock",
        "google_vertex",
        "ollama",
        "litellm",
    ])


def _span_name(span_type: SpanType) -> st.SearchStrategy[str]:
    """Strategy for span names keyed by type."""
    prefixes = {
        SpanType.AGENT: "agent ",
        SpanType.LLM: "chat ",
        SpanType.TOOL: "tool ",
        SpanType.EVALUATION: "evaluation ",
    }
    prefix = prefixes.get(span_type, "span_")
    suffix = st.text(
        alphabet=string.ascii_lowercase + "_",
        min_size=3,
        max_size=20,
    )
    return suffix.map(lambda s: f"{prefix}{s}")


# --------------------------------------------------------------------------- #
# TokenUsage
# --------------------------------------------------------------------------- #

@st.composite
def token_usage(draw: st.DrawFn) -> TokenUsage:
    """Strategy for generating ``TokenUsage`` instances.

    Guarantees total_tokens >= prompt_tokens + completion_tokens.
    """
    prompt = draw(st.integers(min_value=0, max_value=10_000))
    completion = draw(st.integers(min_value=0, max_value=10_000))
    total = prompt + completion
    cached = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=prompt)))
    reasoning = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=completion)))

    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )


# --------------------------------------------------------------------------- #
# SimulatedSpan
# --------------------------------------------------------------------------- #

@st.composite
def simulated_span(
    draw: st.DrawFn,
    span_type: SpanType | None = None,
    parent_span_id: str | None = None,
) -> SimulatedSpan:
    """Strategy for generating ``SimulatedSpan`` instances.

    Parameters
    ----------
    span_type:
        Fix the span type. When ``None`` a random type is drawn.
    parent_span_id:
        Fix the parent span ID. When ``None`` the span is either a root
        (no parent) or is given a random parent ID.
    """
    stype = span_type or draw(st.sampled_from(list(SpanType)))
    span_id = draw(_hex_string(8))
    name = draw(_span_name(stype))

    # Timestamps: start in realistic nanosecond range, duration 10ms-10s
    start_ns = draw(st.integers(
        min_value=1_700_000_000_000_000_000,
        max_value=1_800_000_000_000_000_000,
    ))
    duration_ns = draw(st.integers(min_value=10_000_000, max_value=10_000_000_000))
    end_ns = start_ns + duration_ns

    kind = SPAN_TYPE_TO_KIND.get(stype, SpanKind.INTERNAL)
    status = draw(st.sampled_from(list(SpanStatus)))

    # Build kwargs based on span type
    kwargs: dict = dict(
        span_id=span_id,
        parent_span_id=parent_span_id,
        span_type=stype,
        name=name,
        start_time_unix_nano=start_ns,
        end_time_unix_nano=end_ns,
        kind=kind,
        status=status,
    )

    if stype == SpanType.LLM:
        kwargs["provider"] = draw(_provider_name())
        kwargs["model"] = draw(_model_name())
        kwargs["operation"] = draw(st.sampled_from(["chat", "text_completion"]))
        kwargs["token_usage"] = draw(token_usage())
        kwargs["temperature"] = draw(st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
        ))
        kwargs["max_tokens"] = draw(st.one_of(
            st.none(),
            st.integers(min_value=1, max_value=4096),
        ))
        kwargs["top_p"] = draw(st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ))
        kwargs["finish_reasons"] = draw(st.just(["stop"]))

    elif stype == SpanType.TOOL:
        kwargs["tool_name"] = draw(st.text(
            alphabet=string.ascii_lowercase + "_",
            min_size=3,
            max_size=30,
        ))
        kwargs["tool_description"] = draw(st.one_of(
            st.none(),
            st.text(min_size=5, max_size=100),
        ))

    elif stype == SpanType.AGENT:
        kwargs["agent_name"] = draw(st.text(
            alphabet=string.ascii_letters + "_",
            min_size=3,
            max_size=30,
        ))

    elif stype == SpanType.EVALUATION:
        kwargs["eval_dimension"] = draw(st.sampled_from([
            "factual_accuracy",
            "helpfulness",
            "safety",
            "relevance",
            "compliance",
        ]))
        kwargs["eval_score"] = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        kwargs["eval_label"] = draw(st.sampled_from(["pass", "fail"]))

    return SimulatedSpan(**kwargs)


# --------------------------------------------------------------------------- #
# SimulatedTrace
# --------------------------------------------------------------------------- #

@st.composite
def simulated_trace(
    draw: st.DrawFn,
    min_spans: int = 1,
    max_spans: int = 10,
) -> SimulatedTrace:
    """Strategy for generating ``SimulatedTrace`` instances.

    Produces a trace with a single root (agent) span and *n-1* child spans
    of mixed types.
    """
    trace_id = draw(_hex_string(16))
    n_spans = draw(st.integers(min_value=max(min_spans, 1), max_value=max_spans))

    # First span is always the root agent span
    root = draw(simulated_span(span_type=SpanType.AGENT, parent_span_id=None))
    spans = [root]

    # Remaining spans are children of the root
    for _ in range(n_spans - 1):
        child = draw(simulated_span(parent_span_id=root.span_id))
        # Ensure child timestamps are within root span bounds
        child.start_time_unix_nano = max(
            child.start_time_unix_nano,
            root.start_time_unix_nano + 1_000_000,
        )
        child.end_time_unix_nano = max(
            child.end_time_unix_nano,
            child.start_time_unix_nano + 1_000_000,
        )
        # Extend root if children go beyond it
        if child.end_time_unix_nano >= root.end_time_unix_nano:
            root.end_time_unix_nano = child.end_time_unix_nano + 1_000_000
        spans.append(child)

    source_format = draw(st.one_of(
        st.none(),
        st.sampled_from([sf.value for sf in SourceFormat]),
    ))
    scenario = draw(st.one_of(
        st.none(),
        st.sampled_from([sn.value for sn in ScenarioName]),
    ))
    seed = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=2**31)))

    return SimulatedTrace(
        trace_id=trace_id,
        spans=spans,
        source_format=source_format,
        scenario=scenario,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# SimulatorConfig
# --------------------------------------------------------------------------- #

@st.composite
def simulator_config(draw: st.DrawFn) -> SimulatorConfig:
    """Strategy for generating ``SimulatorConfig`` instances."""
    source = draw(st.sampled_from(list(SourceFormat)))
    output = draw(st.sampled_from(list(OutputFormat)))
    scenario = draw(st.sampled_from(list(ScenarioName)))
    seed = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=2**31)))
    count = draw(st.integers(min_value=1, max_value=50))
    include_content = draw(st.booleans())

    errors = draw(error_config())
    streaming = draw(_streaming_config())
    conversation = draw(_conversation_config())

    return SimulatorConfig(
        source_format=source,
        output_format=output,
        scenario=scenario,
        seed=seed,
        count=count,
        include_content=include_content,
        errors=errors,
        streaming=streaming,
        conversation=conversation,
        content=ContentConfig(tier=ContentTier.TEMPLATE),
    )


# --------------------------------------------------------------------------- #
# ErrorConfig
# --------------------------------------------------------------------------- #

@st.composite
def error_config(draw: st.DrawFn) -> ErrorConfig:
    """Strategy for generating ``ErrorConfig`` instances."""
    enabled = draw(st.booleans())
    return ErrorConfig(
        enabled=enabled,
        rate_limit_probability=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        timeout_probability=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        auth_failure_probability=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        content_filter_probability=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        server_error_probability=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    )


# --------------------------------------------------------------------------- #
# Internal: StreamingConfig, ConversationConfig
# --------------------------------------------------------------------------- #

@st.composite
def _streaming_config(draw: st.DrawFn) -> StreamingConfig:
    """Strategy for generating ``StreamingConfig``."""
    enabled = draw(st.booleans())
    ttft_min = draw(st.floats(min_value=0.0, max_value=500.0, allow_nan=False))
    ttft_max = draw(st.floats(min_value=ttft_min, max_value=1000.0, allow_nan=False))
    tpot_min = draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False))
    tpot_max = draw(st.floats(min_value=tpot_min, max_value=100.0, allow_nan=False))
    chunks_min = draw(st.integers(min_value=1, max_value=50))
    chunks_max = draw(st.integers(min_value=chunks_min, max_value=100))
    return StreamingConfig(
        enabled=enabled,
        ttft_ms_min=ttft_min,
        ttft_ms_max=ttft_max,
        tpot_ms_min=tpot_min,
        tpot_ms_max=tpot_max,
        chunks_min=chunks_min,
        chunks_max=chunks_max,
    )


@st.composite
def _conversation_config(draw: st.DrawFn) -> ConversationConfig:
    """Strategy for generating ``ConversationConfig``."""
    enabled = draw(st.booleans())
    turns_min = draw(st.integers(min_value=1, max_value=10))
    turns_max = draw(st.integers(min_value=turns_min, max_value=20))
    return ConversationConfig(
        enabled=enabled,
        turns_min=turns_min,
        turns_max=turns_max,
    )
