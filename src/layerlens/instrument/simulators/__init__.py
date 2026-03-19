"""STRATIX Multi-Source OTel Trace Simulator SDK.

A commercial SDK-quality simulator supporting all 12 ingestion sources,
5 scenarios, 3 output formats, 3-tier content generation, error injection,
streaming, multi-turn conversations, and comprehensive testing utilities.

Quick start:
    from layerlens.instrument.simulators import TraceSimulator, SimulatorConfig

    config = SimulatorConfig.minimal()
    simulator = TraceSimulator(config)
    traces, result = simulator.generate_and_format()
"""

from ._version import VERSION
from .base import BaseSimulator, SimulatorResult, TraceSimulator
from .clock import DeterministicClock
from .config import (
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
from .identifiers import IDGenerator
from .run_store import RunRecord, RunStore
from .span_model import (
    SimulatedSpan,
    SimulatedTrace,
    SpanKind,
    SpanStatus,
    SpanType,
    TokenUsage,
)
from .trace_builder import TraceBuilder

__version__ = VERSION

__all__ = [
    # Core
    "TraceSimulator",
    "BaseSimulator",
    "SimulatorResult",
    "SimulatorConfig",
    "TraceBuilder",
    # Models
    "SimulatedTrace",
    "SimulatedSpan",
    "TokenUsage",
    "SpanType",
    "SpanKind",
    "SpanStatus",
    # Config
    "SourceFormat",
    "OutputFormat",
    "ScenarioName",
    "ContentTier",
    "ContentConfig",
    "ConversationConfig",
    "StreamingConfig",
    "ErrorConfig",
    # Infrastructure
    "DeterministicClock",
    "IDGenerator",
    "RunStore",
    "RunRecord",
    # Version
    "VERSION",
]
