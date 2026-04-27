"""Unit tests for the OpenAI Agents SDK framework adapter.

Mocked at the SDK shape level — no real ``agents`` runtime needed. The
adapter dispatches by ``type(span_data).__name__``, so each test span
uses a class with the right name (AgentSpanData, GenerationSpanData, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.openai_agents import (
    ADAPTER_CLASS,
    OpenAIAgentsAdapter,
    instrument_runner,
)


class _RecordingStratix:
    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this stratix pass the
    # BaseAdapter fail-fast check. Tests asserting cross-tenant
    # isolation override this default.
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


# Span data classes — names must match what the adapter dispatches on.
class AgentSpanData:
    def __init__(self, name: str, output: Any = None, tools: Any = None, model: Any = None) -> None:
        self.name = name
        self.output = output
        self.tools = tools
        self.model = model


class GenerationSpanData:
    def __init__(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FunctionSpanData:
    def __init__(self, name: str, input: Any = None, output: Any = None) -> None:
        self.name = name
        self.input = input
        self.output = output


class HandoffSpanData:
    def __init__(self, from_agent: str, to_agent: str) -> None:
        self.from_agent = from_agent
        self.to_agent = to_agent


class GuardrailSpanData:
    def __init__(self, name: str, triggered: bool, output: Any = None) -> None:
        self.name = name
        self.triggered = triggered
        self.output = output


class _Span:
    def __init__(self, span_data: Any, span_id: str = "span-1", duration_ms: float = 100.0) -> None:
        self.span_data = span_data
        self.span_id = span_id
        self.duration_ms = duration_ms


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is OpenAIAgentsAdapter


def test_lifecycle() -> None:
    a = OpenAIAgentsAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = OpenAIAgentsAdapter(org_id="test-org")
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "openai_agents"
    assert info.name == "OpenAIAgentsAdapter"
    health = a.health_check()
    assert health.framework_name == "openai_agents"


def test_agent_span_emits_input_output_and_config() -> None:
    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    data = AgentSpanData(name="planner", output="response", model="gpt-5")
    span = _Span(data, span_id="span-1")

    adapter._on_span_start(span)
    adapter._on_span_end(span)

    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["agent_name"] == "planner"
    assert out["payload"]["output"] == "response"


def test_generation_span_emits_model_invoke_and_cost() -> None:
    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    data = GenerationSpanData(model="gpt-5", input_tokens=10, output_tokens=20)
    adapter._on_span_end(_Span(data, duration_ms=42.0))

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["model"] == "gpt-5"
    assert invoke["payload"]["tokens_prompt"] == 10
    assert invoke["payload"]["latency_ms"] == 42.0

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["tokens_total"] == 30


def test_function_span_emits_tool_call() -> None:
    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter._on_span_end(_Span(FunctionSpanData(name="calc", input={"x": 1}, output=42)))

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert evt["payload"]["tool_name"] == "calc"
    assert evt["payload"]["tool_output"] == 42


def test_handoff_span_emits_agent_handoff() -> None:
    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter._on_span_end(_Span(HandoffSpanData(from_agent="a", to_agent="b")))

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "a"
    assert evt["payload"]["to_agent"] == "b"


def test_guardrail_span_emits_policy_violation() -> None:
    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter._on_span_end(_Span(GuardrailSpanData(name="profanity", triggered=True, output="blocked")))

    evt = next(e for e in stratix.events if e["event_type"] == "policy.violation")
    assert evt["payload"]["guardrail_name"] == "profanity"
    assert evt["payload"]["triggered"] is True


def test_capture_config_gates_l3_model_metadata() -> None:
    """When l3_model_metadata is disabled, model.invoke does NOT fire (handoff still does)."""
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l3_model_metadata=False)
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    adapter._on_span_end(_Span(GenerationSpanData(model="gpt-5", input_tokens=10, output_tokens=5)))
    adapter._on_span_end(_Span(HandoffSpanData(from_agent="a", to_agent="b")))

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" not in types
    # handoff is cross-cutting / always enabled.
    assert "agent.handoff" in types


def test_trace_start_end_emits_state_change() -> None:
    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    class _Trace:
        trace_id = "trace-1"

    adapter._on_trace_start(_Trace())
    adapter._on_trace_end(_Trace())

    states = [e for e in stratix.events if e["event_type"] == "agent.state.change"]
    subtypes = {s["payload"]["event_subtype"] for s in states}
    assert "trace_start" in subtypes
    assert "trace_end" in subtypes


def test_instrument_runner_helper() -> None:
    """Convenience function returns a connected adapter even without agents installed."""
    adapter = instrument_runner(org_id="test-org")
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY


def test_serialize_for_replay() -> None:
    adapter = OpenAIAgentsAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "openai_agents"
    assert rt.adapter_name == "OpenAIAgentsAdapter"
    assert "capture_config" in rt.config
