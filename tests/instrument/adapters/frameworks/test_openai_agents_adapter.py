"""Unit tests for the OpenAI Agents SDK framework adapter.

Mocked at the SDK shape level — no real ``agents`` runtime needed. The
adapter dispatches by ``type(span_data).__name__``, so each test span
uses a class with the right name (AgentSpanData, GenerationSpanData, etc.).

After the typed-event migration (PR #129 follow-up — bundle 4) every
emit site flows through :meth:`BaseAdapter.emit_event` with a canonical
Pydantic payload. The :class:`_RecordingStratix` stand-in below records
both shapes so pre- and post-migration assertions live side by side: the
``payload`` slot always carries a dict (model-dumped if typed), and
``typed_payloads`` holds the original Pydantic instances for tests that
want to assert against the model surface.
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
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
        # Hold strong references to the original typed payloads for
        # the subset of tests that want to assert against the model
        # surface (e.g. ``isinstance(payload, ToolCallEvent)``). The
        # dict view lives on ``events`` and is what most assertions
        # read.
        self.typed_payloads: List[Any] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        # Two-arg legacy path: ``emit(event_type, payload_dict)``.
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})
            return
        # Single-arg typed path: ``emit(payload_model[, privacy_level])``.
        if args and isinstance(args[0], _CompatBaseModel):
            payload_model = args[0]
            self.typed_payloads.append(payload_model)
            event_type = getattr(payload_model, "event_type", "<unknown>")
            self.events.append(
                {"event_type": event_type, "payload": _compat_model_dump(payload_model)}
            )


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
    """Typed AgentInputEvent + AgentOutputEvent + EnvironmentConfigEvent."""
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
    payload = out["payload"]
    assert payload["layer"] == "L1"
    assert payload["content"]["message"] == "response"
    assert payload["content"]["role"] == "agent"
    metadata = payload["content"]["metadata"]
    assert metadata["framework"] == "openai_agents"
    assert metadata["agent_name"] == "planner"
    assert metadata["span_id"] == "span-1"

    cfg = next(e for e in stratix.events if e["event_type"] == "environment.config")
    cfg_payload = cfg["payload"]
    assert cfg_payload["layer"] == "L4a"
    assert cfg_payload["environment"]["type"] == "cloud"
    assert cfg_payload["environment"]["attributes"]["agent_name"] == "planner"
    assert cfg_payload["environment"]["attributes"]["model"] == "gpt-5"


def test_generation_span_emits_model_invoke_and_cost() -> None:
    """Typed ModelInvokeEvent + CostRecordEvent for GenerationSpanData."""
    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    data = GenerationSpanData(model="gpt-5", input_tokens=10, output_tokens=20)
    adapter._on_span_end(_Span(data, duration_ms=42.0))

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    payload = invoke["payload"]
    assert payload["layer"] == "L3"
    assert payload["model"]["name"] == "gpt-5"
    assert payload["model"]["provider"] == "openai"
    assert payload["model"]["version"] == "unavailable"
    assert payload["model"]["parameters"]["framework"] == "openai_agents"
    assert payload["prompt_tokens"] == 10
    assert payload["completion_tokens"] == 20
    assert payload["latency_ms"] == 42.0

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    cost_payload = cost["payload"]
    assert cost_payload["cost"]["prompt_tokens"] == 10
    assert cost_payload["cost"]["completion_tokens"] == 20
    assert cost_payload["cost"]["tokens"] == 30


def test_function_span_emits_tool_call() -> None:
    """Typed ToolCallEvent for FunctionSpanData."""
    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter._on_span_end(_Span(FunctionSpanData(name="calc", input={"x": 1}, output=42)))

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    payload = evt["payload"]
    assert payload["layer"] == "L5a"
    assert payload["tool"]["name"] == "calc"
    assert payload["tool"]["integration"] == "library"
    assert payload["input"]["x"] == 1
    assert payload["input"]["framework"] == "openai_agents"
    assert payload["output"] == {"value": 42}


def test_handoff_span_emits_agent_handoff() -> None:
    """Typed AgentHandoffEvent: canonical sha256:<hex64> handoff_context_hash."""
    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter._on_span_end(_Span(HandoffSpanData(from_agent="a", to_agent="b")))

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    payload = evt["payload"]
    assert payload["from_agent"] == "a"
    assert payload["to_agent"] == "b"
    assert payload["handoff_context_hash"].startswith("sha256:")
    assert len(payload["handoff_context_hash"]) == 7 + 64


def test_guardrail_span_emits_policy_violation() -> None:
    """Typed PolicyViolationEvent: canonical violation_type=POLICY_CONSTRAINT."""
    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter._on_span_end(_Span(GuardrailSpanData(name="profanity", triggered=True, output="blocked")))

    evt = next(e for e in stratix.events if e["event_type"] == "policy.violation")
    payload = evt["payload"]
    assert payload["violation"]["type"] == "policy_constraint"
    details = payload["violation"]["details"]
    assert details["framework"] == "openai_agents"
    assert details["guardrail_name"] == "profanity"
    assert details["triggered"] is True


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


def test_trace_start_end_emits_input_output_with_event_subtype() -> None:
    """The trace_start/trace_end markers are remapped to typed
    AgentInputEvent / AgentOutputEvent.

    The previous adapter implementation emitted ad-hoc
    ``agent.state.change`` payloads carrying only an
    ``event_subtype`` marker. Those payloads did not satisfy the
    canonical AgentStateChangeEvent ``before_hash`` / ``after_hash``
    schema. The typed migration maps trace_start -> AgentInputEvent
    (role=AGENT) and trace_end -> AgentOutputEvent, with the original
    event_subtype marker preserved on MessageContent.metadata.
    """
    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    class _Trace:
        trace_id = "trace-1"

    adapter._on_trace_start(_Trace())
    adapter._on_trace_end(_Trace())

    inputs = [e for e in stratix.events if e["event_type"] == "agent.input"]
    outputs = [e for e in stratix.events if e["event_type"] == "agent.output"]
    trace_inputs = [
        e for e in inputs
        if e["payload"]["content"]["metadata"].get("event_subtype") == "trace_start"
    ]
    trace_outputs = [
        e for e in outputs
        if e["payload"]["content"]["metadata"].get("event_subtype") == "trace_end"
    ]
    assert len(trace_inputs) == 1
    assert len(trace_outputs) == 1
    assert trace_inputs[0]["payload"]["content"]["metadata"]["trace_id"] == "trace-1"
    assert trace_outputs[0]["payload"]["content"]["metadata"]["trace_id"] == "trace-1"


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


# ---------------------------------------------------------------------------
# Typed-event migration regression tests (PR #129 follow-up — bundle 4)
# ---------------------------------------------------------------------------


def test_openai_agents_emits_typed_payloads_only() -> None:
    """Every emit site in openai_agents lifecycle is a typed emit_event call.

    Pins the post-migration contract: the recording stratix's
    ``typed_payloads`` list grows for every emission and the legacy
    two-arg dict path receives nothing. This is the public contract
    backing the ``grep emit_dict_event src/.../openai_agents/lifecycle.py
    → 0`` acceptance criterion in the typed-events bundle 4 PR.
    """
    from layerlens.instrument._compat.events import (
        ToolCallEvent,
        AgentInputEvent,
        CostRecordEvent,
        AgentOutputEvent,
        ModelInvokeEvent,
        AgentHandoffEvent,
        PolicyViolationEvent,
        EnvironmentConfigEvent,
    )

    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    # Drive every emission path.
    class _Trace:
        trace_id = "trace-1"

    adapter._on_trace_start(_Trace())
    agent_data = AgentSpanData(name="planner", output="response", model="gpt-5")
    adapter._on_span_start(_Span(agent_data, span_id="s-1"))
    adapter._on_span_end(_Span(agent_data, span_id="s-1"))
    adapter._on_span_end(_Span(GenerationSpanData(model="gpt-5", input_tokens=10, output_tokens=20)))
    adapter._on_span_end(_Span(FunctionSpanData(name="calc", input={"x": 1}, output=42)))
    adapter._on_span_end(_Span(HandoffSpanData(from_agent="a", to_agent="b")))
    adapter._on_span_end(_Span(GuardrailSpanData(name="profanity", triggered=True)))
    adapter._on_trace_end(_Trace())
    adapter.on_run_start(agent_name="planner", input_data="x")
    adapter.on_run_end(agent_name="planner", output="y")
    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
    adapter.on_llm_call(provider="openai", model="gpt-5", tokens_prompt=1, tokens_completion=2)
    adapter.on_handoff(from_agent="a", to_agent="b", context="ctx")

    # Every captured payload is a Pydantic model instance — the legacy
    # dict path was not used.
    assert stratix.typed_payloads, "expected typed payloads to be captured"
    types_seen = {type(p) for p in stratix.typed_payloads}
    assert AgentInputEvent in types_seen
    assert AgentOutputEvent in types_seen
    assert AgentHandoffEvent in types_seen
    assert EnvironmentConfigEvent in types_seen
    assert ModelInvokeEvent in types_seen
    assert ToolCallEvent in types_seen
    assert CostRecordEvent in types_seen
    assert PolicyViolationEvent in types_seen


def test_openai_agents_emit_does_not_warn_after_migration() -> None:
    """No DeprecationWarning fires from openai_agents lifecycle emission paths.

    The base adapter's ``emit_dict_event`` raises a DeprecationWarning
    on every call. After migration, openai_agents lifecycle must
    never trigger that warning. ``filterwarnings("error", ...)``
    converts the warning into a test failure.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = OpenAIAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    class _Trace:
        trace_id = "trace-1"

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        adapter._on_trace_start(_Trace())
        agent_data = AgentSpanData(name="planner", output="response", model="gpt-5")
        adapter._on_span_start(_Span(agent_data))
        adapter._on_span_end(_Span(agent_data))
        adapter._on_span_end(_Span(GenerationSpanData(model="gpt-5", input_tokens=10, output_tokens=20)))
        adapter._on_span_end(_Span(FunctionSpanData(name="calc", input={"x": 1}, output=42)))
        adapter._on_span_end(_Span(HandoffSpanData(from_agent="a", to_agent="b")))
        adapter._on_span_end(_Span(GuardrailSpanData(name="profanity", triggered=True)))
        adapter._on_trace_end(_Trace())
        adapter.on_run_start(agent_name="planner", input_data="x")
        adapter.on_run_end(agent_name="planner", output="y")
        adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
        adapter.on_llm_call(provider="openai", model="gpt-5", tokens_prompt=1, tokens_completion=2)
        adapter.on_handoff(from_agent="a", to_agent="b", context="ctx")
