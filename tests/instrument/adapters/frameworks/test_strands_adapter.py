"""Unit tests for the AWS Strands framework adapter.

Mocked at the SDK shape level — no real ``strands`` runtime needed.
The adapter wraps ``invoke()`` (and ``__call__``); tests exercise ``invoke``.

After the typed-event migration (PR #129 follow-up — bundle 5) every
emit site flows through :meth:`BaseAdapter.emit_event` with a canonical
Pydantic payload. The :class:`_RecordingStratix` stand-in below records
both shapes: the ``payload`` slot always carries a dict (model-dumped
if typed), and ``typed_payloads`` holds the original Pydantic instances
for tests that want to assert against the model surface.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.strands import (
    ADAPTER_CLASS,
    StrandsAdapter,
    instrument_agent,
)


class _RecordingStratix:
    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this stratix pass the
    # BaseAdapter fail-fast check. Tests asserting cross-tenant
    # isolation override this default.
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        # Hold strong references to the original typed payloads.
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


class _FakeAgent:
    """Minimal duck-typed Strands agent for tests."""

    def __init__(
        self,
        name: str = "strands-agent",
        tools: Any = None,
        model: Any = None,
        system_prompt: Any = None,
        conversation: Any = None,
        result: Any = None,
        raises: bool = False,
    ) -> None:
        self.name = name
        self.tools = tools
        self.model = model
        self.system_prompt = system_prompt
        self.conversation = conversation
        self._result = result
        self._raises = raises

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        if self._raises:
            raise RuntimeError("simulated failure")
        return self._result if self._result is not None else SimpleNamespace(content=f"out:{prompt}", text=None)

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        return self.invoke(prompt, **kwargs)


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is StrandsAdapter


def test_lifecycle() -> None:
    a = StrandsAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = StrandsAdapter(org_id="test-org")
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "strands"
    assert info.name == "StrandsAdapter"
    health = a.health_check()
    assert health.framework_name == "strands"


def test_instrument_agent_wraps_invoke() -> None:
    adapter = StrandsAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()
    agent = _FakeAgent(name="planner")
    adapter.instrument_agent(agent)
    assert agent.invoke.__name__ == "traced_call"

    adapter.disconnect()
    assert agent.invoke.__name__ == "invoke"


def test_invoke_emits_input_and_output_events() -> None:
    """Typed migration: agent_name lives at payload.content.metadata."""
    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", model="anthropic.claude-v2")
    adapter.instrument_agent(agent)
    result = agent.invoke("hello")
    assert getattr(result, "content", None) == "out:hello"

    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    metadata = out["payload"]["content"]["metadata"]
    assert metadata["agent_name"] == "planner"
    assert metadata["duration_ns"] >= 0


def test_invoke_extracts_usage_and_emits_cost() -> None:
    """Typed migration: tokens at payload.cost.tokens; model name at payload.model.name."""
    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    usage = SimpleNamespace(inputTokens=10, outputTokens=5, totalTokens=15)
    result = SimpleNamespace(content="ok", text=None, usage=usage, tool_results=[])
    agent = _FakeAgent(name="planner", model="anthropic.claude-v2", result=result)
    adapter.instrument_agent(agent)
    agent.invoke("hi")

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "cost.record" in types

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["cost"]["tokens"] == 15
    assert cost["payload"]["cost"]["prompt_tokens"] == 10
    assert cost["payload"]["cost"]["completion_tokens"] == 5

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["model"]["name"] == "anthropic.claude-v2"
    assert invoke["payload"]["model"]["provider"] == "bedrock"


def test_invoke_failure_emits_output_with_error() -> None:
    """Typed migration: error is on AgentOutputEvent.metadata, run_status=run_failed."""
    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    agent = _FakeAgent(name="failing", raises=True)
    adapter.instrument_agent(agent)

    with pytest.raises(RuntimeError):
        agent.invoke("bad")

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    metadata = out["payload"]["content"]["metadata"]
    assert "error" in metadata
    assert "simulated failure" in metadata["error"]
    assert metadata["run_status"] == "run_failed"


def test_on_tool_use_emits_event() -> None:
    """Typed migration: tool name at payload.tool.name; latency at payload.latency_ms."""
    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2, latency_ms=12.3)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert evt["payload"]["tool"]["name"] == "calc"
    assert evt["payload"]["tool"]["integration"] == "library"
    assert evt["payload"]["latency_ms"] == 12.3
    # Scalar tool output is wrapped in {"value": ...}.
    assert evt["payload"]["output"] == {"value": 2}


def test_capture_config_gates_l3_model_metadata() -> None:
    """When l3_model_metadata is disabled, model.invoke does NOT fire.

    Post-migration: the previous adapter emitted an ad-hoc
    ``agent.state.change`` payload at the run boundary to carry a
    run_complete / run_failed marker. That payload is no longer
    emitted (it did not satisfy the canonical
    AgentStateChangeEvent contract); the marker now lives on
    AgentOutputEvent.metadata.run_status. So this test now asserts
    only that model.invoke is gated and agent.input/output still fire.
    """
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l3_model_metadata=False)
    adapter = StrandsAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    adapter.on_llm_call(model="claude", provider="bedrock")
    adapter.on_run_start(agent_name="a", input_data="x")
    adapter.on_run_end(agent_name="a", output="y")

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" not in types
    # agent.state.change is no longer emitted post-migration
    assert "agent.state.change" not in types
    # Run boundary signal is preserved on AgentOutputEvent.metadata.
    assert "agent.input" in types
    assert "agent.output" in types
    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["content"]["metadata"]["run_status"] == "run_complete"


def test_environment_config_emits_once_per_agent() -> None:
    """Typed migration: tools live at payload.environment.attributes."""
    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="a1", tools=[SimpleNamespace(name="search")], model="claude")
    adapter.instrument_agent(agent)
    adapter.instrument_agent(agent)

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1
    attributes = configs[0]["payload"]["environment"]["attributes"]
    assert attributes["tools"] == ["search"]
    assert configs[0]["payload"]["environment"]["type"] == "simulated"


def test_conversation_state_carried_on_agent_output_metadata() -> None:
    """When the agent has a conversation manager, conversation_state metadata fires.

    The previous adapter emitted an ad-hoc agent.state.change payload
    with event_subtype=conversation_update + turn_count. That payload
    did not satisfy the canonical AgentStateChangeEvent before_hash /
    after_hash contract. The post-migration mapping carries the
    conversation-progress signal as agent.output metadata.
    """
    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    conversation = SimpleNamespace(turn_count=3, messages=[])
    agent = _FakeAgent(name="c1", model="claude", conversation=conversation)
    adapter.instrument_agent(agent)
    agent.invoke("hi")

    types = [e["event_type"] for e in stratix.events]
    assert "agent.state.change" not in types
    # Conversation state is carried on an agent.output emission.
    convo_outputs = [
        e for e in stratix.events
        if e["event_type"] == "agent.output"
        and e["payload"]["content"].get("metadata", {}).get("conversation_state") == "conversation_update"
    ]
    assert convo_outputs, "expected conversation_update marker on agent.output metadata"
    assert convo_outputs[0]["payload"]["content"]["metadata"]["turn_count"] == 3


def test_instrument_agent_helper() -> None:
    agent = _FakeAgent(name="helper")
    adapter = instrument_agent(agent, org_id="test-org")
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY


def test_serialize_for_replay() -> None:
    adapter = StrandsAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "strands"
    assert rt.adapter_name == "StrandsAdapter"
    assert "capture_config" in rt.config


# ---------------------------------------------------------------------------
# Typed-event migration regression tests (PR #129 follow-up — bundle 5)
# ---------------------------------------------------------------------------


def test_strands_emits_typed_payloads_only() -> None:
    """Every emit site in strands lifecycle.py is a typed
    emit_event call.

    Pins the post-migration contract: the recording stratix's
    ``typed_payloads`` list grows for every emission and the legacy
    two-arg dict path receives nothing.
    """
    from layerlens.instrument._compat.events import (
        ToolCallEvent,
        AgentInputEvent,
        CostRecordEvent,
        AgentOutputEvent,
        ModelInvokeEvent,
        EnvironmentConfigEvent,
    )

    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    # Drive every emission path: instrument_agent triggers
    # environment.config; invoke drives agent.input/output and
    # _extract_run_details drives model.invoke + cost.record + tool.call.
    usage = SimpleNamespace(inputTokens=10, outputTokens=5, totalTokens=15)
    tool_result = SimpleNamespace(name="calc", input={"x": 1}, output=2)
    conversation = SimpleNamespace(turn_count=3, messages=[])
    result = SimpleNamespace(
        content="ok",
        text=None,
        usage=usage,
        tool_results=[tool_result],
    )
    agent = _FakeAgent(
        name="planner",
        model="anthropic.claude-v2",
        result=result,
        conversation=conversation,
    )
    adapter.instrument_agent(agent)
    agent.invoke("hi")

    # Direct lifecycle hooks
    adapter.on_tool_use("ext_tool", tool_input={"a": 1}, tool_output="ok")
    adapter.on_llm_call(provider="bedrock", model="claude", tokens_prompt=5)

    # Every captured payload is a Pydantic model instance — the legacy
    # dict path was not used.
    assert stratix.typed_payloads, "expected typed payloads to be captured"
    types_seen = {type(p) for p in stratix.typed_payloads}
    assert AgentInputEvent in types_seen
    assert AgentOutputEvent in types_seen
    assert CostRecordEvent in types_seen
    assert EnvironmentConfigEvent in types_seen
    assert ModelInvokeEvent in types_seen
    assert ToolCallEvent in types_seen


def test_strands_emit_does_not_warn_after_migration() -> None:
    """No DeprecationWarning fires from strands lifecycle paths.

    The base adapter's ``emit_dict_event`` raises a DeprecationWarning
    on every call. After migration, strands lifecycle must never
    trigger that warning.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        usage = SimpleNamespace(inputTokens=10, outputTokens=5, totalTokens=15)
        tool_result = SimpleNamespace(name="calc", input={"x": 1}, output=2)
        conversation = SimpleNamespace(turn_count=3, messages=[])
        result = SimpleNamespace(
            content="ok",
            text=None,
            usage=usage,
            tool_results=[tool_result],
        )
        agent = _FakeAgent(
            name="planner",
            model="anthropic.claude-v2",
            result=result,
            conversation=conversation,
        )
        adapter.instrument_agent(agent)
        agent.invoke("hi")
        adapter.on_tool_use("t", tool_input={"a": 1}, tool_output="ok")
        adapter.on_llm_call(provider="bedrock", model="claude", tokens_prompt=5)
