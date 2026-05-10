"""Unit tests for the Microsoft Agent Framework adapter.

Mocked at the SDK shape level — no real ``semantic_kernel.agents`` runtime
needed. The adapter wraps ``invoke()`` async generators on chat instances;
tests exercise ``_process_message`` and the lifecycle hooks directly.

After the typed-event migration (PR #129 follow-up — bundle 3) every
emit site flows through :meth:`BaseAdapter.emit_event` with a canonical
Pydantic payload. The :class:`_RecordingStratix` stand-in below records
both shapes: the ``payload`` slot always carries a dict (model-dumped
if typed), and ``typed_payloads`` holds the original Pydantic instances
for tests that want to assert against the model surface.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.ms_agent_framework import (
    ADAPTER_CLASS,
    MSAgentAdapter,
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


# Item types — name-driven dispatch in adapter
class FunctionCallContent:
    def __init__(self, name: str, arguments: Any) -> None:
        self.name = name
        self.arguments = arguments


class FunctionResultContent:
    def __init__(self, name: str, result: Any) -> None:
        self.name = name
        self.result = result


class _FakeChat:
    def __init__(self, name: str = "ms-chat", agents: Any = None, agent: Any = None) -> None:
        self.name = name
        self.agents = agents
        self.agent = agent

    async def invoke(self, *args: Any, **kwargs: Any) -> Any:
        # async generator stub
        if False:
            yield None  # type: ignore[unreachable]

    async def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        if False:
            yield None  # type: ignore[unreachable]


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is MSAgentAdapter


def test_lifecycle() -> None:
    a = MSAgentAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = MSAgentAdapter(org_id="test-org")
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "ms_agent_framework"
    assert info.name == "MSAgentAdapter"
    health = a.health_check()
    assert health.framework_name == "ms_agent_framework"


def test_instrument_chat_wraps_invoke_and_emits_config() -> None:
    """Typed EnvironmentConfigEvent: chat_name lives at
    payload.environment.attributes.
    """
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    chat = _FakeChat(name="planner-chat")
    adapter.instrument_chat(chat)

    # Wrapped: name is now traced.
    assert chat.invoke.__name__ == "traced_invoke"
    assert chat.invoke_stream.__name__ == "traced_invoke_stream"

    cfg = next(e for e in stratix.events if e["event_type"] == "environment.config")
    attributes = cfg["payload"]["environment"]["attributes"]
    assert attributes["chat_name"] == "planner-chat"
    assert cfg["payload"]["environment"]["type"] == "simulated"

    adapter.disconnect()
    # Restored.
    assert chat.invoke.__name__ == "invoke"


def test_process_message_emits_handoff_on_agent_change() -> None:
    """Typed AgentHandoffEvent: handoff_context_hash is sha256:<hex64>."""
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    msg = SimpleNamespace(agent_name="bob", items=[], metadata={})
    adapter._process_message(_FakeChat(), msg, current_agent="alice")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    payload = evt["payload"]
    assert payload["from_agent"] == "alice"
    assert payload["to_agent"] == "bob"
    assert payload["handoff_context_hash"].startswith("sha256:")


def test_process_message_emits_tool_calls_from_function_items() -> None:
    """Typed ToolCallEvent: tool name lives at payload.tool.name.
    Scalar result is wrapped in {"value": ...}.
    """
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    msg = SimpleNamespace(
        items=[
            FunctionCallContent(name="calc", arguments={"x": 1}),
            FunctionResultContent(name="calc", result=42),
        ],
        metadata={},
    )
    adapter._process_message(_FakeChat(), msg, current_agent="alice")

    tool_calls = [e for e in stratix.events if e["event_type"] == "tool.call"]
    assert len(tool_calls) == 2
    assert tool_calls[0]["payload"]["tool"]["name"] == "calc"
    assert tool_calls[0]["payload"]["tool"]["integration"] == "library"
    # Scalar result is wrapped in {"value": ...}.
    assert tool_calls[1]["payload"]["output"] == {"value": 42}


def test_process_message_emits_model_and_cost_from_metadata() -> None:
    """Typed ModelInvokeEvent + CostRecordEvent.
    Model name lives at payload.model.name; tokens at payload.cost.*.
    """
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    msg = SimpleNamespace(
        items=[],
        metadata={"model": "gpt-5", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )
    adapter._process_message(_FakeChat(), msg, current_agent="alice")

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    inv_payload = invoke["payload"]
    assert inv_payload["model"]["name"] == "gpt-5"
    assert inv_payload["model"]["provider"] == "openai"

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["cost"]["prompt_tokens"] == 10
    assert cost["payload"]["cost"]["completion_tokens"] == 5


def test_on_run_start_end_emits_input_output() -> None:
    """Typed migration: on_run_end → AgentOutputEvent with
    run_status=run_complete on metadata.

    The previous adapter also emitted an ad-hoc
    ``agent.state.change`` payload with ``event_subtype`` —
    that did not satisfy the canonical
    :class:`AgentStateChangeEvent` ``before_hash`` / ``after_hash``
    contract. The completion marker is now carried as
    ``run_status`` on :class:`MessageContent.metadata` of the
    :class:`AgentOutputEvent`.
    """
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_run_start(agent_name="planner", input_data="hi")
    adapter.on_run_end(agent_name="planner", output="bye")

    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" in types
    assert "agent.output" in types
    # agent.state.change is NO LONGER emitted — the completion marker
    # is preserved on the agent.output metadata's run_status field.
    assert "agent.state.change" not in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    metadata = out["payload"]["content"]["metadata"]
    assert metadata["run_status"] == "run_complete"


def test_on_run_end_failure_carries_run_failed() -> None:
    """When an error is supplied, run_status carries ``run_failed``."""
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_run_start(agent_name="planner", input_data="hi")
    adapter.on_run_end(agent_name="planner", output=None, error=RuntimeError("boom"))

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    metadata = out["payload"]["content"]["metadata"]
    assert metadata["run_status"] == "run_failed"
    assert "boom" in metadata["error"]


def test_capture_config_gates_l5a_tool_calls() -> None:
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l5a_tool_calls=False)
    adapter = MSAgentAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    msg = SimpleNamespace(
        items=[FunctionCallContent(name="calc", arguments={"x": 1})],
        metadata={},
    )
    adapter._process_message(_FakeChat(), msg, current_agent="alice")
    adapter.on_handoff(from_agent="a", to_agent="b", context="x")

    types = [e["event_type"] for e in stratix.events]
    assert "tool.call" not in types
    # handoff is cross-cutting / always enabled.
    assert "agent.handoff" in types


def test_on_handoff_emits_event_with_context_hash() -> None:
    """Typed AgentHandoffEvent: handoff_context_hash is sha256:<hex64>."""
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context="some context")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    payload = evt["payload"]
    assert payload["from_agent"] == "a"
    assert payload["to_agent"] == "b"
    assert payload["handoff_context_hash"].startswith("sha256:")
    assert len(payload["handoff_context_hash"]) == 7 + 64


def test_handoff_emits_canonical_hash_for_empty_context() -> None:
    """Empty context still produces a well-formed sha256 hash."""
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context=None)

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["handoff_context_hash"].startswith("sha256:")


def test_instrument_agent_helper() -> None:
    chat = _FakeChat(name="helper")
    adapter = instrument_agent(chat, org_id="test-org")
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY


def test_serialize_for_replay() -> None:
    adapter = MSAgentAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "ms_agent_framework"
    assert rt.adapter_name == "MSAgentAdapter"
    assert "capture_config" in rt.config


# ---------------------------------------------------------------------------
# Typed-event migration regression tests (PR #129 follow-up — bundle 3)
# ---------------------------------------------------------------------------


def test_ms_agent_framework_lifecycle_emits_typed_payloads_only() -> None:
    """Every emit site in ms_agent_framework lifecycle.py is a typed
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
        AgentHandoffEvent,
        EnvironmentConfigEvent,
    )

    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    chat = _FakeChat(name="planner-chat")
    adapter.instrument_chat(chat)

    # Process a message with all the typed emission paths.
    msg = SimpleNamespace(
        agent_name="bob",
        items=[
            FunctionCallContent(name="calc", arguments={"x": 1}),
            FunctionResultContent(name="calc", result=42),
        ],
        metadata={"model": "gpt-5", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )
    adapter._process_message(chat, msg, current_agent="alice")

    # Direct lifecycle hooks
    adapter.on_run_start(agent_name="planner", input_data="hi")
    adapter.on_run_end(agent_name="planner", output="bye")
    adapter.on_tool_use("ext_tool", tool_input={"a": 1}, tool_output="ok")
    adapter.on_llm_call(provider="openai", model="gpt-5", tokens_prompt=5)
    adapter.on_handoff(from_agent="planner", to_agent="executor", context="ctx")

    # Every captured payload is a Pydantic model instance — the legacy
    # dict path was not used.
    assert stratix.typed_payloads, "expected typed payloads to be captured"
    types_seen = {type(p) for p in stratix.typed_payloads}
    assert AgentInputEvent in types_seen
    assert AgentOutputEvent in types_seen
    assert AgentHandoffEvent in types_seen
    assert CostRecordEvent in types_seen
    assert EnvironmentConfigEvent in types_seen
    assert ModelInvokeEvent in types_seen
    assert ToolCallEvent in types_seen


def test_ms_agent_framework_emit_does_not_warn_after_migration() -> None:
    """No DeprecationWarning fires from ms_agent_framework lifecycle paths.

    The base adapter's ``emit_dict_event`` raises a DeprecationWarning
    on every call. After migration, ms_agent_framework lifecycle must
    never trigger that warning.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    chat = _FakeChat(name="planner-chat")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        adapter.instrument_chat(chat)
        msg = SimpleNamespace(
            agent_name="bob",
            items=[
                FunctionCallContent(name="calc", arguments={"x": 1}),
                FunctionResultContent(name="calc", result=42),
            ],
            metadata={"model": "gpt-5", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        )
        adapter._process_message(chat, msg, current_agent="alice")
        adapter.on_run_start(agent_name="p", input_data="i")
        adapter.on_run_end(agent_name="p", output="o")
        adapter.on_tool_use("t", tool_input={"a": 1}, tool_output="ok")
        adapter.on_llm_call(provider="openai", model="gpt-5", tokens_prompt=5)
        adapter.on_handoff(from_agent="a", to_agent="b", context="ctx")
