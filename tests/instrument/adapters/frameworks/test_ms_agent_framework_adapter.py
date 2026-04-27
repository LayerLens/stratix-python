"""Unit tests for the Microsoft Agent Framework adapter.

Mocked at the SDK shape level — no real ``semantic_kernel.agents`` runtime
needed. The adapter wraps ``invoke()`` async generators on chat instances;
tests exercise ``_process_message`` and the lifecycle hooks directly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.ms_agent_framework import (
    ADAPTER_CLASS,
    MSAgentAdapter,
    instrument_agent,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


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
    a = MSAgentAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = MSAgentAdapter()
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "ms_agent_framework"
    assert info.name == "MSAgentAdapter"
    health = a.health_check()
    assert health.framework_name == "ms_agent_framework"


def test_instrument_chat_wraps_invoke_and_emits_config() -> None:
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    chat = _FakeChat(name="planner-chat")
    adapter.instrument_chat(chat)

    # Wrapped: name is now traced.
    assert chat.invoke.__name__ == "traced_invoke"
    assert chat.invoke_stream.__name__ == "traced_invoke_stream"

    cfg = next(e for e in stratix.events if e["event_type"] == "environment.config")
    assert cfg["payload"]["chat_name"] == "planner-chat"

    adapter.disconnect()
    # Restored.
    assert chat.invoke.__name__ == "invoke"


def test_process_message_emits_handoff_on_agent_change() -> None:
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    msg = SimpleNamespace(agent_name="bob", items=[], metadata={})
    adapter._process_message(_FakeChat(), msg, current_agent="alice")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "alice"
    assert evt["payload"]["to_agent"] == "bob"


def test_process_message_emits_tool_calls_from_function_items() -> None:
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
    assert tool_calls[0]["payload"]["tool_name"] == "calc"
    assert tool_calls[1]["payload"]["tool_output"] == 42


def test_process_message_emits_model_and_cost_from_metadata() -> None:
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    msg = SimpleNamespace(
        items=[],
        metadata={"model": "gpt-5", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )
    adapter._process_message(_FakeChat(), msg, current_agent="alice")

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["model"] == "gpt-5"
    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["tokens_prompt"] == 10


def test_on_run_start_end_emits_input_output_and_state() -> None:
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_run_start(agent_name="planner", input_data="hi")
    adapter.on_run_end(agent_name="planner", output="bye")

    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" in types
    assert "agent.output" in types
    assert "agent.state.change" in types


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
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context="some context")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "a"
    assert evt["payload"]["context_hash"] is not None


def test_instrument_agent_helper() -> None:
    chat = _FakeChat(name="helper")
    adapter = instrument_agent(chat)
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


ADAPTER_CLS = MSAgentAdapter



# ---------------------------------------------------------------------------
# Field-specific truncation policy (cross-pollination audit §2.4)
# ---------------------------------------------------------------------------


def test_truncation_policy_is_default_after_construction() -> None:
    """The adapter wires :data:`DEFAULT_POLICY` in its constructor.

    Without this, large prompts / tool I/O / state values would flow
    through to ``Stratix.emit`` unbounded — see audit §2.4.
    """
    from layerlens.instrument.adapters._base import DEFAULT_POLICY

    adapter = ADAPTER_CLS()
    assert adapter._truncation_policy is DEFAULT_POLICY


def test_truncation_clips_oversize_prompt_via_emit_dict_event() -> None:
    """A 10 000-char prompt is truncated to the policy cap on emit."""
    stratix = _RecordingStratix()
    adapter = ADAPTER_CLS(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.emit_dict_event("model.invoke", {"prompt": "p" * 10000})

    assert stratix.events
    payload = stratix.events[-1]["payload"]
    assert isinstance(payload["prompt"], str)
    assert payload["prompt"].startswith("p" * 4096)
    assert "more chars truncated" in payload["prompt"]
    audit = payload.get("_truncated_fields", [])
    assert any("prompt:chars-10000->4096" in entry for entry in audit), audit


def test_truncation_drops_screenshot_with_hash_reference() -> None:
    """``screenshot`` field is replaced with a SHA-256 reference string."""
    stratix = _RecordingStratix()
    adapter = ADAPTER_CLS(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.emit_dict_event(
        "tool.call",
        {"tool_name": "snap", "screenshot": b"FAKE_PNG_BYTES" * 1000},
    )

    payload = stratix.events[-1]["payload"]
    assert isinstance(payload["screenshot"], str)
    assert payload["screenshot"].startswith("<dropped:screenshot:sha256:")


def test_truncation_short_payload_no_audit_attached() -> None:
    """Payloads under cap do NOT receive a ``_truncated_fields`` key."""
    stratix = _RecordingStratix()
    adapter = ADAPTER_CLS(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.emit_dict_event("model.invoke", {"prompt": "short"})

    payload = stratix.events[-1]["payload"]
    assert "_truncated_fields" not in payload
