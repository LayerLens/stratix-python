"""Unit tests for the PydanticAI framework adapter.

Mocked at the SDK shape level — no real ``pydantic_ai`` runtime needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.pydantic_ai import (
    ADAPTER_CLASS,
    PydanticAIAdapter,
    instrument_agent,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _FakeAgent:
    """Minimal duck-typed PydanticAI agent for tests."""

    def __init__(
        self,
        name: str = "pa-agent",
        tools: Any = None,
        model: Any = None,
        system_prompt: Any = None,
        result_type: Any = None,
        result: Any = None,
        raises: bool = False,
    ) -> None:
        self.name = name
        self.tools = tools
        self.model = model
        self.system_prompt = system_prompt
        self.result_type = result_type
        self._result = result
        self._raises = raises

    def run_sync(self, user_prompt: str, **kwargs: Any) -> Any:
        if self._raises:
            raise RuntimeError("simulated failure")
        return self._result if self._result is not None else SimpleNamespace(data=f"out:{user_prompt}")


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is PydanticAIAdapter


def test_lifecycle() -> None:
    a = PydanticAIAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = PydanticAIAdapter()
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "pydantic_ai"
    assert info.name == "PydanticAIAdapter"
    health = a.health_check()
    assert health.framework_name == "pydantic_ai"


def test_instrument_agent_wraps_run_sync() -> None:
    adapter = PydanticAIAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner")
    adapter.instrument_agent(agent)
    assert agent.run_sync.__name__ == "traced_run_sync"

    adapter.disconnect()
    # Restored to original.
    assert agent.run_sync.__name__ == "run_sync"


def test_run_emits_input_and_output_events() -> None:
    stratix = _RecordingStratix()
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", model="gpt-5")
    adapter.instrument_agent(agent)
    result = agent.run_sync("hello")
    assert getattr(result, "data", None) == "out:hello"

    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["agent_name"] == "planner"
    assert out["payload"]["duration_ns"] >= 0


def test_run_failure_emits_output_with_error() -> None:
    stratix = _RecordingStratix()
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    agent = _FakeAgent(name="failing", raises=True)
    adapter.instrument_agent(agent)

    with pytest.raises(RuntimeError):
        agent.run_sync("bad")

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert "error" in out["payload"]
    assert "simulated failure" in out["payload"]["error"]


def test_run_extracts_usage_and_messages() -> None:
    """When the result has usage and a tool-return message, cost.record + tool.call fire."""
    stratix = _RecordingStratix()
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    usage = SimpleNamespace(request_tokens=10, response_tokens=5, total_tokens=15)
    response_msg = SimpleNamespace(kind="response")
    tool_msg = SimpleNamespace(kind="tool-return", tool_name="calc", content=42)
    result = SimpleNamespace(
        data="ok",
        usage=usage,
        all_messages=[response_msg, tool_msg],
        model_name="gpt-5",
    )
    agent = _FakeAgent(name="planner", result=result)
    adapter.instrument_agent(agent)
    agent.run_sync("hi")

    types = [e["event_type"] for e in stratix.events]
    assert "cost.record" in types
    assert "model.invoke" in types
    assert "tool.call" in types

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["tokens_total"] == 15
    tool = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert tool["payload"]["tool_name"] == "calc"


def test_on_handoff_emits_event_with_context_hash() -> None:
    stratix = _RecordingStratix()
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context="some context")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "a"
    assert evt["payload"]["context_hash"] is not None


def test_capture_config_gates_l1_agent_io() -> None:
    """When l1_agent_io is disabled, agent.input/output do NOT fire (state.change still does)."""
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l1_agent_io=False)
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    adapter.on_run_start(agent_name="a", input_data="x")
    adapter.on_run_end(agent_name="a", output="y")

    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" not in types
    assert "agent.output" not in types
    # state.change is cross-cutting / always enabled.
    assert "agent.state.change" in types


def test_environment_config_emits_once_per_agent() -> None:
    stratix = _RecordingStratix()
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="a1", tools=[SimpleNamespace(name="search")], model="gpt-5")
    adapter.instrument_agent(agent)
    adapter.instrument_agent(agent)  # idempotent

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1
    assert configs[0]["payload"]["agent_name"] == "a1"
    assert configs[0]["payload"]["tools"] == ["search"]


def test_instrument_agent_helper() -> None:
    agent = _FakeAgent(name="helper")
    adapter = instrument_agent(agent)
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY


def test_serialize_for_replay() -> None:
    adapter = PydanticAIAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "pydantic_ai"
    assert rt.adapter_name == "PydanticAIAdapter"
    assert "capture_config" in rt.config


# --- Cross-pollination #2: error-aware emission ----------------------------


def test_pydantic_ai_run_sync_failure_emits_error_event() -> None:
    """run_sync raise → policy.violation event with framework attribution."""
    stratix = _RecordingStratix()
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    agent = _FakeAgent(name="failing", raises=True)
    adapter.instrument_agent(agent)

    with pytest.raises(RuntimeError):
        agent.run_sync("bad")

    error_events = [e for e in stratix.events if e["event_type"] == "policy.violation"]
    assert len(error_events) == 1
    payload = error_events[0]["payload"]
    assert payload["framework"] == "pydantic_ai"
    assert payload["agent_name"] == "failing"
    assert payload["phase"] == "agent.run"


def test_pydantic_ai_on_tool_use_with_error_emits_tool_error() -> None:
    stratix = _RecordingStratix()
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_tool_use("calc", error=ValueError("nope"))

    error_events = [e for e in stratix.events if e["event_type"] == "tool.error"]
    assert len(error_events) == 1
    assert error_events[0]["payload"]["tool_name"] == "calc"
