"""Unit tests for the Google Agent Development Kit (ADK) framework adapter.

Mocked at the SDK shape level — no real ``google.adk`` runtime needed.
The adapter integrates via 6 native callbacks (before/after agent/model/tool).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.google_adk import (
    ADAPTER_CLASS,
    GoogleADKAdapter,
    instrument_agent,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _FakeAgent:
    """Minimal duck-typed Google ADK agent for tests."""

    def __init__(
        self,
        name: str = "adk-agent",
        tools: Any = None,
        model: Any = None,
        description: Any = None,
        instruction: Any = None,
        sub_agents: Any = None,
    ) -> None:
        self.name = name
        self.tools = tools
        self.model = model
        self.description = description
        self.instruction = instruction
        self.sub_agents = sub_agents
        self.before_agent_callback: Any = None
        self.after_agent_callback: Any = None
        self.before_model_callback: Any = None
        self.after_model_callback: Any = None
        self.before_tool_callback: Any = None
        self.after_tool_callback: Any = None


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is GoogleADKAdapter


def test_lifecycle() -> None:
    a = GoogleADKAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = GoogleADKAdapter()
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "google_adk"
    assert info.name == "GoogleADKAdapter"
    health = a.health_check()
    assert health.framework_name == "google_adk"


def test_instrument_agent_attaches_callbacks() -> None:
    adapter = GoogleADKAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner")
    adapter.instrument_agent(agent)
    # All six callbacks attached. Bound methods compare equal but not identical.
    assert agent.before_agent_callback == adapter._before_agent_callback
    assert agent.after_agent_callback == adapter._after_agent_callback
    assert agent.before_model_callback == adapter._before_model_callback
    assert agent.after_model_callback == adapter._after_model_callback
    assert agent.before_tool_callback == adapter._before_tool_callback
    assert agent.after_tool_callback == adapter._after_tool_callback


def test_before_after_agent_emits_input_output() -> None:
    stratix = _RecordingStratix()
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", model="gemini-2", tools=[SimpleNamespace(name="search")])
    callback_context = SimpleNamespace(agent=agent, user_content="hello world", agent_output="response", session=None)

    adapter._before_agent_callback(callback_context)
    adapter._after_agent_callback(callback_context)

    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    assert "agent.input" in types
    assert "agent.output" in types

    inp = next(e for e in stratix.events if e["event_type"] == "agent.input")
    assert inp["payload"]["agent_name"] == "planner"
    assert inp["payload"]["input"] == "hello world"

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["output"] == "response"
    assert out["payload"]["duration_ns"] >= 0


def test_after_model_emits_invoke_and_cost() -> None:
    stratix = _RecordingStratix()
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    callback_context = SimpleNamespace(model="gemini-2", agent=None)
    llm_request = SimpleNamespace()
    adapter._before_model_callback(callback_context, llm_request)

    llm_response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20),
    )
    adapter._after_model_callback(callback_context, llm_response)

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["model"] == "gemini-2"
    assert invoke["payload"]["provider"] == "google"
    assert invoke["payload"]["tokens_prompt"] == 10

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["tokens_total"] == 30


def test_after_tool_emits_tool_call() -> None:
    stratix = _RecordingStratix()
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    inp = {"x": 1}
    adapter._before_tool_callback(SimpleNamespace(), "calc", inp)
    adapter._after_tool_callback(SimpleNamespace(), "calc", inp, 42)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert evt["payload"]["tool_name"] == "calc"
    assert evt["payload"]["tool_output"] == 42
    assert evt["payload"]["latency_ms"] is not None


def test_on_handoff_emits_event_with_context_hash() -> None:
    stratix = _RecordingStratix()
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context="some context")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "a"
    assert evt["payload"]["to_agent"] == "b"
    assert evt["payload"]["context_hash"] is not None


def test_capture_config_gates_l3_model_metadata() -> None:
    """When l3_model_metadata is disabled, model.invoke does NOT fire (handoff still does)."""
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l3_model_metadata=False)
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    callback_context = SimpleNamespace(model="gemini-2", agent=None)
    adapter._before_model_callback(callback_context, SimpleNamespace())
    adapter._after_model_callback(
        callback_context,
        SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5)),
    )
    adapter.on_handoff(from_agent="a", to_agent="b", context="x")

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" not in types
    assert "agent.handoff" in types


def test_environment_config_emits_once_per_agent() -> None:
    stratix = _RecordingStratix()
    adapter = GoogleADKAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="a1", tools=[SimpleNamespace(name="search")])
    cb = SimpleNamespace(agent=agent, user_content="hi", agent_output=None, session=None)
    adapter._before_agent_callback(cb)
    # second call should not re-emit environment.config
    adapter._before_agent_callback(cb)

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1
    assert configs[0]["payload"]["agent_name"] == "a1"


def test_instrument_agent_helper() -> None:
    """Top-level convenience function returns a connected adapter."""
    agent = _FakeAgent(name="helper")
    adapter = instrument_agent(agent)
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY


def test_serialize_for_replay() -> None:
    adapter = GoogleADKAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "google_adk"
    assert rt.adapter_name == "GoogleADKAdapter"
    assert "capture_config" in rt.config
