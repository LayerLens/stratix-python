"""Unit tests for the SmolAgents framework adapter.

Mocked at the SDK shape level — no real ``smolagents`` runtime needed.
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.smolagents import (
    ADAPTER_CLASS,
    SmolAgentsAdapter,
    instrument_agent,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _FakeAgent:
    """Minimal duck-typed SmolAgents agent for tests."""

    def __init__(
        self,
        name: str = "test-agent",
        tools: Any = None,
        managed_agents: Any = None,
        model: Any = None,
        system_prompt: Any = None,
    ) -> None:
        self.name = name
        self.tools = tools
        self.managed_agents = managed_agents
        self.model = model
        self.system_prompt = system_prompt
        self._raised = False

    def run(self, task: str, **kwargs: Any) -> Any:
        if self._raised:
            raise RuntimeError("simulated failure")
        return f"result for {task}"


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is SmolAgentsAdapter


def test_lifecycle() -> None:
    a = SmolAgentsAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_declares_replay_capability() -> None:
    """SmolAgents adapter implements ``serialize_for_replay`` so REPLAY
    must appear in the declared capabilities.
    """
    from layerlens.instrument.adapters._base.adapter import AdapterCapability

    info = SmolAgentsAdapter().get_adapter_info()
    assert AdapterCapability.REPLAY in info.capabilities


def test_instrument_agent_wraps_run() -> None:
    adapter = SmolAgentsAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner")
    adapter.instrument_agent(agent)
    # Wrapped: the bound method's underlying function is now ``traced_run``.
    assert agent.run.__name__ == "traced_run"

    adapter.disconnect()
    # Restored: name is back to the original.
    assert agent.run.__name__ == "run"


def test_run_emits_input_and_output_events() -> None:
    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner")
    adapter.instrument_agent(agent)
    result = agent.run("compute 2+2")

    assert result == "result for compute 2+2"

    types = [e["event_type"] for e in stratix.events]
    # First event is environment.config from initial agent registration.
    assert "environment.config" in types
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["agent_name"] == "planner"
    assert out["payload"]["duration_ns"] >= 0


def test_run_failure_emits_output_with_error() -> None:
    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="failing")
    agent._raised = True
    adapter.instrument_agent(agent)

    import pytest

    with pytest.raises(RuntimeError):
        agent.run("bad task")

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert "error" in out["payload"]
    assert "simulated failure" in out["payload"]["error"]


def test_managed_agents_recursively_instrumented() -> None:
    adapter = SmolAgentsAdapter()
    adapter.connect()

    sub = _FakeAgent(name="sub")
    parent = _FakeAgent(name="parent", managed_agents={"sub": sub})

    adapter.instrument_agent(parent)
    # Both wrapped.
    assert parent.run.__name__ == "traced_run"
    assert sub.run.__name__ == "traced_run"


def test_environment_config_emits_once_per_agent() -> None:
    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(
        name="a1",
        tools=["search", "calc"],
        model="some-model",
        system_prompt="you are helpful",
    )
    adapter.instrument_agent(agent)
    # Re-instrument should not re-emit config.
    adapter.instrument_agent(agent)

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1
    cfg = configs[0]["payload"]
    assert cfg["agent_name"] == "a1"
    assert cfg["tools"] == ["search", "calc"]


def test_on_tool_use_emits_event() -> None:
    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2, latency_ms=12.3)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert evt["payload"]["tool_name"] == "calc"
    assert evt["payload"]["latency_ms"] == 12.3


def test_on_handoff_emits_event_with_context_hash() -> None:
    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_handoff(from_agent="a", to_agent="b", context="some context")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "a"
    assert evt["payload"]["to_agent"] == "b"
    assert evt["payload"]["context_hash"] is not None
    # Capture content on => preview included.
    assert evt["payload"]["context_preview"] == "some context"


def test_handoff_redacts_context_when_capture_content_disabled() -> None:
    stratix = _RecordingStratix()
    adapter = SmolAgentsAdapter(
        stratix=stratix,
        capture_config=CaptureConfig(capture_content=False),
    )
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context="secret")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["context_preview"] is None
    # Hash still present (it's not content).
    assert evt["payload"]["context_hash"] is not None


def test_instrument_agent_helper() -> None:
    """Top-level convenience function returns a connected adapter."""
    agent = _FakeAgent(name="helper")
    adapter = instrument_agent(agent)
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY


def test_serialize_for_replay() -> None:
    adapter = SmolAgentsAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    rt = adapter.serialize_for_replay()
    assert rt.framework == "smolagents"
    assert "capture_config" in rt.config
