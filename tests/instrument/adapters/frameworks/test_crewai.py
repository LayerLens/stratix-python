"""Unit tests for the CrewAI framework adapter.

Mocked at the SDK shape level — no real ``crewai`` runtime needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.crewai import (
    ADAPTER_CLASS,
    CrewAIAdapter,
    LayerLensCrewCallback,
    instrument_crew,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _FakeCrew:
    def __init__(self, agents: Any = None, process: Any = None) -> None:
        self.agents = agents or []
        self.process = process
        self.step_callback: Any = None
        self.task_callback: Any = None


def _make_agent(role: str = "researcher", tools: Any = None, llm: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        role=role,
        goal="goal",
        backstory="back",
        verbose=False,
        allow_delegation=False,
        max_iter=5,
        memory=False,
        tools=tools,
        llm=llm,
    )


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is CrewAIAdapter


def test_lifecycle() -> None:
    a = CrewAIAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = CrewAIAdapter()
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "crewai"
    assert info.name == "CrewAIAdapter"
    health = a.health_check()
    assert health.framework_name == "crewai"


def test_instrument_crew_attaches_callback_and_emits_config() -> None:
    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    crew = _FakeCrew(
        agents=[_make_agent(role="researcher"), _make_agent(role="writer")],
        process="sequential",
    )
    instrumented = adapter.instrument_crew(crew)

    # Callbacks attached.
    assert instrumented.step_callback is not None
    assert instrumented.task_callback is not None
    assert isinstance(instrumented._stratix_callback, LayerLensCrewCallback)

    # Two environment.config events — one per agent role.
    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 2
    roles = {c["payload"]["agent_role"] for c in configs}
    assert roles == {"researcher", "writer"}


def test_environment_config_idempotent_per_role() -> None:
    """Re-instrumenting a crew with same agents should not re-emit configs."""
    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    crew = _FakeCrew(agents=[_make_agent(role="researcher")])
    adapter.instrument_crew(crew)
    adapter.instrument_crew(crew)

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1


def test_on_crew_start_end_emits_input_output() -> None:
    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_crew_start(crew_input="research topic")
    adapter.on_crew_end(crew_output="report")

    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["output"] == "report"
    assert out["payload"]["duration_ns"] >= 0


def test_on_task_start_end_emits_code_and_state_change() -> None:
    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_task_start("research", agent_role="researcher", task_order=1)

    # Build a task_output with token_usage to also verify cost.record fires.
    task_output = SimpleNamespace(
        token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    adapter.on_task_end(task_output=task_output, agent_role="researcher", task_order=1)

    types = [e["event_type"] for e in stratix.events]
    assert "agent.code" in types
    assert "agent.state.change" in types
    assert "cost.record" in types

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["tokens_total"] == 15


def test_on_tool_use_emits_event() -> None:
    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2, latency_ms=12.3)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert evt["payload"]["tool_name"] == "calc"
    assert evt["payload"]["latency_ms"] == 12.3


def test_on_delegation_emits_handoff() -> None:
    stratix = _RecordingStratix()
    adapter = CrewAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_delegation(from_agent="researcher", to_agent="writer", context="findings")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "researcher"
    assert evt["payload"]["to_agent"] == "writer"


def test_capture_config_gates_l5a_tool_calls() -> None:
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l5a_tool_calls=False)
    adapter = CrewAIAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
    adapter.on_delegation(from_agent="a", to_agent="b", context="x")

    types = [e["event_type"] for e in stratix.events]
    assert "tool.call" not in types
    # handoff is cross-cutting / always enabled.
    assert "agent.handoff" in types


def test_instrument_crew_helper() -> None:
    """Top-level convenience function returns the instrumented crew."""
    crew = _FakeCrew(agents=[_make_agent(role="r1")])
    result = instrument_crew(crew)
    # The helper returns the crew itself (with callbacks attached).
    assert result is crew
    assert result._stratix_callback is not None


def test_serialize_for_replay() -> None:
    adapter = CrewAIAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "crewai"
    assert rt.adapter_name == "CrewAIAdapter"
    assert "capture_config" in rt.config
