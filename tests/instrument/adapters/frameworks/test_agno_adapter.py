"""Unit tests for the Agno framework adapter.

Mocked at the SDK shape level — no real ``agno`` runtime needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.agno import (
    ADAPTER_CLASS,
    AgnoAdapter,
    instrument_agent,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _FakeAgent:
    """Minimal duck-typed Agno agent for tests."""

    def __init__(
        self,
        name: str = "test-agent",
        tools: Any = None,
        model: Any = None,
        description: Any = None,
        instructions: Any = None,
        team: Any = None,
        knowledge: Any = None,
        result: Any = None,
        raises: bool = False,
    ) -> None:
        self.name = name
        self.tools = tools
        self.model = model
        self.description = description
        self.instructions = instructions
        self.team = team
        self.knowledge = knowledge
        self._result = result
        self._raises = raises

    def run(self, message: str, **kwargs: Any) -> Any:
        if self._raises:
            raise RuntimeError("simulated failure")
        return self._result if self._result is not None else SimpleNamespace(content=f"out:{message}")


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is AgnoAdapter


def test_lifecycle() -> None:
    a = AgnoAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    assert a.is_connected is True
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED
    assert a.is_connected is False


def test_adapter_info_and_health() -> None:
    a = AgnoAdapter()
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "agno"
    assert info.name == "AgnoAdapter"
    assert info.version == AgnoAdapter.VERSION
    assert info.capabilities  # non-empty list
    health = a.health_check()
    assert health.framework_name == "agno"
    assert health.status == AdapterStatus.HEALTHY


def test_instrument_agent_wraps_run() -> None:
    adapter = AgnoAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner")
    adapter.instrument_agent(agent)
    # Wrapped: function name is now traced.
    assert agent.run.__name__ == "traced_run_sync"

    adapter.disconnect()
    # Restored: name is back to the original.
    assert agent.run.__name__ == "run"


def test_run_emits_input_and_output_events() -> None:
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", model="gpt-5")
    adapter.instrument_agent(agent)
    result = agent.run("hello")

    assert getattr(result, "content", None) == "out:hello"

    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["agent_name"] == "planner"
    assert out["payload"]["duration_ns"] >= 0
    assert out["payload"]["framework"] == "agno"


def test_run_failure_emits_output_with_error() -> None:
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="failing", raises=True)
    adapter.instrument_agent(agent)

    with pytest.raises(RuntimeError):
        agent.run("bad")

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert "error" in out["payload"]
    assert "simulated failure" in out["payload"]["error"]


def test_environment_config_emits_once_per_agent() -> None:
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="a1", tools=[SimpleNamespace(name="search")], model="gpt-5")
    adapter.instrument_agent(agent)
    adapter.instrument_agent(agent)  # idempotent

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1
    cfg = configs[0]["payload"]
    assert cfg["agent_name"] == "a1"
    assert cfg["tools"] == ["search"]


def test_on_tool_use_emits_event() -> None:
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2, latency_ms=12.3)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert evt["payload"]["tool_name"] == "calc"
    assert evt["payload"]["latency_ms"] == 12.3


def test_on_handoff_emits_event_with_context_hash() -> None:
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context="some context")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "a"
    assert evt["payload"]["to_agent"] == "b"
    assert evt["payload"]["context_hash"] is not None


def test_capture_config_gates_l5a_tool_calls() -> None:
    """When l5a_tool_calls is disabled, tool.call events do NOT fire."""
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l5a_tool_calls=False)
    adapter = AgnoAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2)
    # And handoffs (cross-cutting) should still fire.
    adapter.on_handoff(from_agent="a", to_agent="b", context="x")

    types = [e["event_type"] for e in stratix.events]
    assert "tool.call" not in types
    assert "agent.handoff" in types


def test_instrument_agent_helper() -> None:
    """Top-level convenience function returns a connected adapter."""
    agent = _FakeAgent(name="helper")
    adapter = instrument_agent(agent)
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY


def test_serialize_for_replay() -> None:
    adapter = AgnoAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    rt = adapter.serialize_for_replay()
    assert rt.framework == "agno"
    assert rt.adapter_name == "AgnoAdapter"
    assert "capture_config" in rt.config


# ---------------------------------------------------------------------------
# Streaming tests (cross-pollination #9: shared SSE parser integration)
# ---------------------------------------------------------------------------


class _StreamingAgent:
    """Agno agent stub that returns an iterator when ``stream=True``."""

    def __init__(self, name: str, chunks: List[Any]) -> None:
        self.name = name
        self.tools = None
        self.model = "gpt-5"
        self.description = None
        self.instructions = None
        self.team = None
        self.knowledge = None
        self._chunks = chunks

    def run(self, message: str, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return iter(self._chunks)
        return SimpleNamespace(content=f"out:{message}")


def test_stream_emits_one_event_per_object_chunk() -> None:
    """Object chunks (typical agno path) emit one model.stream.chunk each."""
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    chunks = [
        SimpleNamespace(content="Hello"),
        SimpleNamespace(content=" "),
        SimpleNamespace(content="world"),
    ]
    agent = _StreamingAgent(name="streamer", chunks=chunks)
    adapter.instrument_agent(agent)

    stream = agent.run("hi", stream=True)
    consumed = list(stream)

    assert len(consumed) == 3
    chunk_events = [e for e in stratix.events if e["event_type"] == "model.stream.chunk"]
    assert len(chunk_events) == 3
    assert chunk_events[0]["payload"]["agent_name"] == "streamer"
    # Final on_run_end fires after iterator exhaustion.
    assert any(e["event_type"] == "agent.output" for e in stratix.events)


def test_stream_emits_multiple_events_per_sse_chunk() -> None:
    """A single bytes chunk containing multiple SSE events emits multiple events."""
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    # One network chunk holds two complete SSE events.
    chunks = [b"data: first\n\ndata: second\n\n"]
    agent = _StreamingAgent(name="sse-streamer", chunks=chunks)
    adapter.instrument_agent(agent)

    list(agent.run("go", stream=True))

    chunk_events = [e for e in stratix.events if e["event_type"] == "model.stream.chunk"]
    # Two SSE events extracted from one network chunk -> two emitted events.
    assert len(chunk_events) == 2
    assert chunk_events[0]["payload"]["chunk"] == "first"
    assert chunk_events[1]["payload"]["chunk"] == "second"


def test_stream_handles_partial_sse_across_chunks() -> None:
    """A single SSE event split across two network chunks emits one event."""
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    chunks = [b"data: hel", b"lo\n\n"]
    agent = _StreamingAgent(name="partial", chunks=chunks)
    adapter.instrument_agent(agent)

    list(agent.run("go", stream=True))

    chunk_events = [e for e in stratix.events if e["event_type"] == "model.stream.chunk"]
    assert len(chunk_events) == 1
    assert chunk_events[0]["payload"]["chunk"] == "hello"


def test_stream_passthrough_does_not_break_iterator() -> None:
    """The wrapped stream still yields the original chunks to the caller."""
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    chunks = [SimpleNamespace(content="a"), SimpleNamespace(content="b")]
    agent = _StreamingAgent(name="pt", chunks=chunks)
    adapter.instrument_agent(agent)

    stream = agent.run("hi", stream=True)
    consumed = [getattr(c, "content", c) for c in stream]
    assert consumed == ["a", "b"]


def test_non_stream_path_unchanged() -> None:
    """When stream=True is NOT passed, no model.stream.chunk events fire."""
    stratix = _RecordingStratix()
    adapter = AgnoAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _StreamingAgent(name="nostream", chunks=[])
    adapter.instrument_agent(agent)
    agent.run("hi")  # no stream kwarg

    chunk_events = [e for e in stratix.events if e["event_type"] == "model.stream.chunk"]
    assert chunk_events == []
