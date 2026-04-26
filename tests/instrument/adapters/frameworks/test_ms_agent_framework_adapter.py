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


# ---------------------------------------------------------------------------
# Streaming tests (cross-pollination #9: shared SSE parser integration)
# ---------------------------------------------------------------------------


import asyncio  # noqa: E402  (positioned with the streaming tests it serves)


class _StreamingChat:
    """Chat stub whose ``invoke_stream`` yields a configurable list of chunks."""

    def __init__(self, name: str, chunks: List[Any]) -> None:
        self.name = name
        self.agents = None
        self.agent = None
        self._chunks = chunks

    async def invoke(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        if False:
            yield None  # type: ignore[unreachable]

    async def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        for chunk in self._chunks:
            yield chunk


def _drain_async(gen: Any) -> List[Any]:
    """Drain an async generator into a list synchronously."""

    async def runner() -> List[Any]:
        out: List[Any] = []
        async for item in gen:
            out.append(item)
        return out

    return asyncio.run(runner())


def test_invoke_stream_emits_one_event_per_object_chunk() -> None:
    """Object chunks (typical Microsoft Agent Framework path) emit one event each."""
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    chunks = [
        SimpleNamespace(content="Hi"),
        SimpleNamespace(content=" "),
        SimpleNamespace(content="there"),
    ]
    chat = _StreamingChat(name="ms-stream", chunks=chunks)
    adapter.instrument_chat(chat)

    consumed = _drain_async(chat.invoke_stream())
    assert len(consumed) == 3

    chunk_events = [e for e in stratix.events if e["event_type"] == "model.stream.chunk"]
    assert len(chunk_events) == 3
    assert chunk_events[0]["payload"]["framework"] == "ms_agent_framework"
    assert chunk_events[0]["payload"]["agent_name"] == "ms-stream"
    # on_run_end fires after iterator exhaustion.
    assert any(e["event_type"] == "agent.output" for e in stratix.events)


def test_invoke_stream_emits_multiple_events_per_sse_chunk() -> None:
    """One bytes chunk holding multiple SSE events emits multiple per-event events."""
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    chunks = [b"data: alpha\n\ndata: beta\n\n"]
    chat = _StreamingChat(name="ms-sse", chunks=chunks)
    adapter.instrument_chat(chat)

    _drain_async(chat.invoke_stream())

    chunk_events = [e for e in stratix.events if e["event_type"] == "model.stream.chunk"]
    assert len(chunk_events) == 2
    assert [e["payload"]["chunk"] for e in chunk_events] == ["alpha", "beta"]


def test_invoke_stream_handles_partial_sse_across_chunks() -> None:
    """One SSE event split across two network chunks emits exactly one event."""
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    chunks = [b"data: par", b"tial\n\n"]
    chat = _StreamingChat(name="ms-partial", chunks=chunks)
    adapter.instrument_chat(chat)

    _drain_async(chat.invoke_stream())

    chunk_events = [e for e in stratix.events if e["event_type"] == "model.stream.chunk"]
    assert len(chunk_events) == 1
    assert chunk_events[0]["payload"]["chunk"] == "partial"


def test_invoke_stream_passthrough_yields_original_chunks() -> None:
    """The wrapped invoke_stream still yields the original chunks to the caller."""
    stratix = _RecordingStratix()
    adapter = MSAgentAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    chunks = [SimpleNamespace(content="x"), SimpleNamespace(content="y")]
    chat = _StreamingChat(name="ms-pt", chunks=chunks)
    adapter.instrument_chat(chat)

    consumed = [getattr(c, "content", c) for c in _drain_async(chat.invoke_stream())]
    assert consumed == ["x", "y"]
