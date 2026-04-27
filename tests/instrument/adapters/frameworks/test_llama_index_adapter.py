"""Unit tests for the LlamaIndex framework adapter.

Mocked at the SDK shape level — no real ``llama_index`` runtime needed.
Internal dispatch is by ``type(event).__name__``, so each test event uses
a minimally-shaped class with the right name.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.llama_index import (
    ADAPTER_CLASS,
    LlamaIndexAdapter,
    instrument_workflow,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


# Minimal classes shaped like LlamaIndex events. The adapter dispatches by
# ``type(event).__name__``, so the class name is what matters.
class LLMChatEndEvent:
    def __init__(self, model: str, response: Any = None) -> None:
        self.model = model
        self.response = response


class ToolCallEvent:
    def __init__(self, tool_name: str, tool_input: Any = None, tool_output: Any = None) -> None:
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.tool_output = tool_output


class RetrievalEndEvent:
    def __init__(self, nodes: List[Any]) -> None:
        self.nodes = nodes


class AgentRunStepStartEvent:
    def __init__(self, agent_id: str, step: int = 0, tools: Any = None) -> None:
        self.agent_id = agent_id
        self.step = step
        self.tools = tools


class AgentRunStepEndEvent:
    def __init__(self, agent_id: str, response: Any = None) -> None:
        self.agent_id = agent_id
        self.response = response


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is LlamaIndexAdapter


def test_lifecycle() -> None:
    a = LlamaIndexAdapter()
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = LlamaIndexAdapter()
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "llama_index"
    assert info.name == "LlamaIndexAdapter"
    health = a.health_check()
    assert health.framework_name == "llama_index"


def test_handle_llm_end_emits_model_invoke_and_cost() -> None:
    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    raw = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
    response = SimpleNamespace(raw=raw)
    adapter._handle_event(LLMChatEndEvent(model="gpt-5", response=response))

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "cost.record" in types

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["model"] == "gpt-5"
    assert invoke["payload"]["tokens_prompt"] == 10

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["tokens_total"] == 15


def test_handle_tool_call_event_emits_tool_call() -> None:
    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter._handle_event(ToolCallEvent(tool_name="calc", tool_input={"x": 1}, tool_output=2))

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert evt["payload"]["tool_name"] == "calc"
    assert evt["payload"]["tool_output"] == 2


def test_handle_retrieval_end_emits_retrieval_tool_call() -> None:
    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    nodes = [SimpleNamespace(score=0.9), SimpleNamespace(score=0.8)]
    adapter._handle_event(RetrievalEndEvent(nodes=nodes))

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert evt["payload"]["tool_type"] == "retrieval"
    assert evt["payload"]["result_count"] == 2


def test_agent_step_start_end_emits_input_output_and_config() -> None:
    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter._handle_event(AgentRunStepStartEvent(agent_id="myagent", step=1))
    adapter._handle_event(AgentRunStepEndEvent(agent_id="myagent", response="result"))

    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["agent_name"] == "myagent"
    assert out["payload"]["duration_ns"] >= 0


def test_on_handoff_emits_event_with_context_hash() -> None:
    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context="some context")

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["from_agent"] == "a"
    assert evt["payload"]["to_agent"] == "b"
    assert evt["payload"]["context_hash"] is not None


def test_capture_config_gates_l5a_tool_calls() -> None:
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l5a_tool_calls=False)
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    adapter._handle_event(ToolCallEvent(tool_name="calc", tool_input={"x": 1}, tool_output=2))
    adapter.on_handoff(from_agent="a", to_agent="b", context="x")

    types = [e["event_type"] for e in stratix.events]
    assert "tool.call" not in types
    assert "agent.handoff" in types


def test_unknown_event_type_does_nothing() -> None:
    """Events the adapter does not recognize should be silently ignored."""
    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter._handle_event(SimpleNamespace())  # type name 'SimpleNamespace' — unhandled

    assert stratix.events == []


def test_instrument_workflow_helper_returns_connected_adapter() -> None:
    """Convenience function returns a connected adapter even without llama_index installed."""
    adapter = instrument_workflow()
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY


def test_serialize_for_replay() -> None:
    adapter = LlamaIndexAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "llama_index"
    assert rt.adapter_name == "LlamaIndexAdapter"
    assert "capture_config" in rt.config


# --- Cross-pollination #2: error-aware emission ----------------------------


class _LLMChatEndEvent:
    """Shape-matched fake of a LlamaIndex LLMChatEndEvent carrying an exception."""

    def __init__(self, exc: BaseException, model: str = "gpt-5") -> None:
        self.exception = exc
        self.model = model


def test_llama_index_event_with_exception_emits_model_error() -> None:
    """LlamaIndex events with an ``exception`` attribute become model.error events."""
    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter._handle_event(_LLMChatEndEvent(RuntimeError("rate limit hit"), model="gpt-5"))

    error_events = [e for e in stratix.events if e["event_type"] == "model.error"]
    assert len(error_events) == 1
    payload = error_events[0]["payload"]
    assert payload["framework"] == "llama_index"
    assert payload["model"] == "gpt-5"
    assert payload["phase"] == "model.invoke"
    assert "rate limit" in payload["message"]


def test_llama_index_on_agent_end_with_error_emits_agent_error() -> None:
    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_agent_end(agent_name="planner", error=RuntimeError("workflow halted"))

    error_events = [e for e in stratix.events if e["event_type"] == "agent.error"]
    assert len(error_events) == 1
    assert error_events[0]["payload"]["agent_name"] == "planner"


def test_llama_index_on_tool_use_with_error_emits_tool_error() -> None:
    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_tool_use("search", error=ConnectionError("no network"))

    error_events = [e for e in stratix.events if e["event_type"] == "tool.error"]
    assert len(error_events) == 1
    payload = error_events[0]["payload"]
    assert payload["tool_name"] == "search"
    assert payload["exception_type"] == "ConnectionError"
