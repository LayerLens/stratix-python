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


ADAPTER_CLS = LlamaIndexAdapter



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
