"""Unit tests for the LlamaIndex framework adapter.

Mocked at the SDK shape level — no real ``llama_index`` runtime needed.
Internal dispatch is by ``type(event).__name__``, so each test event uses
a minimally-shaped class with the right name.

After the typed-event migration (PR #129 follow-up — bundle 3) every
emit site flows through :meth:`BaseAdapter.emit_event` with a canonical
Pydantic payload. The :class:`_RecordingStratix` stand-in below records
both shapes so pre- and post-migration assertions live side by side: the
``payload`` slot always carries a dict (model-dumped if typed), and
``typed_payloads`` holds the original Pydantic instances for tests that
want to assert against the model surface.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.llama_index import (
    ADAPTER_CLASS,
    LlamaIndexAdapter,
    instrument_workflow,
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
    a = LlamaIndexAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = LlamaIndexAdapter(org_id="test-org")
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "llama_index"
    assert info.name == "LlamaIndexAdapter"
    health = a.health_check()
    assert health.framework_name == "llama_index"


def test_handle_llm_end_emits_model_invoke_and_cost() -> None:
    """Typed ModelInvokeEvent + CostRecordEvent.
    Model name lives at payload.model.name; tokens at payload.cost.*.
    """
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
    inv_payload = invoke["payload"]
    assert inv_payload["layer"] == "L3"
    assert inv_payload["model"]["name"] == "gpt-5"
    assert inv_payload["model"]["provider"] == "openai"
    assert inv_payload["model"]["version"] == "unavailable"
    assert inv_payload["prompt_tokens"] == 10
    assert inv_payload["completion_tokens"] == 5

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    cost_payload = cost["payload"]
    assert cost_payload["cost"]["prompt_tokens"] == 10
    assert cost_payload["cost"]["completion_tokens"] == 5
    assert cost_payload["cost"]["tokens"] == 15


def test_handle_tool_call_event_emits_tool_call() -> None:
    """Typed ToolCallEvent: tool name lives at payload.tool.name.
    Scalar output is wrapped in {"value": ...}.
    """
    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter._handle_event(ToolCallEvent(tool_name="calc", tool_input={"x": 1}, tool_output=2))

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    payload = evt["payload"]
    assert payload["layer"] == "L5a"
    assert payload["tool"]["name"] == "calc"
    assert payload["tool"]["integration"] == "library"
    assert payload["output"] == {"value": 2}


def test_handle_retrieval_end_emits_retrieval_tool_call() -> None:
    """Retrieval is mapped onto ToolCallEvent with tool.name='retrieval'.
    Adapter-specific tool_type/result_count live on payload.input.
    """
    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    nodes = [SimpleNamespace(score=0.9), SimpleNamespace(score=0.8)]
    adapter._handle_event(RetrievalEndEvent(nodes=nodes))

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    payload = evt["payload"]
    assert payload["tool"]["name"] == "retrieval"
    assert payload["input"]["tool_type"] == "retrieval"
    assert payload["input"]["result_count"] == 2


def test_agent_step_start_end_emits_input_output_and_config() -> None:
    """Typed AgentInputEvent + AgentOutputEvent + EnvironmentConfigEvent.
    LlamaIndex-specific provenance lives on MessageContent.metadata
    and EnvironmentInfo.attributes.
    """
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
    out_payload = out["payload"]
    assert out_payload["content"]["message"] == "result"
    metadata = out_payload["content"]["metadata"]
    assert metadata["agent_name"] == "myagent"
    assert metadata["framework"] == "llama_index"
    assert metadata["duration_ns"] >= 0


def test_on_handoff_emits_event_with_context_hash() -> None:
    """Typed AgentHandoffEvent: handoff_context_hash is sha256:<hex64>."""
    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
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
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context=None)

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["handoff_context_hash"].startswith("sha256:")


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
    adapter = instrument_workflow(org_id="test-org")
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


# ---------------------------------------------------------------------------
# Typed-event migration regression tests (PR #129 follow-up — bundle 3)
# ---------------------------------------------------------------------------


def test_llama_index_lifecycle_emits_typed_payloads_only() -> None:
    """Every emit site in llama_index lifecycle.py is a typed emit_event call.

    Pins the post-migration contract: the recording stratix's
    ``typed_payloads`` list grows for every emission and the legacy
    two-arg dict path receives nothing.
    """
    from layerlens.instrument._compat.events import (
        AgentInputEvent,
        CostRecordEvent,
        AgentOutputEvent,
        ModelInvokeEvent,
        AgentHandoffEvent,
        EnvironmentConfigEvent,
    )
    from layerlens.instrument._compat.events import (
        ToolCallEvent as CanonicalToolCallEvent,
    )

    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    # LLM end
    raw = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
    response = SimpleNamespace(raw=raw)
    adapter._handle_event(LLMChatEndEvent(model="gpt-5", response=response))

    # Tool call
    adapter._handle_event(ToolCallEvent(tool_name="calc", tool_input={"x": 1}, tool_output=2))

    # Retrieval
    adapter._handle_event(RetrievalEndEvent(nodes=[SimpleNamespace(score=0.9)]))

    # Agent step
    adapter._handle_event(AgentRunStepStartEvent(agent_id="myagent", step=1))
    adapter._handle_event(AgentRunStepEndEvent(agent_id="myagent", response="ok"))

    # Direct lifecycle hooks
    adapter.on_agent_start(agent_name="other", input_data="task")
    adapter.on_agent_end(agent_name="other", output="done")
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
    assert CanonicalToolCallEvent in types_seen


def test_llama_index_emit_does_not_warn_after_migration() -> None:
    """No DeprecationWarning fires from llama_index lifecycle paths.

    The base adapter's ``emit_dict_event`` raises a DeprecationWarning
    on every call. After migration, llama_index lifecycle must never
    trigger that warning.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = LlamaIndexAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    raw = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
    response = SimpleNamespace(raw=raw)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        adapter._handle_event(LLMChatEndEvent(model="gpt-5", response=response))
        adapter._handle_event(
            ToolCallEvent(tool_name="calc", tool_input={"x": 1}, tool_output=2)
        )
        adapter._handle_event(RetrievalEndEvent(nodes=[SimpleNamespace(score=0.9)]))
        adapter._handle_event(AgentRunStepStartEvent(agent_id="myagent", step=1))
        adapter._handle_event(AgentRunStepEndEvent(agent_id="myagent", response="ok"))
        adapter.on_agent_start(agent_name="o", input_data="i")
        adapter.on_agent_end(agent_name="o", output="o")
        adapter.on_tool_use("t", tool_input={"a": 1}, tool_output="ok")
        adapter.on_llm_call(provider="openai", model="gpt-5", tokens_prompt=5)
        adapter.on_handoff(from_agent="a", to_agent="b", context="ctx")
