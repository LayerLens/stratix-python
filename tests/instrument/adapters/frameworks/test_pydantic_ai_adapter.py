"""Unit tests for the PydanticAI framework adapter.

Mocked at the SDK shape level — no real ``pydantic_ai`` runtime needed.

After the typed-event migration (PR #129 follow-up — bundle 5) every
emit site flows through :meth:`BaseAdapter.emit_event` with a canonical
Pydantic payload. The :class:`_RecordingStratix` stand-in below records
both shapes: the ``payload`` slot always carries a dict (model-dumped
if typed), and ``typed_payloads`` holds the original Pydantic instances
for tests that want to assert against the model surface.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.pydantic_ai import (
    ADAPTER_CLASS,
    PydanticAIAdapter,
    instrument_agent,
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
    a = PydanticAIAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = PydanticAIAdapter(org_id="test-org")
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
    """Typed migration: agent_name lives at payload.content.metadata."""
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
    metadata = out["payload"]["content"]["metadata"]
    assert metadata["agent_name"] == "planner"
    assert metadata["duration_ns"] >= 0


def test_run_failure_emits_output_with_error() -> None:
    """Typed migration: error is on AgentOutputEvent.metadata, run_status=run_failed."""
    stratix = _RecordingStratix()
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    agent = _FakeAgent(name="failing", raises=True)
    adapter.instrument_agent(agent)

    with pytest.raises(RuntimeError):
        agent.run_sync("bad")

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    metadata = out["payload"]["content"]["metadata"]
    assert "error" in metadata
    assert "simulated failure" in metadata["error"]
    assert metadata["run_status"] == "run_failed"


def test_run_extracts_usage_and_messages() -> None:
    """When the result has usage and a tool-return message, cost.record + tool.call fire.

    Typed shape: tokens at payload.cost.tokens, model at payload.model.name,
    tool name at payload.tool.name.
    """
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
    assert cost["payload"]["cost"]["tokens"] == 15
    assert cost["payload"]["cost"]["prompt_tokens"] == 10
    assert cost["payload"]["cost"]["completion_tokens"] == 5

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["model"]["name"] == "gpt-5"
    assert invoke["payload"]["model"]["provider"] == "openai"

    tool = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert tool["payload"]["tool"]["name"] == "calc"
    assert tool["payload"]["tool"]["integration"] == "library"
    # Scalar tool-return content is wrapped in {"value": ...}.
    assert tool["payload"]["output"] == {"value": 42}


def test_on_handoff_emits_event_with_context_hash() -> None:
    """Typed AgentHandoffEvent: handoff_context_hash is sha256:<hex64>."""
    stratix = _RecordingStratix()
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
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
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_handoff(from_agent="a", to_agent="b", context=None)

    evt = next(e for e in stratix.events if e["event_type"] == "agent.handoff")
    assert evt["payload"]["handoff_context_hash"].startswith("sha256:")


def test_capture_config_gates_l1_agent_io() -> None:
    """When l1_agent_io is disabled, agent.input/output do NOT fire.

    Post-migration: the previous adapter emitted an ad-hoc
    ``agent.state.change`` payload alongside agent.output to carry a
    run_complete / run_failed marker. That payload is no longer
    emitted (it did not satisfy the canonical
    AgentStateChangeEvent contract); the marker now lives on
    AgentOutputEvent.metadata.run_status. So when l1_agent_io is
    disabled, no agent.* events fire at all from on_run_*.
    """
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l1_agent_io=False)
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    adapter.on_run_start(agent_name="a", input_data="x")
    adapter.on_run_end(agent_name="a", output="y")

    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" not in types
    assert "agent.output" not in types
    # agent.state.change is NO LONGER emitted post-migration —
    # see the on_run_end docstring for the rationale.
    assert "agent.state.change" not in types


def test_environment_config_emits_once_per_agent() -> None:
    """Typed migration: agent_name + tools live at payload.environment.attributes."""
    stratix = _RecordingStratix()
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="a1", tools=[SimpleNamespace(name="search")], model="gpt-5")
    adapter.instrument_agent(agent)
    adapter.instrument_agent(agent)  # idempotent

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1
    attributes = configs[0]["payload"]["environment"]["attributes"]
    assert attributes["agent_name"] == "a1"
    assert attributes["tools"] == ["search"]
    assert configs[0]["payload"]["environment"]["type"] == "simulated"


def test_instrument_agent_helper() -> None:
    agent = _FakeAgent(name="helper")
    adapter = instrument_agent(agent, org_id="test-org")
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


# ---------------------------------------------------------------------------
# Typed-event migration regression tests (PR #129 follow-up — bundle 5)
# ---------------------------------------------------------------------------


def test_pydantic_ai_emits_typed_payloads_only() -> None:
    """Every emit site in pydantic_ai lifecycle.py is a typed
    emit_event call.

    Pins the post-migration contract: the recording stratix's
    ``typed_payloads`` list grows for every emission and the legacy
    two-arg dict path receives nothing.
    """
    from layerlens.instrument._compat.events import (
        ToolCallEvent,
        AgentInputEvent,
        CostRecordEvent,
        AgentOutputEvent,
        ModelInvokeEvent,
        AgentHandoffEvent,
        EnvironmentConfigEvent,
    )

    stratix = _RecordingStratix()
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    # Drive every emission path: instrument_agent triggers
    # environment.config; run_sync drives agent.input/output and
    # _extract_run_usage drives cost.record + model.invoke + tool.call.
    usage = SimpleNamespace(request_tokens=10, response_tokens=5, total_tokens=15)
    response_msg = SimpleNamespace(kind="response")
    tool_msg = SimpleNamespace(kind="tool-return", tool_name="calc", content=42)
    result = SimpleNamespace(
        data="ok",
        usage=usage,
        all_messages=[response_msg, tool_msg],
        model_name="gpt-5",
    )
    agent = _FakeAgent(name="planner", model="gpt-5", result=result)
    adapter.instrument_agent(agent)
    agent.run_sync("hi")

    # Direct lifecycle hooks
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
    assert ToolCallEvent in types_seen


def test_pydantic_ai_emit_does_not_warn_after_migration() -> None:
    """No DeprecationWarning fires from pydantic_ai lifecycle paths.

    The base adapter's ``emit_dict_event`` raises a DeprecationWarning
    on every call. After migration, pydantic_ai lifecycle must never
    trigger that warning.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = PydanticAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        usage = SimpleNamespace(request_tokens=10, response_tokens=5, total_tokens=15)
        response_msg = SimpleNamespace(kind="response")
        tool_msg = SimpleNamespace(kind="tool-return", tool_name="calc", content=42)
        result = SimpleNamespace(
            data="ok",
            usage=usage,
            all_messages=[response_msg, tool_msg],
            model_name="gpt-5",
        )
        agent = _FakeAgent(name="planner", model="gpt-5", result=result)
        adapter.instrument_agent(agent)
        agent.run_sync("hi")
        adapter.on_tool_use("t", tool_input={"a": 1}, tool_output="ok")
        adapter.on_llm_call(provider="openai", model="gpt-5", tokens_prompt=5)
        adapter.on_handoff(from_agent="a", to_agent="b", context="ctx")
