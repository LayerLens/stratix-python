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


ADAPTER_CLS = PydanticAIAdapter



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
