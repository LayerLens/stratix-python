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


ADAPTER_CLS = AgnoAdapter



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
