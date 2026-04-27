"""Unit tests for the AWS Strands framework adapter.

Mocked at the SDK shape level — no real ``strands`` runtime needed.
The adapter wraps ``invoke()`` (and ``__call__``); tests exercise ``invoke``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.strands import (
    ADAPTER_CLASS,
    StrandsAdapter,
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

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _FakeAgent:
    """Minimal duck-typed Strands agent for tests."""

    def __init__(
        self,
        name: str = "strands-agent",
        tools: Any = None,
        model: Any = None,
        system_prompt: Any = None,
        conversation: Any = None,
        result: Any = None,
        raises: bool = False,
    ) -> None:
        self.name = name
        self.tools = tools
        self.model = model
        self.system_prompt = system_prompt
        self.conversation = conversation
        self._result = result
        self._raises = raises

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        if self._raises:
            raise RuntimeError("simulated failure")
        return self._result if self._result is not None else SimpleNamespace(content=f"out:{prompt}", text=None)

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        return self.invoke(prompt, **kwargs)


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is StrandsAdapter


def test_lifecycle() -> None:
    a = StrandsAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = StrandsAdapter(org_id="test-org")
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "strands"
    assert info.name == "StrandsAdapter"
    health = a.health_check()
    assert health.framework_name == "strands"


def test_instrument_agent_wraps_invoke() -> None:
    adapter = StrandsAdapter(stratix=_RecordingStratix(), capture_config=CaptureConfig.full())
    adapter.connect()
    agent = _FakeAgent(name="planner")
    adapter.instrument_agent(agent)
    assert agent.invoke.__name__ == "traced_call"

    adapter.disconnect()
    assert agent.invoke.__name__ == "invoke"


def test_invoke_emits_input_and_output_events() -> None:
    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="planner", model="anthropic.claude-v2")
    adapter.instrument_agent(agent)
    result = agent.invoke("hello")
    assert getattr(result, "content", None) == "out:hello"

    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["agent_name"] == "planner"
    assert out["payload"]["duration_ns"] >= 0


def test_invoke_extracts_usage_and_emits_cost() -> None:
    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    usage = SimpleNamespace(inputTokens=10, outputTokens=5, totalTokens=15)
    result = SimpleNamespace(content="ok", text=None, usage=usage, tool_results=[])
    agent = _FakeAgent(name="planner", model="anthropic.claude-v2", result=result)
    adapter.instrument_agent(agent)
    agent.invoke("hi")

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "cost.record" in types

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["tokens_total"] == 15


def test_invoke_failure_emits_output_with_error() -> None:
    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    agent = _FakeAgent(name="failing", raises=True)
    adapter.instrument_agent(agent)

    with pytest.raises(RuntimeError):
        agent.invoke("bad")

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert "error" in out["payload"]
    assert "simulated failure" in out["payload"]["error"]


def test_on_tool_use_emits_event() -> None:
    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_tool_use("calc", tool_input={"x": 1}, tool_output=2, latency_ms=12.3)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert evt["payload"]["tool_name"] == "calc"
    assert evt["payload"]["latency_ms"] == 12.3


def test_capture_config_gates_l3_model_metadata() -> None:
    """When l3_model_metadata is disabled, model.invoke does NOT fire (state.change still does)."""
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l3_model_metadata=False)
    adapter = StrandsAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    adapter.on_llm_call(model="claude", provider="bedrock")
    adapter.on_run_start(agent_name="a", input_data="x")
    adapter.on_run_end(agent_name="a", output="y")

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" not in types
    # state.change is cross-cutting — always fires.
    assert "agent.state.change" in types


def test_environment_config_emits_once_per_agent() -> None:
    stratix = _RecordingStratix()
    adapter = StrandsAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    agent = _FakeAgent(name="a1", tools=[SimpleNamespace(name="search")], model="claude")
    adapter.instrument_agent(agent)
    adapter.instrument_agent(agent)

    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(configs) == 1
    assert configs[0]["payload"]["tools"] == ["search"]


def test_instrument_agent_helper() -> None:
    agent = _FakeAgent(name="helper")
    adapter = instrument_agent(agent, org_id="test-org")
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY


def test_serialize_for_replay() -> None:
    adapter = StrandsAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "strands"
    assert rt.adapter_name == "StrandsAdapter"
    assert "capture_config" in rt.config
