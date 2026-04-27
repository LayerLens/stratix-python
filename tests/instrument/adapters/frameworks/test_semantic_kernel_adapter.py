"""Unit tests for the Microsoft Semantic Kernel adapter.

Mocked at the SDK shape level — no real ``semantic_kernel`` runtime needed.
The adapter wires filters via ``kernel.add_filter(...)`` and exposes a
suite of lifecycle hooks (``on_function_start``, ``on_model_invoke``,
``on_planner_step``, etc.) that are called by those filters. Tests
exercise the lifecycle hooks directly + verify filter wiring.
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.semantic_kernel import (
    ADAPTER_CLASS,
    SemanticKernelAdapter,
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


class _FakeKernel:
    def __init__(self, plugins: Any = None) -> None:
        self.plugins = plugins or {}
        self._added_filters: List[Dict[str, Any]] = []

    def add_filter(self, filter_type: str, filter_obj: Any) -> None:
        self._added_filters.append({"type": filter_type, "filter": filter_obj})


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is SemanticKernelAdapter


def test_lifecycle() -> None:
    a = SemanticKernelAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_adapter_info_and_health() -> None:
    a = SemanticKernelAdapter(org_id="test-org")
    a.connect()
    info = a.get_adapter_info()
    assert info.framework == "semantic_kernel"
    assert info.name == "SemanticKernelAdapter"
    health = a.health_check()
    assert health.framework_name == "semantic_kernel"


def test_instrument_kernel_registers_filters_and_discovers_plugins() -> None:
    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    kernel = _FakeKernel(plugins={"math": object(), "search": object()})
    adapter.instrument_kernel(kernel)

    filter_types = {f["type"] for f in kernel._added_filters}
    assert filter_types == {"function_invocation", "prompt_rendering", "auto_function_invocation"}

    # Plugin discovery emits environment.config events.
    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    plugin_names = {c["payload"].get("plugin_name") for c in configs}
    assert "math" in plugin_names
    assert "search" in plugin_names


def test_on_function_start_end_emits_tool_call() -> None:
    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    ctx = adapter.on_function_start(plugin_name="math", function_name="add", arguments={"a": 1, "b": 2})
    adapter.on_function_end(context=ctx, result=3)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert evt["payload"]["tool_name"] == "math.add"
    assert evt["payload"]["plugin_name"] == "math"
    assert evt["payload"]["function_name"] == "add"
    assert evt["payload"]["latency_ms"] >= 0


def test_on_model_invoke_emits_invoke_and_cost() -> None:
    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_model_invoke(
        provider="azure_openai",
        model="gpt-5",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=20.0,
    )

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["model"] == "gpt-5"
    assert invoke["payload"]["latency_ms"] == 20.0

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["total_tokens"] == 15


def test_on_prompt_render_emits_agent_code() -> None:
    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_prompt_render(
        template="Hello {{name}}",
        rendered_prompt="Hello world",
        function_name="greet",
    )

    evt = next(e for e in stratix.events if e["event_type"] == "agent.code")
    assert evt["payload"]["event_subtype"] == "prompt_render"
    assert evt["payload"]["function_name"] == "greet"


def test_on_planner_step_emits_agent_code() -> None:
    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_planner_step(
        planner_type="HandlebarsPlanner",
        step_index=1,
        thought="I need to search",
        action="search",
        observation="found results",
        status="completed",
    )

    evt = next(e for e in stratix.events if e["event_type"] == "agent.code")
    assert evt["payload"]["event_subtype"] == "planner_step"
    assert evt["payload"]["planner_type"] == "HandlebarsPlanner"
    assert evt["payload"]["step_index"] == 1


def test_on_memory_operation_emits_tool_call() -> None:
    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_memory_operation(
        operation="search",
        collection="facts",
        query="capital of France",
        result_count=3,
        relevance_scores=[0.9, 0.8, 0.7],
        backend_type="qdrant",
    )

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert evt["payload"]["tool_name"] == "memory.search"
    assert evt["payload"]["result_count"] == 3
    assert evt["payload"]["backend_type"] == "qdrant"


def test_on_kernel_invoke_start_end_emits_input_output() -> None:
    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_kernel_invoke_start(input_text="hello")
    adapter.on_kernel_invoke_end(output="world")

    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["output"] == "world"
    assert out["payload"]["duration_ns"] >= 0


def test_capture_config_gates_l5a_tool_calls() -> None:
    """When l5a_tool_calls is disabled, tool.call does NOT fire (model.invoke still does)."""
    stratix = _RecordingStratix()
    cfg = CaptureConfig(l5a_tool_calls=False)
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=cfg)
    adapter.connect()

    ctx = adapter.on_function_start(plugin_name="math", function_name="add")
    adapter.on_function_end(context=ctx, result=3)
    adapter.on_model_invoke(model="gpt-5", prompt_tokens=10, completion_tokens=5)

    types = [e["event_type"] for e in stratix.events]
    assert "tool.call" not in types
    assert "model.invoke" in types


def test_serialize_for_replay() -> None:
    adapter = SemanticKernelAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert rt.framework == "semantic_kernel"
    assert rt.adapter_name == "SemanticKernelAdapter"
    assert "capture_config" in rt.config
