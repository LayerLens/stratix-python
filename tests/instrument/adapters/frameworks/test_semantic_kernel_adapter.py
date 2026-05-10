"""Unit tests for the Microsoft Semantic Kernel adapter.

Mocked at the SDK shape level — no real ``semantic_kernel`` runtime needed.
The adapter wires filters via ``kernel.add_filter(...)`` and exposes a
suite of lifecycle hooks (``on_function_start``, ``on_model_invoke``,
``on_planner_step``, etc.) that are called by those filters. Tests
exercise the lifecycle hooks directly + verify filter wiring.

After the typed-event migration (PR #129 follow-up — bundle 5) every
emit site flows through :meth:`BaseAdapter.emit_event` with a canonical
Pydantic payload. The :class:`_RecordingStratix` stand-in below records
both shapes: the ``payload`` slot always carries a dict (model-dumped
if typed), and ``typed_payloads`` holds the original Pydantic instances
for tests that want to assert against the model surface.

Notable mapping decision: the legacy ``agent.code`` event type used
for prompt rendering and planner steps is NOT in the canonical
13-event taxonomy. The migration re-maps those boundaries onto
:class:`ToolLogicEvent` (L5b — tool business logic). Test coverage
asserts the L5b fields (``description``, ``rules``) carry the
expected provenance.
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
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


class _FakeKernel:
    def __init__(self, plugins: Any = None) -> None:
        self.plugins = plugins or {}
        self._added_filters: List[Dict[str, Any]] = []

    def add_filter(self, filter_type: str, filter_obj: Any) -> None:
        self._added_filters.append({"type": filter_type, "filter": filter_obj})


def _rule_value(rules: List[str], key: str) -> str | None:
    """Find the JSON-encoded value in the L5b rules list for a given key."""
    prefix = f"{key}="
    for rule in rules:
        if rule.startswith(prefix):
            return rule[len(prefix):]
    return None


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
    """Typed migration: plugin_name lives at payload.environment.attributes."""
    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    kernel = _FakeKernel(plugins={"math": object(), "search": object()})
    adapter.instrument_kernel(kernel)

    filter_types = {f["type"] for f in kernel._added_filters}
    assert filter_types == {"function_invocation", "prompt_rendering", "auto_function_invocation"}

    # Plugin discovery emits environment.config events.
    configs = [e for e in stratix.events if e["event_type"] == "environment.config"]
    plugin_names = {c["payload"]["environment"]["attributes"].get("plugin_name") for c in configs}
    assert "math" in plugin_names
    assert "search" in plugin_names


def test_on_function_start_end_emits_tool_call() -> None:
    """Typed migration: tool name lives at payload.tool.name; latency at payload.latency_ms."""
    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    ctx = adapter.on_function_start(plugin_name="math", function_name="add", arguments={"a": 1, "b": 2})
    adapter.on_function_end(context=ctx, result=3)

    evt = next(e for e in stratix.events if e["event_type"] == "tool.call")
    payload = evt["payload"]
    assert payload["tool"]["name"] == "math.add"
    assert payload["tool"]["integration"] == "library"
    assert payload["input"]["plugin_name"] == "math"
    assert payload["input"]["function_name"] == "add"
    assert payload["latency_ms"] >= 0


def test_on_model_invoke_emits_invoke_and_cost() -> None:
    """Typed migration: model name at payload.model.name; tokens at payload.cost.tokens."""
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
    inv_payload = invoke["payload"]
    assert inv_payload["model"]["name"] == "gpt-5"
    assert inv_payload["model"]["provider"] == "azure_openai"
    assert inv_payload["latency_ms"] == 20.0

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["cost"]["tokens"] == 15
    assert cost["payload"]["cost"]["prompt_tokens"] == 10
    assert cost["payload"]["cost"]["completion_tokens"] == 5


def test_on_prompt_render_emits_tool_logic() -> None:
    """Typed migration: agent.code → ToolLogicEvent (L5b).

    The legacy adapter emitted an ad-hoc ``agent.code`` event for
    prompt template rendering; that event type is NOT in the
    canonical 13-event taxonomy. The post-migration mapping carries
    rendering as L5b business logic with the rendering operation as
    ``description`` and per-event provenance encoded as ``rules``.
    """
    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.on_prompt_render(
        template="Hello {{name}}",
        rendered_prompt="Hello world",
        function_name="greet",
    )

    # agent.code is no longer emitted post-migration
    types = [e["event_type"] for e in stratix.events]
    assert "agent.code" not in types
    assert "tool.logic" in types

    evt = next(e for e in stratix.events if e["event_type"] == "tool.logic")
    payload = evt["payload"]
    assert "prompt_render" in payload["logic"]["description"]
    rules = payload["logic"]["rules"]
    assert _rule_value(rules, "event_subtype") == '"prompt_render"'
    assert _rule_value(rules, "function_name") == '"greet"'


def test_on_planner_step_emits_tool_logic() -> None:
    """Typed migration: agent.code → ToolLogicEvent for planner steps."""
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

    types = [e["event_type"] for e in stratix.events]
    assert "agent.code" not in types
    assert "tool.logic" in types

    evt = next(e for e in stratix.events if e["event_type"] == "tool.logic")
    payload = evt["payload"]
    assert "planner_step" in payload["logic"]["description"]
    assert "HandlebarsPlanner" in payload["logic"]["description"]
    rules = payload["logic"]["rules"]
    assert _rule_value(rules, "event_subtype") == '"planner_step"'
    assert _rule_value(rules, "planner_type") == '"HandlebarsPlanner"'
    assert _rule_value(rules, "step_index") == "1"
    assert _rule_value(rules, "action") == '"search"'


def test_on_memory_operation_emits_tool_call() -> None:
    """Typed migration: memory operations remain L5a tool.call (in-process API)."""
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
    payload = evt["payload"]
    assert payload["tool"]["name"] == "memory.search"
    assert payload["tool"]["integration"] == "library"
    assert payload["input"]["backend_type"] == "qdrant"
    assert payload["output"]["result_count"] == 3


def test_on_kernel_invoke_start_end_emits_input_output() -> None:
    """Typed migration: input/output text lives at payload.content.message."""
    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_kernel_invoke_start(input_text="hello")
    adapter.on_kernel_invoke_end(output="world")

    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" in types
    assert "agent.output" in types

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["content"]["message"] == "world"
    assert out["payload"]["content"]["metadata"]["duration_ns"] >= 0


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


# ---------------------------------------------------------------------------
# Typed-event migration regression tests (PR #129 follow-up — bundle 5)
# ---------------------------------------------------------------------------


def test_semantic_kernel_emits_typed_payloads_only() -> None:
    """Every emit site in semantic_kernel lifecycle.py is a typed
    emit_event call.

    Pins the post-migration contract: the recording stratix's
    ``typed_payloads`` list grows for every emission and the legacy
    two-arg dict path receives nothing.
    """
    from layerlens.instrument._compat.events import (
        ToolCallEvent,
        AgentInputEvent,
        CostRecordEvent,
        ToolLogicEvent,
        AgentOutputEvent,
        ModelInvokeEvent,
        EnvironmentConfigEvent,
    )

    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    kernel = _FakeKernel(plugins={"math": object()})
    adapter.instrument_kernel(kernel)

    # Drive every emission path
    ctx = adapter.on_function_start(plugin_name="math", function_name="add")
    adapter.on_function_end(context=ctx, result=3)
    adapter.on_prompt_render(template="hello", rendered_prompt="hi", function_name="greet")
    adapter.on_model_invoke(provider="openai", model="gpt-5", prompt_tokens=10, completion_tokens=5)
    adapter.on_planner_step(planner_type="HandlebarsPlanner", step_index=0, thought="t", action="a")
    adapter.on_memory_operation(operation="search", collection="facts", query="q", result_count=3)
    adapter.on_kernel_invoke_start(input_text="hello")
    adapter.on_kernel_invoke_end(output="world")

    # Every captured payload is a Pydantic model instance — the legacy
    # dict path was not used.
    assert stratix.typed_payloads, "expected typed payloads to be captured"
    types_seen = {type(p) for p in stratix.typed_payloads}
    assert AgentInputEvent in types_seen
    assert AgentOutputEvent in types_seen
    assert CostRecordEvent in types_seen
    assert EnvironmentConfigEvent in types_seen
    assert ModelInvokeEvent in types_seen
    assert ToolCallEvent in types_seen
    assert ToolLogicEvent in types_seen


def test_semantic_kernel_emit_does_not_warn_after_migration() -> None:
    """No DeprecationWarning fires from semantic_kernel lifecycle paths.

    The base adapter's ``emit_dict_event`` raises a DeprecationWarning
    on every call. After migration, semantic_kernel lifecycle must
    never trigger that warning.
    """
    import warnings

    stratix = _RecordingStratix()
    adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        kernel = _FakeKernel(plugins={"math": object()})
        adapter.instrument_kernel(kernel)
        ctx = adapter.on_function_start(plugin_name="math", function_name="add")
        adapter.on_function_end(context=ctx, result=3)
        adapter.on_prompt_render(template="t", rendered_prompt="r", function_name="f")
        adapter.on_model_invoke(provider="openai", model="gpt-5", prompt_tokens=10, completion_tokens=5)
        adapter.on_planner_step(planner_type="P", step_index=0)
        adapter.on_memory_operation(operation="search", collection="c", query="q", result_count=1)
        adapter.on_kernel_invoke_start(input_text="hi")
        adapter.on_kernel_invoke_end(output="bye")
