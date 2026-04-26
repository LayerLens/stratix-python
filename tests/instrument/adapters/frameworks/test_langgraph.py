"""Unit tests for the LangGraph framework adapter.

Mocked at the SDK shape level — no real LangGraph runtime needed. Each
test exercises one slice of the adapter surface using a duck-typed
``_FakeCompiledGraph`` that mimics ``langgraph.graph.StateGraph.compile()``.

Coverage:

* Lifecycle (connect → healthy → disconnect → disconnected).
* ``ADAPTER_CLASS`` registry export.
* ``info()`` / ``get_adapter_info()`` reports ``requires_pydantic = V2_ONLY``
  per Round-2 deliberation item 20.
* ``STRATIXLangGraphAdapter`` alias resolves to ``LayerLensLangGraphAdapter``
  and emits a ``DeprecationWarning`` per Round-2 deliberation item 23.
* ``wrap_graph(...).invoke(...)`` emits ``environment.config`` +
  ``agent.input`` + ``agent.output`` (and ``agent.state.change`` when the
  state hash changes).
* ``ainvoke(...)`` mirrors ``invoke(...)``.
* Errors during graph execution still emit ``agent.output`` with the
  exception captured.
* ``HandoffDetector`` integration — node transitions emit ``agent.handoff``.
* ``trace_node`` / ``trace_langgraph_tool`` decorator emit the right
  legacy events when used standalone.
* ``wrap_llm_for_langgraph`` / ``TracedLLM`` emits ``model.invoke``.
* ``serialize_for_replay`` returns a ``ReplayableTrace`` with the
  framework string and ``capture_config`` payload.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.frameworks.langgraph import (
    ADAPTER_CLASS,
    TracedLLM,
    NodeTracer,
    HandoffDetector,
    LangGraphStateAdapter,
    LayerLensLangGraphAdapter,
    trace_node,
    detect_handoff,
    trace_langgraph_tool,
    wrap_llm_for_langgraph,
)
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

# ---------------------------------------------------------------------------
# Helpers — minimal duck-typed LangGraph + STRATIX surfaces.
# ---------------------------------------------------------------------------


class _RecordingStratix:
    """Duck-typed STRATIX client that captures ``emit(event_type, payload)``."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _FakeCompiledGraph:
    """Duck-typed ``StateGraph.compile()`` result used in tests.

    Calling ``invoke`` runs an in-process function that returns the new
    state. ``ainvoke`` is provided as the async coroutine equivalent.
    """

    def __init__(
        self,
        name: str,
        run: Any,
    ) -> None:
        self.name = name
        self._run = run

    def invoke(self, state: Any, config: Any = None) -> Any:
        return self._run(state)

    async def ainvoke(self, state: Any, config: Any = None) -> Any:
        return self._run(state)


class _FakeLLM:
    """Duck-typed LangChain chat model (``invoke`` / ``ainvoke``)."""

    model_name = "gpt-test"

    class _Response:
        def __init__(self, content: str) -> None:
            self.content = content
            self.type = "ai"
            self.usage_metadata = {"input_tokens": 4, "output_tokens": 2}

    def invoke(self, messages: Any, **kwargs: Any) -> "_FakeLLM._Response":
        del messages, kwargs
        return _FakeLLM._Response("hi from fake llm")

    async def ainvoke(self, messages: Any, **kwargs: Any) -> "_FakeLLM._Response":
        del messages, kwargs
        return _FakeLLM._Response("hi from fake llm async")


# ---------------------------------------------------------------------------
# Registry / metadata
# ---------------------------------------------------------------------------


def test_adapter_class_export() -> None:
    """Registry lazy-loading expects ``ADAPTER_CLASS`` to point at the adapter."""
    assert ADAPTER_CLASS is LayerLensLangGraphAdapter


def test_framework_and_version_constants() -> None:
    assert LayerLensLangGraphAdapter.FRAMEWORK == "langgraph"
    assert LayerLensLangGraphAdapter.VERSION


def test_class_attribute_requires_pydantic_v2() -> None:
    """Round-2 item 20: every framework adapter must declare its compat."""
    assert LayerLensLangGraphAdapter.requires_pydantic == PydanticCompat.V2_ONLY


def test_get_adapter_info_reports_v2_only() -> None:
    """Round-2 item 20: catalog manifest reads ``info()`` — must report v2-only."""
    a = LayerLensLangGraphAdapter()
    info = a.get_adapter_info()
    assert info.framework == "langgraph"
    assert info.name == "LayerLensLangGraphAdapter"
    assert info.requires_pydantic == PydanticCompat.V2_ONLY


def test_info_wrapper_also_reports_v2_only() -> None:
    """``info()`` (the BaseAdapter wrapper) must agree with the class attribute."""
    a = LayerLensLangGraphAdapter()
    assert a.info().requires_pydantic == PydanticCompat.V2_ONLY


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_lifecycle() -> None:
    a = LayerLensLangGraphAdapter()
    a.connect()
    assert a.is_connected is True
    assert a.status == AdapterStatus.HEALTHY

    health = a.health_check()
    assert health.framework_name == "langgraph"
    assert health.adapter_version == LayerLensLangGraphAdapter.VERSION

    a.disconnect()
    assert a.is_connected is False
    assert a.status == AdapterStatus.DISCONNECTED


def test_constructor_accepts_capture_config() -> None:
    cfg = CaptureConfig.standard()
    a = LayerLensLangGraphAdapter(capture_config=cfg)
    assert a.capture_config is cfg


def test_constructor_legacy_kwargs_map_to_capture_config() -> None:
    """Legacy ``emit_environment_config`` / ``emit_agent_code`` flags are honoured."""
    a = LayerLensLangGraphAdapter(
        stratix_instance=_RecordingStratix(),
        emit_environment_config=False,
        emit_agent_code=True,
    )
    assert a.capture_config.l4a_environment_config is False
    assert a.capture_config.l2_agent_code is True


# ---------------------------------------------------------------------------
# Backward-compat alias (Round-2 item 23)
# ---------------------------------------------------------------------------


def test_stratix_alias_resolves_to_layerlens_class() -> None:
    """The ``STRATIX*`` name still resolves for existing ateam users."""
    import layerlens.instrument.adapters.frameworks.langgraph as mod

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cls = mod.STRATIXLangGraphAdapter
    assert cls is LayerLensLangGraphAdapter


def test_stratix_alias_emits_deprecation_warning() -> None:
    """Round-2 item 23: alias access must raise ``DeprecationWarning``."""
    import layerlens.instrument.adapters.frameworks.langgraph as mod

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = mod.STRATIXLangGraphAdapter
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 1
    msg = str(deprecation_warnings[0].message)
    assert "STRATIXLangGraphAdapter" in msg
    assert "LayerLensLangGraphAdapter" in msg
    assert "v2.0" in msg


def test_unknown_attribute_still_raises_attribute_error() -> None:
    """``__getattr__`` must NOT silently swallow unknown names."""
    import layerlens.instrument.adapters.frameworks.langgraph as mod

    with pytest.raises(AttributeError):
        _ = mod.NoSuchSymbol  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Graph wrapping → event emission
# ---------------------------------------------------------------------------


def test_wrap_graph_invoke_emits_lifecycle_events() -> None:
    """Wrapping a compiled graph emits config + input + output events."""
    stratix = _RecordingStratix()
    adapter = LayerLensLangGraphAdapter(
        stratix_instance=stratix,
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    graph = _FakeCompiledGraph(
        name="test-graph",
        run=lambda s: {**s, "result": "done"},
    )
    traced = adapter.wrap_graph(graph)
    out = traced.invoke({"count": 0})

    assert out == {"count": 0, "result": "done"}
    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    assert "agent.input" in types
    assert "agent.output" in types
    # State changed (added "result"), so a state-change event must fire.
    assert "agent.state.change" in types


def test_wrap_graph_emits_state_change_only_when_hash_changes() -> None:
    """Identity-mapping nodes should NOT emit ``agent.state.change``."""
    stratix = _RecordingStratix()
    adapter = LayerLensLangGraphAdapter(
        stratix_instance=stratix,
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    graph = _FakeCompiledGraph(name="passthrough", run=lambda s: s)
    traced = adapter.wrap_graph(graph)
    traced.invoke({"k": "v"})

    types = [e["event_type"] for e in stratix.events]
    assert "agent.state.change" not in types


def test_wrap_graph_invoke_failure_emits_output_with_error() -> None:
    """An exception inside the graph still emits ``agent.output`` with the error."""

    def _boom(_: Any) -> Any:
        raise RuntimeError("graph blew up")

    stratix = _RecordingStratix()
    adapter = LayerLensLangGraphAdapter(
        stratix_instance=stratix,
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    graph = _FakeCompiledGraph(name="failing", run=_boom)
    traced = adapter.wrap_graph(graph)

    with pytest.raises(RuntimeError):
        traced.invoke({"x": 1})

    out = next(e for e in stratix.events if e["event_type"] == "agent.output")
    assert out["payload"]["error"] == "graph blew up"


def test_wrap_graph_ainvoke_emits_lifecycle_events() -> None:
    """Async graph invocation emits the same lifecycle events as ``invoke``."""
    stratix = _RecordingStratix()
    adapter = LayerLensLangGraphAdapter(
        stratix_instance=stratix,
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    graph = _FakeCompiledGraph(
        name="async-graph",
        run=lambda s: {**s, "answer": 42},
    )
    traced = adapter.wrap_graph(graph)
    out = asyncio.run(traced.ainvoke({"q": "what?"}))

    assert out == {"q": "what?", "answer": 42}
    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" in types
    assert "agent.output" in types


def test_traced_graph_proxies_unknown_attributes_to_underlying_graph() -> None:
    """Unknown attribute access on the wrapper falls through to the graph."""
    adapter = LayerLensLangGraphAdapter()
    adapter.connect()

    graph = _FakeCompiledGraph(name="proxy", run=lambda s: s)
    graph.custom_attr = "hello"  # type: ignore[attr-defined]
    traced = adapter.wrap_graph(graph)
    assert traced.custom_attr == "hello"


# ---------------------------------------------------------------------------
# Handoff detection
# ---------------------------------------------------------------------------


def test_handoff_detector_emits_handoff_on_agent_change() -> None:
    """Routing from one agent to another emits a single handoff event."""
    stratix = _RecordingStratix()
    detector = HandoffDetector(stratix_instance=stratix)
    detector.register_agents("researcher", "writer")
    detector.set_current_agent("researcher")

    handoff = detector.detect_handoff("writer", state={"task": "summarize"})
    assert handoff is not None
    assert handoff.from_agent == "researcher"
    assert handoff.to_agent == "writer"

    handoffs = [e for e in stratix.events if e["event_type"] == "agent.handoff"]
    assert len(handoffs) == 1
    assert handoffs[0]["payload"]["from_agent"] == "researcher"
    assert handoffs[0]["payload"]["to_agent"] == "writer"


def test_handoff_detector_returns_none_on_same_agent() -> None:
    """No event when staying with the same agent."""
    stratix = _RecordingStratix()
    detector = HandoffDetector(stratix_instance=stratix)
    detector.set_current_agent("a")
    assert detector.detect_handoff("a") is None
    assert all(e["event_type"] != "agent.handoff" for e in stratix.events)


def test_detect_handoff_helper_emits_when_agents_differ() -> None:
    stratix = _RecordingStratix()
    h = detect_handoff(
        from_agent="a",
        to_agent="b",
        stratix_instance=stratix,
        reason="testing",
    )
    assert h is not None
    handoffs = [e for e in stratix.events if e["event_type"] == "agent.handoff"]
    assert len(handoffs) == 1
    assert handoffs[0]["payload"]["reason"] == "testing"


def test_adapter_with_handoff_detector_emits_on_node_transition() -> None:
    """When attached, the detector fires on every ``on_node_start``."""
    stratix = _RecordingStratix()
    detector = HandoffDetector(stratix_instance=stratix)
    detector.register_agents("planner", "executor")
    detector.set_current_agent("planner")

    adapter = LayerLensLangGraphAdapter(
        stratix_instance=stratix,
        capture_config=CaptureConfig.full(),
        handoff_detector=detector,
    )
    adapter.connect()

    execution = adapter.on_graph_start(
        graph_id="g1",
        execution_id="g1:1",
        initial_state={"step": 0},
    )
    # First node: same as current agent, no handoff.
    adapter.on_node_start(execution, "planner", {"step": 1})
    # Second node: triggers handoff.
    adapter.on_node_start(execution, "executor", {"step": 2})

    handoffs = [e for e in stratix.events if e["event_type"] == "agent.handoff"]
    assert len(handoffs) == 1
    assert handoffs[0]["payload"]["to_agent"] == "executor"


# ---------------------------------------------------------------------------
# Standalone decorators
# ---------------------------------------------------------------------------


def test_trace_node_decorator_emits_state_change() -> None:
    """``@trace_node(stratix)`` decorates a function and emits state change."""
    stratix = _RecordingStratix()

    @trace_node(stratix_instance=stratix)
    def my_node(state: dict[str, Any]) -> dict[str, Any]:
        return {**state, "touched": True}

    out = my_node({"x": 1})
    assert out == {"x": 1, "touched": True}

    state_changes = [e for e in stratix.events if e["event_type"] == "agent.state.change"]
    assert len(state_changes) == 1
    assert state_changes[0]["payload"]["node_name"] == "my_node"


def test_node_tracer_no_state_change_when_state_unchanged() -> None:
    """Identity nodes should not emit ``agent.state.change``."""
    stratix = _RecordingStratix()
    tracer = NodeTracer(stratix_instance=stratix)

    @tracer.decorate
    def passthrough(state: dict[str, Any]) -> dict[str, Any]:
        return state

    passthrough({"k": "v"})
    assert all(e["event_type"] != "agent.state.change" for e in stratix.events)


def test_trace_langgraph_tool_decorator_emits_tool_call() -> None:
    """``@trace_langgraph_tool`` emits ``tool.call`` per invocation."""
    stratix = _RecordingStratix()

    @trace_langgraph_tool(stratix_instance=stratix)
    def search(query: str) -> str:
        return f"results for {query}"

    out = search("python")
    assert out == "results for python"

    tool_calls = [e for e in stratix.events if e["event_type"] == "tool.call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["payload"]["tool_name"] == "search"
    assert tool_calls[0]["payload"]["error"] is None


def test_trace_langgraph_tool_captures_exception() -> None:
    stratix = _RecordingStratix()

    @trace_langgraph_tool(stratix_instance=stratix)
    def failing(query: str) -> str:
        raise ValueError(f"bad query: {query}")

    with pytest.raises(ValueError):
        failing("x")

    tool_calls = [e for e in stratix.events if e["event_type"] == "tool.call"]
    assert len(tool_calls) == 1
    assert "bad query" in tool_calls[0]["payload"]["error"]


# ---------------------------------------------------------------------------
# LLM wrapping
# ---------------------------------------------------------------------------


def test_wrap_llm_for_langgraph_emits_model_invoke() -> None:
    """``TracedLLM`` emits ``model.invoke`` for each invocation."""
    stratix = _RecordingStratix()
    llm = _FakeLLM()
    traced = wrap_llm_for_langgraph(llm, stratix_instance=stratix)
    assert isinstance(traced, TracedLLM)

    response = traced.invoke([{"role": "user", "content": "hi"}])
    assert response.content == "hi from fake llm"

    invokes = [e for e in stratix.events if e["event_type"] == "model.invoke"]
    assert len(invokes) == 1
    payload = invokes[0]["payload"]
    assert payload["model"] == "gpt-test"
    assert payload["provider"] == "openai" or payload["provider"] == "unknown"
    assert payload["error"] is None


def test_traced_llm_ainvoke_emits_model_invoke() -> None:
    stratix = _RecordingStratix()
    traced = TracedLLM(_FakeLLM(), stratix_instance=stratix)

    response = asyncio.run(traced.ainvoke([{"role": "user", "content": "hi"}]))
    assert response.content == "hi from fake llm async"

    invokes = [e for e in stratix.events if e["event_type"] == "model.invoke"]
    assert len(invokes) == 1


# ---------------------------------------------------------------------------
# State adapter (sanity)
# ---------------------------------------------------------------------------


def test_state_adapter_get_hash_stable_across_calls() -> None:
    sa = LangGraphStateAdapter()
    h1 = sa.get_hash({"a": 1, "b": [1, 2, 3]})
    h2 = sa.get_hash({"b": [1, 2, 3], "a": 1})
    assert h1 == h2  # canonical (sort_keys) JSON


def test_state_adapter_diff_reports_changes() -> None:
    sa = LangGraphStateAdapter()
    before = sa.snapshot({"x": 1, "y": 2})
    after = sa.snapshot({"x": 1, "z": 3})
    diff = sa.diff(before, after)
    assert "z" in diff["added"]
    assert "y" in diff["removed"]


# ---------------------------------------------------------------------------
# Replay serialization
# ---------------------------------------------------------------------------


def test_serialize_for_replay_returns_replayable_trace() -> None:
    adapter = LayerLensLangGraphAdapter(
        stratix_instance=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
    )
    adapter.connect()

    rt = adapter.serialize_for_replay()
    assert rt.framework == "langgraph"
    assert rt.adapter_name == "LayerLensLangGraphAdapter"
    assert "capture_config" in rt.config
