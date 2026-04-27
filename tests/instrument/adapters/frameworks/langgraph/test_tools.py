"""Tests for LangGraph tool tracing.

Ported from ``ateam/tests/adapters/langgraph/test_tools.py``.

All public symbols (``LangGraphToolNode``, ``ToolTracer``,
``trace_langgraph_tool``) exist in stratix-python under
``layerlens.instrument.adapters.frameworks.langgraph.tools``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from layerlens.instrument.adapters.frameworks.langgraph.tools import (
    ToolTracer,
    LangGraphToolNode,
    trace_langgraph_tool,
)


class _MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class TestToolTracer:
    """Tests for ToolTracer."""

    def test_tracer_initialization(self) -> None:
        """Test tracer initializes correctly."""
        stratix = _MockStratix()
        tracer = ToolTracer(stratix)

        assert tracer._stratix is stratix
        assert tracer._executions == []

    def test_trace_decorator(self) -> None:
        """Test trace decorator wraps function."""
        stratix = _MockStratix()
        tracer = ToolTracer(stratix)

        @tracer.trace
        def search(query: str) -> str:
            return f"Results for: {query}"

        result = search("test query")

        assert result == "Results for: test query"
        assert len(tracer._executions) == 1

    def test_trace_emits_tool_call_event(self) -> None:
        """Test trace emits tool.call event."""
        stratix = _MockStratix()
        tracer = ToolTracer(stratix)

        @tracer.trace
        def calculator(a: int, b: int) -> int:
            return a + b

        calculator(2, 3)

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "calculator"
        assert tool_events[0]["payload"]["output"] == 5

    def test_trace_captures_input(self) -> None:
        """Test trace captures input arguments."""
        stratix = _MockStratix()
        tracer = ToolTracer(stratix)

        @tracer.trace
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        greet("World", greeting="Hi")

        tool_events = stratix.get_events("tool.call")
        payload = tool_events[0]["payload"]
        assert payload["input"]["args"] == ["World"]
        assert payload["input"]["kwargs"]["greeting"] == "Hi"

    def test_trace_handles_exception(self) -> None:
        """Test trace handles tool exceptions."""
        stratix = _MockStratix()
        tracer = ToolTracer(stratix)

        @tracer.trace
        def failing_tool() -> str:
            raise ValueError("Tool failed")

        with pytest.raises(ValueError, match="Tool failed"):
            failing_tool()

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["error"] == "Tool failed"

    def test_trace_records_duration(self) -> None:
        """Test trace records execution duration."""
        stratix = _MockStratix()
        tracer = ToolTracer(stratix)

        @tracer.trace
        def quick_tool() -> str:
            return "done"

        quick_tool()

        tool_events = stratix.get_events("tool.call")
        assert isinstance(tool_events[0]["payload"]["duration_ns"], int)


class TestTraceLanggraphToolDecorator:
    """Tests for trace_langgraph_tool decorator."""

    def test_decorator_without_args(self) -> None:
        """Test decorator without arguments."""

        @trace_langgraph_tool
        def my_tool(x: int) -> int:
            return x * 2

        result = my_tool(5)

        assert result == 10

    def test_decorator_with_stratix(self) -> None:
        """Test decorator with STRATIX instance."""
        stratix = _MockStratix()

        @trace_langgraph_tool(stratix_instance=stratix)
        def search(query: str) -> str:
            return f"Found: {query}"

        search("test")

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 1

    def test_decorator_with_custom_name(self) -> None:
        """Test decorator with custom tool name."""
        stratix = _MockStratix()

        @trace_langgraph_tool(stratix_instance=stratix, tool_name="custom_search")
        def search(query: str) -> str:
            return query

        search("test")

        tool_events = stratix.get_events("tool.call")
        assert tool_events[0]["payload"]["tool_name"] == "custom_search"

    def test_decorator_preserves_function_metadata(self) -> None:
        """Test decorator preserves function metadata."""

        @trace_langgraph_tool
        def documented_tool(x: int) -> int:
            """This is a documented tool."""
            return x

        assert documented_tool.__name__ == "documented_tool"
        assert documented_tool.__doc__ == "This is a documented tool."

    def test_decorator_handles_complex_types(self) -> None:
        """Test decorator handles complex input/output types."""
        stratix = _MockStratix()

        @trace_langgraph_tool(stratix_instance=stratix)
        def complex_tool(data: dict[str, Any]) -> list[str]:
            return list(data.keys())

        result = complex_tool({"a": 1, "b": 2})

        assert result == ["a", "b"]

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 1


class TestLangGraphToolNode:
    """Tests for LangGraphToolNode."""

    def test_node_initialization(self) -> None:
        """Test node initializes correctly."""

        def my_tool(x: int) -> int:
            return x * 2

        node = LangGraphToolNode(my_tool)

        assert node._tool_name == "my_tool"

    def test_node_callable_interface(self) -> None:
        """Test node implements callable interface."""

        def double(x: int) -> int:
            return x * 2

        node = LangGraphToolNode(double, state_key="input")

        result = node({"input": 5})

        assert result["tool_output"] == 10

    def test_node_emits_tool_call(self) -> None:
        """Test node emits tool.call event."""
        stratix = _MockStratix()

        def search(query: str) -> str:
            return f"Results: {query}"

        node = LangGraphToolNode(search, stratix_instance=stratix, state_key="query")
        node({"query": "test"})

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "search"

    def test_node_custom_name(self) -> None:
        """Test node with custom tool name."""
        stratix = _MockStratix()

        def func(x: Any) -> Any:
            return x

        node = LangGraphToolNode(func, stratix_instance=stratix, tool_name="custom")
        node({"input": "test"})

        tool_events = stratix.get_events("tool.call")
        assert tool_events[0]["payload"]["tool_name"] == "custom"

    def test_node_handles_exception(self) -> None:
        """Test node handles tool exceptions."""
        stratix = _MockStratix()

        def failing(x: Any) -> Any:
            raise RuntimeError("Failed")

        node = LangGraphToolNode(failing, stratix_instance=stratix, state_key="input")

        with pytest.raises(RuntimeError, match="Failed"):
            node({"input": "test"})

        tool_events = stratix.get_events("tool.call")
        assert tool_events[0]["payload"]["error"] == "Failed"

    def test_node_full_state_input(self) -> None:
        """Test node uses full state when no state_key."""

        def process(state: dict[str, Any]) -> dict[str, Any]:
            return {"count": state.get("count", 0) + 1}

        node = LangGraphToolNode(process)
        result = node({"count": 5})

        assert result["tool_output"]["count"] == 6


class TestToolTracerAsync:
    """Async tests for ToolTracer.

    Uses ``asyncio.run`` rather than ``@pytest.mark.asyncio`` because
    ``pytest-asyncio`` is not in the project's dev requirements.
    """

    def test_trace_async_decorator(self) -> None:
        """Test trace_async decorator."""
        stratix = _MockStratix()
        tracer = ToolTracer(stratix)

        @tracer.trace_async
        async def async_search(query: str) -> str:
            return f"Async results: {query}"

        result = asyncio.run(async_search("test"))

        assert result == "Async results: test"

        tool_events = stratix.get_events("tool.call")
        assert len(tool_events) == 1

    def test_trace_async_handles_exception(self) -> None:
        """Test trace_async handles exceptions."""
        stratix = _MockStratix()
        tracer = ToolTracer(stratix)

        @tracer.trace_async
        async def failing_async() -> str:
            raise ValueError("Async failed")

        async def _run() -> None:
            await failing_async()

        with pytest.raises(ValueError, match="Async failed"):
            asyncio.run(_run())

        tool_events = stratix.get_events("tool.call")
        assert tool_events[0]["payload"]["error"] == "Async failed"
