"""
STRATIX LangGraph Tool Tracing

Provides decorators and wrappers for tracing LangGraph tool nodes.
"""

from __future__ import annotations

import time
import logging
from typing import TYPE_CHECKING, Any, TypeVar
from functools import wraps
from dataclasses import dataclass
from collections.abc import Callable

if TYPE_CHECKING:
    from layerlens.instrument.adapters._base.adapter import BaseAdapter

logger = logging.getLogger(__name__)


StateT = TypeVar("StateT")
ToolFunc = Callable[..., Any]


@dataclass
class ToolExecution:
    """Tracks a single tool execution."""

    tool_name: str
    start_time_ns: int
    end_time_ns: int | None = None
    input_args: dict[str, Any] | None = None
    output: Any | None = None
    error: str | None = None


class ToolTracer:
    """
    Tracer for LangGraph tool executions.

    Emits tool.call (L5a) events for each tool invocation.

    Usage:
        tracer = ToolTracer(stratix_instance)

        @tracer.trace
        def my_tool(query: str) -> str:
            return search(query)
    """

    def __init__(
        self,
        stratix_instance: Any = None,
        adapter: BaseAdapter | None = None,
    ) -> None:
        """
        Initialize the tool tracer.

        Args:
            stratix_instance: STRATIX SDK instance (legacy)
            adapter: BaseAdapter instance (new-style). When provided,
                     typed event emission is used.
        """
        self._stratix = stratix_instance
        self._adapter = adapter
        self._executions: list[ToolExecution] = []

    def trace(self, func: ToolFunc) -> ToolFunc:
        """
        Decorate a tool function with tracing.

        Emits tool.call event capturing input/output.

        Args:
            func: Tool function

        Returns:
            Decorated function
        """
        tool_name = func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            execution = ToolExecution(
                tool_name=tool_name,
                start_time_ns=time.time_ns(),
                input_args=self._capture_input(args, kwargs),
            )
            self._executions.append(execution)

            try:
                result = func(*args, **kwargs)
                execution.end_time_ns = time.time_ns()
                execution.output = self._safe_output(result)
                self._emit_tool_call(execution)
                return result

            except Exception as e:
                execution.end_time_ns = time.time_ns()
                execution.error = str(e)
                self._emit_tool_call(execution)
                raise

        return wrapper

    def trace_async(self, func: ToolFunc) -> ToolFunc:
        """
        Decorate an async tool function with tracing.

        Args:
            func: Async tool function

        Returns:
            Decorated async function
        """
        tool_name = func.__name__

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            execution = ToolExecution(
                tool_name=tool_name,
                start_time_ns=time.time_ns(),
                input_args=self._capture_input(args, kwargs),
            )
            self._executions.append(execution)

            try:
                result = await func(*args, **kwargs)
                execution.end_time_ns = time.time_ns()
                execution.output = self._safe_output(result)
                self._emit_tool_call(execution)
                return result

            except Exception as e:
                execution.end_time_ns = time.time_ns()
                execution.error = str(e)
                self._emit_tool_call(execution)
                raise

        return wrapper

    def _capture_input(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Capture tool input arguments."""
        return {
            "args": [self._safe_serialize(a) for a in args],
            "kwargs": {k: self._safe_serialize(v) for k, v in kwargs.items()},
        }

    def _safe_serialize(self, value: Any) -> Any:
        """Safely serialize a value."""
        try:
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            elif isinstance(value, (list, tuple)):
                return [self._safe_serialize(v) for v in value]
            elif isinstance(value, dict):
                return {k: self._safe_serialize(v) for k, v in value.items()}
            else:
                return str(value)
        except Exception:
            return "<unserializable>"

    def _safe_output(self, value: Any) -> Any:
        """Safely capture output value."""
        return self._safe_serialize(value)

    def _emit_tool_call(self, execution: ToolExecution) -> None:
        """Emit tool.call event via adapter (preferred) or legacy path."""
        duration_ns = (execution.end_time_ns or 0) - execution.start_time_ns
        payload_dict = {
            "tool_name": execution.tool_name,
            "input": execution.input_args,
            "output": execution.output,
            "duration_ns": duration_ns,
            "error": execution.error,
        }

        # New-style: route through adapter.emit_event
        if self._adapter is not None:
            try:
                from layerlens.instrument._vendored.events import (
                    ToolCallEvent,
                    IntegrationType,
                )

                typed_payload = ToolCallEvent.create(  # type: ignore[call-arg,unused-ignore]
                    tool_name=execution.tool_name,
                    integration_type=IntegrationType.LIBRARY,
                    input_data=execution.input_args or {},
                    output_data=execution.output,
                    duration_ns=duration_ns,
                    error=execution.error,
                )
                self._adapter.emit_event(typed_payload)
                return
            except Exception:
                logger.debug("Typed event emission failed, falling back to legacy", exc_info=True)

        # Legacy fallback
        if self._stratix and hasattr(self._stratix, "emit"):
            self._stratix.emit("tool.call", payload_dict)


def trace_langgraph_tool(
    func: ToolFunc | None = None,
    *,
    stratix_instance: Any = None,
    adapter: BaseAdapter | None = None,
    tool_name: str | None = None,
) -> ToolFunc | Callable[[ToolFunc], ToolFunc]:
    """
    Decorator for tracing LangGraph tool functions.

    Can be used with or without arguments:

        @trace_langgraph_tool
        def my_tool(query: str) -> str:
            ...

        @trace_langgraph_tool(stratix_instance=stratix)
        def my_tool(query: str) -> str:
            ...

    Args:
        func: Tool function (when used without arguments)
        stratix_instance: STRATIX SDK instance
        adapter: BaseAdapter instance (new-style)
        tool_name: Custom name for the tool

    Returns:
        Decorated function or decorator
    """
    tracer = ToolTracer(stratix_instance, adapter=adapter)

    def decorator(f: ToolFunc) -> ToolFunc:
        name = tool_name or f.__name__

        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            execution = ToolExecution(
                tool_name=name,
                start_time_ns=time.time_ns(),
                input_args=tracer._capture_input(args, kwargs),
            )

            try:
                result = f(*args, **kwargs)
                execution.end_time_ns = time.time_ns()
                execution.output = tracer._safe_output(result)
                tracer._emit_tool_call(execution)
                return result

            except Exception as e:
                execution.end_time_ns = time.time_ns()
                execution.error = str(e)
                tracer._emit_tool_call(execution)
                raise

        return wrapper

    if func is not None:
        # Called without arguments: @trace_langgraph_tool
        return decorator(func)
    else:
        # Called with arguments: @trace_langgraph_tool(...)
        return decorator


class LangGraphToolNode:
    """
    Wrapper for creating traced LangGraph tool nodes.

    This creates a node that wraps a tool function and automatically
    emits tool.call events.

    Usage:
        # Create a traced tool node
        search_node = LangGraphToolNode(
            tool_func=search_function,
            stratix_instance=stratix,
        )

        # Use in graph
        graph.add_node("search", search_node)
    """

    def __init__(
        self,
        tool_func: ToolFunc,
        stratix_instance: Any = None,
        adapter: BaseAdapter | None = None,
        tool_name: str | None = None,
        state_key: str | None = None,
    ) -> None:
        """
        Initialize the tool node.

        Args:
            tool_func: The tool function to wrap
            stratix_instance: STRATIX SDK instance
            adapter: BaseAdapter instance (new-style)
            tool_name: Name for the tool (defaults to function name)
            state_key: Key in state to use as tool input (if None, uses full state)
        """
        self._tool_func = tool_func
        self._stratix = stratix_instance
        self._tool_name = tool_name or tool_func.__name__
        self._state_key = state_key
        self._tracer = ToolTracer(stratix_instance, adapter=adapter)

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the tool node.

        Args:
            state: LangGraph state

        Returns:
            Updated state
        """
        # Get input from state
        tool_input = state.get(self._state_key) if self._state_key else state

        execution = ToolExecution(
            tool_name=self._tool_name,
            start_time_ns=time.time_ns(),
            input_args={"state_input": self._tracer._safe_serialize(tool_input)},
        )

        try:
            # Call the tool
            result = self._tool_func(tool_input)
            execution.end_time_ns = time.time_ns()
            execution.output = self._tracer._safe_output(result)
            self._tracer._emit_tool_call(execution)

            # Return updated state
            return {"tool_output": result}

        except Exception as e:
            execution.end_time_ns = time.time_ns()
            execution.error = str(e)
            self._tracer._emit_tool_call(execution)
            raise
