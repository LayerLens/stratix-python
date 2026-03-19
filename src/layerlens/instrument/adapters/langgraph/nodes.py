"""
STRATIX LangGraph Node Tracing

Provides node entry/exit hooks and decorators for tracing node execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from layerlens.instrument.adapters.langgraph.state import LangGraphStateAdapter

if TYPE_CHECKING:
    from layerlens.instrument.adapters._base import BaseAdapter

logger = logging.getLogger(__name__)


StateT = TypeVar("StateT")
NodeFunc = Callable[[StateT], StateT]


@dataclass
class NodeExecution:
    """Tracks a single node execution."""
    node_name: str
    start_time_ns: int
    end_time_ns: int | None = None
    state_hash_before: str | None = None
    state_hash_after: str | None = None
    error: str | None = None


class NodeTracer:
    """
    Tracer for LangGraph node executions.

    Provides hooks for node entry/exit and automatic state change detection.

    Usage:
        tracer = NodeTracer(stratix_instance)

        # Manual tracking
        with tracer.trace_node("my_node", state):
            # Node logic here
            new_state = process(state)

        # Or use the decorator
        @tracer.decorate
        def my_node(state):
            return process(state)
    """

    def __init__(
        self,
        stratix_instance: Any = None,
        state_adapter: LangGraphStateAdapter | None = None,
        adapter: BaseAdapter | None = None,
    ):
        """
        Initialize the node tracer.

        Args:
            stratix_instance: STRATIX SDK instance (legacy)
            state_adapter: State adapter for change detection
            adapter: BaseAdapter instance (new-style)
        """
        self._stratix = stratix_instance
        self._adapter = adapter
        self._state_adapter = state_adapter or LangGraphStateAdapter()
        self._executions: list[NodeExecution] = []

    def trace_node(self, node_name: str, state: Any) -> "_NodeContext":
        """
        Create a context manager for tracing a node.

        Args:
            node_name: Name of the node
            state: Current state

        Returns:
            Context manager for node tracing
        """
        return _NodeContext(
            tracer=self,
            node_name=node_name,
            state=state,
        )

    def decorate(self, func: NodeFunc) -> NodeFunc:
        """
        Decorate a node function with tracing.

        Args:
            func: Node function

        Returns:
            Decorated function
        """
        node_name = func.__name__

        @wraps(func)
        def wrapper(state: StateT) -> StateT:
            with self.trace_node(node_name, state) as ctx:
                result = func(state)
                ctx.set_result(result)
                return result

        return wrapper

    def on_node_enter(self, node_name: str, state: Any) -> NodeExecution:
        """
        Called when entering a node.

        Emits agent.state.change event tracking entry.

        Args:
            node_name: Name of the node
            state: Current state

        Returns:
            NodeExecution tracking object
        """
        execution = NodeExecution(
            node_name=node_name,
            start_time_ns=time.time_ns(),
            state_hash_before=self._state_adapter.get_hash(state),
        )
        self._executions.append(execution)

        return execution

    def on_node_exit(
        self,
        execution: NodeExecution,
        state: Any,
        error: Exception | None = None,
    ) -> None:
        """
        Called when exiting a node.

        Emits agent.state.change event if state changed.

        Args:
            execution: Execution tracking object
            state: State after node execution
            error: Exception if node failed
        """
        execution.end_time_ns = time.time_ns()
        execution.state_hash_after = self._state_adapter.get_hash(state)

        if error:
            execution.error = str(error)

        # Emit state change event if state changed
        if execution.state_hash_before != execution.state_hash_after:
            self._emit_state_change(execution)

    def _emit_state_change(self, execution: NodeExecution) -> None:
        """Emit state change event via adapter (preferred) or legacy path."""
        duration_ns = (execution.end_time_ns or 0) - execution.start_time_ns

        # New-style: route through adapter.emit_event
        if self._adapter is not None:
            try:
                from layerlens.instrument.schema.events import AgentStateChangeEvent, StateType
                typed_payload = AgentStateChangeEvent.create(
                    state_type=StateType.INTERNAL,
                    before_hash=execution.state_hash_before or "sha256:" + "0" * 64,
                    after_hash=execution.state_hash_after or "sha256:" + "0" * 64,
                )
                self._adapter.emit_event(typed_payload)
                return
            except Exception:
                logger.debug("Typed event emission failed, falling back to legacy", exc_info=True)

        # Legacy fallback
        if self._stratix and hasattr(self._stratix, "emit"):
            self._stratix.emit("agent.state.change", {
                "node_name": execution.node_name,
                "before_hash": execution.state_hash_before,
                "after_hash": execution.state_hash_after,
                "duration_ns": duration_ns,
            })


class _NodeContext:
    """Context manager for node tracing."""

    def __init__(self, tracer: NodeTracer, node_name: str, state: Any):
        self._tracer = tracer
        self._node_name = node_name
        self._state = state
        self._result_state: Any = None
        self._execution: NodeExecution | None = None

    def __enter__(self) -> "_NodeContext":
        self._execution = self._tracer.on_node_enter(self._node_name, self._state)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._execution:
            # Use result state if set, otherwise use original state
            final_state = self._result_state if self._result_state is not None else self._state
            error = exc_val if exc_val else None
            self._tracer.on_node_exit(self._execution, final_state, error)

    def set_result(self, state: Any) -> None:
        """Set the result state for tracking."""
        self._result_state = state


def trace_node(
    stratix_instance: Any = None,
    state_adapter: LangGraphStateAdapter | None = None,
    adapter: BaseAdapter | None = None,
) -> Callable[[NodeFunc], NodeFunc]:
    """
    Decorator factory for tracing node functions.

    Usage:
        @trace_node(stratix)
        def my_node(state):
            return new_state

    Args:
        stratix_instance: STRATIX SDK instance
        state_adapter: State adapter for change detection
        adapter: BaseAdapter instance (new-style)

    Returns:
        Decorator function
    """
    tracer = NodeTracer(stratix_instance, state_adapter, adapter=adapter)

    def decorator(func: NodeFunc) -> NodeFunc:
        return tracer.decorate(func)

    return decorator


def create_traced_node(
    func: NodeFunc,
    stratix_instance: Any = None,
    adapter: BaseAdapter | None = None,
    node_name: str | None = None,
) -> NodeFunc:
    """
    Create a traced version of a node function.

    This is useful when you want to trace existing functions without
    modifying them.

    Args:
        func: Original node function
        stratix_instance: STRATIX SDK instance
        adapter: BaseAdapter instance (new-style)
        node_name: Name to use for tracing (defaults to function name)

    Returns:
        Traced node function
    """
    tracer = NodeTracer(stratix_instance, adapter=adapter)
    name = node_name or func.__name__

    @wraps(func)
    def traced_func(state: Any) -> Any:
        with tracer.trace_node(name, state) as ctx:
            result = func(state)
            ctx.set_result(result)
            return result

    return traced_func
