"""
STRATIX LangGraph Lifecycle Hooks

Provides graph start/end hooks for STRATIX tracing.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, TypeVar
from dataclasses import field, dataclass

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat
from layerlens.instrument.adapters.frameworks.langgraph.state import LangGraphStateAdapter

if TYPE_CHECKING:
    from layerlens.instrument.adapters.frameworks.langgraph.handoff import HandoffDetector


StateT = TypeVar("StateT")
GraphT = TypeVar("GraphT")


@dataclass
class GraphExecution:
    """Represents a single graph execution."""

    graph_id: str
    execution_id: str
    start_time_ns: int
    end_time_ns: int | None = None
    initial_state_hash: str | None = None
    final_state_hash: str | None = None
    node_executions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class LayerLensLangGraphAdapter(BaseAdapter):
    """
    Main adapter for integrating STRATIX with LangGraph.

    This adapter wraps LangGraph graphs to automatically emit STRATIX events
    for graph execution, node transitions, and state changes.

    Supports both the new BaseAdapter interface and the legacy constructor
    for backward compatibility.

    Usage (new):
        from stratix import STRATIX
        from layerlens.instrument.adapters.frameworks.langgraph import LayerLensLangGraphAdapter

        stratix = STRATIX(policy_ref="my-policy")
        adapter = LayerLensLangGraphAdapter(stratix=stratix)
        adapter.connect()
        traced_graph = adapter.wrap_graph(my_graph)
        result = traced_graph.invoke(initial_state)

    Usage (legacy — still supported):
        adapter = LayerLensLangGraphAdapter(stratix_instance=stratix)
        traced_graph = adapter.wrap_graph(my_graph)
    """

    FRAMEWORK = "langgraph"
    VERSION = "0.1.0"
    # LangGraph >=0.2 (pyproject pin: langgraph>=0.2,<0.4) depends on
    # langchain-core>=0.2 which is Pydantic v2 only. LangGraph's own
    # state schema (StateGraph, MessagesState) uses v2-style typing.
    requires_pydantic = PydanticCompat.V2_ONLY

    def __init__(
        self,
        # New-style params
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
        handoff_detector: HandoffDetector | None = None,
        # Legacy params (backward compat)
        stratix_instance: Any | None = None,
        state_adapter: LangGraphStateAdapter | None = None,
        emit_environment_config: bool = True,
        emit_agent_code: bool = False,
    ) -> None:
        """
        Initialize the LangGraph adapter.

        Accepts both new-style (stratix, capture_config) and legacy-style
        (stratix_instance, boolean flags) parameters. When legacy params are
        provided, they are mapped to CaptureConfig equivalents.

        Args:
            stratix: STRATIX SDK instance (new-style)
            capture_config: CaptureConfig (new-style)
            handoff_detector: HandoffDetector for automatic handoff detection
                during node transitions (optional)
            stratix_instance: STRATIX SDK instance (legacy)
            state_adapter: Custom state adapter (uses default if not provided)
            emit_environment_config: Whether to emit environment.config (legacy)
            emit_agent_code: Whether to emit agent.code (legacy)
        """
        # Resolve STRATIX instance: new-style takes priority
        resolved_stratix = stratix or stratix_instance

        # Map legacy booleans → CaptureConfig when any flag differs from default
        if capture_config is None:
            any_legacy = not emit_environment_config or emit_agent_code
            if any_legacy or stratix_instance is not None:
                capture_config = CaptureConfig(
                    l4a_environment_config=emit_environment_config,
                    l2_agent_code=emit_agent_code,
                )

        super().__init__(stratix=resolved_stratix, capture_config=capture_config)

        self._state_adapter = state_adapter or LangGraphStateAdapter()
        self._executions: list[GraphExecution] = []
        self._handoff_detector: HandoffDetector | None = handoff_detector

        # Legacy compat: keep booleans accessible for code that reads them
        self._emit_environment_config = emit_environment_config
        self._emit_agent_code = emit_agent_code

    # --- BaseAdapter lifecycle ---

    def connect(self) -> None:
        """Verify LangGraph is importable and mark as connected."""
        try:
            import langgraph  # type: ignore[import-not-found,unused-ignore]  # noqa: F401

            self._connected = True
            self._status = AdapterStatus.HEALTHY
        except ImportError:
            # Still usable without LangGraph installed (for mock/test use)
            self._connected = True
            self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        """Flush and disconnect."""
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def health_check(self) -> AdapterHealth:
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            framework_version=self._detect_framework_version(),
            adapter_version=self.VERSION,
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )

    def get_adapter_info(self) -> AdapterInfo:
        # Round-2 deliberation item 20: surface the v2-only requirement on
        # both ``info()`` (the manifest wrapper that re-applies the class
        # attribute) and ``get_adapter_info()`` (the subclass override) so
        # the atlas-app catalog UI shows the correct compat badge regardless
        # of which entrypoint the manifest emitter ends up calling.
        return AdapterInfo(
            name="LayerLensLangGraphAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._detect_framework_version(),
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_HANDOFFS,
                AdapterCapability.REPLAY,
            ],
            description="LayerLens adapter for LangGraph agent framework",
            requires_pydantic=PydanticCompat.V2_ONLY,
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        trace_id = str(uuid.uuid4())
        return ReplayableTrace(
            adapter_name="LayerLensLangGraphAdapter",
            framework=self.FRAMEWORK,
            trace_id=trace_id,
            events=list(self._trace_events),
            state_snapshots=[],
            config={
                "capture_config": self._capture_config.model_dump(),
            },
        )

    # --- Handoff detection ---

    def set_handoff_detector(self, detector: HandoffDetector) -> None:
        """
        Attach a HandoffDetector to this adapter.

        When set, ``on_node_start`` will automatically call
        ``detector.detect_handoff(node_name, state)`` on every
        node transition, emitting handoff events when the active
        agent changes.

        Args:
            detector: HandoffDetector instance (should already have
                agents registered via ``register_agent`` /
                ``register_agents``)
        """
        self._handoff_detector = detector

    @property
    def handoff_detector(self) -> HandoffDetector | None:
        """Return the attached HandoffDetector, or None."""
        return self._handoff_detector

    # --- Graph wrapping ---

    def wrap_graph(self, graph: GraphT) -> GraphT:
        """
        Wrap a LangGraph compiled graph with STRATIX tracing.

        Args:
            graph: Compiled LangGraph graph

        Returns:
            Wrapped graph with same interface
        """
        return _TracedGraph(  # type: ignore[return-value]
            graph=graph,
            adapter=self,
            state_adapter=self._state_adapter,
        )

    # --- Lifecycle hooks ---

    def on_graph_start(
        self,
        graph_id: str,
        execution_id: str,
        initial_state: Any,
        config: dict[str, Any] | None = None,
    ) -> GraphExecution:
        """
        Handle graph execution start.

        Emits:
        - environment.config (if enabled)
        - agent.input

        Args:
            graph_id: Identifier for the graph
            execution_id: Unique execution ID
            initial_state: Initial graph state
            config: Graph execution config

        Returns:
            GraphExecution tracking object
        """
        execution = GraphExecution(
            graph_id=graph_id,
            execution_id=execution_id,
            start_time_ns=time.time_ns(),
            initial_state_hash=self._state_adapter.get_hash(initial_state),
        )
        self._executions.append(execution)

        # Emit environment config (gated by CaptureConfig inside emit_dict_event)
        self.emit_dict_event(
            "environment.config",
            {
                "framework": "langgraph",
                "graph_id": graph_id,
                "config": config,
            },
        )

        # Emit agent input
        self.emit_dict_event(
            "agent.input",
            {
                "graph_id": graph_id,
                "execution_id": execution_id,
                "initial_state": self._safe_serialize(initial_state),
            },
        )

        return execution

    def on_graph_end(
        self,
        execution: GraphExecution,
        final_state: Any,
        error: Exception | None = None,
    ) -> None:
        """
        Handle graph execution end.

        Emits:
        - agent.output
        - agent.state.change (if state changed)

        Args:
            execution: Execution tracking object
            final_state: Final graph state
            error: Exception if execution failed
        """
        execution.end_time_ns = time.time_ns()
        execution.final_state_hash = self._state_adapter.get_hash(final_state)

        if error:
            execution.error = str(error)

        # Emit agent output (gated by CaptureConfig inside emit_dict_event)
        self.emit_dict_event(
            "agent.output",
            {
                "graph_id": execution.graph_id,
                "execution_id": execution.execution_id,
                "final_state": self._safe_serialize(final_state),
                "duration_ns": execution.end_time_ns - execution.start_time_ns,
                "error": execution.error,
            },
        )

        # Emit state change if state changed (cross-cutting — always enabled)
        if execution.initial_state_hash != execution.final_state_hash:
            self.emit_dict_event(
                "agent.state.change",
                {
                    "graph_id": execution.graph_id,
                    "execution_id": execution.execution_id,
                    "before_hash": execution.initial_state_hash,
                    "after_hash": execution.final_state_hash,
                },
            )

    def on_node_start(
        self,
        execution: GraphExecution,
        node_name: str,
        state: Any,
    ) -> dict[str, Any]:
        """
        Handle node execution start.

        If a HandoffDetector is attached, automatically feeds the node
        transition to it so that agent-to-agent handoffs are detected
        and emitted.

        Args:
            execution: Execution tracking object
            node_name: Name of the node
            state: Current state

        Returns:
            Node execution context for tracking
        """
        node_context = {
            "node_name": node_name,
            "start_time_ns": time.time_ns(),
            "state_hash_before": self._state_adapter.get_hash(state),
        }

        if self._handoff_detector is not None:
            self._handoff_detector.detect_handoff(
                node_name,
                state if isinstance(state, dict) else None,
            )

        return node_context

    def on_node_end(
        self,
        execution: GraphExecution,
        node_context: dict[str, Any],
        state: Any,
        error: Exception | None = None,
    ) -> None:
        """
        Handle node execution end.

        Emits:
        - agent.state.change (if state changed at this node)

        Args:
            execution: Execution tracking object
            node_context: Node context from on_node_start
            state: State after node execution
            error: Exception if node failed
        """
        node_context["end_time_ns"] = time.time_ns()
        node_context["state_hash_after"] = self._state_adapter.get_hash(state)
        node_context["duration_ns"] = node_context["end_time_ns"] - node_context["start_time_ns"]

        if error:
            node_context["error"] = str(error)

        execution.node_executions.append(node_context)

        # Emit state change if node modified state (cross-cutting — always enabled)
        if node_context["state_hash_before"] != node_context["state_hash_after"]:
            self.emit_dict_event(
                "agent.state.change",
                {
                    "graph_id": execution.graph_id,
                    "execution_id": execution.execution_id,
                    "node_name": node_context["node_name"],
                    "before_hash": node_context["state_hash_before"],
                    "after_hash": node_context["state_hash_after"],
                },
            )

    # --- Internal helpers ---

    def _safe_serialize(self, value: Any) -> Any:
        """Safely serialize a value for events."""
        try:
            if hasattr(value, "dict"):
                return value.dict()
            elif isinstance(value, dict):
                return dict(value)
            else:
                return str(value)
        except Exception:
            return str(value)

    @staticmethod
    def _detect_framework_version() -> str | None:
        try:
            import langgraph  # type: ignore[import-not-found,unused-ignore]

            return getattr(langgraph, "__version__", None)
        except ImportError:
            return None


class _TracedGraph:
    """
    Wrapper around a LangGraph compiled graph that adds STRATIX tracing.
    """

    def __init__(
        self,
        graph: Any,
        adapter: LayerLensLangGraphAdapter,
        state_adapter: LangGraphStateAdapter,
    ) -> None:
        self._graph = graph
        self._adapter = adapter
        self._state_adapter = state_adapter
        self._execution_count = 0

    def invoke(self, state: Any, config: dict[str, Any] | None = None) -> Any:
        """
        Invoke the graph with tracing.

        Args:
            state: Initial state
            config: Execution config

        Returns:
            Final state
        """
        self._execution_count += 1
        graph_id = self._get_graph_id()
        execution_id = f"{graph_id}:{self._execution_count}"

        # Start tracking
        execution = self._adapter.on_graph_start(
            graph_id=graph_id,
            execution_id=execution_id,
            initial_state=state,
            config=config,
        )

        try:
            # Execute the actual graph
            result = self._graph.invoke(state, config)

            # End tracking
            self._adapter.on_graph_end(execution, result)

            return result

        except Exception as e:
            # End tracking with error
            self._adapter.on_graph_end(execution, state, error=e)
            raise

    async def ainvoke(self, state: Any, config: dict[str, Any] | None = None) -> Any:
        """
        Async invoke the graph with tracing.

        Args:
            state: Initial state
            config: Execution config

        Returns:
            Final state
        """
        self._execution_count += 1
        graph_id = self._get_graph_id()
        execution_id = f"{graph_id}:{self._execution_count}"

        # Start tracking
        execution = self._adapter.on_graph_start(
            graph_id=graph_id,
            execution_id=execution_id,
            initial_state=state,
            config=config,
        )

        try:
            # Execute the actual graph
            result = await self._graph.ainvoke(state, config)

            # End tracking
            self._adapter.on_graph_end(execution, result)

            return result

        except Exception as e:
            # End tracking with error
            self._adapter.on_graph_end(execution, state, error=e)
            raise

    def _get_graph_id(self) -> str:
        """Get the graph identifier."""
        if hasattr(self._graph, "name"):
            return self._graph.name  # type: ignore[no-any-return]
        elif hasattr(self._graph, "__class__"):
            return self._graph.__class__.__name__  # type: ignore[no-any-return]
        return "langgraph"

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to underlying graph."""
        return getattr(self._graph, name)
