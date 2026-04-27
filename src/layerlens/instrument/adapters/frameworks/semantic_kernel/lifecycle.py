"""
STRATIX Semantic Kernel Lifecycle Hooks

Provides the main SemanticKernelAdapter class. Instruments SK Kernel
instances via the official filter API (FunctionInvocationFilter,
PromptRenderFilter, AutoFunctionInvocationFilter).
"""

from __future__ import annotations

import time
import uuid
import logging
import threading
from typing import Any

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

logger = logging.getLogger(__name__)


class SemanticKernelAdapter(BaseAdapter):
    """
    Main adapter for integrating STRATIX with Microsoft Semantic Kernel.

    Instruments Kernel instances via the official SK filter API to capture
    plugin invocations, planner executions, memory operations, and LLM calls.

    Usage:
        adapter = SemanticKernelAdapter(stratix=stratix_instance)
        adapter.connect()
        kernel = adapter.instrument_kernel(kernel)
        result = await kernel.invoke(my_function, arg1=val1)
    """

    FRAMEWORK = "semantic_kernel"
    VERSION = "0.1.0"
    # The adapter source files import nothing from ``pydantic`` directly
    # (verified by grep across ``frameworks/semantic_kernel/``). The
    # adapter only registers SK filter callbacks and emits dict events;
    # it never touches Semantic Kernel's own Pydantic models. SK 1.0+ is
    # internally Pydantic v2, but customers running older SK 0.x with
    # Pydantic v1 can still use this adapter.
    requires_pydantic = PydanticCompat.V1_OR_V2

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
        memory_service: Any | None = None,
        *,
        org_id: str | None = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config, org_id=org_id)

        self._adapter_lock = threading.Lock()
        self._seen_plugins: set[str] = set()
        self._invocation_count: int = 0
        self._kernel_start_ns: int = 0
        self._framework_version: str | None = None
        self._filters_registered: list[Any] = []
        self._memory_service = memory_service

    # --- BaseAdapter lifecycle ---

    def connect(self) -> None:
        """Verify Semantic Kernel is importable and mark as connected."""
        try:
            import semantic_kernel  # type: ignore[import-not-found,unused-ignore]  # noqa: F401

            version = getattr(semantic_kernel, "__version__", "unknown")
            logger.debug("Semantic Kernel %s detected", version)
        except ImportError:
            logger.debug("Semantic Kernel not installed; adapter usable in mock/test mode")
        self._framework_version = self._detect_framework_version()
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        """Disconnect and clear state."""
        self._filters_registered.clear()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def health_check(self) -> AdapterHealth:
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            framework_version=self._framework_version,
            adapter_version=self.VERSION,
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="SemanticKernelAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
            ],
            description="LayerLens adapter for Microsoft Semantic Kernel",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="SemanticKernelAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={
                "capture_config": self._capture_config.model_dump(),
            },
        )

    # --- Kernel instrumentation ---

    def instrument_kernel(self, kernel: Any) -> Any:
        """
        Instrument a Semantic Kernel instance with STRATIX tracing.

        Registers filter instances on the kernel for function invocations,
        prompt rendering, and auto-function invocations.

        Args:
            kernel: A semantic_kernel.Kernel instance

        Returns:
            The modified kernel (same object, with filters attached)
        """
        from layerlens.instrument.adapters.frameworks.semantic_kernel.filters import (
            LayerLensFunctionFilter,
            LayerLensAutoFunctionFilter,
            LayerLensPromptRenderFilter,
        )

        func_filter = LayerLensFunctionFilter(adapter=self)
        prompt_filter = LayerLensPromptRenderFilter(adapter=self)
        auto_filter = LayerLensAutoFunctionFilter(adapter=self)

        # Register filters via SK's filter API
        try:
            if hasattr(kernel, "add_filter"):
                kernel.add_filter("function_invocation", func_filter)
                kernel.add_filter("prompt_rendering", prompt_filter)
                kernel.add_filter("auto_function_invocation", auto_filter)
                self._filters_registered = [func_filter, prompt_filter, auto_filter]
            else:
                # Fallback: store on kernel for callback-based approach
                kernel._stratix_filters = [func_filter, prompt_filter, auto_filter]
                self._filters_registered = [func_filter, prompt_filter, auto_filter]
        except Exception:
            logger.warning("Could not register filters on kernel", exc_info=True)

        kernel._stratix_adapter = self

        # Discover registered plugins
        self._discover_plugins(kernel)

        return kernel

    # --- Lifecycle hooks (called by filters) ---

    def on_function_start(
        self,
        plugin_name: str,
        function_name: str,
        arguments: dict[str, Any] | None = None,
        auto_invoked: bool = False,
    ) -> dict[str, Any]:
        """
        Handle function invocation start.

        Returns context dict for correlation with on_function_end.
        """
        with self._adapter_lock:
            self._invocation_count += 1
            invocation_seq = self._invocation_count

        context = {
            "start_ns": time.time_ns(),
            "invocation_seq": invocation_seq,
            "plugin_name": plugin_name,
            "function_name": function_name,
        }

        # Emit agent config on first plugin encounter
        with self._adapter_lock:
            if plugin_name not in self._seen_plugins:
                self._seen_plugins.add(plugin_name)
                self.emit_dict_event(
                    "environment.config",
                    {
                        "framework": "semantic_kernel",
                        "plugin_name": plugin_name,
                        "function_name": function_name,
                    },
                )

        return context

    def on_function_end(
        self,
        context: dict[str, Any],
        result: Any = None,
        error: Exception | None = None,
        auto_invoked: bool = False,
    ) -> None:
        """
        Handle function invocation end.

        Emits tool.call (L5a) for plugin functions.
        """
        start_ns = context.get("start_ns", 0)
        elapsed_ms = (time.time_ns() - start_ns) / 1_000_000 if start_ns else 0

        payload: dict[str, Any] = {
            "framework": "semantic_kernel",
            "tool_name": f"{context.get('plugin_name', '')}.{context.get('function_name', '')}",
            "plugin_name": context.get("plugin_name"),
            "function_name": context.get("function_name"),
            "latency_ms": elapsed_ms,
            "invocation_seq": context.get("invocation_seq"),
        }

        if auto_invoked:
            payload["auto_invoked"] = True

        if result is not None:
            payload["result_preview"] = self._truncate(self._safe_serialize(result))

        if error:
            payload["error"] = str(error)

        self.emit_dict_event("tool.call", payload)

    def on_prompt_render(
        self,
        template: str | None = None,
        rendered_prompt: str | None = None,
        function_name: str | None = None,
    ) -> None:
        """
        Handle prompt template rendering.

        Emits agent.code (L2) for template rendering events.
        """
        payload: dict[str, Any] = {
            "framework": "semantic_kernel",
            "event_subtype": "prompt_render",
        }
        if function_name:
            payload["function_name"] = function_name
        if template:
            payload["template_preview"] = self._truncate(template, 500)
        if rendered_prompt:
            payload["rendered_preview"] = self._truncate(rendered_prompt, 500)

        self.emit_dict_event("agent.code", payload)

    def on_model_invoke(
        self,
        provider: str | None = None,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        """
        Handle LLM call from SK service.

        Emits model.invoke (L3) and cost.record (cross-cutting).
        """
        payload: dict[str, Any] = {
            "framework": "semantic_kernel",
        }
        if provider:
            payload["provider"] = provider
        if model:
            payload["model"] = model
        if prompt_tokens is not None:
            payload["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            payload["completion_tokens"] = completion_tokens
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if error:
            payload["error"] = error
        if self._capture_config.capture_content and messages:
            payload["messages"] = messages

        self.emit_dict_event("model.invoke", payload)

        # Emit cost record
        if prompt_tokens or completion_tokens:
            self.emit_dict_event(
                "cost.record",
                {
                    "framework": "semantic_kernel",
                    "provider": provider,
                    "model": model,
                    "prompt_tokens": prompt_tokens or 0,
                    "completion_tokens": completion_tokens or 0,
                    "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
                },
            )

    def on_planner_step(
        self,
        planner_type: str,
        step_index: int | None = None,
        plan: Any = None,
        thought: str | None = None,
        action: str | None = None,
        observation: str | None = None,
        status: str | None = None,
    ) -> None:
        """
        Handle planner execution step.

        Emits agent.code (L2) for plan generation and step execution.
        """
        payload: dict[str, Any] = {
            "framework": "semantic_kernel",
            "event_subtype": "planner_step",
            "planner_type": planner_type,
        }
        if step_index is not None:
            payload["step_index"] = step_index
        if plan is not None:
            payload["plan_preview"] = self._truncate(str(plan), 1000)
        if thought:
            payload["thought"] = self._truncate(thought)
        if action:
            payload["action"] = action
        if observation:
            payload["observation"] = self._truncate(observation)
        if status:
            payload["status"] = status

        self.emit_dict_event("agent.code", payload)

    def on_memory_operation(
        self,
        operation: str,
        collection: str | None = None,
        key: str | None = None,
        query: str | None = None,
        result_count: int | None = None,
        relevance_scores: list[float] | None = None,
        backend_type: str | None = None,
    ) -> None:
        """
        Handle memory operation (save, search, get).

        Emits tool.call (L5a) for memory operations.
        """
        payload: dict[str, Any] = {
            "framework": "semantic_kernel",
            "tool_name": f"memory.{operation}",
            "operation": operation,
        }
        if collection:
            payload["collection"] = collection
        if key:
            payload["key"] = key
        if query:
            payload["query_preview"] = self._truncate(query, 200)
        if result_count is not None:
            payload["result_count"] = result_count
        if relevance_scores:
            payload["relevance_scores"] = relevance_scores[:10]
        if backend_type:
            payload["backend_type"] = backend_type

        self.emit_dict_event("tool.call", payload)

    def on_kernel_invoke_start(self, input_text: Any = None) -> None:
        """Handle kernel invocation start. Emits agent.input (L1)."""
        with self._adapter_lock:
            self._kernel_start_ns = time.time_ns()

        self.emit_dict_event(
            "agent.input",
            {
                "framework": "semantic_kernel",
                "input": self._safe_serialize(input_text),
                "timestamp_ns": self._kernel_start_ns,
            },
        )

    def on_kernel_invoke_end(
        self,
        output: Any = None,
        error: Exception | None = None,
    ) -> None:
        """Handle kernel invocation end. Emits agent.output (L1)."""
        end_ns = time.time_ns()
        duration_ns = end_ns - self._kernel_start_ns if self._kernel_start_ns else 0

        payload: dict[str, Any] = {
            "framework": "semantic_kernel",
            "output": self._safe_serialize(output),
            "duration_ns": duration_ns,
        }
        if error:
            payload["error"] = str(error)

        self.emit_dict_event("agent.output", payload)

    # --- Plugin discovery ---

    def _discover_plugins(self, kernel: Any) -> None:
        """Discover and register plugins from the kernel."""
        try:
            plugins = getattr(kernel, "plugins", None)
            if plugins is None:
                return
            if isinstance(plugins, dict) or hasattr(plugins, "keys"):
                plugin_names = list(plugins.keys())
            else:
                plugin_names = [str(p) for p in plugins]

            for name in plugin_names:
                with self._adapter_lock:
                    if name not in self._seen_plugins:
                        self._seen_plugins.add(name)
                        self.emit_dict_event(
                            "environment.config",
                            {
                                "framework": "semantic_kernel",
                                "plugin_name": name,
                                "event_subtype": "plugin_registered",
                            },
                        )
        except Exception:
            logger.debug("Error discovering SK plugins", exc_info=True)

    # --- Internal helpers ---

    def _safe_serialize(self, value: Any) -> Any:
        """Safely serialize a value for events."""
        try:
            if value is None:
                return None
            if hasattr(value, "model_dump"):
                return value.model_dump()
            if hasattr(value, "dict"):
                return value.dict()
            if isinstance(value, dict):
                return dict(value)
            if isinstance(value, (str, int, float, bool)):
                return value
            return str(value)
        except Exception:
            return str(value)

    def _truncate(self, text: Any, max_len: int = 500) -> str:
        """Truncate text to max_len."""
        text_str = str(text) if not isinstance(text, str) else text
        if len(text_str) <= max_len:
            return text_str
        return text_str[:max_len] + "..."

    @staticmethod
    def _detect_framework_version() -> str | None:
        try:
            import semantic_kernel  # type: ignore[import-not-found,unused-ignore]

            return getattr(semantic_kernel, "__version__", None)
        except ImportError:
            return None


class StratixMemoryStore:
    """Semantic Kernel memory store backed by AgentMemoryService.

    Implements the SK memory store interface (``save_information``,
    ``get_nearest_matches``) by delegating to the STRATIX
    ``AgentMemoryService``.  This allows SK applications to use
    STRATIX persistent memory without changing their code.

    Usage::

        from stratix.memory.service import AgentMemoryService  # type: ignore[import-not-found,import-untyped,unused-ignore]

        memory_svc = AgentMemoryService(crud_store)
        store = StratixMemoryStore(memory_svc, agent_id="my-agent", org_id="org-1")

        # Inside SK:
        await store.save_information(
            collection="facts",
            text="Paris is the capital of France",
            id="fact-1",
        )
        matches = await store.get_nearest_matches(
            collection="facts",
            query="capital of France",
            limit=3,
        )
    """

    def __init__(
        self,
        memory_service: Any,
        agent_id: str = "semantic_kernel",
        org_id: str = "",
    ) -> None:
        """Initialise the memory store.

        Args:
            memory_service: An ``AgentMemoryService`` instance.
            agent_id: Agent identifier used for all memory entries.
            org_id: Organisation identifier used for all memory entries.
        """
        self._memory_service = memory_service
        self._agent_id = agent_id
        self._org_id = org_id

    async def save_information(
        self,
        collection: str,
        text: str,
        id: str,  # noqa: A002  — matches SK interface
        description: str | None = None,
        additional_metadata: str | None = None,
    ) -> None:
        """Save a piece of information into the memory store.

        Delegates to ``AgentMemoryService.store()`` with
        ``memory_type="semantic"`` and the collection as namespace.

        Args:
            collection: SK memory collection name (mapped to namespace).
            text: Text content to store.
            id: Unique identifier for this memory.
            description: Optional description (stored in metadata).
            additional_metadata: Optional extra metadata string.
        """
        from layerlens.instrument._vendored.memory_models import MemoryEntry

        metadata: dict[str, Any] = {"source": "semantic_kernel_memory_store"}
        if description:
            metadata["description"] = description
        if additional_metadata:
            metadata["additional"] = additional_metadata

        entry = MemoryEntry(
            id=id,
            org_id=self._org_id,
            agent_id=self._agent_id,
            memory_type="semantic",
            namespace=collection,
            key=id,
            content=text,
            importance=0.5,
            metadata=metadata,
        )
        self._memory_service.store(entry)

    async def get_nearest_matches(
        self,
        collection: str,
        query: str,
        limit: int = 5,
        min_relevance_score: float = 0.0,
    ) -> list[tuple[Any, float]]:
        """Retrieve the nearest matches for a query.

        Delegates to ``AgentMemoryService.search()`` and returns results
        in the SK-expected format of ``(MemoryEntry, relevance_score)``
        tuples.

        Args:
            collection: SK memory collection name (used as search context).
            query: Text query to match against memory content.
            limit: Maximum number of results to return.
            min_relevance_score: Minimum relevance threshold (reserved for
                future vector search support; currently unused).

        Returns:
            List of ``(MemoryEntry, score)`` tuples ordered by importance.
        """
        results = self._memory_service.search(self._agent_id, query, limit=limit)
        # Filter to the requested collection/namespace
        filtered = [r for r in results if r.namespace == collection]
        # Return as (entry, relevance) tuples — importance serves as proxy score
        return [(entry, entry.importance) for entry in filtered]
