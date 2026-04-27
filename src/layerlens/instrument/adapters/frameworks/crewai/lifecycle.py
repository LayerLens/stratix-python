"""
STRATIX CrewAI Lifecycle Hooks

Provides the main CrewAIAdapter class and crew instrumentation.
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
from layerlens.instrument.adapters.frameworks.crewai.metadata import AgentMetadataExtractor
from layerlens.instrument.adapters.frameworks.crewai.delegation import CrewDelegationTracker

logger = logging.getLogger(__name__)


class CrewAIAdapter(BaseAdapter):
    """
    Main adapter for integrating STRATIX with CrewAI.

    Instruments CrewAI crews, agents, and tasks to emit STRATIX telemetry events.
    Uses the CrewAI callback protocol (v0.41+) via LayerLensCrewCallback.

    Supports both new-style (stratix, capture_config) and legacy (stratix_instance)
    constructor parameters.

    Usage:
        adapter = CrewAIAdapter(stratix=stratix_instance)
        adapter.connect()
        instrumented_crew = adapter.instrument_crew(my_crew)
        result = instrumented_crew.kickoff()
    """

    FRAMEWORK = "crewai"
    VERSION = "0.1.0"
    # CrewAI >=0.30 (pyproject pin: crewai>=0.30,<0.90) is Pydantic v2
    # only — see crewai's pyproject which pins ``pydantic = "^2.4.2"``.
    # Importing crewai under v1 fails inside crewai's own model layer.
    requires_pydantic = PydanticCompat.V2_ONLY

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
        # Legacy param
        stratix_instance: Any | None = None,
        memory_service: Any | None = None,
        *,
        org_id: str | None = None,
    ) -> None:
        resolved_stratix = stratix or stratix_instance
        super().__init__(stratix=resolved_stratix, capture_config=capture_config, org_id=org_id)

        self._metadata_extractor = AgentMetadataExtractor()
        self._delegation_tracker = CrewDelegationTracker(self)
        self._adapter_lock = threading.Lock()
        self._seen_agents: set[str] = set()
        self._crew_start_ns: int = 0
        self._framework_version: str | None = None
        self._memory_service = memory_service

    # --- BaseAdapter lifecycle ---

    def connect(self) -> None:
        """Verify CrewAI is importable and mark as connected."""
        try:
            import crewai  # type: ignore[import-not-found,unused-ignore]  # noqa: F401

            version = getattr(crewai, "__version__", "unknown")
            logger.debug("CrewAI %s detected", version)
        except ImportError:
            logger.debug("CrewAI not installed; adapter usable in mock/test mode")
        self._framework_version = self._detect_framework_version()
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
            framework_version=self._framework_version,
            adapter_version=self.VERSION,
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="CrewAIAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_HANDOFFS,
            ],
            description="LayerLens adapter for CrewAI agent framework",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="CrewAIAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={
                "capture_config": self._capture_config.model_dump(),
            },
        )

    # --- Crew instrumentation ---

    def instrument_crew(self, crew: Any) -> Any:
        """
        Instrument a CrewAI Crew with STRATIX tracing.

        Registers LayerLensCrewCallback on the crew. Records process type
        and agent metadata.

        Args:
            crew: A CrewAI Crew instance

        Returns:
            The modified crew (same object, with callback attached)
        """
        from layerlens.instrument.adapters.frameworks.crewai.callbacks import LayerLensCrewCallback

        callback = LayerLensCrewCallback(adapter=self)

        # Record process type
        process_type = getattr(crew, "process", None)
        if process_type is not None:
            process_type = str(process_type)

        # Attach callback - CrewAI supports step_callback and task_callback
        try:
            if hasattr(crew, "step_callback"):
                crew.step_callback = callback.on_step
            if hasattr(crew, "task_callback"):
                crew.task_callback = callback.on_task_complete
        except Exception:
            logger.debug("Could not attach callbacks to crew", exc_info=True)

        # Store callback reference for lifecycle hooks
        crew._stratix_callback = callback
        crew._stratix_adapter = self

        # Extract agent metadata on first encounter
        agents = getattr(crew, "agents", []) or []
        for agent in agents:
            self._emit_agent_config(agent, process_type)

        return crew

    # --- Lifecycle hooks (called by callback) ---

    def on_crew_start(self, crew_input: Any = None) -> None:
        """
        Handle crew execution start.

        Emits agent.input (L1).
        """
        with self._adapter_lock:
            self._crew_start_ns = time.time_ns()

        self.emit_dict_event(
            "agent.input",
            {
                "framework": "crewai",
                "input": self._safe_serialize(crew_input),
                "timestamp_ns": self._crew_start_ns,
            },
        )

    def on_crew_end(
        self,
        crew_output: Any = None,
        error: Exception | None = None,
    ) -> None:
        """
        Handle crew execution end.

        Emits agent.output (L1).
        """
        end_ns = time.time_ns()
        duration_ns = end_ns - self._crew_start_ns if self._crew_start_ns else 0

        payload: dict[str, Any] = {
            "framework": "crewai",
            "output": self._safe_serialize(crew_output),
            "duration_ns": duration_ns,
        }
        if error:
            payload["error"] = str(error)

        self.emit_dict_event("agent.output", payload)

    def on_task_start(
        self,
        task_description: str,
        agent_role: str | None = None,
        expected_output: str | None = None,
        task_order: int | None = None,
    ) -> None:
        """
        Handle task start.

        Emits agent.code (L2) as dict event with task metadata.
        """
        payload: dict[str, Any] = {
            "framework": "crewai",
            "task_description": task_description,
            "event_subtype": "task_start",
        }
        if agent_role:
            payload["agent_role"] = agent_role
        if expected_output:
            payload["expected_output"] = expected_output
        if task_order is not None:
            payload["task_order"] = task_order

        self.emit_dict_event("agent.code", payload)

    def on_task_end(
        self,
        task_output: Any = None,
        agent_role: str | None = None,
        task_order: int | None = None,
        error: Exception | None = None,
    ) -> None:
        """
        Handle task completion.

        Emits agent.state.change (cross-cutting) and cost.record (cross-cutting)
        if token costs are available.
        """
        payload: dict[str, Any] = {
            "framework": "crewai",
            "task_output": self._safe_serialize(task_output),
            "event_subtype": "task_complete",
        }
        if agent_role:
            payload["agent_role"] = agent_role
        if task_order is not None:
            payload["task_order"] = task_order
        if error:
            payload["error"] = str(error)

        self.emit_dict_event("agent.state.change", payload)

        # Emit cost record if token usage available
        token_usage = self._extract_token_usage(task_output)
        if token_usage:
            self.emit_dict_event(
                "cost.record",
                {
                    "framework": "crewai",
                    "agent_role": agent_role,
                    **token_usage,
                },
            )

    def on_tool_use(
        self,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        error: Exception | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """
        Handle tool usage.

        Emits tool.call (L5a).
        """
        payload: dict[str, Any] = {
            "framework": "crewai",
            "tool_name": tool_name,
            "tool_input": self._safe_serialize(tool_input),
            "tool_output": self._safe_serialize(tool_output),
        }
        if error:
            payload["error"] = str(error)
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms

        self.emit_dict_event("tool.call", payload)

    def on_llm_call(
        self,
        provider: str | None = None,
        model: str | None = None,
        tokens_prompt: int | None = None,
        tokens_completion: int | None = None,
        latency_ms: float | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        """
        Handle LLM invocation.

        Emits model.invoke (L3).
        """
        payload: dict[str, Any] = {
            "framework": "crewai",
        }
        if provider:
            payload["provider"] = provider
        if model:
            payload["model"] = model
        if tokens_prompt is not None:
            payload["tokens_prompt"] = tokens_prompt
        if tokens_completion is not None:
            payload["tokens_completion"] = tokens_completion
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if self._capture_config.capture_content and messages:
            payload["messages"] = messages

        self.emit_dict_event("model.invoke", payload)

    def on_delegation(
        self,
        from_agent: str,
        to_agent: str,
        context: Any = None,
    ) -> None:
        """
        Handle agent delegation.

        Emits agent.handoff (cross-cutting, always enabled).
        """
        self._delegation_tracker.track_delegation(from_agent, to_agent, context)

    # --- Memory integration ---

    def inject_memory_context(
        self,
        agent_id: str,
        task_context: str,
    ) -> str:
        """Retrieve relevant semantic memories and prepend them to the task context.

        When no ``memory_service`` is configured the original ``task_context``
        is returned unmodified (backward compatible).

        Args:
            agent_id: Agent whose memories to retrieve.
            task_context: Original task context string.

        Returns:
            Enriched context with relevant memories prepended, or the
            original context when the memory service is unavailable or
            no memories are found.
        """
        if self._memory_service is None:
            return task_context

        try:
            memories = self._memory_service.search(agent_id, task_context, limit=5)
            if not memories:
                return task_context

            memory_lines = [f"- [{m.key}]: {m.content[:200]}" for m in memories]
            header = "Relevant memories:\n" + "\n".join(memory_lines) + "\n\n"
            return header + task_context
        except Exception:
            logger.debug(
                "Failed to inject memory context for agent %s",
                agent_id,
                exc_info=True,
            )
            return task_context

    def store_task_result(
        self,
        agent_id: str,
        task_name: str,
        result: Any,
    ) -> None:
        """Store a task result as procedural memory.

        Only active when ``memory_service`` is provided. Failures are
        logged and swallowed.

        Args:
            agent_id: Agent that completed the task.
            task_name: Name or description of the task.
            result: Task result to persist.
        """
        if self._memory_service is None:
            return

        try:
            from layerlens.instrument._vendored.memory_models import MemoryEntry

            content = self._safe_serialize(result)
            entry = MemoryEntry(
                org_id=getattr(self._stratix, "org_id", ""),
                agent_id=agent_id,
                memory_type="procedural",
                key=f"task_result_{task_name}",
                content=str(content),
                importance=0.6,
                metadata={"source": "crewai_adapter", "task_name": task_name},
            )
            self._memory_service.store(entry)
        except Exception:
            logger.debug(
                "Failed to store task result memory for agent %s task %s",
                agent_id,
                task_name,
                exc_info=True,
            )

    # --- Agent config emission ---

    def _emit_agent_config(
        self,
        agent: Any,
        process_type: str | None = None,
    ) -> None:
        """Emit environment.config for an agent on first encounter."""
        role = getattr(agent, "role", None) or str(agent)
        with self._adapter_lock:
            if role in self._seen_agents:
                return
            self._seen_agents.add(role)

        metadata = self._metadata_extractor.extract(agent)
        if process_type:
            metadata["process_type"] = process_type

        self.emit_dict_event(
            "environment.config",
            {
                "framework": "crewai",
                "agent_role": role,
                **metadata,
            },
        )

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

    def _extract_token_usage(self, task_output: Any) -> dict[str, Any] | None:
        """Extract token usage from task output if available."""
        if task_output is None:
            return None
        try:
            usage = getattr(task_output, "token_usage", None)
            if usage and isinstance(usage, dict):
                return {
                    "tokens_prompt": usage.get("prompt_tokens"),
                    "tokens_completion": usage.get("completion_tokens"),
                    "tokens_total": usage.get("total_tokens"),
                }
        except Exception:
            pass
        return None

    @staticmethod
    def _detect_framework_version() -> str | None:
        try:
            import crewai  # type: ignore[import-not-found,unused-ignore]

            return getattr(crewai, "__version__", None)
        except ImportError:
            return None
