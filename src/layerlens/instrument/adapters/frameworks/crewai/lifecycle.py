"""
STRATIX CrewAI Lifecycle Hooks

Provides the main CrewAIAdapter class and crew instrumentation.

Typed-event status (post PR #129 migration, bundle 1):

* Every emission flows through :meth:`BaseAdapter.emit_event` with a
  canonical Pydantic payload imported from
  :mod:`layerlens.instrument._compat.events`.
* CrewAI-specific provenance (``framework``, ``agent_role``,
  ``task_description``, ``task_order``, ``event_subtype``,
  ``process_type``) is carried in the canonical model's metadata /
  attributes / parameters / input slots — the canonical schema does
  not expose these as top-level fields.
* The previous adapter emitted an ad-hoc ``agent.code`` event for
  task-start (not in the canonical 13-event taxonomy). The typed
  migration maps the task-start boundary onto :class:`AgentInputEvent`
  with ``role=AGENT`` (the task description is logically input to the
  receiving agent) and carries the original ``event_subtype="task_start"``
  marker on :class:`MessageContent.metadata`.
* The previous adapter emitted an ``agent.state.change`` for task-end
  with no real ``before_hash`` / ``after_hash`` (the canonical schema
  rejects partial-hash payloads). The typed migration maps task-end
  onto :class:`AgentOutputEvent` with ``run_status=task_complete`` (or
  ``task_failed``) on :class:`MessageContent.metadata`.
"""

from __future__ import annotations

import time
import uuid
import logging
import threading
from typing import Any

from layerlens.instrument._compat.events import (
    MessageRole,
    ToolCallEvent,
    AgentInputEvent,
    CostRecordEvent,
    EnvironmentType,
    IntegrationType,
    AgentOutputEvent,
    ModelInvokeEvent,
    EnvironmentConfigEvent,
)
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


def _stringify(value: Any) -> str:
    """Return a string view of ``value`` suitable for the canonical
    :class:`MessageContent.message` field.

    The canonical schema requires :class:`AgentInputEvent` and
    :class:`AgentOutputEvent` to carry a ``message: str``. CrewAI
    callbacks deliver inputs/outputs as arbitrary Python objects
    (TaskOutput, dicts, ``None``); this helper converts each to a
    (possibly empty) string so the typed event always validates. The
    original payload is preserved on
    :class:`MessageContent.metadata.raw_input` / ``raw_output``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # CrewAI TaskOutput.model_dump() commonly produces
        # ``{"raw": ..., "agent": ..., "description": ..., ...}``;
        # surface the raw slot when present.
        raw = value.get("raw") or value.get("output") or value.get("content")
        if isinstance(raw, str):
            return raw
    return str(value)


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

    # Per-adapter ``extra="allow"`` decision: CrewAI targets the
    # canonical 13-event taxonomy exclusively. Unknown event types
    # must be rejected by the base adapter's typed-event validator,
    # so this stays ``False``. The previous ``agent.code`` event
    # emitted on task-start is migrated to :class:`AgentInputEvent`
    # (with ``role=AGENT``) on the typed path — see
    # :meth:`on_task_start`. See ``docs/adapters/typed-events.md``
    # for the opt-in policy.
    ALLOW_UNREGISTERED_EVENTS: bool = False

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
        """Emit a typed :class:`AgentInputEvent` for crew execution start.

        CrewAI-specific provenance (``framework``, ``timestamp_ns``,
        ``raw_input``) lives on :class:`MessageContent.metadata` —
        the canonical schema does not declare these as top-level
        fields.
        """
        with self._adapter_lock:
            self._crew_start_ns = time.time_ns()
        serialised_input = self._safe_serialize(crew_input)

        self.emit_event(
            AgentInputEvent.create(
                message=_stringify(serialised_input),
                role=MessageRole.HUMAN,
                metadata={
                    "framework": "crewai",
                    "timestamp_ns": self._crew_start_ns,
                    "raw_input": serialised_input,
                },
            )
        )

    def on_crew_end(
        self,
        crew_output: Any = None,
        error: Exception | None = None,
    ) -> None:
        """Emit a typed :class:`AgentOutputEvent` for crew execution end.

        Termination metadata (``duration_ns``, ``framework``,
        ``raw_output``, ``error``) is carried on
        :class:`MessageContent.metadata` — the canonical
        :class:`AgentOutputEvent` has no top-level slot for these
        CrewAI-specific signals.
        """
        end_ns = time.time_ns()
        duration_ns = end_ns - self._crew_start_ns if self._crew_start_ns else 0
        serialised_output = self._safe_serialize(crew_output)

        metadata: dict[str, Any] = {
            "framework": "crewai",
            "duration_ns": duration_ns,
            "raw_output": serialised_output,
            "run_status": "crew_failed" if error else "crew_complete",
        }
        if error:
            metadata["error"] = str(error)

        self.emit_event(
            AgentOutputEvent.create(
                message=_stringify(serialised_output),
                metadata=metadata,
            )
        )

    def on_task_start(
        self,
        task_description: str,
        agent_role: str | None = None,
        expected_output: str | None = None,
        task_order: int | None = None,
    ) -> None:
        """Emit a typed :class:`AgentInputEvent` for task start.

        The previous adapter implementation emitted an ad-hoc
        ``agent.code`` event type that is NOT in the canonical
        13-event taxonomy. The typed migration maps the task-start
        boundary onto :class:`AgentInputEvent` with ``role=AGENT``
        — a task description is logically input to the receiving
        agent. The original ``event_subtype="task_start"`` marker
        and CrewAI-specific provenance (``agent_role``,
        ``expected_output``, ``task_order``) live on
        :class:`MessageContent.metadata`.
        """
        metadata: dict[str, Any] = {
            "framework": "crewai",
            "event_subtype": "task_start",
        }
        if agent_role:
            metadata["agent_role"] = agent_role
        if expected_output:
            metadata["expected_output"] = expected_output
        if task_order is not None:
            metadata["task_order"] = task_order

        self.emit_event(
            AgentInputEvent.create(
                message=task_description,
                role=MessageRole.AGENT,
                metadata=metadata,
            )
        )

    def on_task_end(
        self,
        task_output: Any = None,
        agent_role: str | None = None,
        task_order: int | None = None,
        error: Exception | None = None,
    ) -> None:
        """Emit a typed :class:`AgentOutputEvent` and optional
        :class:`CostRecordEvent` for task completion.

        The previous adapter implementation emitted an
        ``agent.state.change`` payload carrying only ``event_subtype``
        and ``task_output``. That payload did not satisfy the
        canonical :class:`AgentStateChangeEvent` schema's
        ``before_hash`` / ``after_hash`` requirement (the task
        completion boundary has no real state mutation to hash).

        The typed migration maps task-end onto
        :class:`AgentOutputEvent` with ``run_status=task_complete`` (or
        ``task_failed``) on :class:`MessageContent.metadata`. Cost
        records use the canonical :class:`CostRecordEvent` with
        ``prompt_tokens`` / ``completion_tokens`` / ``tokens`` slots;
        ``agent_role`` is preserved on the canonical
        :class:`AgentOutputEvent.metadata` slot since the cost event
        has no metadata field.
        """
        serialised_output = self._safe_serialize(task_output)
        metadata: dict[str, Any] = {
            "framework": "crewai",
            "event_subtype": "task_complete",
            "raw_output": serialised_output,
            "run_status": "task_failed" if error else "task_complete",
        }
        if agent_role:
            metadata["agent_role"] = agent_role
        if task_order is not None:
            metadata["task_order"] = task_order
        if error:
            metadata["error"] = str(error)

        self.emit_event(
            AgentOutputEvent.create(
                message=_stringify(serialised_output),
                metadata=metadata,
            )
        )

        # Emit cost record if token usage available — uses the
        # canonical :class:`CostRecordEvent` slots.
        token_usage = self._extract_token_usage(task_output)
        if token_usage:
            self.emit_event(
                CostRecordEvent.create(
                    prompt_tokens=token_usage.get("tokens_prompt"),
                    completion_tokens=token_usage.get("tokens_completion"),
                    tokens=token_usage.get("tokens_total"),
                )
            )

    def on_tool_use(
        self,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        error: Exception | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Emit a typed :class:`ToolCallEvent` for tool usage.

        CrewAI does not expose tool versions on the
        ``CrewAgentExecutor`` callback signature, so ``version``
        falls back to ``"unavailable"`` per the canonical schema's
        NORMATIVE rule. Scalar tool inputs/outputs are wrapped in
        ``{"value": ...}`` so the canonical ``input`` / ``output``
        dict slots are satisfied (mirrors the agno reference
        pattern).
        """
        serialised_input = self._safe_serialize(tool_input)
        serialised_output = self._safe_serialize(tool_output)
        input_data: dict[str, Any]
        if isinstance(serialised_input, dict):
            input_data = dict(serialised_input)
        elif serialised_input is None:
            input_data = {}
        else:
            input_data = {"value": serialised_input}
        output_data: dict[str, Any] | None
        if isinstance(serialised_output, dict):
            output_data = dict(serialised_output)
        elif serialised_output is None:
            output_data = None
        else:
            output_data = {"value": serialised_output}

        self.emit_event(
            ToolCallEvent.create(
                name=tool_name,
                version="unavailable",
                integration=IntegrationType.LIBRARY,
                input_data=input_data,
                output_data=output_data,
                error=str(error) if error else None,
                latency_ms=latency_ms,
            )
        )

    def on_llm_call(
        self,
        provider: str | None = None,
        model: str | None = None,
        tokens_prompt: int | None = None,
        tokens_completion: int | None = None,
        latency_ms: float | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        """Emit a typed :class:`ModelInvokeEvent` for LLM invocation.

        Provider / model identifiers fall back to ``"unknown"`` when
        not supplied so the canonical schema validators are
        satisfied. Token counts use the canonical ``prompt_tokens`` /
        ``completion_tokens`` slots.
        """
        input_messages: list[dict[str, str]] | None = None
        if self._capture_config.capture_content and messages:
            input_messages = messages

        self.emit_event(
            ModelInvokeEvent.create(
                provider=provider or "unknown",
                name=model or "unknown",
                version="unavailable",
                parameters={"framework": "crewai"},
                prompt_tokens=tokens_prompt,
                completion_tokens=tokens_completion,
                latency_ms=latency_ms,
                input_messages=input_messages,
            )
        )

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
        """Emit a typed :class:`EnvironmentConfigEvent` per agent.

        Idempotent per agent role — only the first call for a given
        role actually emits. CrewAI's runtime is treated as a
        ``simulated`` environment by default; the real production
        environment (``cloud`` / ``on_prem``) is the responsibility
        of the host application's environment.config emission, not
        this framework adapter (mirrors the agno reference pattern).
        """
        role = getattr(agent, "role", None) or str(agent)
        with self._adapter_lock:
            if role in self._seen_agents:
                return
            self._seen_agents.add(role)

        metadata = self._metadata_extractor.extract(agent)
        if process_type:
            metadata["process_type"] = process_type

        attributes: dict[str, Any] = {
            "framework": "crewai",
            "agent_role": role,
            **metadata,
        }
        self.emit_event(
            EnvironmentConfigEvent.create(
                env_type=EnvironmentType.SIMULATED,
                attributes=attributes,
            )
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
