"""
Agno adapter lifecycle.

Instrumentation strategy: Agent wrapper (run/arun wrapping)
  Agent.run() start     -> agent.input (L1)
  Agent.run() end       -> agent.output (L1)
  Tool execution        -> tool.call (L5a)
  Model invocation      -> model.invoke (L3)
  Team delegation       -> agent.handoff (L2)
  Agent config          -> environment.config (L4a)
"""

from __future__ import annotations

import time
import uuid
import hashlib
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
    AgentHandoffEvent,
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
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

logger = logging.getLogger(__name__)


def _stringify(value: Any) -> str:
    """Return a string view of ``value`` suitable for the canonical
    :class:`MessageContent.message` field.

    The canonical schema requires :class:`AgentInputEvent` and
    :class:`AgentOutputEvent` to carry a ``message: str``. Agno
    callbacks deliver the underlying input/output as arbitrary
    Python objects (dicts, model responses, ``None``); this helper
    converts each to a non-empty string so the typed event always
    validates. The original payload is preserved on
    ``MessageContent.metadata.raw_input`` / ``raw_output``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _sha256_of(value: str) -> str:
    """Return a canonical ``sha256:<hex64>`` hash string for ``value``.

    The canonical schema's :class:`AgentHandoffEvent` requires
    ``handoff_context_hash`` to start with ``sha256:`` and have a
    64-character hex tail (see
    ``ateam/stratix/core/events/cross_cutting.py``). Centralising the
    format here ensures every emit site uses the same wire format.
    """
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


class AgnoAdapter(BaseAdapter):
    """LayerLens adapter for Agno.

    Reference adapter for the typed-event foundation. Every emission
    site flows through :meth:`emit_event` with a canonical Pydantic
    payload from :mod:`layerlens.instrument._compat.events`. No call
    site uses :meth:`emit_dict_event` — verified by the
    ``test_agno_no_dict_emits`` test in
    ``tests/instrument/adapters/frameworks/test_agno_adapter.py``.
    """

    FRAMEWORK = "agno"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/agno/``). Agno itself uses Pydantic v2
    # internally but the adapter only wraps ``Agent.run`` / ``Agent.arun``
    # and emits typed events, never touching framework Pydantic models.
    requires_pydantic = PydanticCompat.V1_OR_V2

    # Per-adapter ``extra="allow"`` decision: agno targets the canonical
    # 13-event taxonomy exclusively. Unknown event types must be
    # rejected by the base adapter's typed-event validator, so this
    # stays ``False``. See ``docs/adapters/typed-events.md`` for the
    # opt-in policy.
    ALLOW_UNREGISTERED_EVENTS: bool = False

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: Any | None = None,
        stratix_instance: Any | None = None,
        *,
        org_id: str | None = None,
    ) -> None:
        resolved = stratix or stratix_instance
        super().__init__(stratix=resolved, capture_config=capture_config, org_id=org_id)
        self._originals: dict[int, dict[str, Any]] = {}  # id(agent) -> {method: original}
        self._wrapped_agents: list[Any] = []  # strong refs for disconnect unwrap
        self._adapter_lock = threading.Lock()
        self._seen_agents: set[str] = set()
        self._framework_version: str | None = None
        self._run_starts: dict[int, int] = {}  # thread_id -> start_ns

    def connect(self) -> None:
        """Verify Agno availability and prepare the adapter."""
        try:
            import agno  # type: ignore[import-not-found,unused-ignore]

            self._framework_version = getattr(agno, "__version__", "unknown")
        except ImportError:
            logger.debug("agno not installed")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        """Unwrap all instrumented agents and release resources."""
        for agent in self._wrapped_agents:
            self._unwrap_agent(agent)
        self._wrapped_agents.clear()
        self._originals.clear()
        self._seen_agents.clear()
        self._run_starts.clear()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def _unwrap_agent(self, agent: Any) -> None:
        """Restore original methods on a wrapped agent."""
        agent_id = id(agent)
        originals = self._originals.get(agent_id)
        if not originals:
            return
        for method_name, original in originals.items():
            try:
                setattr(agent, method_name, original)
            except Exception:
                logger.debug("Could not unwrap %s.%s", agent_id, method_name, exc_info=True)

    def health_check(self) -> AdapterHealth:
        """Return a health snapshot."""
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            framework_version=self._framework_version,
            adapter_version=self.VERSION,
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )

    def get_adapter_info(self) -> AdapterInfo:
        """Return metadata about this adapter."""
        return AdapterInfo(
            name="AgnoAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_HANDOFFS,
            ],
            description="LayerLens adapter for Agno",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        """Serialize the current trace data for replay."""
        return ReplayableTrace(
            adapter_name="AgnoAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
        )

    # --- Framework Integration ---

    def instrument_agent(self, agent: Any) -> Any:
        """Wrap Agno agent.run() and agent.arun() methods to capture lifecycle events."""
        agent_id = id(agent)
        if agent_id in self._originals:
            return agent
        originals: dict[str, Any] = {}
        # Wrap run() (sync)
        if hasattr(agent, "run"):
            originals["run"] = agent.run
            agent.run = self._create_traced_run_sync(agent, agent.run)
        # Wrap arun() (async)
        if hasattr(agent, "arun"):
            originals["arun"] = agent.arun
            agent.arun = self._create_traced_run(agent, agent.arun)
        self._originals[agent_id] = originals
        self._wrapped_agents.append(agent)
        agent_name = getattr(agent, "name", None) or str(type(agent).__name__)
        self._emit_agent_config(agent_name, agent)
        return agent

    def _create_traced_run(self, agent: Any, original_run: Any) -> Any:
        """Create an async traced wrapper for agent.arun()."""
        adapter = self

        async def traced_run(*args: Any, **kwargs: Any) -> Any:
            agent_name = getattr(agent, "name", None) or "agno_agent"
            input_data = kwargs.get("message") or (args[0] if args else None)
            adapter.on_run_start(agent_name=agent_name, input_data=input_data)
            error: Exception | None = None
            result = None
            try:
                result = await original_run(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                output = None
                if result is not None:
                    output = getattr(result, "content", result)
                adapter.on_run_end(agent_name=agent_name, output=output, error=error)
                adapter._extract_run_details(agent, result)
            return result

        traced_run._layerlens_original = original_run  # type: ignore[attr-defined]
        return traced_run

    def _create_traced_run_sync(self, agent: Any, original_run: Any) -> Any:
        """Create a sync traced wrapper for agent.run()."""
        adapter = self

        def traced_run_sync(*args: Any, **kwargs: Any) -> Any:
            agent_name = getattr(agent, "name", None) or "agno_agent"
            input_data = kwargs.get("message") or (args[0] if args else None)
            adapter.on_run_start(agent_name=agent_name, input_data=input_data)
            error: Exception | None = None
            result = None
            try:
                result = original_run(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                output = None
                if result is not None:
                    output = getattr(result, "content", result)
                adapter.on_run_end(agent_name=agent_name, output=output, error=error)
                adapter._extract_run_details(agent, result)
            return result

        traced_run_sync._layerlens_original = original_run  # type: ignore[attr-defined]
        return traced_run_sync

    def _extract_run_details(self, agent: Any, result: Any) -> None:
        """Extract tool calls, model invocations, and team handoffs from run result.

        Each extracted signal is emitted as a typed canonical event via
        :meth:`emit_event`. Agno-specific provenance (e.g. ``framework``,
        ``agent_name``) is carried in the model's ``metadata`` /
        ``parameters`` slots — the canonical schema does not expose
        these as top-level fields.
        """
        if result is None:
            return
        try:
            # Extract model invocation details. The canonical schema
            # requires ``provider`` and ``name``; ``version`` falls back
            # to ``"unavailable"`` when agno cannot surface it (per the
            # NORMATIVE rule in events_l3_model.py).
            model = getattr(agent, "model", None)
            if model:
                model_name = getattr(model, "id", None) or str(model)
                provider = self._detect_provider(model_name) or "unknown"
                self.emit_event(
                    ModelInvokeEvent.create(
                        provider=provider,
                        name=model_name,
                        version="unavailable",
                        parameters={"framework": "agno"},
                    )
                )

            # Extract usage/token info from result.
            usage = getattr(result, "metrics", None) or getattr(result, "usage", None)
            if usage:
                self.emit_event(
                    CostRecordEvent.create(
                        prompt_tokens=getattr(usage, "input_tokens", None)
                        or getattr(usage, "prompt_tokens", None),
                        completion_tokens=getattr(usage, "output_tokens", None)
                        or getattr(usage, "completion_tokens", None),
                        tokens=getattr(usage, "total_tokens", None),
                    )
                )

            # Extract tool calls from messages.
            messages = getattr(result, "messages", None) or []
            for msg in messages:
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        function_obj = getattr(tc, "function", None)
                        if isinstance(function_obj, dict):
                            tool_name = function_obj.get("name", "unknown")
                            raw_args = function_obj.get("arguments")
                        else:
                            tool_name = getattr(function_obj, "name", "unknown")
                            raw_args = None
                        serialised_args = self._safe_serialize(raw_args)
                        input_data: dict[str, Any]
                        if isinstance(serialised_args, dict):
                            input_data = dict(serialised_args)
                        elif serialised_args is None:
                            input_data = {}
                        else:
                            input_data = {"value": serialised_args}
                        self.emit_event(
                            ToolCallEvent.create(
                                name=tool_name,
                                version="unavailable",
                                integration=IntegrationType.LIBRARY,
                                input_data=input_data,
                            )
                        )

            # Detect team delegation (multi-agent handoffs). The canonical
            # schema requires a sha256 ``handoff_context_hash`` — we hash
            # the (from_agent, to_agent, reason) tuple deterministically
            # so the same delegation produces the same hash.
            team = getattr(agent, "team", None)
            if team:
                members = getattr(team, "members", None) or getattr(team, "agents", None) or []
                from_name = getattr(agent, "name", "leader") or "leader"
                for member in members:
                    member_name = getattr(member, "name", None) or str(member)
                    self.emit_event(
                        AgentHandoffEvent.create(
                            from_agent=from_name,
                            to_agent=member_name,
                            handoff_context_hash=_sha256_of(
                                f"team_delegation::{from_name}::{member_name}"
                            ),
                        )
                    )
        except Exception:
            logger.debug("Could not extract run details", exc_info=True)

    # --- Lifecycle Hooks ---

    def on_run_start(self, agent_name: str | None = None, input_data: Any = None) -> None:
        """Emit a typed :class:`AgentInputEvent` when an agent run starts.

        Agno-specific provenance (``framework``, ``agent_name``,
        ``timestamp_ns``) is carried on the canonical
        :class:`MessageContent.metadata` slot — the canonical schema
        does not declare these as top-level fields.
        """
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._run_starts[tid] = start_ns
            serialised_input = self._safe_serialize(input_data)
            self.emit_event(
                AgentInputEvent.create(
                    message=_stringify(serialised_input),
                    role=MessageRole.HUMAN,
                    metadata={
                        "framework": "agno",
                        "agent_name": agent_name,
                        "timestamp_ns": start_ns,
                        "raw_input": serialised_input,
                    },
                )
            )
        except Exception:
            logger.warning("Error in on_run_start", exc_info=True)

    def on_run_end(
        self,
        agent_name: str | None = None,
        output: Any = None,
        error: Exception | None = None,
    ) -> None:
        """Emit a typed :class:`AgentOutputEvent` when an agent run ends.

        The previous adapter implementation also emitted a
        ``agent.state.change`` event carrying only an ``event_subtype``
        marker. That payload did not satisfy the canonical schema's
        ``before_hash`` / ``after_hash`` requirement (the event is
        defined for *real* state mutations with computable hashes —
        see ``ateam/stratix/core/events/cross_cutting.py``). Rather
        than synthesise placeholder hashes, the lifecycle marker is
        now folded into the :class:`AgentOutputEvent.metadata` slot
        as ``run_status``. Real state hashing requires upstream agno
        instrumentation that is not available today.
        """
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._run_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            serialised_output = self._safe_serialize(output)
            metadata: dict[str, Any] = {
                "framework": "agno",
                "agent_name": agent_name,
                "duration_ns": duration_ns,
                "raw_output": serialised_output,
                "run_status": "run_failed" if error else "run_complete",
            }
            if error:
                metadata["error"] = str(error)
            self.emit_event(
                AgentOutputEvent.create(
                    message=_stringify(serialised_output),
                    metadata=metadata,
                )
            )
        except Exception:
            logger.warning("Error in on_run_end", exc_info=True)

    def on_tool_use(
        self,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        error: Exception | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Emit a typed :class:`ToolCallEvent` for a tool invocation."""
        if not self._connected:
            return
        try:
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
        except Exception:
            logger.warning("Error in on_tool_use", exc_info=True)

    def on_llm_call(
        self,
        provider: str | None = None,
        model: str | None = None,
        tokens_prompt: int | None = None,
        tokens_completion: int | None = None,
        latency_ms: float | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        """Emit a typed :class:`ModelInvokeEvent` for an LLM call."""
        if not self._connected:
            return
        try:
            self.emit_event(
                ModelInvokeEvent.create(
                    provider=provider or "unknown",
                    name=model or "unknown",
                    version="unavailable",
                    parameters={"framework": "agno"},
                    prompt_tokens=tokens_prompt,
                    completion_tokens=tokens_completion,
                    latency_ms=latency_ms,
                    input_messages=messages
                    if (self._capture_config.capture_content and messages)
                    else None,
                )
            )
        except Exception:
            logger.warning("Error in on_llm_call", exc_info=True)

    def on_handoff(self, from_agent: str, to_agent: str, context: Any = None) -> None:
        """Emit a typed :class:`AgentHandoffEvent` for team delegation.

        The canonical schema requires ``handoff_context_hash`` to be a
        ``sha256:<hex64>`` string — empty contexts are still hashed
        (over the empty string) so the wire format is uniform. The
        previous adapter implementation emitted ``None`` when context
        was missing; that was non-conformant.
        """
        if not self._connected:
            return
        try:
            context_str = str(context) if context else ""
            self.emit_event(
                AgentHandoffEvent.create(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    handoff_context_hash=_sha256_of(context_str),
                )
            )
        except Exception:
            logger.warning("Error in on_handoff", exc_info=True)

    # --- Helpers ---

    def _detect_provider(self, model: str | None) -> str | None:
        """Detect the LLM provider from a model identifier."""
        if not model:
            return None
        model_lower = model.lower()
        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return "openai"
        if "claude" in model_lower:
            return "anthropic"
        if "gemini" in model_lower:
            return "google"
        if "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        if "llama" in model_lower:
            return "meta"
        if "command" in model_lower:
            return "cohere"
        return None

    def _emit_agent_config(self, agent_name: str, agent: Any) -> None:
        """Emit a typed :class:`EnvironmentConfigEvent` for agent configuration.

        Idempotent per agent — only the first call for a given
        ``agent_name`` actually emits. Agno's runtime is treated as a
        ``simulated`` environment by default; the real production
        environment (cloud / on_prem) is the responsibility of the
        host application's environment.config emission, not this
        framework adapter.
        """
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        attributes: dict[str, Any] = {
            "framework": "agno",
            "agent_name": agent_name,
        }
        model = getattr(agent, "model", None)
        if model:
            attributes["model"] = str(model)
        description = getattr(agent, "description", None)
        if description:
            attributes["description"] = str(description)[:500]
        instructions = getattr(agent, "instructions", None)
        if instructions and self._capture_config.capture_content:
            attributes["instructions"] = str(instructions)[:500]
        tools = getattr(agent, "tools", None)
        if tools:
            attributes["tools"] = [getattr(t, "name", str(t)) for t in tools]
        knowledge = getattr(agent, "knowledge", None)
        if knowledge:
            attributes["knowledge"] = str(type(knowledge).__name__)
        team = getattr(agent, "team", None)
        if team:
            members = getattr(team, "members", None) or getattr(team, "agents", None) or []
            attributes["team_members"] = [getattr(m, "name", str(m)) for m in members]
        self.emit_event(
            EnvironmentConfigEvent.create(
                env_type=EnvironmentType.SIMULATED,
                attributes=attributes,
            )
        )

    def _safe_serialize(self, value: Any) -> Any:
        """Safely serialize a value for event payloads."""
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
