"""
AWS Strands adapter lifecycle.

Instrumentation strategy: Agent wrapper (run wrapping) + callback hooks
  Agent start            -> agent.input (L1)
  Agent end              -> agent.output (L1)
  Tool call              -> tool.call (L5a)
  Model invoke (Bedrock) -> model.invoke (L3)
  Conversation state     -> agent.state.change (Cross)
  Cost (Bedrock pricing) -> cost.record (Cross)

Typed-event status (post PR #129 migration, bundle 5):

* Every emission flows through :meth:`BaseAdapter.emit_event` with a
  canonical Pydantic payload imported from
  :mod:`layerlens.instrument._compat.events`.
* AWS Strands-specific provenance (``framework``, ``agent_name``,
  ``timestamp_ns``, ``duration_ns``, ``model``, ``tools``,
  ``conversation_type``, ``turn_count``, ``system_prompt``) is
  carried in the canonical model's metadata / attributes /
  parameters / input slots.
* AWS Strands tool execution: Strands lets agents declare tools as
  Python callables that run in the host runtime — even when the
  underlying capability is an AWS service, the tool *call* is an
  in-process Python invocation. The L5a integration is therefore
  :class:`IntegrationType.LIBRARY`. (This deliberately differs
  from the bedrock_agents adapter's :class:`IntegrationType.SERVICE`
  mapping — bedrock_agents tool execution is performed by the
  Bedrock service via Lambda action groups, not in-process.)
* The ad-hoc ``agent.state.change`` ``run_complete`` / ``run_failed``
  marker emitted by :meth:`on_run_end` does NOT satisfy the canonical
  :class:`AgentStateChangeEvent` ``before_hash`` / ``after_hash``
  contract (the run boundary has no real state mutation to hash).
  Following the PR #151 ms_agent_framework precedent, the marker is
  carried as ``run_status`` on :class:`AgentOutputEvent.metadata`,
  preserving the cross-cutting completion signal without violating
  the canonical schema.
* The conversation-update ``agent.state.change`` emitted by
  :meth:`_extract_run_details` carries Strands ``turn_count`` data
  but not the canonical ``before_hash`` / ``after_hash`` (the
  framework does not surface a hashable state snapshot at the
  conversation boundary). It is mapped onto
  :class:`AgentOutputEvent.metadata.conversation_state` so the
  conversation-progress signal is preserved without violating the
  canonical schema.
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
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

logger = logging.getLogger(__name__)


def _stringify(value: Any) -> str:
    """Return a string view of ``value`` suitable for the canonical
    :class:`MessageContent.message` field.

    The canonical schema requires :class:`AgentInputEvent` and
    :class:`AgentOutputEvent` to carry a ``message: str``. AWS
    Strands delivers user prompts and run outputs as arbitrary Python
    objects (strings, message objects, ``None``); this helper
    converts each to a (possibly empty) string so the typed event
    always validates. The original payload is preserved on
    :class:`MessageContent.metadata.raw_input` / ``raw_output``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_to_dict(value: Any) -> dict[str, Any]:
    """Coerce ``value`` into a dict suitable for the canonical
    :class:`ToolCallEvent` ``input`` / ``output`` slots.

    The canonical schema requires ``input: dict[str, Any]`` and
    ``output: dict[str, Any] | None``. Strands tool returns can be
    arbitrary Python values. This helper wraps non-dict values in
    ``{"value": ...}`` so the canonical slot is always satisfied.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


class StrandsAdapter(BaseAdapter):
    """LayerLens adapter for AWS Strands."""

    FRAMEWORK = "strands"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/strands/``). Strands instrumentation hooks
    # into agent callbacks and emits typed events without crossing the
    # framework's Pydantic boundary.
    requires_pydantic = PydanticCompat.V1_OR_V2

    # Per-adapter ``extra="allow"`` decision: strands targets the
    # canonical 13-event taxonomy exclusively. Unknown event types
    # must be rejected by the base adapter's typed-event validator,
    # so this stays ``False``. See ``docs/adapters/typed-events.md``
    # for the opt-in policy.
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
        """Verify AWS Strands availability and prepare the adapter."""
        try:
            import strands  # type: ignore[import-not-found,unused-ignore]

            self._framework_version = getattr(strands, "__version__", "unknown")
        except ImportError:
            logger.debug("strands-agents not installed")
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
            name="StrandsAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
            ],
            description="LayerLens adapter for AWS Strands",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        """Serialize the current trace data for replay."""
        return ReplayableTrace(
            adapter_name="StrandsAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
        )

    # --- Framework Integration ---

    def instrument_agent(self, agent: Any) -> Any:
        """Wrap AWS Strands agent __call__ and invoke methods to capture lifecycle events."""
        agent_id = id(agent)
        if agent_id in self._originals:
            return agent
        originals: dict[str, Any] = {}
        # Strands Agent uses __call__ as the primary invocation method
        if callable(agent):
            originals["__call__"] = agent.__call__
            agent.__call__ = self._create_traced_call(agent, agent.__call__)
        # Also wrap invoke() if present
        if hasattr(agent, "invoke"):
            originals["invoke"] = agent.invoke
            agent.invoke = self._create_traced_call(agent, agent.invoke)
        self._originals[agent_id] = originals
        self._wrapped_agents.append(agent)
        agent_name = getattr(agent, "name", None) or str(type(agent).__name__)
        self._emit_agent_config(agent_name, agent)
        return agent

    def _create_traced_call(self, agent: Any, original_call: Any) -> Any:
        """Create a traced wrapper for agent invocation."""
        adapter = self

        def traced_call(*args: Any, **kwargs: Any) -> Any:
            agent_name = getattr(agent, "name", None) or "strands_agent"
            input_data = args[0] if args else kwargs.get("prompt") or kwargs.get("message")
            adapter.on_run_start(agent_name=agent_name, input_data=input_data)
            error: Exception | None = None
            result = None
            try:
                result = original_call(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                output = None
                if result is not None:
                    output = getattr(result, "content", None) or getattr(result, "text", result)
                adapter.on_run_end(agent_name=agent_name, output=output, error=error)
                adapter._extract_run_details(agent, result)
            return result

        traced_call._layerlens_original = original_call  # type: ignore[attr-defined]
        return traced_call

    def _extract_run_details(self, agent: Any, result: Any) -> None:
        """Extract tool calls, model invocations, and cost from a run result.

        Emits typed :class:`ModelInvokeEvent`, :class:`CostRecordEvent`,
        :class:`ToolCallEvent`, and a conversation-state-change marker
        (carried as :class:`AgentOutputEvent.metadata.conversation_state`)
        when the framework surfaces the corresponding data on the run
        result.
        """
        if result is None:
            return
        try:
            # Extract model invocation details
            model = getattr(agent, "model", None) or getattr(agent, "model_id", None)
            if model:
                model_name = str(model)
                provider = self._detect_provider(model_name) or "bedrock"
                self.emit_event(
                    ModelInvokeEvent.create(
                        provider=provider,
                        name=model_name,
                        version="unavailable",
                        parameters={"framework": "strands"},
                    )
                )

            # Extract usage/token info from result
            usage = getattr(result, "usage", None) or getattr(result, "metrics", None)
            if usage:
                tokens_prompt = getattr(usage, "inputTokens", None) or getattr(usage, "prompt_tokens", None)
                tokens_completion = getattr(usage, "outputTokens", None) or getattr(usage, "completion_tokens", None)
                tokens_total = getattr(usage, "totalTokens", None) or getattr(usage, "total_tokens", None)
                self.emit_event(
                    CostRecordEvent.create(
                        prompt_tokens=tokens_prompt,
                        completion_tokens=tokens_completion,
                        tokens=tokens_total,
                    )
                )

            # Extract tool calls from result. Strands tool-result entries
            # come through as objects (with ``name``/``input``/``output``
            # attributes) or dicts; normalise both shapes here.
            tool_results = getattr(result, "tool_results", None) or []
            for tr in tool_results:
                if isinstance(tr, dict):
                    tool_name_raw: Any = tr.get("name", "unknown")
                    raw_input = tr.get("input")
                    raw_output = tr.get("output")
                else:
                    tool_name_raw = getattr(tr, "name", None) or "unknown"
                    raw_input = getattr(tr, "input", None)
                    raw_output = getattr(tr, "output", None)
                tool_name = str(tool_name_raw) if tool_name_raw else "unknown"
                serialised_input = self._safe_serialize(raw_input)
                serialised_output = self._safe_serialize(raw_output)
                input_data = _coerce_to_dict(serialised_input)
                input_data.setdefault("framework", "strands")
                output_data: dict[str, Any] | None = (
                    _coerce_to_dict(serialised_output)
                    if serialised_output is not None
                    else None
                )
                self.emit_event(
                    ToolCallEvent.create(
                        name=tool_name,
                        version="unavailable",
                        integration=IntegrationType.LIBRARY,
                        input_data=input_data,
                        output_data=output_data,
                    )
                )

            # Emit conversation state change. The previous adapter emitted
            # an ad-hoc agent.state.change payload with only
            # event_subtype + turn_count. That payload did not satisfy
            # the canonical AgentStateChangeEvent before_hash / after_hash
            # contract — the framework does not surface a hashable state
            # snapshot at the conversation boundary. The post-migration
            # mapping carries the conversation-progress signal as
            # AgentOutputEvent.metadata.conversation_state, preserving
            # the marker without violating the canonical schema.
            conversation = getattr(agent, "conversation", None) or getattr(agent, "conversation_manager", None)
            if conversation:
                turn_count = getattr(conversation, "turn_count", None) or len(
                    getattr(conversation, "messages", [])
                )
                self.emit_event(
                    AgentOutputEvent.create(
                        message="",
                        metadata={
                            "framework": "strands",
                            "agent_name": getattr(agent, "name", "strands_agent"),
                            "conversation_state": "conversation_update",
                            "turn_count": turn_count,
                        },
                    )
                )
        except Exception:
            logger.debug("Could not extract run details", exc_info=True)

    # --- Lifecycle Hooks ---

    def on_run_start(self, agent_name: str | None = None, input_data: Any = None) -> None:
        """Emit a typed :class:`AgentInputEvent` when an agent run starts."""
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._run_starts[tid] = start_ns
            raw_input = self._safe_serialize(input_data)
            self.emit_event(
                AgentInputEvent.create(
                    message=_stringify(raw_input),
                    role=MessageRole.HUMAN,
                    metadata={
                        "framework": "strands",
                        "agent_name": agent_name,
                        "timestamp_ns": start_ns,
                        "raw_input": raw_input,
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

        The previous adapter implementation also emitted a separate
        ``agent.state.change`` payload carrying only an
        ``event_subtype`` marker (``run_complete`` / ``run_failed``).
        That payload did not satisfy the canonical
        :class:`AgentStateChangeEvent` ``before_hash`` / ``after_hash``
        contract — the run boundary has no real state mutation to
        hash. The post-migration mapping (matching the PR #151
        ms_agent_framework precedent) carries the same signal as
        ``run_status`` on :class:`MessageContent.metadata`, preserving
        the cross-cutting completion marker without violating the
        canonical schema.
        """
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._run_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            raw_output = self._safe_serialize(output)
            metadata: dict[str, Any] = {
                "framework": "strands",
                "agent_name": agent_name,
                "duration_ns": duration_ns,
                "raw_output": raw_output,
                "run_status": "run_complete" if not error else "run_failed",
            }
            if error:
                metadata["error"] = str(error)
            self.emit_event(
                AgentOutputEvent.create(
                    message=_stringify(raw_output),
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
        """Emit a typed :class:`ToolCallEvent` for a tool invocation.

        AWS Strands tools execute as in-process Python callables in
        the host runtime — even when the underlying capability is an
        AWS service, the *call* boundary instrumented here is the
        in-process invocation. Integration is therefore
        :class:`IntegrationType.LIBRARY`. Tool versions are not
        surfaced by Strands so ``version`` falls back to
        ``"unavailable"`` per the canonical NORMATIVE rule.
        """
        if not self._connected:
            return
        try:
            serialised_input = self._safe_serialize(tool_input)
            serialised_output = self._safe_serialize(tool_output)
            input_data = _coerce_to_dict(serialised_input)
            input_data.setdefault("framework", "strands")
            output_data: dict[str, Any] | None = (
                _coerce_to_dict(serialised_output) if serialised_output is not None else None
            )
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
        """Emit a typed :class:`ModelInvokeEvent` for an LLM call.

        Strands defaults to AWS Bedrock when no provider is
        explicitly supplied, matching the framework's primary
        deployment target.
        """
        if not self._connected:
            return
        try:
            model_name = model or "unknown"
            resolved_provider = provider or self._detect_provider(model_name) or "bedrock"
            input_messages: list[dict[str, str]] | None = None
            if self._capture_config.capture_content and messages:
                input_messages = messages
            self.emit_event(
                ModelInvokeEvent.create(
                    provider=resolved_provider,
                    name=model_name,
                    version="unavailable",
                    parameters={"framework": "strands"},
                    prompt_tokens=tokens_prompt,
                    completion_tokens=tokens_completion,
                    latency_ms=latency_ms,
                    input_messages=input_messages,
                )
            )
        except Exception:
            logger.warning("Error in on_llm_call", exc_info=True)

    # --- Helpers ---

    def _detect_provider(self, model: str | None) -> str | None:
        """Detect the LLM provider from a model identifier."""
        if not model:
            return None
        model_lower = model.lower()
        # Strands defaults to Bedrock
        if "anthropic" in model_lower or "claude" in model_lower:
            return "bedrock"
        if "amazon" in model_lower or "titan" in model_lower:
            return "bedrock"
        if "meta" in model_lower or "llama" in model_lower:
            return "bedrock"
        if "mistral" in model_lower or "mixtral" in model_lower:
            return "bedrock"
        if "cohere" in model_lower or "command" in model_lower:
            return "bedrock"
        if "ai21" in model_lower or "jamba" in model_lower:
            return "bedrock"
        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return "openai"
        if "gemini" in model_lower:
            return "google"
        return "bedrock"  # Default to Bedrock for Strands

    def _emit_agent_config(self, agent_name: str, agent: Any) -> None:
        """Emit a typed :class:`EnvironmentConfigEvent` per agent.

        Idempotent per agent — only the first call for a given agent
        name actually emits. Strands agents typically run in a
        ``cloud`` environment (AWS), but the adapter cannot
        confidently distinguish cloud vs. local execution, so the
        canonical ``simulated`` env_type is used by default — the
        host application is responsible for emitting the real
        environment.config record. Strands-specific provenance
        (``framework``, ``agent_name``, ``model``, ``tools``,
        ``conversation_type``, ``system_prompt``) lives on
        :attr:`EnvironmentInfo.attributes`.
        """
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        attributes: dict[str, Any] = {
            "framework": "strands",
            "agent_name": agent_name,
        }
        model = getattr(agent, "model", None) or getattr(agent, "model_id", None)
        if model:
            attributes["model"] = str(model)
        system_prompt = getattr(agent, "system_prompt", None)
        if system_prompt and self._capture_config.capture_content:
            attributes["system_prompt"] = str(system_prompt)[:500]
        tools = getattr(agent, "tools", None)
        if tools:
            if isinstance(tools, dict):
                attributes["tools"] = list(tools.keys())
            else:
                attributes["tools"] = [
                    getattr(t, "name", None) or getattr(t, "tool_name", str(t)) for t in tools
                ]
        conversation = getattr(agent, "conversation", None) or getattr(agent, "conversation_manager", None)
        if conversation:
            attributes["conversation_type"] = str(type(conversation).__name__)
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
