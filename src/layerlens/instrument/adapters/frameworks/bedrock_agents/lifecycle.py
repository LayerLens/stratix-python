"""
AWS Bedrock Agents adapter lifecycle.

Instrumentation strategy: boto3 event hooks + OTel (ADOT integration)
  invoke_agent request  → agent.input (L1)
  invoke_agent response → agent.output (L1)
  Action Group          → tool.call (L5a)
  Knowledge Base query  → tool.call (L5a, retrieval)
  Model invocation      → model.invoke (L3)
  Supervisor→Collaborator → agent.handoff (Cross)

Typed-event status (post PR #129 migration, bundle 4):

* Every emission flows through :meth:`BaseAdapter.emit_event` with a
  canonical Pydantic payload imported from
  :mod:`layerlens.instrument._compat.events`.
* Bedrock-specific provenance (``framework``, ``agent_id``,
  ``session_id``, ``timestamp_ns``, ``duration_ns``, ``enable_trace``)
  is carried in the canonical model's metadata / attributes /
  parameters / input slots — the canonical schema does not expose
  these as top-level fields.
* Action group + knowledge base steps map to
  :class:`ToolCallEvent` with ``integration=IntegrationType.SERVICE``
  (AWS-side execution, not in-process library).
* The handoff context hash is generated via SHA-256 over the context
  string (or the empty string when no context is available) so the
  canonical :class:`AgentHandoffEvent.handoff_context_hash` validator
  always passes.
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
    :class:`AgentOutputEvent` to carry a ``message: str``. Bedrock
    callbacks deliver inputs/outputs as arbitrary Python objects
    (parsed boto3 dicts, ``None``); this helper converts each to a
    (possibly empty) string so the typed event always validates. The
    original payload is preserved on
    :class:`MessageContent.metadata.raw_input` / ``raw_output``.
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
    format here ensures every emit site uses the same wire format —
    including the empty-string fallback used when Bedrock has no
    handoff context to hash.
    """
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


class BedrockAgentsAdapter(BaseAdapter):
    """LayerLens adapter for AWS Bedrock Agents."""

    FRAMEWORK = "bedrock_agents"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/bedrock_agents/``). Bedrock Agents is a
    # remote AWS service consumed via boto3 hooks — boto3 does not use
    # Pydantic. Adapter emits typed events through the canonical
    # schema (PR #129), never touching the framework's Pydantic models.
    requires_pydantic = PydanticCompat.V1_OR_V2

    # Per-adapter ``extra="allow"`` decision: bedrock_agents targets the
    # canonical 13-event taxonomy exclusively. Unknown event types must
    # be rejected by the base adapter's typed-event validator, so this
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
        self._originals: dict[str, Any] = {}
        self._adapter_lock = threading.Lock()
        self._seen_agents: set[str] = set()
        self._framework_version: str | None = None
        self._invoke_starts: dict[int, int] = {}

    def connect(self) -> None:
        try:
            import boto3  # type: ignore[import-untyped,unused-ignore]

            self._framework_version = boto3.__version__
        except ImportError:
            logger.debug("boto3 not installed")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        # Unregister boto3 event hooks
        client = self._originals.get("client")
        if client is not None:
            try:
                event_system = client.meta.events
                event_system.unregister(
                    "provide-client-params.bedrock-agent-runtime.InvokeAgent",
                    self._before_invoke_agent,
                )
                event_system.unregister(
                    "after-call.bedrock-agent-runtime.InvokeAgent",
                    self._after_invoke_agent,
                )
            except Exception:
                logger.debug("Could not unregister boto3 event hooks", exc_info=True)
        self._originals.clear()
        self._seen_agents.clear()
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
            name="BedrockAgentsAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_HANDOFFS,
            ],
            description="LayerLens adapter for AWS Bedrock Agents",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="BedrockAgentsAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
        )

    # --- Framework Integration ---

    def instrument_client(self, client: Any) -> Any:
        """Register boto3 event hooks on a bedrock-agent-runtime client."""
        try:
            event_system = client.meta.events
            event_system.register(
                "provide-client-params.bedrock-agent-runtime.InvokeAgent",
                self._before_invoke_agent,
            )
            event_system.register(
                "after-call.bedrock-agent-runtime.InvokeAgent",
                self._after_invoke_agent,
            )
            self._originals["client"] = client
        except Exception:
            logger.warning("Failed to register boto3 event hooks", exc_info=True)
        return client

    # --- boto3 Event Hooks ---

    def _before_invoke_agent(self, **kwargs: Any) -> None:
        """Emit a typed :class:`AgentInputEvent` for an invoke_agent boto3 call.

        Bedrock-specific provenance (``framework``, ``agent_id``,
        ``session_id``, ``enable_trace``, ``timestamp_ns``) is carried
        on :class:`MessageContent.metadata`. The canonical ``message``
        field carries the inbound input text (or empty string when
        Bedrock supplies no inputText).
        """
        if not self._connected:
            return
        try:
            params = kwargs.get("params", {})
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._invoke_starts[tid] = start_ns
            agent_id = params.get("agentId", "unknown")
            self._emit_agent_config(agent_id, params)
            input_text = params.get("inputText")
            self.emit_event(
                AgentInputEvent.create(
                    message=_stringify(input_text),
                    role=MessageRole.HUMAN,
                    metadata={
                        "framework": "bedrock_agents",
                        "agent_id": agent_id,
                        "session_id": params.get("sessionId"),
                        "enable_trace": params.get("enableTrace", False),
                        "timestamp_ns": start_ns,
                        "raw_input": input_text,
                    },
                )
            )
        except Exception:
            logger.warning("Error in _before_invoke_agent", exc_info=True)

    def _after_invoke_agent(self, **kwargs: Any) -> None:
        """Emit a typed :class:`AgentOutputEvent` for an invoke_agent response.

        The canonical ``message`` slot carries the extracted completion
        text; Bedrock-specific provenance (``framework``,
        ``session_id``, ``duration_ns``, ``raw_output``) lives on
        :class:`MessageContent.metadata`.
        """
        if not self._connected:
            return
        try:
            parsed = kwargs.get("parsed", {})
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._invoke_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            output = self._extract_completion(parsed)
            self.emit_event(
                AgentOutputEvent.create(
                    message=_stringify(output),
                    metadata={
                        "framework": "bedrock_agents",
                        "session_id": parsed.get("sessionId"),
                        "duration_ns": duration_ns,
                        "raw_output": output,
                    },
                )
            )
            # Extract trace steps if available
            self._process_trace(parsed)
        except Exception:
            logger.warning("Error in _after_invoke_agent", exc_info=True)

    def _process_trace(self, parsed: dict[str, Any]) -> None:
        """Extract trace steps from Bedrock response and emit events."""
        trace = parsed.get("trace", {})
        steps = trace.get("trace", {}).get("orchestrationTrace", {}).get("steps", [])
        if not steps and isinstance(trace, dict):
            # Try alternative trace structure
            steps = trace.get("steps", [])
        for step in steps:
            step_type = step.get("type", "")
            if step_type == "ACTION_GROUP":
                self._emit_action_group(step)
            elif step_type == "KNOWLEDGE_BASE":
                self._emit_knowledge_base(step)
            elif step_type == "MODEL_INVOCATION":
                self._emit_model_invocation(step)
            elif step_type == "AGENT_COLLABORATOR":
                self._emit_collaborator_handoff(step)

    def _emit_action_group(self, step: dict[str, Any]) -> None:
        """Emit a typed :class:`ToolCallEvent` for an action group invocation.

        Bedrock action groups are AWS-managed Lambda functions
        invoked by the agent — this maps to ``integration=SERVICE``
        (the closest canonical match: action groups run as managed
        cloud services, not in-process libraries). Bedrock-specific
        provenance (``framework``, ``tool_type``) is folded into the
        canonical ``input`` dict.
        """
        action = step.get("actionGroupInvocationOutput", {})
        action_input = self._safe_serialize(step.get("actionGroupInput"))
        action_output = self._safe_serialize(action.get("output"))
        input_data: dict[str, Any]
        if isinstance(action_input, dict):
            input_data = dict(action_input)
        elif action_input is None:
            input_data = {}
        else:
            input_data = {"value": action_input}
        input_data["framework"] = "bedrock_agents"
        input_data["tool_type"] = "action_group"
        output_data: dict[str, Any] | None
        if isinstance(action_output, dict):
            output_data = dict(action_output)
        elif action_output is None:
            output_data = None
        else:
            output_data = {"value": action_output}
        self.emit_event(
            ToolCallEvent.create(
                name=step.get("actionGroupName", "unknown"),
                version="unavailable",
                integration=IntegrationType.SERVICE,
                input_data=input_data,
                output_data=output_data,
            )
        )

    def _emit_knowledge_base(self, step: dict[str, Any]) -> None:
        """Emit a typed :class:`ToolCallEvent` for a knowledge base lookup.

        Knowledge base retrievals are AWS-managed services
        (``integration=SERVICE``). The retrieved references list
        moves into the canonical ``output`` dict, and Bedrock-specific
        provenance (``framework``, ``tool_type``) is folded onto the
        canonical ``input``.
        """
        kb = step.get("knowledgeBaseLookupOutput", {})
        kb_input = self._safe_serialize(step.get("knowledgeBaseLookupInput"))
        retrieved = self._safe_serialize(kb.get("retrievedReferences"))
        input_data: dict[str, Any]
        if isinstance(kb_input, dict):
            input_data = dict(kb_input)
        elif kb_input is None:
            input_data = {}
        else:
            input_data = {"value": kb_input}
        input_data["framework"] = "bedrock_agents"
        input_data["tool_type"] = "knowledge_base_retrieval"
        output_data: dict[str, Any] | None
        if isinstance(retrieved, dict):
            output_data = dict(retrieved)
        elif retrieved is None:
            output_data = None
        else:
            output_data = {"value": retrieved}
        self.emit_event(
            ToolCallEvent.create(
                name=step.get("knowledgeBaseId", "knowledge_base"),
                version="unavailable",
                integration=IntegrationType.SERVICE,
                input_data=input_data,
                output_data=output_data,
            )
        )

    def _emit_model_invocation(self, step: dict[str, Any]) -> None:
        """Emit typed :class:`ModelInvokeEvent` (and :class:`CostRecordEvent`).

        AWS Bedrock is the canonical provider here (``aws_bedrock``);
        the model identifier is the foundation model id (e.g.
        ``anthropic.claude-v2``). The canonical schema requires
        ``provider`` + ``name``; ``version`` falls back to
        ``"unavailable"`` per the NORMATIVE rule. Bedrock-specific
        provenance (``framework``) is carried on
        :attr:`ModelInfo.parameters`. Token usage is mirrored onto the
        canonical ``prompt_tokens`` / ``completion_tokens`` slots, and
        a paired :class:`CostRecordEvent` is emitted when the
        underlying Bedrock response contains usage metrics.
        """
        invocation = step.get("modelInvocationOutput", {})
        model_id = step.get("foundationModel") or "unknown"
        usage = invocation.get("usage", {})
        prompt_tokens = usage.get("inputTokens") if usage else None
        completion_tokens = usage.get("outputTokens") if usage else None
        self.emit_event(
            ModelInvokeEvent.create(
                provider="aws_bedrock",
                name=model_id,
                version="unavailable",
                parameters={"framework": "bedrock_agents"},
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        if usage:
            total = (usage.get("inputTokens") or 0) + (usage.get("outputTokens") or 0)
            self.emit_event(
                CostRecordEvent.create(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    tokens=total,
                )
            )

    def _emit_collaborator_handoff(self, step: dict[str, Any]) -> None:
        """Emit a typed :class:`AgentHandoffEvent` for a Bedrock supervisor
        delegation.

        The canonical schema requires ``handoff_context_hash`` in
        ``sha256:<hex64>`` format. Bedrock supervisor→collaborator
        steps do not carry an explicit context payload, so the hash
        is computed deterministically from the
        (supervisor, collaborator, reason) tuple — this preserves the
        wire-format guarantee while remaining stable across replays.
        """
        from_agent = step.get("supervisorAgentId", "supervisor")
        to_agent = step.get("collaboratorAgentId", "collaborator")
        reason = "supervisor_delegation"
        self.emit_event(
            AgentHandoffEvent.create(
                from_agent=from_agent,
                to_agent=to_agent,
                handoff_context_hash=_sha256_of(
                    f"{reason}::{from_agent}::{to_agent}"
                ),
            )
        )

    # --- Lifecycle Hooks ---

    def on_invoke_start(self, agent_id: str | None = None, input_text: str | None = None) -> None:
        """Emit a typed :class:`AgentInputEvent` for a manual invoke start.

        Bedrock-specific provenance (``framework``, ``agent_id``,
        ``timestamp_ns``, ``raw_input``) lives on
        :class:`MessageContent.metadata`.
        """
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._invoke_starts[tid] = start_ns
            self.emit_event(
                AgentInputEvent.create(
                    message=_stringify(input_text),
                    role=MessageRole.HUMAN,
                    metadata={
                        "framework": "bedrock_agents",
                        "agent_id": agent_id,
                        "timestamp_ns": start_ns,
                        "raw_input": input_text,
                    },
                )
            )
        except Exception:
            logger.warning("Error in on_invoke_start", exc_info=True)

    def on_invoke_end(
        self,
        agent_id: str | None = None,
        output: Any = None,
        error: Exception | None = None,
    ) -> None:
        """Emit a typed :class:`AgentOutputEvent` for a manual invoke end.

        Bedrock-specific provenance (``framework``, ``agent_id``,
        ``duration_ns``, ``raw_output``, ``error``, ``run_status``)
        lives on :class:`MessageContent.metadata`.
        """
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._invoke_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            serialised_output = self._safe_serialize(output)
            metadata: dict[str, Any] = {
                "framework": "bedrock_agents",
                "agent_id": agent_id,
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
            logger.warning("Error in on_invoke_end", exc_info=True)

    def on_tool_use(
        self,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        error: Exception | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Emit a typed :class:`ToolCallEvent` for a tool invocation.

        Bedrock surfaces tool calls primarily through action groups
        (cloud services). The manual ``on_tool_use`` hook is generic,
        so ``integration`` defaults to :attr:`IntegrationType.LIBRARY`
        — callers needing a different integration kind can wrap the
        adapter and emit a typed event directly. Bedrock-specific
        provenance (``framework``) is folded onto the canonical
        ``input``.
        """
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
            input_data["framework"] = "bedrock_agents"
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
        """Emit a typed :class:`ModelInvokeEvent` for a manual LLM call.

        Bedrock-specific provenance (``framework``) is carried on
        :attr:`ModelInfo.parameters`. Provider falls back to
        ``aws_bedrock`` when the caller supplies none — the manual
        hook is exclusively for Bedrock-routed model calls.
        """
        if not self._connected:
            return
        try:
            self.emit_event(
                ModelInvokeEvent.create(
                    provider=provider or "aws_bedrock",
                    name=model or "unknown",
                    version="unavailable",
                    parameters={"framework": "bedrock_agents"},
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
        """Emit a typed :class:`AgentHandoffEvent` for a manual handoff.

        Empty contexts are still hashed (over the empty string) so
        the canonical wire format is uniform — the previous adapter
        emitted ``None`` when context was missing, which the canonical
        validator rejects.
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

    def _extract_completion(self, parsed: dict[str, Any]) -> str | None:
        """Extract completion text from the boto3 parsed response.

        IMPORTANT: We do NOT consume the 'completion' EventStream directly
        as that would prevent the caller from reading the response. Instead
        we extract from already-parsed metadata fields that boto3 populates.
        """
        # Try the output text field (populated by boto3 after-call parsing)
        output_text = parsed.get("outputText")
        if output_text:
            return str(output_text)
        # Try the output field
        output = parsed.get("output", {})
        if isinstance(output, dict):
            text = output.get("text")
            if text:
                return str(text)
        # Fallback: serialize whatever non-stream data is available
        for key in ("returnControlInvocationResults", "sessionAttributes"):
            val = parsed.get(key)
            if val:
                serialized = self._safe_serialize(val)
                return str(serialized) if serialized is not None else None
        return None

    def _emit_agent_config(self, agent_id: str, params: dict[str, Any]) -> None:
        """Emit a typed :class:`EnvironmentConfigEvent` once per agent.

        Bedrock Agents runs in AWS, so the canonical
        :attr:`EnvironmentType.CLOUD` enum value is used. Agent
        configuration (``agent_id``, ``agent_alias_id``,
        ``enable_trace``) lives on :attr:`EnvironmentInfo.attributes`.
        """
        with self._adapter_lock:
            if agent_id in self._seen_agents:
                return
            self._seen_agents.add(agent_id)
        self.emit_event(
            EnvironmentConfigEvent.create(
                env_type=EnvironmentType.CLOUD,
                attributes={
                    "framework": "bedrock_agents",
                    "agent_id": agent_id,
                    "agent_alias_id": params.get("agentAliasId"),
                    "enable_trace": params.get("enableTrace", False),
                },
            )
        )

    def _safe_serialize(self, value: Any) -> Any:
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
