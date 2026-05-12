"""
AWS Bedrock Agents adapter lifecycle.

Instrumentation strategy: boto3 event hooks + OTel (ADOT integration)
  invoke_agent request  → agent.input (L1)
  invoke_agent response → agent.output (L1)
  Action Group          → tool.call (L5a)
  Knowledge Base query  → tool.call (L5a, retrieval)
  Model invocation      → model.invoke (L3)
  Supervisor→Collaborator → agent.handoff (Cross)
"""

from __future__ import annotations

import time
import uuid
import hashlib
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
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

logger = logging.getLogger(__name__)


class BedrockAgentsAdapter(BaseAdapter):
    """LayerLens adapter for AWS Bedrock Agents."""

    FRAMEWORK = "bedrock_agents"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/bedrock_agents/``). Bedrock Agents is a
    # remote AWS service consumed via boto3 hooks — boto3 does not use
    # Pydantic. Adapter emits plain dict events.
    requires_pydantic = PydanticCompat.V1_OR_V2

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
        # Per-thread cache of input + tool list for memory persistence.
        # boto3 hooks fire on the thread that invoked the SDK call, so
        # thread-id keying is safe and matches _invoke_starts.
        self._invoke_inputs: dict[int, Any] = {}
        self._invoke_agent_ids: dict[int, str] = {}
        self._invoke_tools: dict[int, list[str]] = {}

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
        self._invoke_inputs.clear()
        self._invoke_agent_ids.clear()
        self._invoke_tools.clear()
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
        """Serialize the current trace data for replay.

        Includes the per-adapter memory snapshot (cross-poll #1) under
        ``metadata["memory_snapshot"]``. Bedrock session attributes are
        consolidated into the snapshot at after_invoke so a replay
        engine can reconstruct cross-session recall.
        """
        return ReplayableTrace(
            adapter_name="BedrockAgentsAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
            metadata={"memory_snapshot": self.memory_snapshot_dict()},
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
        if not self._connected:
            return
        try:
            params = kwargs.get("params", {})
            tid = threading.get_ident()
            start_ns = time.time_ns()
            agent_id = params.get("agentId", "unknown")
            input_text = params.get("inputText")
            with self._adapter_lock:
                self._invoke_starts[tid] = start_ns
                self._invoke_inputs[tid] = input_text
                self._invoke_agent_ids[tid] = agent_id
                self._invoke_tools[tid] = []
            self._emit_agent_config(agent_id, params)
            self.emit_dict_event(
                "agent.input",
                {
                    "framework": "bedrock_agents",
                    "agent_id": agent_id,
                    "session_id": params.get("sessionId"),
                    "input": input_text,
                    "enable_trace": params.get("enableTrace", False),
                    "timestamp_ns": start_ns,
                },
            )
        except Exception:
            logger.warning("Error in _before_invoke_agent", exc_info=True)

    def _after_invoke_agent(self, **kwargs: Any) -> None:
        if not self._connected:
            return
        try:
            parsed = kwargs.get("parsed", {})
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._invoke_starts.pop(tid, 0)
                input_text = self._invoke_inputs.pop(tid, None)
                agent_id = self._invoke_agent_ids.pop(tid, "")
                tool_names = self._invoke_tools.pop(tid, [])
            duration_ns = end_ns - start_ns if start_ns else 0
            output = self._extract_completion(parsed)
            self.emit_dict_event(
                "agent.output",
                {
                    "framework": "bedrock_agents",
                    "output": output,
                    "duration_ns": duration_ns,
                    "session_id": parsed.get("sessionId"),
                },
            )
            # Extract trace steps if available — also populates
            # _invoke_tools[tid] for the in-flight invocation when
            # supervisor/collaborator orchestration emits action-group
            # steps. The caller's ``tool_names`` list is appended to
            # in-place by _process_trace.
            self._process_trace(parsed, tool_names=tool_names)
            # Cross-poll #1: persist this turn into memory.
            self.record_memory_turn(
                agent_name=agent_id,
                input_data=self._safe_serialize(input_text),
                output_data=self._safe_serialize(output),
                tools=tool_names or None,
            )
        except Exception:
            logger.warning("Error in _after_invoke_agent", exc_info=True)

    def _process_trace(
        self,
        parsed: dict[str, Any],
        tool_names: list[str] | None = None,
    ) -> None:
        """Extract trace steps from Bedrock response and emit events.

        Args:
            parsed: The boto3-parsed InvokeAgent response.
            tool_names: Mutable list the caller passed in to receive
                the tool-name roll-up. When supplied, action-group /
                knowledge-base step names are appended for memory
                persistence at the caller's :meth:`record_memory_turn`.
        """
        trace = parsed.get("trace", {})
        steps = trace.get("trace", {}).get("orchestrationTrace", {}).get("steps", [])
        if not steps and isinstance(trace, dict):
            # Try alternative trace structure
            steps = trace.get("steps", [])
        for step in steps:
            step_type = step.get("type", "")
            if step_type == "ACTION_GROUP":
                self._emit_action_group(step)
                if tool_names is not None:
                    name = step.get("actionGroupName")
                    if name:
                        tool_names.append(str(name))
            elif step_type == "KNOWLEDGE_BASE":
                self._emit_knowledge_base(step)
                if tool_names is not None:
                    name = step.get("knowledgeBaseId") or "knowledge_base"
                    tool_names.append(str(name))
            elif step_type == "MODEL_INVOCATION":
                self._emit_model_invocation(step)
            elif step_type == "AGENT_COLLABORATOR":
                self._emit_collaborator_handoff(step)

    def _emit_action_group(self, step: dict[str, Any]) -> None:
        action = step.get("actionGroupInvocationOutput", {})
        self.emit_dict_event(
            "tool.call",
            {
                "framework": "bedrock_agents",
                "tool_name": step.get("actionGroupName", "unknown"),
                "tool_input": self._safe_serialize(step.get("actionGroupInput")),
                "tool_output": self._safe_serialize(action.get("output")),
                "tool_type": "action_group",
            },
        )

    def _emit_knowledge_base(self, step: dict[str, Any]) -> None:
        kb = step.get("knowledgeBaseLookupOutput", {})
        self.emit_dict_event(
            "tool.call",
            {
                "framework": "bedrock_agents",
                "tool_name": step.get("knowledgeBaseId", "knowledge_base"),
                "tool_input": self._safe_serialize(step.get("knowledgeBaseLookupInput")),
                "tool_output": self._safe_serialize(kb.get("retrievedReferences")),
                "tool_type": "knowledge_base_retrieval",
            },
        )

    def _emit_model_invocation(self, step: dict[str, Any]) -> None:
        invocation = step.get("modelInvocationOutput", {})
        payload: dict[str, Any] = {
            "framework": "bedrock_agents",
            "provider": "aws_bedrock",
        }
        model_id = step.get("foundationModel")
        if model_id:
            payload["model"] = model_id
        usage = invocation.get("usage", {})
        if usage:
            payload["tokens_prompt"] = usage.get("inputTokens")
            payload["tokens_completion"] = usage.get("outputTokens")
        self.emit_dict_event("model.invoke", payload)
        if usage:
            self.emit_dict_event(
                "cost.record",
                {
                    "framework": "bedrock_agents",
                    "model": model_id,
                    "tokens_prompt": usage.get("inputTokens"),
                    "tokens_completion": usage.get("outputTokens"),
                    "tokens_total": (usage.get("inputTokens") or 0) + (usage.get("outputTokens") or 0),
                },
            )

    def _emit_collaborator_handoff(self, step: dict[str, Any]) -> None:
        self.emit_dict_event(
            "agent.handoff",
            {
                "from_agent": step.get("supervisorAgentId", "supervisor"),
                "to_agent": step.get("collaboratorAgentId", "collaborator"),
                "reason": "supervisor_delegation",
                "framework": "bedrock_agents",
            },
        )

    # --- Lifecycle Hooks ---

    def on_invoke_start(self, agent_id: str | None = None, input_text: str | None = None) -> None:
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._invoke_starts[tid] = start_ns
            self.emit_dict_event(
                "agent.input",
                {
                    "framework": "bedrock_agents",
                    "agent_id": agent_id,
                    "input": input_text,
                    "timestamp_ns": start_ns,
                },
            )
        except Exception:
            logger.warning("Error in on_invoke_start", exc_info=True)

    def on_invoke_end(
        self,
        agent_id: str | None = None,
        output: Any = None,
        error: Exception | None = None,
        input_data: Any = None,
        tools: list[str] | None = None,
    ) -> None:
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._invoke_starts.pop(tid, 0)
                if input_data is None:
                    input_data = self._invoke_inputs.pop(tid, None)
                else:
                    self._invoke_inputs.pop(tid, None)
                if tools is None:
                    tools = self._invoke_tools.pop(tid, []) or None
                else:
                    self._invoke_tools.pop(tid, None)
                self._invoke_agent_ids.pop(tid, None)
            duration_ns = end_ns - start_ns if start_ns else 0
            payload: dict[str, Any] = {
                "framework": "bedrock_agents",
                "agent_id": agent_id,
                "output": self._safe_serialize(output),
                "duration_ns": duration_ns,
            }
            if error:
                payload["error"] = str(error)
            self.emit_dict_event("agent.output", payload)
            # Cross-poll #1: persist this turn into memory.
            self.record_memory_turn(
                agent_name=agent_id,
                input_data=self._safe_serialize(input_data),
                output_data=self._safe_serialize(output),
                error=str(error) if error else None,
                tools=tools,
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
        if not self._connected:
            return
        try:
            payload: dict[str, Any] = {
                "framework": "bedrock_agents",
                "tool_name": tool_name,
                "tool_input": self._safe_serialize(tool_input),
                "tool_output": self._safe_serialize(tool_output),
            }
            if error:
                payload["error"] = str(error)
            if latency_ms is not None:
                payload["latency_ms"] = latency_ms
            self.emit_dict_event("tool.call", payload)
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
        if not self._connected:
            return
        try:
            payload: dict[str, Any] = {"framework": "bedrock_agents"}
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
        except Exception:
            logger.warning("Error in on_llm_call", exc_info=True)

    def on_handoff(self, from_agent: str, to_agent: str, context: Any = None) -> None:
        if not self._connected:
            return
        try:
            context_str = str(context) if context else ""
            self.emit_dict_event(
                "agent.handoff",
                {
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "reason": "supervisor_delegation",
                    "context_hash": hashlib.sha256(context_str.encode()).hexdigest() if context_str else None,
                },
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
        with self._adapter_lock:
            if agent_id in self._seen_agents:
                return
            self._seen_agents.add(agent_id)
        self.emit_dict_event(
            "environment.config",
            {
                "framework": "bedrock_agents",
                "agent_id": agent_id,
                "agent_alias_id": params.get("agentAliasId"),
                "enable_trace": params.get("enableTrace", False),
            },
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
