from __future__ import annotations

import logging
from typing import Any, Set, Dict, Optional

from ._utils import safe_serialize
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    import boto3  # pyright: ignore[reportMissingImports]  # noqa: F401

    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False


_BEFORE_HOOK = "provide-client-params.bedrock-agent-runtime.InvokeAgent"
_AFTER_HOOK = "after-call.bedrock-agent-runtime.InvokeAgent"

_STEP_DISPATCH = {
    "ACTION_GROUP": "_on_action_group",
    "KNOWLEDGE_BASE": "_on_knowledge_base",
    "MODEL_INVOCATION": "_on_model_invocation",
    "AGENT_COLLABORATOR": "_on_collaborator_handoff",
}


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _extract_completion(parsed: Dict[str, Any]) -> Optional[str]:
    output_text = parsed.get("outputText")
    if output_text:
        return str(output_text)
    output = parsed.get("output", {})
    if isinstance(output, dict):
        text = output.get("text")
        if text:
            return str(text)
    for key in ("returnControlInvocationResults", "sessionAttributes"):
        val = parsed.get(key)
        if val:
            return str(safe_serialize(val))
    return None


def _collect_steps(parsed: Dict[str, Any]) -> list:
    trace = parsed.get("trace", {})
    if not isinstance(trace, dict):
        return []
    steps = []
    inner = trace.get("trace", {})
    if isinstance(inner, dict):
        orch = inner.get("orchestrationTrace", {})
        if isinstance(orch, dict):
            steps.extend(orch.get("steps", []))
    steps.extend(trace.get("steps", []))
    return steps


class BedrockAgentsAdapter(FrameworkAdapter):
    """AWS Bedrock Agents adapter using boto3 event hooks.

    Registers ``provide-client-params`` and ``after-call`` hooks on a
    ``bedrock-agent-runtime`` client to capture agent invocations, trace
    steps, and emit flat events.

    Uses ``_begin_run`` / ``_end_run`` per ``InvokeAgent`` call — boto3
    hooks fire synchronously in the calling thread, so ContextVars work.

    Usage::

        client = boto3.client("bedrock-agent-runtime")
        adapter = BedrockAgentsAdapter(ll_client)
        adapter.connect(target=client)
        response = client.invoke_agent(agentId=..., ...)
        adapter.disconnect()
    """

    name = "bedrock_agents"
    package = "bedrock"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._boto_client: Optional[Any] = None
        self._seen_agents: Set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_BOTO3)
        if target is None:
            raise ValueError("connect() requires a bedrock-agent-runtime boto3 client as target")
        self._boto_client = target
        event_system = target.meta.events
        event_system.register(_BEFORE_HOOK, self._before_invoke)
        event_system.register(_AFTER_HOOK, self._after_invoke)

    def _on_disconnect(self) -> None:
        if self._boto_client is not None:
            try:
                ev = self._boto_client.meta.events
                ev.unregister(_BEFORE_HOOK, self._before_invoke)
                ev.unregister(_AFTER_HOOK, self._after_invoke)
            except Exception:
                log.debug("layerlens: could not unregister boto3 event hooks", exc_info=True)
            self._boto_client = None
        with self._lock:
            self._seen_agents.clear()

    # ------------------------------------------------------------------
    # boto3 event hooks
    # ------------------------------------------------------------------

    def _before_invoke(self, **kwargs: Any) -> None:
        if not self._connected:
            return
        try:
            params = kwargs.get("params", {})
            agent_id = params.get("agentId", "unknown")

            self._begin_run()
            self._start_timer("invoke")

            self._emit_agent_config(agent_id, params)

            root = self._get_root_span()
            payload = self._payload(
                agent_id=agent_id,
                session_id=params.get("sessionId"),
                enable_trace=params.get("enableTrace", False),
            )
            self._set_if_capturing(payload, "input", params.get("inputText"))
            self._emit(
                "agent.input",
                payload,
                span_id=root,
                parent_span_id=None,
                span_name="bedrock.invoke_agent",
            )
        except Exception:
            log.warning("layerlens: error in _before_invoke", exc_info=True)

    def _after_invoke(self, **kwargs: Any) -> None:
        if not self._connected:
            return
        try:
            parsed = kwargs.get("parsed", {})
            latency_ms = self._stop_timer("invoke")
            output = _extract_completion(parsed)

            root = self._get_root_span()
            payload = self._payload(session_id=parsed.get("sessionId"))
            if latency_ms is not None:
                payload["latency_ms"] = latency_ms
            self._set_if_capturing(payload, "output", output)
            self._emit(
                "agent.output",
                payload,
                span_id=root,
                parent_span_id=None,
                span_name="bedrock.invoke_agent",
            )

            for step in _collect_steps(parsed):
                self._process_step(step)
        except Exception:
            log.warning("layerlens: error in _after_invoke", exc_info=True)
        finally:
            self._end_run()

    # ------------------------------------------------------------------
    # Trace step processing
    # ------------------------------------------------------------------

    def _process_step(self, step: Dict[str, Any]) -> None:
        handler_name = _STEP_DISPATCH.get(step.get("type", ""))
        if handler_name is not None:
            getattr(self, handler_name)(step)

    def _on_action_group(self, step: Dict[str, Any]) -> None:
        action_output = step.get("actionGroupInvocationOutput", {})
        payload = self._payload(
            tool_name=step.get("actionGroupName", "unknown"),
            tool_type="action_group",
        )
        self._set_if_capturing(payload, "input", safe_serialize(step.get("actionGroupInput")))
        output = action_output.get("output") if isinstance(action_output, dict) else None
        self._set_if_capturing(payload, "output", safe_serialize(output))
        self._emit("tool.call", payload, span_name="bedrock.action_group")

    def _on_knowledge_base(self, step: Dict[str, Any]) -> None:
        kb_output = step.get("knowledgeBaseLookupOutput", {})
        payload = self._payload(
            tool_name=step.get("knowledgeBaseId", "knowledge_base"),
            tool_type="knowledge_base_retrieval",
        )
        self._set_if_capturing(payload, "input", safe_serialize(step.get("knowledgeBaseLookupInput")))
        refs = kb_output.get("retrievedReferences") if isinstance(kb_output, dict) else None
        self._set_if_capturing(payload, "output", safe_serialize(refs))
        self._emit("tool.call", payload, span_name="bedrock.knowledge_base")

    def _on_model_invocation(self, step: Dict[str, Any]) -> None:
        invocation = step.get("modelInvocationOutput", {})
        model_id = step.get("foundationModel")
        usage = invocation.get("usage", {}) if isinstance(invocation, dict) else {}

        tokens_prompt = usage.get("inputTokens", 0) or 0 if isinstance(usage, dict) else 0
        tokens_completion = usage.get("outputTokens", 0) or 0 if isinstance(usage, dict) else 0

        span_id = self._new_span_id()
        payload = self._payload(provider="aws_bedrock")
        if model_id:
            payload["model"] = model_id
        if tokens_prompt:
            payload["tokens_prompt"] = tokens_prompt
        if tokens_completion:
            payload["tokens_completion"] = tokens_completion
        if tokens_prompt or tokens_completion:
            payload["tokens_total"] = tokens_prompt + tokens_completion
        self._emit("model.invoke", payload, span_id=span_id, span_name="bedrock.model")

        if tokens_prompt or tokens_completion:
            cost_payload = self._payload(
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                tokens_total=tokens_prompt + tokens_completion,
            )
            if model_id:
                cost_payload["model"] = model_id
            self._emit("cost.record", cost_payload, span_id=span_id)

    def _on_collaborator_handoff(self, step: Dict[str, Any]) -> None:
        self._emit(
            "agent.handoff",
            self._payload(
                from_agent=step.get("supervisorAgentId", "supervisor"),
                to_agent=step.get("collaboratorAgentId", "collaborator"),
                reason="supervisor_delegation",
            ),
            span_name="bedrock.handoff",
        )

    # ------------------------------------------------------------------
    # Environment config
    # ------------------------------------------------------------------

    def _emit_agent_config(self, agent_id: str, params: Dict[str, Any]) -> None:
        with self._lock:
            if agent_id in self._seen_agents:
                return
            self._seen_agents.add(agent_id)
        self._emit(
            "environment.config",
            self._payload(
                agent_id=agent_id,
                agent_alias_id=params.get("agentAliasId"),
                enable_trace=params.get("enableTrace", False),
            ),
            span_name="bedrock.config",
        )
