from __future__ import annotations

import logging
from typing import Any, Set, Dict, Optional

from .._base import resilient_callback
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

    @resilient_callback(callback_name="_before_invoke")
    def _before_invoke(self, **kwargs: Any) -> None:
        if not self._connected:
            return
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

    def _after_invoke(self, **kwargs: Any) -> None:
        # _end_run() MUST run regardless of telemetry failures (otherwise
        # collector/span ContextVars leak across boto3 calls). Keep the
        # ``finally`` here at the OUTER level and delegate the resilient
        # body to a helper wrapped with @resilient_callback.
        if not self._connected:
            return
        try:
            self._after_invoke_body(**kwargs)
        finally:
            self._end_run()

    @resilient_callback(callback_name="_after_invoke")
    def _after_invoke_body(self, **kwargs: Any) -> None:
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

    # ------------------------------------------------------------------
    # Trace step processing
    # ------------------------------------------------------------------

    def _process_step(self, step: Dict[str, Any]) -> None:
        handler_name = _STEP_DISPATCH.get(step.get("type", ""))
        if handler_name is not None:
            getattr(self, handler_name)(step)

    def _on_action_group(self, step: Dict[str, Any]) -> None:
        action_output = (
            step.get("actionGroupInvocationOutput", {})
            if isinstance(step.get("actionGroupInvocationOutput"), dict)
            else {}
        )
        payload = self._payload(
            tool_name=step.get("actionGroupName", "unknown"),
            tool_type="action_group",
        )
        # Function / API schema introspection: Bedrock action groups ship both
        # the called function name and the HTTP verb + resource path when the
        # schema is an OpenAPI doc. Surface both so action-group tool.call events
        # can be cross-referenced with the action group definition.
        function = step.get("function") or action_output.get("function")
        if function:
            payload["function"] = str(function)
        verb = step.get("verb") or action_output.get("verb")
        if verb:
            payload["verb"] = str(verb)
        api_path = step.get("apiPath") or action_output.get("apiPath")
        if api_path:
            payload["api_path"] = str(api_path)
        execution_type = step.get("executionType") or action_output.get("executionType")
        if execution_type:
            payload["execution_type"] = str(execution_type)
        invocation_id = action_output.get("invocationId") or step.get("invocationId")
        if invocation_id:
            payload["invocation_id"] = str(invocation_id)
        status = action_output.get("responseState") or action_output.get("status")
        if status:
            payload["status"] = str(status)
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
        # Retrieval ranking: surface the number of references + their scores so
        # we can measure retrieval quality across runs without capturing raw
        # chunks (which tend to be large and may contain PII).
        if isinstance(refs, list):
            payload["num_results"] = len(refs)
            scores: list[float] = []
            sources: list[str] = []
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                score = ref.get("score")
                if isinstance(score, (int, float)):
                    scores.append(float(score))
                location = ref.get("location") or {}
                if isinstance(location, dict):
                    s3 = location.get("s3Location") or {}
                    if isinstance(s3, dict) and s3.get("uri"):
                        sources.append(str(s3["uri"]))
            if scores:
                payload["retrieval_scores"] = scores[:20]
                payload["retrieval_score_max"] = max(scores)
                payload["retrieval_score_min"] = min(scores)
            if sources:
                payload["retrieval_sources"] = sources[:20]
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
        payload = self._payload(
            from_agent=step.get("supervisorAgentId", "supervisor"),
            to_agent=step.get("collaboratorAgentId", "collaborator"),
            reason="supervisor_delegation",
        )
        # Collaborator metadata: the supervisor's rationale for delegating
        # ("why this agent?") and the task it's handing off. This is what
        # makes a multi-agent trace readable without replaying every step.
        for key in ("collaboratorName", "collaboratorDescription", "collaboratorInvocationType"):
            val = step.get(key)
            if val:
                payload[_snake(key)] = val
        rationale = step.get("rationale") or step.get("reasoning")
        if rationale:
            self._set_if_capturing(payload, "rationale", str(rationale)[:1000])
        task_input = step.get("invocationInput") or step.get("collaboratorInvocationInput")
        self._set_if_capturing(payload, "input", safe_serialize(task_input))
        self._emit("agent.handoff", payload, span_name="bedrock.handoff")

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


def _snake(camel: str) -> str:
    out = []
    for i, ch in enumerate(camel):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
