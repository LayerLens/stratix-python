"""
AWS Strands adapter lifecycle.

Instrumentation strategy: Agent wrapper (run wrapping) + callback hooks
  Agent start            -> agent.input (L1)
  Agent end              -> agent.output (L1)
  Tool call              -> tool.call (L5a)
  Model invoke (Bedrock) -> model.invoke (L3)
  Conversation state     -> agent.state.change (Cross)
  Cost (Bedrock pricing) -> cost.record (Cross)
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
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

logger = logging.getLogger(__name__)


class StrandsAdapter(BaseAdapter):
    """LayerLens adapter for AWS Strands."""

    FRAMEWORK = "strands"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/strands/``). Strands instrumentation hooks
    # into agent callbacks and emits dict events without crossing the
    # framework's Pydantic boundary.
    requires_pydantic = PydanticCompat.V1_OR_V2

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: Any | None = None,
        stratix_instance: Any | None = None,
    ) -> None:
        resolved = stratix or stratix_instance
        super().__init__(stratix=resolved, capture_config=capture_config)
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
        """Extract tool calls, model invocations, and cost from run result."""
        if result is None:
            return
        try:
            # Extract model invocation details
            model = getattr(agent, "model", None) or getattr(agent, "model_id", None)
            if model:
                model_name = str(model)
                self.emit_dict_event(
                    "model.invoke",
                    {
                        "framework": "strands",
                        "model": model_name,
                        "provider": self._detect_provider(model_name),
                    },
                )

            # Extract usage/token info from result
            usage = getattr(result, "usage", None) or getattr(result, "metrics", None)
            if usage:
                tokens_prompt = getattr(usage, "inputTokens", None) or getattr(
                    usage, "prompt_tokens", None
                )
                tokens_completion = getattr(usage, "outputTokens", None) or getattr(
                    usage, "completion_tokens", None
                )
                tokens_total = getattr(usage, "totalTokens", None) or getattr(
                    usage, "total_tokens", None
                )
                self.emit_dict_event(
                    "cost.record",
                    {
                        "framework": "strands",
                        "model": str(model) if model else None,
                        "tokens_prompt": tokens_prompt,
                        "tokens_completion": tokens_completion,
                        "tokens_total": tokens_total,
                    },
                )

            # Extract tool calls from result
            tool_results = getattr(result, "tool_results", None) or []
            for tr in tool_results:
                self.emit_dict_event(
                    "tool.call",
                    {
                        "framework": "strands",
                        "tool_name": getattr(tr, "name", None) or tr.get("name", "unknown")
                        if isinstance(tr, dict)
                        else "unknown",
                        "tool_input": self._safe_serialize(
                            getattr(tr, "input", None)
                            or (tr.get("input") if isinstance(tr, dict) else None)
                        ),
                        "tool_output": self._safe_serialize(
                            getattr(tr, "output", None)
                            or (tr.get("output") if isinstance(tr, dict) else None)
                        ),
                    },
                )

            # Emit conversation state change
            conversation = getattr(agent, "conversation", None) or getattr(
                agent, "conversation_manager", None
            )
            if conversation:
                turn_count = getattr(conversation, "turn_count", None) or len(
                    getattr(conversation, "messages", [])
                )
                self.emit_dict_event(
                    "agent.state.change",
                    {
                        "framework": "strands",
                        "agent_name": getattr(agent, "name", "strands_agent"),
                        "event_subtype": "conversation_update",
                        "turn_count": turn_count,
                    },
                )
        except Exception:
            logger.debug("Could not extract run details", exc_info=True)

    # --- Lifecycle Hooks ---

    def on_run_start(self, agent_name: str | None = None, input_data: Any = None) -> None:
        """Emit agent.input event when an agent run starts."""
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._run_starts[tid] = start_ns
            self.emit_dict_event(
                "agent.input",
                {
                    "framework": "strands",
                    "agent_name": agent_name,
                    "input": self._safe_serialize(input_data),
                    "timestamp_ns": start_ns,
                },
            )
        except Exception:
            logger.warning("Error in on_run_start", exc_info=True)

    def on_run_end(
        self,
        agent_name: str | None = None,
        output: Any = None,
        error: Exception | None = None,
    ) -> None:
        """Emit agent.output event when an agent run ends."""
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._run_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            payload: dict[str, Any] = {
                "framework": "strands",
                "agent_name": agent_name,
                "output": self._safe_serialize(output),
                "duration_ns": duration_ns,
            }
            if error:
                payload["error"] = str(error)
            self.emit_dict_event("agent.output", payload)
            self.emit_dict_event(
                "agent.state.change",
                {
                    "framework": "strands",
                    "agent_name": agent_name,
                    "event_subtype": "run_complete" if not error else "run_failed",
                },
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
        """Emit tool.call event for a tool invocation."""
        if not self._connected:
            return
        try:
            payload: dict[str, Any] = {
                "framework": "strands",
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
        """Emit model.invoke event for an LLM call."""
        if not self._connected:
            return
        try:
            payload: dict[str, Any] = {"framework": "strands"}
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
        """Emit environment.config event for agent configuration on first encounter."""
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        metadata: dict[str, Any] = {
            "framework": "strands",
            "agent_name": agent_name,
        }
        model = getattr(agent, "model", None) or getattr(agent, "model_id", None)
        if model:
            metadata["model"] = str(model)
        system_prompt = getattr(agent, "system_prompt", None)
        if system_prompt and self._capture_config.capture_content:
            metadata["system_prompt"] = str(system_prompt)[:500]
        tools = getattr(agent, "tools", None)
        if tools:
            if isinstance(tools, dict):
                metadata["tools"] = list(tools.keys())
            else:
                metadata["tools"] = [
                    getattr(t, "name", None) or getattr(t, "tool_name", str(t)) for t in tools
                ]
        conversation = getattr(agent, "conversation", None) or getattr(
            agent, "conversation_manager", None
        )
        if conversation:
            metadata["conversation_type"] = str(type(conversation).__name__)
        self.emit_dict_event("environment.config", metadata)

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
