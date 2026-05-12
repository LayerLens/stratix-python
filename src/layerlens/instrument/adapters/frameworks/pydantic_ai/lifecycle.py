"""
PydanticAI adapter lifecycle.

Instrumentation strategy: OTel wrapper (Logfire-compatible) + Agent wrapper
  Agent.run() start    → agent.input (L1)
  Agent.run() end      → agent.output (L1)
  ModelRequestNode     → model.invoke (L3)
  CallToolsNode        → tool.call (L5a)
  AgentRun transitions → agent.state.change (Cross)
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


class PydanticAIAdapter(BaseAdapter):
    """LayerLens adapter for PydanticAI."""

    FRAMEWORK = "pydantic_ai"
    VERSION = "0.1.0"
    # Pydantic-AI is built on Pydantic v2 from day one — see
    # pydantic-ai's own pyproject which requires ``pydantic>=2.7``.
    # There is no v1 path; the framework cannot be installed alongside v1.
    requires_pydantic = PydanticCompat.V2_ONLY

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
        try:
            import pydantic_ai  # type: ignore[import-not-found,unused-ignore]

            self._framework_version = getattr(pydantic_ai, "__version__", "unknown")
        except ImportError:
            logger.debug("pydantic-ai not installed")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        for agent in self._wrapped_agents:
            self._unwrap_agent(agent)
        self._wrapped_agents.clear()
        self._originals.clear()
        self._seen_agents.clear()
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
            name="PydanticAIAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
            ],
            description="LayerLens adapter for PydanticAI",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="PydanticAIAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
        )

    # --- Framework Integration ---

    def instrument_agent(self, agent: Any) -> Any:
        """Wrap PydanticAI agent.run() methods to capture lifecycle events."""
        agent_id = id(agent)
        if agent_id in self._originals:
            return agent
        originals: dict[str, Any] = {}
        # Wrap run()
        if hasattr(agent, "run"):
            originals["run"] = agent.run
            agent.run = self._create_traced_run(agent, agent.run)
        # Wrap run_sync()
        if hasattr(agent, "run_sync"):
            originals["run_sync"] = agent.run_sync
            agent.run_sync = self._create_traced_run_sync(agent, agent.run_sync)
        self._originals[agent_id] = originals
        self._wrapped_agents.append(agent)
        agent_name = getattr(agent, "name", None) or str(type(agent).__name__)
        self._emit_agent_config(agent_name, agent)
        return agent

    def _create_traced_run(self, agent: Any, original_run: Any) -> Any:
        adapter = self

        async def traced_run(*args: Any, **kwargs: Any) -> Any:
            agent_name = getattr(agent, "name", None) or "pydantic_ai_agent"
            user_prompt = args[0] if args else kwargs.get("user_prompt")
            adapter.on_run_start(agent_name=agent_name, input_data=user_prompt)
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
                    output = getattr(result, "data", result)
                adapter.on_run_end(agent_name=agent_name, output=output, error=error)
                adapter._extract_run_usage(result)
            return result

        traced_run._layerlens_original = original_run  # type: ignore[attr-defined]
        return traced_run

    def _create_traced_run_sync(self, agent: Any, original_run_sync: Any) -> Any:
        adapter = self

        def traced_run_sync(*args: Any, **kwargs: Any) -> Any:
            agent_name = getattr(agent, "name", None) or "pydantic_ai_agent"
            user_prompt = args[0] if args else kwargs.get("user_prompt")
            adapter.on_run_start(agent_name=agent_name, input_data=user_prompt)
            error: Exception | None = None
            result = None
            try:
                result = original_run_sync(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                output = None
                if result is not None:
                    output = getattr(result, "data", result)
                adapter.on_run_end(agent_name=agent_name, output=output, error=error)
                adapter._extract_run_usage(result)
            return result

        traced_run_sync._layerlens_original = original_run_sync  # type: ignore[attr-defined]
        return traced_run_sync

    def _extract_run_usage(self, result: Any) -> None:
        """Extract usage info from PydanticAI RunResult."""
        if result is None:
            return
        try:
            usage = getattr(result, "usage", None) or getattr(result, "_usage", None)
            if usage:
                self.emit_dict_event(
                    "cost.record",
                    {
                        "framework": "pydantic_ai",
                        "tokens_prompt": getattr(usage, "request_tokens", None),
                        "tokens_completion": getattr(usage, "response_tokens", None),
                        "tokens_total": getattr(usage, "total_tokens", None),
                    },
                )
            # Extract model invocation details
            all_messages = getattr(result, "all_messages", None) or []
            for msg in all_messages:
                msg_kind = getattr(msg, "kind", None)
                if msg_kind == "response":
                    model = getattr(result, "model_name", None)
                    self.emit_dict_event(
                        "model.invoke",
                        {
                            "framework": "pydantic_ai",
                            "model": model,
                            "provider": self._detect_provider(model),
                        },
                    )
                elif msg_kind == "tool-return":
                    self.emit_dict_event(
                        "tool.call",
                        {
                            "framework": "pydantic_ai",
                            "tool_name": getattr(msg, "tool_name", "unknown"),
                            "tool_output": self._safe_serialize(getattr(msg, "content", None)),
                        },
                    )
        except Exception:
            logger.debug("Could not extract run usage", exc_info=True)

    # --- Lifecycle Hooks ---

    def on_run_start(self, agent_name: str | None = None, input_data: Any = None) -> None:
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
                    "framework": "pydantic_ai",
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
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._run_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            payload: dict[str, Any] = {
                "framework": "pydantic_ai",
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
                    "framework": "pydantic_ai",
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
        if not self._connected:
            return
        try:
            payload: dict[str, Any] = {
                "framework": "pydantic_ai",
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
            payload: dict[str, Any] = {"framework": "pydantic_ai"}
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
                    "reason": "pydantic_ai_handoff",
                    "context_hash": hashlib.sha256(context_str.encode()).hexdigest()
                    if context_str
                    else None,
                },
            )
        except Exception:
            logger.warning("Error in on_handoff", exc_info=True)

    # --- Helpers ---

    def _detect_provider(self, model: str | None) -> str | None:
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
        return None

    def _emit_agent_config(self, agent_name: str, agent: Any) -> None:
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        metadata: dict[str, Any] = {
            "framework": "pydantic_ai",
            "agent_name": agent_name,
        }
        model = getattr(agent, "model", None)
        if model:
            metadata["model"] = str(model)
        system_prompt = getattr(agent, "system_prompt", None)
        if system_prompt and self._capture_config.capture_content:
            metadata["system_prompt"] = str(system_prompt)[:500]
        tools = getattr(agent, "_function_tools", None) or getattr(agent, "tools", None)
        if tools:
            if isinstance(tools, dict):
                metadata["tools"] = list(tools.keys())
            else:
                metadata["tools"] = [getattr(t, "name", str(t)) for t in tools]
        result_type = getattr(agent, "result_type", None)
        if result_type:
            metadata["result_type"] = str(result_type)
        self.emit_dict_event("environment.config", metadata)

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
