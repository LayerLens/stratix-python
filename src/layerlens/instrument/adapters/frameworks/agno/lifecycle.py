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


class AgnoAdapter(BaseAdapter):
    """LayerLens adapter for Agno."""

    FRAMEWORK = "agno"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/agno/``). Agno itself uses Pydantic v2
    # internally but the adapter only wraps ``Agent.run`` / ``Agent.arun``
    # and emits dict events, never touching framework Pydantic models.
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
        """Extract tool calls, model invocations, and team handoffs from run result."""
        if result is None:
            return
        try:
            # Extract model invocation details
            model = getattr(agent, "model", None)
            if model:
                model_name = getattr(model, "id", None) or str(model)
                self.emit_dict_event(
                    "model.invoke",
                    {
                        "framework": "agno",
                        "model": model_name,
                        "provider": self._detect_provider(model_name),
                    },
                )

            # Extract usage/token info from result
            usage = getattr(result, "metrics", None) or getattr(result, "usage", None)
            if usage:
                self.emit_dict_event(
                    "cost.record",
                    {
                        "framework": "agno",
                        "tokens_prompt": getattr(usage, "input_tokens", None)
                        or getattr(usage, "prompt_tokens", None),
                        "tokens_completion": getattr(usage, "output_tokens", None)
                        or getattr(usage, "completion_tokens", None),
                        "tokens_total": getattr(usage, "total_tokens", None),
                    },
                )

            # Extract tool calls from messages
            messages = getattr(result, "messages", None) or []
            for msg in messages:
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        self.emit_dict_event(
                            "tool.call",
                            {
                                "framework": "agno",
                                "tool_name": getattr(tc, "function", {}).get("name", "unknown")
                                if isinstance(getattr(tc, "function", None), dict)
                                else getattr(getattr(tc, "function", None), "name", "unknown"),
                                "tool_input": self._safe_serialize(
                                    getattr(tc, "function", {}).get("arguments")
                                    if isinstance(getattr(tc, "function", None), dict)
                                    else None
                                ),
                            },
                        )

            # Detect team delegation (multi-agent handoffs)
            team = getattr(agent, "team", None)
            if team:
                members = getattr(team, "members", None) or getattr(team, "agents", None) or []
                for member in members:
                    member_name = getattr(member, "name", None) or str(member)
                    self.emit_dict_event(
                        "agent.handoff",
                        {
                            "from_agent": getattr(agent, "name", "leader"),
                            "to_agent": member_name,
                            "reason": "team_delegation",
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
                    "framework": "agno",
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
                "framework": "agno",
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
                    "framework": "agno",
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
                "framework": "agno",
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
            payload: dict[str, Any] = {"framework": "agno"}
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
        """Emit agent.handoff event for team delegation."""
        if not self._connected:
            return
        try:
            context_str = str(context) if context else ""
            self.emit_dict_event(
                "agent.handoff",
                {
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "reason": "agno_team_delegation",
                    "context_hash": hashlib.sha256(context_str.encode()).hexdigest()
                    if context_str
                    else None,
                },
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
        """Emit environment.config event for agent configuration on first encounter."""
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        metadata: dict[str, Any] = {
            "framework": "agno",
            "agent_name": agent_name,
        }
        model = getattr(agent, "model", None)
        if model:
            metadata["model"] = str(model)
        description = getattr(agent, "description", None)
        if description:
            metadata["description"] = str(description)[:500]
        instructions = getattr(agent, "instructions", None)
        if instructions and self._capture_config.capture_content:
            metadata["instructions"] = str(instructions)[:500]
        tools = getattr(agent, "tools", None)
        if tools:
            metadata["tools"] = [getattr(t, "name", str(t)) for t in tools]
        knowledge = getattr(agent, "knowledge", None)
        if knowledge:
            metadata["knowledge"] = str(type(knowledge).__name__)
        team = getattr(agent, "team", None)
        if team:
            members = getattr(team, "members", None) or getattr(team, "agents", None) or []
            metadata["team_members"] = [getattr(m, "name", str(m)) for m in members]
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
