"""SmolAgents adapter lifecycle.

Instrumentation strategy: agent wrapper + lifecycle hooks (no native callbacks).

* ``Agent.run()`` start → ``agent.input`` (L1)
* ``Agent.run()`` end   → ``agent.output`` (L1)
* Model call            → ``model.invoke`` (L3)
* Tool execution        → ``tool.call`` (L5a)
* Code execution        → ``agent.code`` (L2)
* Manager → managed     → ``agent.handoff`` (cross-cutting)

Ported from ``ateam/stratix/sdk/python/adapters/smolagents/lifecycle.py``.
"""

from __future__ import annotations

import time
import uuid
import hashlib
import logging
import threading
from typing import Any, Set, Dict, List, Optional

from layerlens.instrument.adapters._base.errors import emit_error_event
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


class SmolAgentsAdapter(BaseAdapter):
    """LayerLens adapter for SmolAgents (HuggingFace)."""

    FRAMEWORK = "smolagents"
    VERSION = "0.1.0"
    # The only Pydantic touch in the adapter is
    # ``from layerlens._compat.pydantic import model_dump`` at line 105
    # of this file — the v1/v2 shim itself. SmolAgents 1.x uses Pydantic
    # internally but the adapter only wraps ``Agent.run()`` and never
    # touches framework Pydantic models directly.
    requires_pydantic = PydanticCompat.V1_OR_V2

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Any = None,
        stratix_instance: Any = None,
    ) -> None:
        resolved = stratix or stratix_instance
        super().__init__(stratix=resolved, capture_config=capture_config)
        self._originals: Dict[int, Dict[str, Any]] = {}
        self._adapter_lock = threading.Lock()
        self._seen_agents: Set[str] = set()
        self._framework_version: Optional[str] = None
        self._run_starts: Dict[int, int] = {}
        self._wrapped_agents: List[Any] = []

    def connect(self) -> None:
        try:
            import smolagents  # type: ignore[import-not-found,unused-ignore]

            version = getattr(smolagents, "__version__", "unknown")
            self._framework_version = (
                str(version) if version is not None else "unknown"
            )
        except ImportError:
            logger.debug("smolagents not installed")
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
            name="SmolAgentsAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_HANDOFFS,
            ],
            description="LayerLens adapter for SmolAgents (HuggingFace)",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        from layerlens._compat.pydantic import model_dump

        return ReplayableTrace(
            adapter_name="SmolAgentsAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": model_dump(self._capture_config)},
        )

    # --- Framework integration ---

    def instrument_agent(self, agent: Any) -> Any:
        """Wrap a SmolAgents agent's ``run()`` method."""
        agent_id = id(agent)
        if agent_id in self._originals:
            return agent
        originals: Dict[str, Any] = {}
        if hasattr(agent, "run"):
            originals["run"] = agent.run
            agent.run = self._create_traced_run(agent, agent.run)
        self._originals[agent_id] = originals
        self._wrapped_agents.append(agent)
        agent_name = self._get_agent_name(agent)
        agent_type = type(agent).__name__
        self._emit_agent_config(agent_name, agent, agent_type)
        managed = getattr(agent, "managed_agents", None)
        if managed:
            if isinstance(managed, dict):
                for _name, managed_agent in managed.items():
                    self.instrument_agent(managed_agent)
            elif isinstance(managed, list):
                for managed_agent in managed:
                    self.instrument_agent(managed_agent)
        return agent

    def _create_traced_run(self, agent: Any, original_run: Any) -> Any:
        adapter = self

        def traced_run(*args: Any, **kwargs: Any) -> Any:
            agent_name = adapter._get_agent_name(agent)
            task = args[0] if args else kwargs.get("task")
            adapter.on_run_start(agent_name=agent_name, input_data=task)
            error: Optional[Exception] = None
            result: Any = None
            try:
                result = original_run(*args, **kwargs)
            except Exception as exc:
                error = exc
                emit_error_event(
                    adapter,
                    exc,
                    {"framework": "smolagents", "agent_name": agent_name, "phase": "agent.run"},
                )
                raise
            finally:
                adapter.on_run_end(agent_name=agent_name, output=result, error=error)
                agent_type = type(agent).__name__
                if agent_type == "CodeAgent" and result is not None:
                    adapter._emit_code_execution(agent_name, result)
            return result

        traced_run._layerlens_original = original_run  # type: ignore[attr-defined]
        return traced_run

    def _unwrap_agent(self, agent: Any) -> None:
        agent_id = id(agent)
        originals = self._originals.get(agent_id)
        if not originals:
            return
        for method_name, original in originals.items():
            try:
                setattr(agent, method_name, original)
            except Exception:
                logger.debug("Could not unwrap %s", method_name, exc_info=True)

    # --- Lifecycle hooks ---

    def on_run_start(
        self,
        agent_name: Optional[str] = None,
        input_data: Any = None,
    ) -> None:
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
                    "framework": "smolagents",
                    "agent_name": agent_name,
                    "input": self._safe_serialize(input_data),
                    "timestamp_ns": start_ns,
                },
            )
        except Exception:
            logger.warning("Error in on_run_start", exc_info=True)

    def on_run_end(
        self,
        agent_name: Optional[str] = None,
        output: Any = None,
        error: Optional[Exception] = None,
    ) -> None:
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._run_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            payload: Dict[str, Any] = {
                "framework": "smolagents",
                "agent_name": agent_name,
                "output": self._safe_serialize(output),
                "duration_ns": duration_ns,
            }
            if error:
                payload["error"] = str(error)
            self.emit_dict_event("agent.output", payload)
        except Exception:
            logger.warning("Error in on_run_end", exc_info=True)

    def on_tool_use(
        self,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        error: Optional[Exception] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        if not self._connected:
            return
        try:
            payload: Dict[str, Any] = {
                "framework": "smolagents",
                "tool_name": tool_name,
                "tool_input": self._safe_serialize(tool_input),
                "tool_output": self._safe_serialize(tool_output),
            }
            if error:
                payload["error"] = str(error)
            if latency_ms is not None:
                payload["latency_ms"] = latency_ms
            self.emit_dict_event("tool.call", payload)
            if error is not None:
                emit_error_event(
                    self,
                    error,
                    {"framework": "smolagents", "tool_name": tool_name, "phase": "tool.call"},
                    event_type="tool.error",
                )
        except Exception:
            logger.warning("Error in on_tool_use", exc_info=True)

    def on_llm_call(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
        latency_ms: Optional[float] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        if not self._connected:
            return
        try:
            payload: Dict[str, Any] = {"framework": "smolagents"}
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

    def on_handoff(
        self,
        from_agent: str,
        to_agent: str,
        context: Any = None,
    ) -> None:
        if not self._connected:
            return
        try:
            context_str = str(context) if context else ""
            self.emit_dict_event(
                "agent.handoff",
                {
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "reason": "managed_agent_delegation",
                    "context_hash": (
                        hashlib.sha256(context_str.encode()).hexdigest()
                        if context_str
                        else None
                    ),
                    "context_preview": (
                        context_str[:500]
                        if context_str and self._capture_config.capture_content
                        else None
                    ),
                },
            )
        except Exception:
            logger.warning("Error in on_handoff", exc_info=True)

    # --- Helpers ---

    def _get_agent_name(self, agent: Any) -> str:
        return getattr(agent, "name", None) or type(agent).__name__

    def _emit_agent_config(
        self,
        agent_name: str,
        agent: Any,
        agent_type: str,
    ) -> None:
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        metadata: Dict[str, Any] = {
            "framework": "smolagents",
            "agent_name": agent_name,
            "agent_type": agent_type,
        }
        tools = getattr(agent, "tools", None)
        if tools:
            if isinstance(tools, dict):
                metadata["tools"] = list(tools.keys())
            else:
                metadata["tools"] = [getattr(t, "name", str(t)) for t in tools]
        model = getattr(agent, "model", None)
        if model:
            metadata["model"] = str(model)
        managed = getattr(agent, "managed_agents", None)
        if managed:
            if isinstance(managed, dict):
                metadata["managed_agents"] = list(managed.keys())
            elif isinstance(managed, list):
                metadata["managed_agents"] = [
                    getattr(a, "name", str(a)) for a in managed
                ]
        system_prompt = getattr(agent, "system_prompt", None)
        if system_prompt and self._capture_config.capture_content:
            metadata["system_prompt"] = str(system_prompt)[:500]
        self.emit_dict_event("environment.config", metadata)

    def _emit_code_execution(self, agent_name: str, result: Any) -> None:
        """Emit an L2 code execution event for ``CodeAgent``."""
        try:
            logs = getattr(result, "logs", None) or getattr(result, "inner_messages", None)
            self.emit_dict_event(
                "agent.code",
                {
                    "framework": "smolagents",
                    "agent_name": agent_name,
                    "event_subtype": "code_execution",
                    "output": self._safe_serialize(result),
                    "logs": self._safe_serialize(logs),
                },
            )
        except Exception:
            logger.debug("Could not emit code execution event", exc_info=True)

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


# Registry lazy-loading convention.
ADAPTER_CLASS = SmolAgentsAdapter
