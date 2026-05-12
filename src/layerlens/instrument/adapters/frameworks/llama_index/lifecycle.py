"""
LlamaIndex adapter lifecycle.

Instrumentation strategy: Instrumentation Module (modern event-driven, v0.10.20+)
  Agent start          → agent.input (L1)
  Agent end            → agent.output (L1)
  LLM call             → model.invoke (L3)
  Tool call            → tool.call (L5a)
  Query/retrieval      → tool.call (L5a, retrieval)
  Agent handoff        → agent.handoff (Cross)
  Workflow event       → agent.state.change (Cross)
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


class LlamaIndexAdapter(BaseAdapter):
    """LayerLens adapter for LlamaIndex."""

    FRAMEWORK = "llama_index"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/llama_index/``). LlamaIndex's
    # Instrumentation Module emits dict-shaped events that the adapter
    # forwards without touching framework Pydantic models.
    requires_pydantic = PydanticCompat.V1_OR_V2

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: Any | None = None,
        stratix_instance: Any | None = None,
    ) -> None:
        resolved = stratix or stratix_instance
        super().__init__(stratix=resolved, capture_config=capture_config)
        self._originals: dict[str, Any] = {}
        self._adapter_lock = threading.Lock()
        self._seen_agents: set[str] = set()
        self._framework_version: str | None = None
        self._event_handler: Any | None = None
        self._agent_starts: dict[int, int] = {}  # thread_id -> start_ns

    def connect(self) -> None:
        try:
            import llama_index.core  # type: ignore[import-not-found,unused-ignore]

            self._framework_version = getattr(llama_index.core, "__version__", "unknown")
        except ImportError:
            try:
                import llama_index  # type: ignore[import-not-found,unused-ignore]

                self._framework_version = getattr(llama_index, "__version__", "unknown")
            except ImportError:
                logger.debug("llama-index not installed")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        if self._event_handler is not None:
            try:
                from llama_index.core.instrumentation import (  # type: ignore[import-not-found,unused-ignore]
                    get_dispatcher,
                )

                dispatcher = get_dispatcher()
                # LlamaIndex dispatcher stores handlers in span_handlers / event_handlers lists
                handlers = getattr(dispatcher, "event_handlers", [])
                if self._event_handler in handlers:
                    handlers.remove(self._event_handler)
            except Exception:
                logger.debug("Could not unregister event handler", exc_info=True)
            self._event_handler = None
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
            name="LlamaIndexAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_HANDOFFS,
            ],
            description="LayerLens adapter for LlamaIndex",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="LlamaIndexAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
        )

    # --- Framework Integration ---

    def instrument_workflow(self, workflow: Any) -> Any:
        """Register Stratix event handler with LlamaIndex instrumentation."""
        try:
            from llama_index.core.instrumentation import get_dispatcher

            dispatcher = get_dispatcher()
            handler = self._create_event_handler()
            if handler is None:
                logger.warning("Could not create event handler (BaseEventHandler not importable)")
                return workflow
            dispatcher.add_event_handler(handler)
            self._event_handler = handler
        except ImportError:
            logger.debug("LlamaIndex instrumentation module not available")
        except Exception:
            logger.warning("Failed to register event handler", exc_info=True)
        return workflow

    def _create_event_handler(self) -> Any:
        """Create a LlamaIndex event handler that routes to Stratix."""
        adapter = self

        try:
            from llama_index.core.instrumentation.events import (  # type: ignore[import-not-found,unused-ignore]
                BaseEvent,
            )
            from llama_index.core.instrumentation.event_handlers import (  # type: ignore[import-not-found,unused-ignore]
                BaseEventHandler,
            )
        except ImportError:
            return None

        class StratixEventHandler(BaseEventHandler):  # type: ignore[misc]
            @classmethod
            def class_name(cls) -> str:
                return "StratixEventHandler"

            def handle(self, event: BaseEvent, **kwargs: Any) -> None:
                try:
                    adapter._handle_event(event)
                except Exception:
                    logger.warning("Error handling LlamaIndex event", exc_info=True)

        return StratixEventHandler()

    def _handle_event(self, event: Any) -> None:
        """Route LlamaIndex events to appropriate Stratix event emission."""
        if not self._connected:
            return
        event_type = type(event).__name__

        if event_type in ("LLMChatStartEvent", "LLMStartEvent"):
            self._on_llm_start(event)
        elif event_type in ("LLMChatEndEvent", "LLMCompletionEndEvent"):
            self._on_llm_end(event)
        elif event_type == "ToolCallEvent":
            self._on_tool_call(event)
        elif event_type in ("RetrievalStartEvent", "QueryStartEvent"):
            self._on_retrieval_start(event)
        elif event_type in ("RetrievalEndEvent", "QueryEndEvent"):
            self._on_retrieval_end(event)
        elif event_type in ("AgentRunStepStartEvent",):
            self._on_agent_step_start(event)
        elif event_type in ("AgentRunStepEndEvent",):
            self._on_agent_step_end(event)

    def _on_llm_start(self, event: Any) -> None:
        pass  # Timing tracked on end

    def _on_llm_end(self, event: Any) -> None:
        payload: dict[str, Any] = {"framework": "llama_index"}
        model = getattr(event, "model", None) or getattr(event, "model_name", None)
        if model:
            payload["model"] = str(model)
        response = getattr(event, "response", None)
        if response:
            raw = getattr(response, "raw", None)
            if raw:
                usage = getattr(raw, "usage", None)
                if usage:
                    payload["tokens_prompt"] = getattr(usage, "prompt_tokens", None)
                    payload["tokens_completion"] = getattr(usage, "completion_tokens", None)
        self.emit_dict_event("model.invoke", payload)
        if "tokens_prompt" in payload or "tokens_completion" in payload:
            self.emit_dict_event(
                "cost.record",
                {
                    "framework": "llama_index",
                    "model": payload.get("model"),
                    "tokens_prompt": payload.get("tokens_prompt"),
                    "tokens_completion": payload.get("tokens_completion"),
                    "tokens_total": (payload.get("tokens_prompt") or 0)
                    + (payload.get("tokens_completion") or 0),
                },
            )

    def _on_tool_call(self, event: Any) -> None:
        self.emit_dict_event(
            "tool.call",
            {
                "framework": "llama_index",
                "tool_name": getattr(event, "tool_name", None) or getattr(event, "name", "unknown"),
                "tool_input": self._safe_serialize(getattr(event, "tool_input", None)),
                "tool_output": self._safe_serialize(getattr(event, "tool_output", None)),
            },
        )

    def _on_retrieval_start(self, event: Any) -> None:
        pass  # Tracked on end

    def _on_retrieval_end(self, event: Any) -> None:
        nodes = getattr(event, "nodes", None) or []
        self.emit_dict_event(
            "tool.call",
            {
                "framework": "llama_index",
                "tool_name": "retrieval",
                "tool_type": "retrieval",
                "tool_output": self._safe_serialize(
                    [{"score": getattr(n, "score", None)} for n in nodes[:10]]
                ),
                "result_count": len(nodes),
            },
        )

    def _on_agent_step_start(self, event: Any) -> None:
        agent_name = getattr(event, "agent_id", None) or "llama_agent"
        self._emit_agent_config(agent_name, event)
        tid = threading.get_ident()
        start_ns = time.time_ns()
        with self._adapter_lock:
            self._agent_starts[tid] = start_ns
        self.emit_dict_event(
            "agent.input",
            {
                "framework": "llama_index",
                "agent_name": agent_name,
                "step": getattr(event, "step", None),
                "timestamp_ns": start_ns,
            },
        )

    def _on_agent_step_end(self, event: Any) -> None:
        agent_name = getattr(event, "agent_id", None) or "llama_agent"
        tid = threading.get_ident()
        end_ns = time.time_ns()
        with self._adapter_lock:
            start_ns = self._agent_starts.pop(tid, 0)
        duration_ns = end_ns - start_ns if start_ns else 0
        self.emit_dict_event(
            "agent.output",
            {
                "framework": "llama_index",
                "agent_name": agent_name,
                "output": self._safe_serialize(getattr(event, "response", None)),
                "duration_ns": duration_ns,
            },
        )

    # --- Lifecycle Hooks ---

    def on_agent_start(self, agent_name: str | None = None, input_data: Any = None) -> None:
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._agent_starts[tid] = start_ns
            self.emit_dict_event(
                "agent.input",
                {
                    "framework": "llama_index",
                    "agent_name": agent_name,
                    "input": self._safe_serialize(input_data),
                    "timestamp_ns": start_ns,
                },
            )
        except Exception:
            logger.warning("Error in on_agent_start", exc_info=True)

    def on_agent_end(
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
                start_ns = self._agent_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            payload: dict[str, Any] = {
                "framework": "llama_index",
                "agent_name": agent_name,
                "output": self._safe_serialize(output),
                "duration_ns": duration_ns,
            }
            if error:
                payload["error"] = str(error)
            self.emit_dict_event("agent.output", payload)
        except Exception:
            logger.warning("Error in on_agent_end", exc_info=True)

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
                "framework": "llama_index",
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
            payload: dict[str, Any] = {"framework": "llama_index"}
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
                    "reason": "agent_workflow_handoff",
                    "context_hash": hashlib.sha256(context_str.encode()).hexdigest()
                    if context_str
                    else None,
                },
            )
        except Exception:
            logger.warning("Error in on_handoff", exc_info=True)

    # --- Helpers ---

    def _emit_agent_config(self, agent_name: str, event_or_agent: Any) -> None:
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        metadata: dict[str, Any] = {
            "framework": "llama_index",
            "agent_name": agent_name,
        }
        tools = getattr(event_or_agent, "tools", None)
        if tools:
            metadata["tools"] = [getattr(t, "name", str(t)) for t in tools]
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
            if isinstance(value, list):
                return [self._safe_serialize(v) for v in value[:100]]
            return str(value)
        except Exception:
            return str(value)
