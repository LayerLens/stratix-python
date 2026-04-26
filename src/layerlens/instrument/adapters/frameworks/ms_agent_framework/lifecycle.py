"""
Microsoft Agent Framework adapter lifecycle.

Instrumentation strategy: Chat wrapper (invoke wrapping)
  Chat.invoke() start      -> agent.input (L1)
  Chat.invoke() end        -> agent.output (L1)
  Agent turn (group chat)  -> agent.handoff (L2)
  Tool call                -> tool.call (L5a)
  Model call               -> model.invoke (L3)
  Channel selection        -> agent.state.change (Cross)
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
from layerlens.instrument.adapters._base.handoff import (
    HandoffSequencer,
    build_handoff_payload,
)
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

logger = logging.getLogger(__name__)


class MSAgentAdapter(BaseAdapter):
    """LayerLens adapter for Microsoft Agent Framework."""

    FRAMEWORK = "ms_agent_framework"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/ms_agent_framework/``). The adapter wraps
    # AgentChat.invoke() and emits dict events. The pyproject extra pulls
    # ``semantic-kernel>=1.0,<2.0`` (SK 1.x is internally Pydantic v2)
    # but the adapter itself stays version-agnostic.
    requires_pydantic = PydanticCompat.V1_OR_V2

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: Any | None = None,
        stratix_instance: Any | None = None,
    ) -> None:
        resolved = stratix or stratix_instance
        super().__init__(stratix=resolved, capture_config=capture_config)
        self._originals: dict[int, dict[str, Any]] = {}  # id(chat) -> {method: original}
        self._wrapped_chats: list[Any] = []  # strong refs for disconnect unwrap
        self._adapter_lock = threading.Lock()
        self._seen_agents: set[str] = set()
        self._framework_version: str | None = None
        self._run_starts: dict[int, int] = {}  # thread_id -> start_ns
        # Standardised handoff sequence counter (cross-pollination #7).
        # Both group-chat-turn detection in ``_process_message`` and the
        # manual ``on_handoff`` hook draw from this single instance so
        # seqs stay monotonic across detection paths.
        self._handoff_sequencer = HandoffSequencer()

    def connect(self) -> None:
        """Verify Microsoft Agent Framework availability and prepare the adapter."""
        try:
            import semantic_kernel.agents  # type: ignore[import-not-found,unused-ignore]

            self._framework_version = getattr(semantic_kernel.agents, "__version__", None)
            if not self._framework_version:
                import semantic_kernel  # type: ignore[import-not-found,unused-ignore]

                self._framework_version = getattr(semantic_kernel, "__version__", "unknown")
        except ImportError:
            logger.debug("semantic-kernel agents not installed")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        """Unwrap all instrumented chats and release resources."""
        for chat in self._wrapped_chats:
            self._unwrap_chat(chat)
        self._wrapped_chats.clear()
        self._originals.clear()
        self._seen_agents.clear()
        self._run_starts.clear()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def _unwrap_chat(self, chat: Any) -> None:
        """Restore original methods on a wrapped chat."""
        chat_id = id(chat)
        originals = self._originals.get(chat_id)
        if not originals:
            return
        for method_name, original in originals.items():
            try:
                setattr(chat, method_name, original)
            except Exception:
                logger.debug("Could not unwrap %s.%s", chat_id, method_name, exc_info=True)

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
            name="MSAgentAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_HANDOFFS,
            ],
            description="LayerLens adapter for Microsoft Agent Framework",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        """Serialize the current trace data for replay."""
        return ReplayableTrace(
            adapter_name="MSAgentAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
        )

    # --- Framework Integration ---

    def instrument_chat(self, chat: Any) -> Any:
        """Wrap AgentChat or AgentGroupChat invoke methods to capture lifecycle events."""
        chat_id = id(chat)
        if chat_id in self._originals:
            return chat
        originals: dict[str, Any] = {}
        # Wrap invoke() (async generator)
        if hasattr(chat, "invoke"):
            originals["invoke"] = chat.invoke
            chat.invoke = self._create_traced_invoke(chat, chat.invoke)
        # Wrap invoke_stream() if present
        if hasattr(chat, "invoke_stream"):
            originals["invoke_stream"] = chat.invoke_stream
            chat.invoke_stream = self._create_traced_invoke_stream(chat, chat.invoke_stream)
        self._originals[chat_id] = originals
        self._wrapped_chats.append(chat)
        chat_name = getattr(chat, "name", None) or str(type(chat).__name__)
        self._emit_chat_config(chat_name, chat)
        return chat

    def instrument_agent(self, agent: Any) -> Any:
        """Convenience alias: wraps instrument_chat for AgentChat instances."""
        return self.instrument_chat(agent)

    def _create_traced_invoke(self, chat: Any, original_invoke: Any) -> Any:
        """Create a traced wrapper for chat.invoke()."""
        adapter = self

        async def traced_invoke(*args: Any, **kwargs: Any) -> Any:
            chat_name = getattr(chat, "name", None) or "ms_agent_chat"
            agent = kwargs.get("agent") or (args[0] if args else None)
            agent_name = getattr(agent, "name", None) or chat_name if agent else chat_name
            input_data = kwargs.get("input") or kwargs.get("message")
            adapter.on_run_start(agent_name=agent_name, input_data=input_data)
            error: Exception | None = None
            results: list[Any] = []
            try:
                # invoke() returns an async iterable of ChatMessageContent
                async for message in original_invoke(*args, **kwargs):
                    results.append(message)
                    adapter._process_message(chat, message, agent_name)
                    yield message
            except Exception as exc:
                error = exc
                raise
            finally:
                output = adapter._safe_serialize(results[-1]) if results else None
                adapter.on_run_end(agent_name=agent_name, output=output, error=error)

        traced_invoke._layerlens_original = original_invoke  # type: ignore[attr-defined]
        return traced_invoke

    def _create_traced_invoke_stream(self, chat: Any, original_invoke_stream: Any) -> Any:
        """Create a traced wrapper for chat.invoke_stream()."""
        adapter = self

        async def traced_invoke_stream(*args: Any, **kwargs: Any) -> Any:
            chat_name = getattr(chat, "name", None) or "ms_agent_chat"
            agent = kwargs.get("agent") or (args[0] if args else None)
            agent_name = getattr(agent, "name", None) or chat_name if agent else chat_name
            adapter.on_run_start(agent_name=agent_name, input_data=None)
            error: Exception | None = None
            last_message = None
            try:
                async for message in original_invoke_stream(*args, **kwargs):
                    last_message = message
                    yield message
            except Exception as exc:
                error = exc
                raise
            finally:
                output = adapter._safe_serialize(last_message) if last_message else None
                adapter.on_run_end(agent_name=agent_name, output=output, error=error)

        traced_invoke_stream._layerlens_original = original_invoke_stream  # type: ignore[attr-defined]
        return traced_invoke_stream

    def _process_message(self, chat: Any, message: Any, current_agent: str) -> None:
        """Process a chat message to extract tool calls, model info, and handoffs."""
        try:
            # Detect agent turn transitions (handoffs in group chat)
            msg_agent_name = getattr(message, "agent_name", None) or getattr(message, "name", None)
            if msg_agent_name and msg_agent_name != current_agent:
                msg_content = getattr(message, "content", None)
                payload = build_handoff_payload(
                    sequencer=self._handoff_sequencer,
                    from_agent=current_agent,
                    to_agent=msg_agent_name,
                    context={
                        "from_agent": current_agent,
                        "to_agent": msg_agent_name,
                        "message_content": msg_content,
                    },
                    preview_text=str(msg_content) if msg_content is not None else None,
                    extra={"reason": "group_chat_turn", "framework": "ms_agent_framework"},
                )
                self.emit_dict_event("agent.handoff", payload)

            # Extract tool calls from message
            items = getattr(message, "items", None) or []
            for item in items:
                item_type = type(item).__name__
                if "FunctionCall" in item_type or "ToolCall" in item_type:
                    self.emit_dict_event(
                        "tool.call",
                        {
                            "framework": "ms_agent_framework",
                            "tool_name": getattr(item, "name", None)
                            or getattr(item, "function_name", "unknown"),
                            "tool_input": self._safe_serialize(getattr(item, "arguments", None)),
                        },
                    )
                elif "FunctionResult" in item_type or "ToolResult" in item_type:
                    self.emit_dict_event(
                        "tool.call",
                        {
                            "framework": "ms_agent_framework",
                            "tool_name": getattr(item, "name", None)
                            or getattr(item, "function_name", "unknown"),
                            "tool_output": self._safe_serialize(getattr(item, "result", None)),
                        },
                    )

            # Extract model info from metadata
            metadata = getattr(message, "metadata", None) or {}
            if isinstance(metadata, dict):
                model = metadata.get("model") or metadata.get("model_id")
                if model:
                    self.emit_dict_event(
                        "model.invoke",
                        {
                            "framework": "ms_agent_framework",
                            "model": str(model),
                            "provider": self._detect_provider(str(model)),
                        },
                    )
                usage = metadata.get("usage")
                if usage:
                    self.emit_dict_event(
                        "cost.record",
                        {
                            "framework": "ms_agent_framework",
                            "model": str(model) if model else None,
                            "tokens_prompt": getattr(usage, "prompt_tokens", None)
                            or (usage.get("prompt_tokens") if isinstance(usage, dict) else None),
                            "tokens_completion": getattr(usage, "completion_tokens", None)
                            or (
                                usage.get("completion_tokens") if isinstance(usage, dict) else None
                            ),
                        },
                    )
        except Exception:
            logger.debug("Could not process message", exc_info=True)

    # --- Lifecycle Hooks ---

    def on_run_start(self, agent_name: str | None = None, input_data: Any = None) -> None:
        """Emit agent.input event when a chat invocation starts."""
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
                    "framework": "ms_agent_framework",
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
        """Emit agent.output event when a chat invocation ends."""
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._run_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            payload: dict[str, Any] = {
                "framework": "ms_agent_framework",
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
                    "framework": "ms_agent_framework",
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
                "framework": "ms_agent_framework",
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
            payload: dict[str, Any] = {"framework": "ms_agent_framework"}
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
        """Emit agent.handoff event for agent turn transitions."""
        if not self._connected:
            return
        try:
            ctx_dict = context if isinstance(context, dict) else (
                {"context": str(context)} if context is not None else None
            )
            payload = build_handoff_payload(
                sequencer=self._handoff_sequencer,
                from_agent=from_agent,
                to_agent=to_agent,
                context=ctx_dict,
                preview_text=str(context) if context is not None else None,
                extra={
                    "reason": "group_chat_turn",
                    "framework": "ms_agent_framework",
                },
            )
            self.emit_dict_event("agent.handoff", payload)
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
        if "phi" in model_lower:
            return "microsoft"
        if "llama" in model_lower:
            return "meta"
        return "azure_openai"  # Default for MS Agent Framework

    def _emit_chat_config(self, chat_name: str, chat: Any) -> None:
        """Emit environment.config event for chat configuration on first encounter."""
        with self._adapter_lock:
            if chat_name in self._seen_agents:
                return
            self._seen_agents.add(chat_name)
        metadata: dict[str, Any] = {
            "framework": "ms_agent_framework",
            "chat_name": chat_name,
            "chat_type": type(chat).__name__,
        }
        # Extract agents from group chat
        agents = getattr(chat, "agents", None)
        if agents:
            metadata["agents"] = [getattr(a, "name", str(a)) for a in agents]
        # Extract agent info from single chat
        agent = getattr(chat, "agent", None)
        if agent:
            metadata["agent_name"] = getattr(agent, "name", str(agent))
            instructions = getattr(agent, "instructions", None)
            if instructions and self._capture_config.capture_content:
                metadata["instructions"] = str(instructions)[:500]
            kernel = getattr(agent, "kernel", None)
            if kernel:
                plugins = getattr(kernel, "plugins", None)
                if plugins:
                    metadata["plugins"] = (
                        list(plugins.keys())
                        if isinstance(plugins, dict)
                        else [str(p) for p in plugins]
                    )
        # Selection strategy for group chats
        selection_strategy = getattr(chat, "selection_strategy", None)
        if selection_strategy:
            metadata["selection_strategy"] = type(selection_strategy).__name__
        termination_strategy = getattr(chat, "termination_strategy", None)
        if termination_strategy:
            metadata["termination_strategy"] = type(termination_strategy).__name__
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
