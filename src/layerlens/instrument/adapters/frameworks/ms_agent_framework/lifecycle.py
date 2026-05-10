"""
Microsoft Agent Framework adapter lifecycle.

Instrumentation strategy: Chat wrapper (invoke wrapping)
  Chat.invoke() start      -> agent.input (L1)
  Chat.invoke() end        -> agent.output (L1)
  Agent turn (group chat)  -> agent.handoff (L2)
  Tool call                -> tool.call (L5a)
  Model call               -> model.invoke (L3)
  Channel selection        -> agent.state.change (Cross)

Typed-event status (post PR #129 migration, bundle 3):

* Every emission flows through :meth:`BaseAdapter.emit_event` with a
  canonical Pydantic payload imported from
  :mod:`layerlens.instrument._compat.events`.
* MS Agent Framework-specific provenance (``framework``, ``agent_name``,
  ``chat_name``, ``chat_type``, ``timestamp_ns``, ``selection_strategy``,
  ``termination_strategy``) is carried in the canonical model's
  metadata / attributes / parameters slots.
* The ``agent.state.change`` "run_complete" / "run_failed" marker
  emitted by :meth:`on_run_end` does not satisfy the canonical
  :class:`AgentStateChangeEvent` ``before_hash`` / ``after_hash``
  contract (the run boundary has no real state mutation to hash).
  It is mapped onto :class:`AgentOutputEvent` metadata as
  ``run_status`` so the cross-cutting completion signal is preserved
  without violating the canonical schema.
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
    :class:`AgentOutputEvent` to carry a ``message: str``. MS Agent
    Framework delivers the underlying input/output as arbitrary Python
    objects (Pydantic ``ChatMessageContent`` models with ``content``
    / ``items``, dicts, ``None``); this helper converts each to a
    (possibly empty) string so the typed event always validates. The
    original payload is preserved on
    :class:`MessageContent.metadata.raw_input` / ``raw_output``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # SK ChatMessageContent serialises with a ``content`` slot.
        content = value.get("content")
        if isinstance(content, str):
            return content
    return str(value)


def _coerce_to_dict(value: Any) -> dict[str, Any]:
    """Coerce ``value`` into a dict suitable for the canonical
    :class:`ToolCallEvent` ``input`` / ``output`` slots.

    The canonical schema requires ``input: dict[str, Any]`` and
    ``output: dict[str, Any] | None``. SK function-call payloads
    deliver arbitrary Python values. This helper wraps non-dict
    values in ``{"value": ...}`` so the canonical slot is always
    satisfied.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


def _sha256_of(value: str) -> str:
    """Return a canonical ``sha256:<hex64>`` hash string for ``value``.

    The canonical schema's :class:`AgentHandoffEvent` requires
    ``handoff_context_hash`` to start with ``sha256:`` and have a
    64-character hex tail. Centralising the format here ensures every
    emit site uses the same wire format — including the empty-string
    fallback used when the framework has no context to hash.
    """
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


class MSAgentAdapter(BaseAdapter):
    """LayerLens adapter for Microsoft Agent Framework."""

    FRAMEWORK = "ms_agent_framework"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/ms_agent_framework/``). The adapter wraps
    # AgentChat.invoke() and emits typed events through the canonical
    # schema (PR #129). The pyproject extra pulls
    # ``semantic-kernel>=1.0,<2.0`` (SK 1.x is internally Pydantic v2)
    # but the adapter itself stays version-agnostic.
    requires_pydantic = PydanticCompat.V1_OR_V2

    # Per-adapter ``extra="allow"`` decision: ms_agent_framework targets
    # the canonical 13-event taxonomy exclusively. Unknown event types
    # must be rejected by the base adapter's typed-event validator, so
    # this stays ``False``.
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
        self._originals: dict[int, dict[str, Any]] = {}  # id(chat) -> {method: original}
        self._wrapped_chats: list[Any] = []  # strong refs for disconnect unwrap
        self._adapter_lock = threading.Lock()
        self._seen_agents: set[str] = set()
        self._framework_version: str | None = None
        self._run_starts: dict[int, int] = {}  # thread_id -> start_ns

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
        """Process a chat message to extract tool calls, model info, and handoffs.

        Emits typed :class:`AgentHandoffEvent`, :class:`ToolCallEvent`,
        :class:`ModelInvokeEvent`, and :class:`CostRecordEvent` based
        on the message shape. SK-specific provenance is carried on
        canonical metadata / parameters / input slots.
        """
        try:
            # Detect agent turn transitions (handoffs in group chat)
            msg_agent_name = getattr(message, "agent_name", None) or getattr(message, "name", None)
            if msg_agent_name and msg_agent_name != current_agent:
                self.emit_event(
                    AgentHandoffEvent.create(
                        from_agent=current_agent,
                        to_agent=msg_agent_name,
                        handoff_context_hash=_sha256_of(""),
                    )
                )

            # Extract tool calls from message
            items = getattr(message, "items", None) or []
            for item in items:
                item_type = type(item).__name__
                tool_name_raw = getattr(item, "name", None) or getattr(
                    item, "function_name", "unknown"
                )
                tool_name: str = str(tool_name_raw) if tool_name_raw else "unknown"
                if "FunctionCall" in item_type or "ToolCall" in item_type:
                    serialized_input = self._safe_serialize(getattr(item, "arguments", None))
                    input_data = _coerce_to_dict(serialized_input)
                    input_data.setdefault("framework", "ms_agent_framework")
                    self.emit_event(
                        ToolCallEvent.create(
                            name=tool_name,
                            version="unavailable",
                            integration=IntegrationType.LIBRARY,
                            input_data=input_data,
                        )
                    )
                elif "FunctionResult" in item_type or "ToolResult" in item_type:
                    serialized_output = self._safe_serialize(getattr(item, "result", None))
                    output_data: dict[str, Any] | None = (
                        _coerce_to_dict(serialized_output)
                        if serialized_output is not None
                        else None
                    )
                    self.emit_event(
                        ToolCallEvent.create(
                            name=tool_name,
                            version="unavailable",
                            integration=IntegrationType.LIBRARY,
                            input_data={"framework": "ms_agent_framework"},
                            output_data=output_data,
                        )
                    )

            # Extract model info from metadata
            metadata = getattr(message, "metadata", None) or {}
            if isinstance(metadata, dict):
                model_raw = metadata.get("model") or metadata.get("model_id")
                if model_raw:
                    model_name = str(model_raw)
                    provider = self._detect_provider(model_name) or "azure_openai"
                    self.emit_event(
                        ModelInvokeEvent.create(
                            provider=provider,
                            name=model_name,
                            version="unavailable",
                            parameters={"framework": "ms_agent_framework"},
                        )
                    )
                usage = metadata.get("usage")
                if usage:
                    prompt_tokens = getattr(usage, "prompt_tokens", None) or (
                        usage.get("prompt_tokens") if isinstance(usage, dict) else None
                    )
                    completion_tokens = getattr(usage, "completion_tokens", None) or (
                        usage.get("completion_tokens") if isinstance(usage, dict) else None
                    )
                    self.emit_event(
                        CostRecordEvent.create(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            tokens=(prompt_tokens or 0) + (completion_tokens or 0)
                            if (prompt_tokens or completion_tokens)
                            else None,
                        )
                    )
        except Exception:
            logger.debug("Could not process message", exc_info=True)

    # --- Lifecycle Hooks ---

    def on_run_start(self, agent_name: str | None = None, input_data: Any = None) -> None:
        """Emit a typed :class:`AgentInputEvent` when a chat invocation starts."""
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._run_starts[tid] = start_ns
            raw_input = self._safe_serialize(input_data)
            self.emit_event(
                AgentInputEvent.create(
                    message=_stringify(raw_input),
                    role=MessageRole.HUMAN,
                    metadata={
                        "framework": "ms_agent_framework",
                        "agent_name": agent_name,
                        "timestamp_ns": start_ns,
                        "raw_input": raw_input,
                    },
                )
            )
        except Exception:
            logger.warning("Error in on_run_start", exc_info=True)

    def on_run_end(
        self,
        agent_name: str | None = None,
        output: Any = None,
        error: Exception | None = None,
    ) -> None:
        """Emit a typed :class:`AgentOutputEvent` when a chat invocation ends.

        The previous adapter implementation also emitted a separate
        ``agent.state.change`` payload carrying only an
        ``event_subtype`` marker (``run_complete`` / ``run_failed``).
        That payload did not satisfy the canonical
        :class:`AgentStateChangeEvent` ``before_hash`` / ``after_hash``
        contract — the run boundary has no real state mutation to
        hash. The post-migration mapping carries the same signal as
        ``run_status`` on :class:`MessageContent.metadata`, preserving
        the cross-cutting completion marker without violating the
        canonical schema.
        """
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._run_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            raw_output = self._safe_serialize(output)
            metadata: dict[str, Any] = {
                "framework": "ms_agent_framework",
                "agent_name": agent_name,
                "duration_ns": duration_ns,
                "raw_output": raw_output,
                "run_status": "run_complete" if not error else "run_failed",
            }
            if error:
                metadata["error"] = str(error)
            self.emit_event(
                AgentOutputEvent.create(
                    message=_stringify(raw_output),
                    metadata=metadata,
                )
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
        """Emit a typed :class:`ToolCallEvent` for a tool invocation."""
        if not self._connected:
            return
        try:
            serialized_input = self._safe_serialize(tool_input)
            serialized_output = self._safe_serialize(tool_output)
            input_data = _coerce_to_dict(serialized_input)
            input_data.setdefault("framework", "ms_agent_framework")
            output_data: dict[str, Any] | None = (
                _coerce_to_dict(serialized_output) if serialized_output is not None else None
            )
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
        """Emit a typed :class:`ModelInvokeEvent` for an LLM call."""
        if not self._connected:
            return
        try:
            model_name = model or "unknown"
            resolved_provider = provider or self._detect_provider(model_name) or "azure_openai"
            input_messages: list[dict[str, str]] | None = None
            if self._capture_config.capture_content and messages:
                input_messages = messages
            self.emit_event(
                ModelInvokeEvent.create(
                    provider=resolved_provider,
                    name=model_name,
                    version="unavailable",
                    parameters={"framework": "ms_agent_framework"},
                    prompt_tokens=tokens_prompt,
                    completion_tokens=tokens_completion,
                    latency_ms=latency_ms,
                    input_messages=input_messages,
                )
            )
        except Exception:
            logger.warning("Error in on_llm_call", exc_info=True)

    def on_handoff(self, from_agent: str, to_agent: str, context: Any = None) -> None:
        """Emit a typed :class:`AgentHandoffEvent` for agent turn transitions."""
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
        """Emit a typed :class:`EnvironmentConfigEvent` per chat.

        Idempotent per chat — only the first call for a given chat
        name actually emits. SK chat instances run in a
        ``simulated`` environment by default; the real production
        environment (``cloud`` / ``on_prem``) is the responsibility
        of the host application's environment.config emission.
        """
        with self._adapter_lock:
            if chat_name in self._seen_agents:
                return
            self._seen_agents.add(chat_name)
        attributes: dict[str, Any] = {
            "framework": "ms_agent_framework",
            "chat_name": chat_name,
            "chat_type": type(chat).__name__,
        }
        # Extract agents from group chat
        agents = getattr(chat, "agents", None)
        if agents:
            attributes["agents"] = [getattr(a, "name", str(a)) for a in agents]
        # Extract agent info from single chat
        agent = getattr(chat, "agent", None)
        if agent:
            attributes["agent_name"] = getattr(agent, "name", str(agent))
            instructions = getattr(agent, "instructions", None)
            if instructions and self._capture_config.capture_content:
                attributes["instructions"] = str(instructions)[:500]
            kernel = getattr(agent, "kernel", None)
            if kernel:
                plugins = getattr(kernel, "plugins", None)
                if plugins:
                    attributes["plugins"] = (
                        list(plugins.keys()) if isinstance(plugins, dict) else [str(p) for p in plugins]
                    )
        # Selection strategy for group chats
        selection_strategy = getattr(chat, "selection_strategy", None)
        if selection_strategy:
            attributes["selection_strategy"] = type(selection_strategy).__name__
        termination_strategy = getattr(chat, "termination_strategy", None)
        if termination_strategy:
            attributes["termination_strategy"] = type(termination_strategy).__name__
        self.emit_event(
            EnvironmentConfigEvent.create(
                env_type=EnvironmentType.SIMULATED,
                attributes=attributes,
            )
        )

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
