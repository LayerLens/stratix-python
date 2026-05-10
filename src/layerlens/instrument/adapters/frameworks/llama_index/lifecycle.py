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

Typed-event status (post PR #129 migration, bundle 3):

* Every emission flows through :meth:`BaseAdapter.emit_event` with a
  canonical Pydantic payload imported from
  :mod:`layerlens.instrument._compat.events`.
* LlamaIndex-specific provenance (``framework``, ``agent_name``,
  ``step``, ``timestamp_ns``, ``tool_type``, ``result_count``) is
  carried in the canonical model's metadata / attributes / parameters
  / input slots — the canonical schema does not expose these as
  top-level fields.
* The handoff context hash is generated via SHA-256 over the context
  string (or the empty string when no context is available) so the
  canonical :class:`AgentHandoffEvent.handoff_context_hash` validator
  always passes.
* Retrieval events map onto :class:`ToolCallEvent` with
  ``name="retrieval"`` and ``integration=IntegrationType.LIBRARY`` —
  the canonical schema has no dedicated retrieval shape, but the
  agno reference adapter follows the same convention.
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
    :class:`AgentOutputEvent` to carry a ``message: str``. LlamaIndex
    delivers the underlying input/output as arbitrary Python objects
    (Pydantic ``Response`` models with ``response`` attribute, dicts,
    ``None``); this helper converts each to a (possibly empty) string
    so the typed event always validates. The original payload is
    preserved on :class:`MessageContent.metadata.raw_input` /
    ``raw_output``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # LlamaIndex Response payloads serialise to ``{"response": ...}``;
        # surface the response slot when present.
        response = value.get("response")
        if isinstance(response, str):
            return response
        content = value.get("content")
        if isinstance(content, str):
            return content
    return str(value)


def _coerce_to_dict(value: Any) -> dict[str, Any]:
    """Coerce ``value`` into a dict suitable for the canonical
    :class:`ToolCallEvent` ``input`` / ``output`` slots.

    The canonical schema requires ``input: dict[str, Any]`` and
    ``output: dict[str, Any] | None``. LlamaIndex tool events deliver
    arbitrary Python values (scalars, dicts, lists, dataclass-like
    objects). This helper wraps non-dict values in ``{"value": ...}``
    so the canonical slot is always satisfied. ``None`` returns an
    empty dict so the caller can pass it positionally without a guard.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return {"value": value}
    return {"value": value}


def _sha256_of(value: str) -> str:
    """Return a canonical ``sha256:<hex64>`` hash string for ``value``.

    The canonical schema's :class:`AgentHandoffEvent` requires
    ``handoff_context_hash`` to start with ``sha256:`` and have a
    64-character hex tail. Centralising the format here ensures every
    emit site uses the same wire format — including the empty-string
    fallback used when LlamaIndex has no context to hash.
    """
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _detect_provider(model: str | None) -> str:
    """Detect the LLM provider from a model identifier.

    LlamaIndex supports OpenAI, Anthropic, Google, Mistral, Cohere,
    LiteLLM, etc. The canonical :class:`ModelInvokeEvent` requires
    both ``provider`` and ``name``, so this heuristic derives the
    provider from well-known model name prefixes. Unknown identifiers
    return ``"unknown"`` per the canonical schema's NORMATIVE rule.
    """
    if not model:
        return "unknown"
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
    return "unknown"


class LlamaIndexAdapter(BaseAdapter):
    """LayerLens adapter for LlamaIndex."""

    FRAMEWORK = "llama_index"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/llama_index/``). LlamaIndex's
    # Instrumentation Module emits dict-shaped events that the adapter
    # forwards through the canonical schema (PR #129) without touching
    # framework Pydantic models.
    requires_pydantic = PydanticCompat.V1_OR_V2

    # Per-adapter ``extra="allow"`` decision: llama_index targets the
    # canonical 13-event taxonomy exclusively. Unknown event types must
    # be rejected by the base adapter's typed-event validator, so this
    # stays ``False``.
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
        """Emit a typed :class:`ModelInvokeEvent` and (optionally)
        :class:`CostRecordEvent` for an LLM completion.

        LlamaIndex-specific provenance (``framework``) lives on
        :attr:`ModelInfo.parameters`. Token usage is extracted from
        the optional ``response.raw.usage`` slot LlamaIndex exposes
        for OpenAI-compatible providers.
        """
        model_raw = getattr(event, "model", None) or getattr(event, "model_name", None)
        model_name = str(model_raw) if model_raw else "unknown"
        provider = _detect_provider(model_name)
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        response = getattr(event, "response", None)
        if response:
            raw = getattr(response, "raw", None)
            if raw:
                usage = getattr(raw, "usage", None)
                if usage:
                    prompt_tokens = getattr(usage, "prompt_tokens", None)
                    completion_tokens = getattr(usage, "completion_tokens", None)
        self.emit_event(
            ModelInvokeEvent.create(
                provider=provider,
                name=model_name,
                version="unavailable",
                parameters={"framework": "llama_index"},
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        if prompt_tokens is not None or completion_tokens is not None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            self.emit_event(
                CostRecordEvent.create(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    tokens=total_tokens,
                )
            )

    def _on_tool_call(self, event: Any) -> None:
        """Emit a typed :class:`ToolCallEvent` for a LlamaIndex tool call."""
        tool_name_raw = getattr(event, "tool_name", None) or getattr(event, "name", "unknown")
        tool_name: str = str(tool_name_raw) if tool_name_raw else "unknown"
        serialized_input = self._safe_serialize(getattr(event, "tool_input", None))
        serialized_output = self._safe_serialize(getattr(event, "tool_output", None))
        input_data = _coerce_to_dict(serialized_input)
        input_data.setdefault("framework", "llama_index")
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
            )
        )

    def _on_retrieval_start(self, event: Any) -> None:
        pass  # Tracked on end

    def _on_retrieval_end(self, event: Any) -> None:
        """Emit a typed :class:`ToolCallEvent` for a retrieval result.

        Retrieval is modelled as a tool call with ``name="retrieval"``
        and ``integration=IntegrationType.LIBRARY``. Adapter-specific
        provenance (``framework``, ``tool_type``, ``result_count``)
        lives on the canonical ``input`` slot; the per-node scores
        list lives on the canonical ``output`` slot.
        """
        nodes = getattr(event, "nodes", None) or []
        node_scores = self._safe_serialize(
            [{"score": getattr(n, "score", None)} for n in nodes[:10]]
        )
        self.emit_event(
            ToolCallEvent.create(
                name="retrieval",
                version="unavailable",
                integration=IntegrationType.LIBRARY,
                input_data={
                    "framework": "llama_index",
                    "tool_type": "retrieval",
                    "result_count": len(nodes),
                },
                output_data={"nodes": node_scores},
            )
        )

    def _on_agent_step_start(self, event: Any) -> None:
        """Emit a typed :class:`AgentInputEvent` for agent step start.

        LlamaIndex-specific provenance (``framework``, ``agent_name``,
        ``step``, ``timestamp_ns``) lives on
        :class:`MessageContent.metadata`. The canonical ``message``
        field carries a string view of the step.
        """
        agent_name = getattr(event, "agent_id", None) or "llama_agent"
        self._emit_agent_config(agent_name, event)
        tid = threading.get_ident()
        start_ns = time.time_ns()
        with self._adapter_lock:
            self._agent_starts[tid] = start_ns
        step = getattr(event, "step", None)
        self.emit_event(
            AgentInputEvent.create(
                message=_stringify(step),
                role=MessageRole.HUMAN,
                metadata={
                    "framework": "llama_index",
                    "agent_name": agent_name,
                    "step": step,
                    "timestamp_ns": start_ns,
                },
            )
        )

    def _on_agent_step_end(self, event: Any) -> None:
        """Emit a typed :class:`AgentOutputEvent` for agent step end."""
        agent_name = getattr(event, "agent_id", None) or "llama_agent"
        tid = threading.get_ident()
        end_ns = time.time_ns()
        with self._adapter_lock:
            start_ns = self._agent_starts.pop(tid, 0)
        duration_ns = end_ns - start_ns if start_ns else 0
        raw_output = self._safe_serialize(getattr(event, "response", None))
        self.emit_event(
            AgentOutputEvent.create(
                message=_stringify(raw_output),
                metadata={
                    "framework": "llama_index",
                    "agent_name": agent_name,
                    "duration_ns": duration_ns,
                    "raw_output": raw_output,
                },
            )
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
            raw_input = self._safe_serialize(input_data)
            self.emit_event(
                AgentInputEvent.create(
                    message=_stringify(raw_input),
                    role=MessageRole.HUMAN,
                    metadata={
                        "framework": "llama_index",
                        "agent_name": agent_name,
                        "timestamp_ns": start_ns,
                        "raw_input": raw_input,
                    },
                )
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
            raw_output = self._safe_serialize(output)
            metadata: dict[str, Any] = {
                "framework": "llama_index",
                "agent_name": agent_name,
                "duration_ns": duration_ns,
                "raw_output": raw_output,
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
            serialized_input = self._safe_serialize(tool_input)
            serialized_output = self._safe_serialize(tool_output)
            input_data = _coerce_to_dict(serialized_input)
            input_data.setdefault("framework", "llama_index")
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
        if not self._connected:
            return
        try:
            model_name = model or "unknown"
            resolved_provider = provider or _detect_provider(model_name)
            input_messages: list[dict[str, str]] | None = None
            if self._capture_config.capture_content and messages:
                input_messages = messages
            self.emit_event(
                ModelInvokeEvent.create(
                    provider=resolved_provider,
                    name=model_name,
                    version="unavailable",
                    parameters={"framework": "llama_index"},
                    prompt_tokens=tokens_prompt,
                    completion_tokens=tokens_completion,
                    latency_ms=latency_ms,
                    input_messages=input_messages,
                )
            )
        except Exception:
            logger.warning("Error in on_llm_call", exc_info=True)

    def on_handoff(self, from_agent: str, to_agent: str, context: Any = None) -> None:
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

    def _emit_agent_config(self, agent_name: str, event_or_agent: Any) -> None:
        """Emit a typed :class:`EnvironmentConfigEvent` per agent.

        Idempotent per agent — only the first call for a given agent
        name actually emits. LlamaIndex's runtime is treated as a
        ``simulated`` environment by default; the real production
        environment (``cloud`` / ``on_prem``) is the responsibility
        of the host application's environment.config emission, not
        this framework adapter (mirrors the agno reference pattern).
        """
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        attributes: dict[str, Any] = {
            "framework": "llama_index",
            "agent_name": agent_name,
        }
        tools = getattr(event_or_agent, "tools", None)
        if tools:
            attributes["tools"] = [getattr(t, "name", str(t)) for t in tools]
        self.emit_event(
            EnvironmentConfigEvent.create(
                env_type=EnvironmentType.SIMULATED,
                attributes=attributes,
            )
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
            if isinstance(value, list):
                return [self._safe_serialize(v) for v in value[:100]]
            return str(value)
        except Exception:
            return str(value)
