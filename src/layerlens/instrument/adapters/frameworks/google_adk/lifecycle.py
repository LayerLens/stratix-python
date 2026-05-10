"""
Google Agent Development Kit (ADK) adapter lifecycle.

Instrumentation strategy: Callback pattern (native first-class support)
  BeforeAgentCallback  → agent.input (L1)
  AfterAgentCallback   → agent.output (L1)
  BeforeModelCallback  → model.invoke start (L3)
  AfterModelCallback   → model.invoke complete (L3)
  BeforeToolCallback   → tool.call start (L5a)
  AfterToolCallback    → tool.call complete (L5a)
  transfer_to_agent    → agent.handoff (Cross)

Typed-event status (post PR #129 migration, bundle 3):

* Every emission flows through :meth:`BaseAdapter.emit_event` with a
  canonical Pydantic payload imported from
  :mod:`layerlens.instrument._compat.events`.
* Google ADK-specific provenance (``framework``, ``agent_name``,
  ``timestamp_ns``, ``session_id``, ``description``, ``instruction``)
  is carried in the canonical model's metadata / attributes /
  parameters slots — the canonical schema does not expose these as
  top-level fields.
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
    :class:`AgentOutputEvent` to carry a ``message: str``. ADK
    callbacks deliver the underlying input/output as arbitrary Python
    objects (Pydantic ``Content`` models with ``parts``, dicts,
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
        # ADK Content payloads serialise to ``{"parts": [...], "role": ...}``;
        # surface a flat string view of the parts when present.
        parts = value.get("parts")
        if isinstance(parts, list) and parts:
            text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in parts]
            return "".join(text_parts)
        content = value.get("content")
        if isinstance(content, str):
            return content
    return str(value)


def _sha256_of(value: str) -> str:
    """Return a canonical ``sha256:<hex64>`` hash string for ``value``.

    The canonical schema's :class:`AgentHandoffEvent` requires
    ``handoff_context_hash`` to start with ``sha256:`` and have a
    64-character hex tail (see
    ``ateam/stratix/core/events/cross_cutting.py``). Centralising the
    format here ensures every emit site uses the same wire format —
    including the empty-string fallback used when ADK has no
    context to hash.
    """
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _coerce_to_dict(value: Any) -> dict[str, Any]:
    """Coerce ``value`` into a dict suitable for the canonical
    :class:`ToolCallEvent` ``input`` / ``output`` slots.

    The canonical schema requires ``input: dict[str, Any]`` and
    ``output: dict[str, Any] | None``. ADK tool callbacks deliver
    arbitrary Python values (scalars, dicts, lists, dataclass-like
    objects). This helper wraps non-dict values in ``{"value": ...}``
    so the canonical slot is always satisfied. ``None`` returns an
    empty dict so the caller can pass it positionally without a guard.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


def _detect_provider(model: str | None) -> str:
    """Detect the LLM provider from a model identifier.

    Google ADK's primary models are Gemini, but ADK supports
    third-party LLMs via LiteLLM. The canonical
    :class:`ModelInvokeEvent` requires both ``provider`` and ``name``,
    so this heuristic derives the provider from well-known model name
    prefixes. Mirrors the agno reference implementation. Unknown
    identifiers return ``"google"`` (the ADK default) since most ADK
    deployments use Gemini.
    """
    if not model:
        return "google"
    model_lower = model.lower()
    if "gemini" in model_lower:
        return "google"
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"
    if "llama" in model_lower:
        return "meta"
    if "command" in model_lower:
        return "cohere"
    return "google"


class GoogleADKAdapter(BaseAdapter):
    """LayerLens adapter for Google Agent Development Kit."""

    FRAMEWORK = "google_adk"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/google_adk/``). The adapter only registers
    # ADK's native 6-callback hooks and emits typed events through the
    # canonical schema (PR #129); it never touches ADK's own Pydantic
    # models.
    requires_pydantic = PydanticCompat.V1_OR_V2

    # Per-adapter ``extra="allow"`` decision: google_adk targets the
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
        self._model_call_starts: dict[int, int] = {}  # thread_id -> start_ns
        self._tool_call_starts: dict[str, int] = {}
        self._agent_starts: dict[int, int] = {}  # thread_id -> start_ns

    def connect(self) -> None:
        try:
            import google.adk  # type: ignore[import-untyped,unused-ignore]

            self._framework_version = getattr(google.adk, "__version__", "unknown")
        except ImportError:
            try:
                import google.genai  # type: ignore[import-untyped,unused-ignore]

                self._framework_version = getattr(google.genai, "__version__", "unknown")
            except ImportError:
                logger.debug("google-adk not installed")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        self._originals.clear()
        self._seen_agents.clear()
        self._model_call_starts.clear()
        self._tool_call_starts.clear()
        self._agent_starts.clear()
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
            name="GoogleADKAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_HANDOFFS,
            ],
            description="LayerLens adapter for Google Agent Development Kit",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="GoogleADKAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
        )

    # --- Framework Integration ---

    def instrument_agent(self, agent: Any) -> Any:
        """Attach Stratix callbacks to a Google ADK agent."""
        try:
            agent.before_agent_callback = self._before_agent_callback
            agent.after_agent_callback = self._after_agent_callback
            agent.before_model_callback = self._before_model_callback
            agent.after_model_callback = self._after_model_callback
            agent.before_tool_callback = self._before_tool_callback
            agent.after_tool_callback = self._after_tool_callback
        except Exception:
            logger.warning("Failed to attach callbacks to agent", exc_info=True)
        return agent

    # --- Callback Implementations ---

    def _before_agent_callback(self, callback_context: Any) -> Any:
        if not self._connected:
            return None
        try:
            agent_name = self._get_agent_name(callback_context)
            self._emit_agent_config(agent_name, callback_context)
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._agent_starts[tid] = start_ns
            raw_input = self._safe_serialize(getattr(callback_context, "user_content", None))
            self.emit_event(
                AgentInputEvent.create(
                    message=_stringify(raw_input),
                    role=MessageRole.HUMAN,
                    metadata={
                        "framework": "google_adk",
                        "agent_name": agent_name,
                        "timestamp_ns": start_ns,
                        "raw_input": raw_input,
                    },
                )
            )
        except Exception:
            logger.warning("Error in before_agent_callback", exc_info=True)
        return None

    def _after_agent_callback(self, callback_context: Any) -> Any:
        if not self._connected:
            return None
        try:
            agent_name = self._get_agent_name(callback_context)
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._agent_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            raw_output = self._safe_serialize(getattr(callback_context, "agent_output", None))
            self.emit_event(
                AgentOutputEvent.create(
                    message=_stringify(raw_output),
                    metadata={
                        "framework": "google_adk",
                        "agent_name": agent_name,
                        "duration_ns": duration_ns,
                        "raw_output": raw_output,
                    },
                )
            )
        except Exception:
            logger.warning("Error in after_agent_callback", exc_info=True)
        return None

    def _before_model_callback(self, callback_context: Any, llm_request: Any) -> Any:
        if not self._connected:
            return None
        try:
            tid = threading.get_ident()
            with self._adapter_lock:
                self._model_call_starts[tid] = time.time_ns()
        except Exception:
            logger.warning("Error in before_model_callback", exc_info=True)
        return None

    def _after_model_callback(self, callback_context: Any, llm_response: Any) -> Any:
        if not self._connected:
            return None
        try:
            tid = threading.get_ident()
            with self._adapter_lock:
                start_ns = self._model_call_starts.pop(tid, None)
            latency_ms: float | None = None
            if start_ns:
                latency_ms = (time.time_ns() - start_ns) / 1_000_000
            model_raw = getattr(callback_context, "model", None) or getattr(
                llm_response, "model", None
            )
            model_name = str(model_raw) if model_raw else "unknown"
            provider = _detect_provider(model_name)
            usage = getattr(llm_response, "usage_metadata", None)
            prompt_tokens = (
                getattr(usage, "prompt_token_count", None) if usage else None
            )
            completion_tokens = (
                getattr(usage, "candidates_token_count", None) if usage else None
            )
            self.emit_event(
                ModelInvokeEvent.create(
                    provider=provider,
                    name=model_name,
                    version="unavailable",
                    parameters={"framework": "google_adk"},
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                )
            )
            if usage:
                total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
                self.emit_event(
                    CostRecordEvent.create(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        tokens=total_tokens,
                    )
                )
        except Exception:
            logger.warning("Error in after_model_callback", exc_info=True)
        return None

    def _before_tool_callback(self, callback_context: Any, tool_name: str, tool_input: Any) -> Any:
        if not self._connected:
            return None
        try:
            call_id = f"{tool_name}_{id(tool_input)}"
            with self._adapter_lock:
                self._tool_call_starts[call_id] = time.time_ns()
        except Exception:
            logger.warning("Error in before_tool_callback", exc_info=True)
        return None

    def _after_tool_callback(
        self,
        callback_context: Any,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
    ) -> Any:
        if not self._connected:
            return None
        try:
            call_id = f"{tool_name}_{id(tool_input)}"
            with self._adapter_lock:
                start_ns = self._tool_call_starts.pop(call_id, None)
            latency_ms: float | None = None
            if start_ns:
                latency_ms = (time.time_ns() - start_ns) / 1_000_000
            serialized_input = self._safe_serialize(tool_input)
            serialized_output = self._safe_serialize(tool_output)
            input_data = _coerce_to_dict(serialized_input)
            input_data.setdefault("framework", "google_adk")
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
                    latency_ms=latency_ms,
                )
            )
        except Exception:
            logger.warning("Error in after_tool_callback", exc_info=True)
        return None

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
                        "framework": "google_adk",
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
                "framework": "google_adk",
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
            input_data.setdefault("framework", "google_adk")
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
                    parameters={"framework": "google_adk"},
                    prompt_tokens=tokens_prompt,
                    completion_tokens=tokens_completion,
                    latency_ms=latency_ms,
                    input_messages=input_messages,
                )
            )
        except Exception:
            logger.warning("Error in on_llm_call", exc_info=True)

    # --- Helpers ---

    def _get_agent_name(self, callback_context: Any) -> str:
        agent = getattr(callback_context, "agent", None)
        if agent:
            return getattr(agent, "name", None) or str(agent)
        return "unknown"

    def _emit_agent_config(self, agent_name: str, callback_context: Any) -> None:
        """Emit a typed :class:`EnvironmentConfigEvent` per agent.

        Idempotent per agent — only the first call for a given agent
        name actually emits. ADK's runtime is treated as a
        ``simulated`` environment by default; the real production
        environment (``cloud`` / ``on_prem``) is the responsibility of
        the host application's environment.config emission, not this
        framework adapter (mirrors the agno reference pattern).
        """
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        agent = getattr(callback_context, "agent", None)
        attributes: dict[str, Any] = {
            "framework": "google_adk",
            "agent_name": agent_name,
        }
        if agent:
            for attr in ("description", "instruction", "model"):
                val = getattr(agent, attr, None)
                if val is not None:
                    attributes[attr] = str(val)
            tools = getattr(agent, "tools", None)
            if tools:
                attributes["tools"] = [getattr(t, "name", str(t)) for t in tools]
            sub_agents = getattr(agent, "sub_agents", None)
            if sub_agents:
                attributes["sub_agents"] = [getattr(a, "name", str(a)) for a in sub_agents]
        session = getattr(callback_context, "session", None)
        if session:
            attributes["session_id"] = getattr(session, "id", None)
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
            return str(value)
        except Exception:
            return str(value)
