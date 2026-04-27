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
"""

from __future__ import annotations

import time
import uuid
import hashlib
import logging
import threading
from typing import Any

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


class GoogleADKAdapter(BaseAdapter):
    """LayerLens adapter for Google Agent Development Kit."""

    FRAMEWORK = "google_adk"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/google_adk/``). The adapter only registers
    # ADK's native 6-callback hooks and emits dict events; it never
    # touches ADK's own Pydantic models.
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
            self.emit_dict_event(
                "agent.input",
                {
                    "framework": "google_adk",
                    "agent_name": agent_name,
                    "input": self._safe_serialize(getattr(callback_context, "user_content", None)),
                    "timestamp_ns": start_ns,
                },
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
            self.emit_dict_event(
                "agent.output",
                {
                    "framework": "google_adk",
                    "agent_name": agent_name,
                    "output": self._safe_serialize(getattr(callback_context, "agent_output", None)),
                    "duration_ns": duration_ns,
                },
            )
            # Surface ADK-side errors propagated via the callback context.
            self._maybe_emit_callback_error(
                callback_context,
                phase="agent.run",
                context={"agent_name": agent_name},
                event_type="agent.error",
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
            latency_ms = None
            if start_ns:
                latency_ms = (time.time_ns() - start_ns) / 1_000_000
            payload: dict[str, Any] = {"framework": "google_adk"}
            model = getattr(callback_context, "model", None) or getattr(llm_response, "model", None)
            if model:
                payload["model"] = str(model)
                payload["provider"] = "google"
            usage = getattr(llm_response, "usage_metadata", None)
            if usage:
                payload["tokens_prompt"] = getattr(usage, "prompt_token_count", None)
                payload["tokens_completion"] = getattr(usage, "candidates_token_count", None)
            if latency_ms is not None:
                payload["latency_ms"] = latency_ms
            self.emit_dict_event("model.invoke", payload)
            if usage:
                self.emit_dict_event(
                    "cost.record",
                    {
                        "framework": "google_adk",
                        "model": payload.get("model"),
                        "tokens_prompt": payload.get("tokens_prompt"),
                        "tokens_completion": payload.get("tokens_completion"),
                        "tokens_total": (
                            (payload.get("tokens_prompt") or 0)
                            + (payload.get("tokens_completion") or 0)
                        ),
                    },
                )
            # ADK exposes failed model calls via llm_response.error /
            # error_message — surface as a discrete event.
            self._maybe_emit_callback_error(
                llm_response,
                phase="model.invoke",
                context={"model": str(model) if model else None},
                event_type="model.error",
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
            latency_ms = None
            if start_ns:
                latency_ms = (time.time_ns() - start_ns) / 1_000_000
            self.emit_dict_event(
                "tool.call",
                {
                    "framework": "google_adk",
                    "tool_name": tool_name,
                    "tool_input": self._safe_serialize(tool_input),
                    "tool_output": self._safe_serialize(tool_output),
                    "latency_ms": latency_ms,
                },
            )
            # tool_output may be an Exception object or a dict containing
            # a top-level "error" / "exception" entry when the function
            # raised — surface as discrete tool.error.
            self._maybe_emit_callback_error(
                tool_output,
                phase="tool.call",
                context={"tool_name": tool_name},
                event_type="tool.error",
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
            self.emit_dict_event(
                "agent.input",
                {
                    "framework": "google_adk",
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
                "framework": "google_adk",
                "agent_name": agent_name,
                "output": self._safe_serialize(output),
                "duration_ns": duration_ns,
            }
            if error:
                payload["error"] = str(error)
            self.emit_dict_event("agent.output", payload)
            if error is not None:
                emit_error_event(
                    self,
                    error,
                    {"framework": "google_adk", "agent_name": agent_name, "phase": "agent.run"},
                    event_type="agent.error",
                )
        except Exception:
            logger.warning("Error in on_agent_end", exc_info=True)

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
                    "reason": "transfer_to_agent",
                    "context_hash": hashlib.sha256(context_str.encode()).hexdigest()
                    if context_str
                    else None,
                    "context_preview": context_str[:500] if context_str else None,
                },
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
            payload: dict[str, Any] = {
                "framework": "google_adk",
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
                    {"framework": "google_adk", "tool_name": tool_name, "phase": "tool.call"},
                    event_type="tool.error",
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
            payload: dict[str, Any] = {"framework": "google_adk"}
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

    def _maybe_emit_callback_error(
        self,
        carrier: Any,
        phase: str,
        context: dict[str, Any],
        event_type: str,
    ) -> None:
        """Emit a discrete error event when ``carrier`` exposes one.

        ADK callbacks receive response objects that may carry the failure
        on ``error`` / ``error_message`` / ``exception`` attributes (the
        SDK uses these instead of raising directly so the agent loop can
        decide whether to retry). We coerce whatever shape we find into
        a real exception and route through :func:`emit_error_event`.
        """
        if carrier is None:
            return
        raw_error: Any = None
        if isinstance(carrier, BaseException):
            raw_error = carrier
        else:
            for attr in ("error", "exception", "error_message"):
                value = getattr(carrier, attr, None)
                if value:
                    raw_error = value
                    break
            if raw_error is None and isinstance(carrier, dict):
                for key in ("error", "exception", "error_message"):
                    if carrier.get(key):
                        raw_error = carrier[key]
                        break
        if raw_error is None:
            return
        if isinstance(raw_error, BaseException):
            exc: BaseException = raw_error
        else:
            exc = RuntimeError(str(raw_error))
        emit_error_event(
            self,
            exc,
            {"framework": "google_adk", "phase": phase, **context},
            event_type=event_type,
        )

    def _get_agent_name(self, callback_context: Any) -> str:
        agent = getattr(callback_context, "agent", None)
        if agent:
            return getattr(agent, "name", None) or str(agent)
        return "unknown"

    def _emit_agent_config(self, agent_name: str, callback_context: Any) -> None:
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        agent = getattr(callback_context, "agent", None)
        metadata: dict[str, Any] = {
            "framework": "google_adk",
            "agent_name": agent_name,
        }
        if agent:
            for attr in ("description", "instruction", "model"):
                val = getattr(agent, attr, None)
                if val is not None:
                    metadata[attr] = str(val)
            tools = getattr(agent, "tools", None)
            if tools:
                metadata["tools"] = [getattr(t, "name", str(t)) for t in tools]
            sub_agents = getattr(agent, "sub_agents", None)
            if sub_agents:
                metadata["sub_agents"] = [getattr(a, "name", str(a)) for a in sub_agents]
        session = getattr(callback_context, "session", None)
        if session:
            metadata["session_id"] = getattr(session, "id", None)
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
