"""
OpenAI Agents SDK adapter lifecycle.

Instrumentation strategy: Dual approach
  1. TraceProcessor (primary) — framework-sanctioned, receives all SDK span events
  2. Runner wrapping (secondary) — execution lifecycle hooks

SDK spans map to Stratix events:
  AgentSpanData      → agent.input / agent.output (L1)
  GenerationSpanData → model.invoke (L3)
  FunctionSpanData   → tool.call (L5a)
  HandoffSpanData    → agent.handoff (Cross)
  GuardrailSpanData  → policy.violation (Cross)
  Runner start/end   → agent.state.change (Cross)
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from typing import Any

from layerlens.instrument.adapters._base import (
    AdapterCapability,
    AdapterHealth,
    AdapterInfo,
    AdapterStatus,
    BaseAdapter,
)
from layerlens.instrument.adapters._base import ReplayableTrace

logger = logging.getLogger(__name__)


class OpenAIAgentsAdapter(BaseAdapter):
    """Stratix adapter for OpenAI Agents SDK."""

    FRAMEWORK = "openai_agents"
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: Any | None = None,
        stratix_instance: Any | None = None,
    ) -> None:
        resolved = stratix or stratix_instance
        super().__init__(stratix=resolved, capture_config=capture_config)
        self._adapter_lock = threading.Lock()
        self._seen_agents: set[str] = set()
        self._framework_version: str | None = None
        self._trace_processor: Any | None = None
        self._run_starts: dict[int, int] = {}  # thread_id -> start_ns

    def connect(self) -> None:
        """Import openai-agents SDK and register trace processor."""
        try:
            import agents

            self._framework_version = getattr(agents, "__version__", "unknown")
        except ImportError:
            logger.debug("openai-agents not installed")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        """Remove trace processor and flush sinks."""
        # Note: OpenAI Agents SDK add_trace_processor() is additive and global.
        # There is no SDK API to remove a processor, so we disable it via the
        # _connected guard in emit_dict_event instead.
        self._trace_processor = None
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
            name="OpenAIAgentsAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.TRACE_HANDOFFS,
            ],
            description="Stratix adapter for OpenAI Agents SDK",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="OpenAIAgentsAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
        )

    # --- Framework Integration ---

    def instrument_runner(self, runner: Any) -> Any:
        """Register Stratix trace processor with the SDK."""
        try:
            from agents import add_trace_processor

            processor = self._create_trace_processor()
            if processor is None:
                logger.warning("Could not create trace processor (TraceProcessor not importable)")
                return runner
            add_trace_processor(processor)
            self._trace_processor = processor
        except ImportError:
            logger.debug("Cannot import agents.add_trace_processor")
        except Exception:
            logger.warning("Failed to register trace processor", exc_info=True)
        return runner

    def _create_trace_processor(self) -> Any:
        """Create a TraceProcessor that routes SDK spans to Stratix events."""
        adapter = self

        try:
            from agents.tracing import TraceProcessor
        except ImportError:
            return None

        class StratixTraceProcessor(TraceProcessor):
            def on_trace_start(self, trace: Any) -> None:
                try:
                    adapter._on_trace_start(trace)
                except Exception:
                    logger.warning("Error in on_trace_start", exc_info=True)

            def on_trace_end(self, trace: Any) -> None:
                try:
                    adapter._on_trace_end(trace)
                except Exception:
                    logger.warning("Error in on_trace_end", exc_info=True)

            def on_span_start(self, span: Any) -> None:
                try:
                    adapter._on_span_start(span)
                except Exception:
                    logger.warning("Error in on_span_start", exc_info=True)

            def on_span_end(self, span: Any) -> None:
                try:
                    adapter._on_span_end(span)
                except Exception:
                    logger.warning("Error in on_span_end", exc_info=True)

        return StratixTraceProcessor()

    # --- Trace Lifecycle ---

    def _on_trace_start(self, trace: Any) -> None:
        if not self._connected:
            return
        tid = threading.get_ident()
        start_ns = time.time_ns()
        with self._adapter_lock:
            self._run_starts[tid] = start_ns
        self.emit_dict_event("agent.state.change", {
            "framework": "openai_agents",
            "event_subtype": "trace_start",
            "trace_id": getattr(trace, "trace_id", None),
            "timestamp_ns": start_ns,
        })

    def _on_trace_end(self, trace: Any) -> None:
        if not self._connected:
            return
        tid = threading.get_ident()
        end_ns = time.time_ns()
        with self._adapter_lock:
            start_ns = self._run_starts.pop(tid, 0)
        duration_ns = end_ns - start_ns if start_ns else 0
        self.emit_dict_event("agent.state.change", {
            "framework": "openai_agents",
            "event_subtype": "trace_end",
            "trace_id": getattr(trace, "trace_id", None),
            "duration_ns": duration_ns,
        })

    def _on_span_start(self, span: Any) -> None:
        span_data = getattr(span, "span_data", None)
        if span_data is None:
            return
        span_type = type(span_data).__name__
        if span_type == "AgentSpanData":
            self._on_agent_span_start(span, span_data)
        elif span_type == "GenerationSpanData":
            pass  # handled on end
        elif span_type == "HandoffSpanData":
            self._on_handoff_span_start(span, span_data)
        elif span_type == "GuardrailSpanData":
            pass  # handled on end

    def _on_span_end(self, span: Any) -> None:
        span_data = getattr(span, "span_data", None)
        if span_data is None:
            return
        span_type = type(span_data).__name__
        if span_type == "AgentSpanData":
            self._on_agent_span_end(span, span_data)
        elif span_type == "GenerationSpanData":
            self._on_generation_span_end(span, span_data)
        elif span_type == "FunctionSpanData":
            self._on_function_span_end(span, span_data)
        elif span_type == "HandoffSpanData":
            self._on_handoff_span_end(span, span_data)
        elif span_type == "GuardrailSpanData":
            self._on_guardrail_span_end(span, span_data)

    # --- Span Type Handlers ---

    def _on_agent_span_start(self, span: Any, data: Any) -> None:
        agent_name = getattr(data, "name", None) or "unknown"
        self._emit_agent_config(agent_name, data)
        self.emit_dict_event("agent.input", {
            "framework": "openai_agents",
            "agent_name": agent_name,
            "span_id": getattr(span, "span_id", None),
            "timestamp_ns": time.time_ns(),
        })

    def _on_agent_span_end(self, span: Any, data: Any) -> None:
        agent_name = getattr(data, "name", None) or "unknown"
        output = getattr(data, "output", None)
        self.emit_dict_event("agent.output", {
            "framework": "openai_agents",
            "agent_name": agent_name,
            "output": self._safe_serialize(output),
            "span_id": getattr(span, "span_id", None),
        })

    def _on_generation_span_end(self, span: Any, data: Any) -> None:
        payload: dict[str, Any] = {"framework": "openai_agents"}
        model = getattr(data, "model", None)
        if model:
            payload["model"] = model
        input_tokens = getattr(data, "input_tokens", None)
        output_tokens = getattr(data, "output_tokens", None)
        if input_tokens is not None:
            payload["tokens_prompt"] = input_tokens
        if output_tokens is not None:
            payload["tokens_completion"] = output_tokens
        duration = getattr(span, "duration_ms", None)
        if duration is not None:
            payload["latency_ms"] = duration
        self.emit_dict_event("model.invoke", payload)
        if input_tokens is not None or output_tokens is not None:
            self.emit_dict_event("cost.record", {
                "framework": "openai_agents",
                "model": model,
                "tokens_prompt": input_tokens,
                "tokens_completion": output_tokens,
                "tokens_total": (input_tokens or 0) + (output_tokens or 0),
            })

    def _on_function_span_end(self, span: Any, data: Any) -> None:
        tool_name = getattr(data, "name", None) or "unknown"
        self.emit_dict_event("tool.call", {
            "framework": "openai_agents",
            "tool_name": tool_name,
            "tool_input": self._safe_serialize(getattr(data, "input", None)),
            "tool_output": self._safe_serialize(getattr(data, "output", None)),
            "latency_ms": getattr(span, "duration_ms", None),
        })

    def _on_handoff_span_start(self, span: Any, data: Any) -> None:
        pass  # Start event captured on end for complete data

    def _on_handoff_span_end(self, span: Any, data: Any) -> None:
        from_agent = getattr(data, "from_agent", None) or "unknown"
        to_agent = getattr(data, "to_agent", None) or "unknown"
        self.emit_dict_event("agent.handoff", {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "reason": "handoff",
            "framework": "openai_agents",
        })

    def _on_guardrail_span_end(self, span: Any, data: Any) -> None:
        guardrail_name = getattr(data, "name", None) or "unknown"
        triggered = getattr(data, "triggered", False)
        self.emit_dict_event("policy.violation", {
            "framework": "openai_agents",
            "guardrail_name": guardrail_name,
            "triggered": triggered,
            "output": self._safe_serialize(getattr(data, "output", None)),
        })

    # --- Lifecycle Hooks (Runner wrapping) ---

    def on_run_start(self, agent_name: str | None = None, input_data: Any = None) -> None:
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._run_starts[tid] = start_ns
            self.emit_dict_event("agent.input", {
                "framework": "openai_agents",
                "agent_name": agent_name,
                "input": self._safe_serialize(input_data),
                "timestamp_ns": start_ns,
            })
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
                "framework": "openai_agents",
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
        error: Exception | None = None,
        latency_ms: float | None = None,
    ) -> None:
        if not self._connected:
            return
        try:
            payload: dict[str, Any] = {
                "framework": "openai_agents",
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
            payload: dict[str, Any] = {"framework": "openai_agents"}
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
            context_hash = hashlib.sha256(context_str.encode("utf-8")).hexdigest() if context_str else None
            self.emit_dict_event("agent.handoff", {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "reason": "handoff",
                "context_hash": context_hash,
                "context_preview": context_str[:500] if context_str else None,
            })
        except Exception:
            logger.warning("Error in on_handoff", exc_info=True)

    # --- Helpers ---

    def _emit_agent_config(self, agent_name: str, data: Any) -> None:
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        metadata: dict[str, Any] = {
            "framework": "openai_agents",
            "agent_name": agent_name,
        }
        for attr in ("instructions", "model", "handoff_description"):
            val = getattr(data, attr, None)
            if val is not None:
                metadata[attr] = str(val)
        tools = getattr(data, "tools", None)
        if tools:
            metadata["tools"] = [getattr(t, "name", str(t)) for t in tools]
        handoffs = getattr(data, "handoffs", None)
        if handoffs:
            metadata["handoffs"] = [getattr(h, "agent_name", str(h)) for h in handoffs]
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
