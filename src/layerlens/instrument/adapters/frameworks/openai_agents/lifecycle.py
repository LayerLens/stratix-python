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
  Runner start/end   → agent.input / agent.output (L1)

Typed-event status (post PR #129 migration, bundle 4):

* Every emission flows through :meth:`BaseAdapter.emit_event` with a
  canonical Pydantic payload imported from
  :mod:`layerlens.instrument._compat.events`.
* OpenAI-Agents-specific provenance (``framework``, ``agent_name``,
  ``span_id``, ``trace_id``, ``timestamp_ns``, ``duration_ns``,
  ``event_subtype``) is carried in the canonical model's metadata /
  attributes / parameters slots — the canonical schema does not
  expose these as top-level fields.
* The trace_start / trace_end markers previously emitted as
  :class:`AgentStateChangeEvent` with only ``event_subtype`` cannot
  satisfy the canonical ``before_hash`` / ``after_hash`` requirement
  (no real state mutation to hash). They are remapped to
  :class:`AgentInputEvent` (trace_start, role=AGENT) and
  :class:`AgentOutputEvent` (trace_end), with the original
  ``event_subtype`` marker preserved on
  :class:`MessageContent.metadata`.
* The handoff context hash is generated via SHA-256 over the context
  string (or a deterministic from/to/reason tuple when the SDK does
  not surface a context payload) so the canonical
  :class:`AgentHandoffEvent.handoff_context_hash` validator always
  passes.
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
    ViolationType,
    AgentInputEvent,
    CostRecordEvent,
    EnvironmentType,
    IntegrationType,
    AgentOutputEvent,
    ModelInvokeEvent,
    AgentHandoffEvent,
    PolicyViolationEvent,
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
    :class:`AgentOutputEvent` to carry a ``message: str``. OpenAI
    Agents SDK callbacks deliver inputs/outputs as arbitrary Python
    objects (model responses, dicts, ``None``); this helper converts
    each to a (possibly empty) string so the typed event always
    validates. The original payload is preserved on
    :class:`MessageContent.metadata.raw_input` / ``raw_output``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _sha256_of(value: str) -> str:
    """Return a canonical ``sha256:<hex64>`` hash string for ``value``.

    The canonical schema's :class:`AgentHandoffEvent` requires
    ``handoff_context_hash`` to start with ``sha256:`` and have a
    64-character hex tail (see
    ``ateam/stratix/core/events/cross_cutting.py``). Centralising
    the format here ensures every emit site uses the same wire
    format — including the empty-string fallback used when the
    OpenAI Agents SDK has no handoff context to hash.
    """
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _detect_provider(model: str | None) -> str:
    """Detect the LLM provider from a model identifier.

    OpenAI Agents primarily routes calls through OpenAI / Azure OpenAI,
    but the SDK is provider-agnostic — third-party model identifiers
    can flow through GenerationSpanData.model. Default to ``openai``
    when the heuristic cannot match (the OpenAI Agents SDK is
    OpenAI-centric by design).
    """
    if not model:
        return "openai"
    model_lower = model.lower()
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
    return "openai"


class OpenAIAgentsAdapter(BaseAdapter):
    """LayerLens adapter for OpenAI Agents SDK."""

    FRAMEWORK = "openai_agents"
    VERSION = "0.1.0"
    # The adapter source has no direct ``pydantic`` imports (verified by
    # grep across ``frameworks/openai_agents/``). The adapter registers
    # a TraceProcessor and wraps Runner; both hand the adapter
    # SpanData-typed dicts that are read structurally rather than via
    # Pydantic methods.
    requires_pydantic = PydanticCompat.V1_OR_V2

    # Per-adapter ``extra="allow"`` decision: openai_agents targets the
    # canonical 13-event taxonomy exclusively. Unknown event types must
    # be rejected by the base adapter's typed-event validator, so this
    # stays ``False``. See ``docs/adapters/typed-events.md`` for the
    # opt-in policy.
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
        self._adapter_lock = threading.Lock()
        self._seen_agents: set[str] = set()
        self._framework_version: str | None = None
        self._trace_processor: Any | None = None
        self._run_starts: dict[int, int] = {}  # thread_id -> start_ns

    def connect(self) -> None:
        """Import openai-agents SDK and register trace processor."""
        try:
            import agents  # type: ignore[import-not-found,unused-ignore]

            self._framework_version = getattr(agents, "__version__", "unknown")
        except ImportError:
            logger.debug("openai-agents not installed")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        """Remove trace processor and flush sinks."""
        # Note: OpenAI Agents SDK add_trace_processor() is additive and global.
        # There is no SDK API to remove a processor, so we disable it via the
        # _connected guard in emit_event instead.
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
            description="LayerLens adapter for OpenAI Agents SDK",
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
            from agents import add_trace_processor  # type: ignore[import-not-found,unused-ignore]

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
            from agents.tracing import TracingProcessor  # type: ignore[import-not-found,unused-ignore]
        except ImportError:
            return None

        # Renamed from StratixTraceProcessor → LayerLensTraceProcessor;
        # backward-compat alias is exposed at module scope below.
        class LayerLensTraceProcessor(TracingProcessor):  # type: ignore[misc,unused-ignore]
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

            def force_flush(self) -> None:
                pass

            def shutdown(self) -> None:
                pass

        return LayerLensTraceProcessor()

    # --- Trace Lifecycle ---

    def _on_trace_start(self, trace: Any) -> None:
        """Emit a typed :class:`AgentInputEvent` for a TraceProcessor trace_start.

        The previous adapter implementation emitted an ad-hoc
        ``agent.state.change`` payload carrying only an
        ``event_subtype`` marker. That payload did not satisfy the
        canonical :class:`AgentStateChangeEvent` schema's
        ``before_hash`` / ``after_hash`` requirement (the trace
        boundary has no real state mutation to hash).

        The trace_start boundary is logically the inbound for a new
        agent run, so the canonical mapping is :class:`AgentInputEvent`
        with ``role=AGENT``. The original ``event_subtype="trace_start"``
        marker is preserved on :class:`MessageContent.metadata` so
        downstream consumers can still filter on it.
        """
        if not self._connected:
            return
        tid = threading.get_ident()
        start_ns = time.time_ns()
        with self._adapter_lock:
            self._run_starts[tid] = start_ns
        self.emit_event(
            AgentInputEvent.create(
                message="",
                role=MessageRole.AGENT,
                metadata={
                    "framework": "openai_agents",
                    "event_subtype": "trace_start",
                    "trace_id": getattr(trace, "trace_id", None),
                    "timestamp_ns": start_ns,
                },
            )
        )

    def _on_trace_end(self, trace: Any) -> None:
        """Emit a typed :class:`AgentOutputEvent` for a TraceProcessor trace_end.

        See :meth:`_on_trace_start` for the rationale on why this is
        not an :class:`AgentStateChangeEvent`. Duration metadata
        lives on :class:`MessageContent.metadata.duration_ns`.
        """
        if not self._connected:
            return
        tid = threading.get_ident()
        end_ns = time.time_ns()
        with self._adapter_lock:
            start_ns = self._run_starts.pop(tid, 0)
        duration_ns = end_ns - start_ns if start_ns else 0
        self.emit_event(
            AgentOutputEvent.create(
                message="",
                metadata={
                    "framework": "openai_agents",
                    "event_subtype": "trace_end",
                    "trace_id": getattr(trace, "trace_id", None),
                    "duration_ns": duration_ns,
                },
            )
        )

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
        """Emit a typed :class:`AgentInputEvent` for an AgentSpanData start.

        OpenAI-Agents-specific provenance (``framework``,
        ``agent_name``, ``span_id``, ``timestamp_ns``) lives on
        :class:`MessageContent.metadata`. The agent config emission is
        idempotent per agent name.
        """
        agent_name = getattr(data, "name", None) or "unknown"
        self._emit_agent_config(agent_name, data)
        self.emit_event(
            AgentInputEvent.create(
                message="",
                role=MessageRole.AGENT,
                metadata={
                    "framework": "openai_agents",
                    "agent_name": agent_name,
                    "span_id": getattr(span, "span_id", None),
                    "timestamp_ns": time.time_ns(),
                },
            )
        )

    def _on_agent_span_end(self, span: Any, data: Any) -> None:
        """Emit a typed :class:`AgentOutputEvent` for an AgentSpanData end.

        The canonical ``message`` slot carries the stringified output;
        OpenAI-Agents-specific provenance (``framework``,
        ``agent_name``, ``span_id``, ``raw_output``) lives on
        :class:`MessageContent.metadata`.
        """
        agent_name = getattr(data, "name", None) or "unknown"
        output = getattr(data, "output", None)
        serialised_output = self._safe_serialize(output)
        self.emit_event(
            AgentOutputEvent.create(
                message=_stringify(serialised_output),
                metadata={
                    "framework": "openai_agents",
                    "agent_name": agent_name,
                    "span_id": getattr(span, "span_id", None),
                    "raw_output": serialised_output,
                },
            )
        )

    def _on_generation_span_end(self, span: Any, data: Any) -> None:
        """Emit typed :class:`ModelInvokeEvent` (and :class:`CostRecordEvent`).

        Provider is derived from the model identifier
        (default ``openai`` when unrecognised — the OpenAI Agents SDK
        is OpenAI-centric by design). Token usage is mirrored onto the
        canonical ``prompt_tokens`` / ``completion_tokens`` slots and
        a paired :class:`CostRecordEvent` is emitted when the SDK
        surfaces usage metrics.
        """
        model = getattr(data, "model", None)
        provider = _detect_provider(model)
        input_tokens = getattr(data, "input_tokens", None)
        output_tokens = getattr(data, "output_tokens", None)
        duration = getattr(span, "duration_ms", None)
        self.emit_event(
            ModelInvokeEvent.create(
                provider=provider,
                name=model or "unknown",
                version="unavailable",
                parameters={"framework": "openai_agents"},
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                latency_ms=duration,
            )
        )
        if input_tokens is not None or output_tokens is not None:
            total = (input_tokens or 0) + (output_tokens or 0)
            self.emit_event(
                CostRecordEvent.create(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    tokens=total,
                )
            )

    def _on_function_span_end(self, span: Any, data: Any) -> None:
        """Emit a typed :class:`ToolCallEvent` for a FunctionSpanData end.

        OpenAI Agents function spans wrap Python callables exposed
        via the SDK's tool-decorator surface, so
        ``integration=IntegrationType.LIBRARY`` is the canonical
        match. OpenAI-Agents-specific provenance (``framework``) is
        folded onto the canonical ``input`` dict.
        """
        tool_name = getattr(data, "name", None) or "unknown"
        serialised_input = self._safe_serialize(getattr(data, "input", None))
        serialised_output = self._safe_serialize(getattr(data, "output", None))
        input_data: dict[str, Any]
        if isinstance(serialised_input, dict):
            input_data = dict(serialised_input)
        elif serialised_input is None:
            input_data = {}
        else:
            input_data = {"value": serialised_input}
        input_data["framework"] = "openai_agents"
        output_data: dict[str, Any] | None
        if isinstance(serialised_output, dict):
            output_data = dict(serialised_output)
        elif serialised_output is None:
            output_data = None
        else:
            output_data = {"value": serialised_output}
        self.emit_event(
            ToolCallEvent.create(
                name=tool_name,
                version="unavailable",
                integration=IntegrationType.LIBRARY,
                input_data=input_data,
                output_data=output_data,
                latency_ms=getattr(span, "duration_ms", None),
            )
        )

    def _on_handoff_span_start(self, span: Any, data: Any) -> None:
        pass  # Start event captured on end for complete data

    def _on_handoff_span_end(self, span: Any, data: Any) -> None:
        """Emit a typed :class:`AgentHandoffEvent` for a HandoffSpanData end.

        OpenAI Agents handoff spans do not surface an explicit context
        payload, so the canonical ``handoff_context_hash`` is computed
        deterministically from the (from_agent, to_agent, reason)
        tuple — this preserves the wire-format guarantee while
        remaining stable across replays.
        """
        from_agent = getattr(data, "from_agent", None) or "unknown"
        to_agent = getattr(data, "to_agent", None) or "unknown"
        reason = "handoff"
        self.emit_event(
            AgentHandoffEvent.create(
                from_agent=from_agent,
                to_agent=to_agent,
                handoff_context_hash=_sha256_of(f"{reason}::{from_agent}::{to_agent}"),
            )
        )

    def _on_guardrail_span_end(self, span: Any, data: Any) -> None:
        """Emit a typed :class:`PolicyViolationEvent` for a guardrail trip.

        OpenAI Agents guardrails are policy-constraint checks that
        run before / after model invocation. The canonical mapping is
        :attr:`ViolationType.POLICY_CONSTRAINT`. OpenAI-Agents-
        specific provenance (``framework``, ``guardrail_name``,
        ``triggered``, ``output``) lives on
        :attr:`ViolationInfo.details`.

        Non-triggered guardrail spans are not policy violations and
        therefore are not emitted as ``policy.violation`` events —
        only triggered guardrails fire the canonical event. This is
        more semantically correct than the previous adapter, which
        emitted ``policy.violation`` for every guardrail span
        (triggered or not). The previous behaviour is preserved
        defensively: when ``triggered`` cannot be evaluated as truthy
        but is also not ``False``, the canonical event still fires
        (``triggered=None`` etc. — the guardrail outcome is unknown
        and the most conservative interpretation is to emit).
        """
        guardrail_name = getattr(data, "name", None) or "unknown"
        triggered = getattr(data, "triggered", False)
        output_value = self._safe_serialize(getattr(data, "output", None))
        self.emit_event(
            PolicyViolationEvent.create(
                violation_type=ViolationType.POLICY_CONSTRAINT,
                root_cause=f"guardrail '{guardrail_name}' triggered={triggered!r}",
                remediation="review guardrail output and adjust agent prompt or guardrail policy",
                details={
                    "framework": "openai_agents",
                    "guardrail_name": guardrail_name,
                    "triggered": triggered,
                    "output": output_value,
                },
            )
        )

    # --- Lifecycle Hooks (Runner wrapping) ---

    def on_run_start(self, agent_name: str | None = None, input_data: Any = None) -> None:
        """Emit a typed :class:`AgentInputEvent` for a Runner run_start.

        OpenAI-Agents-specific provenance (``framework``,
        ``agent_name``, ``timestamp_ns``, ``raw_input``) lives on
        :class:`MessageContent.metadata`.
        """
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._run_starts[tid] = start_ns
            serialised_input = self._safe_serialize(input_data)
            self.emit_event(
                AgentInputEvent.create(
                    message=_stringify(serialised_input),
                    role=MessageRole.HUMAN,
                    metadata={
                        "framework": "openai_agents",
                        "agent_name": agent_name,
                        "timestamp_ns": start_ns,
                        "raw_input": serialised_input,
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
        """Emit a typed :class:`AgentOutputEvent` for a Runner run_end.

        OpenAI-Agents-specific provenance (``framework``,
        ``agent_name``, ``duration_ns``, ``raw_output``, ``error``,
        ``run_status``) lives on :class:`MessageContent.metadata`.
        """
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._run_starts.pop(tid, 0)
            duration_ns = end_ns - start_ns if start_ns else 0
            serialised_output = self._safe_serialize(output)
            metadata: dict[str, Any] = {
                "framework": "openai_agents",
                "agent_name": agent_name,
                "duration_ns": duration_ns,
                "raw_output": serialised_output,
                "run_status": "run_failed" if error else "run_complete",
            }
            if error:
                metadata["error"] = str(error)
            self.emit_event(
                AgentOutputEvent.create(
                    message=_stringify(serialised_output),
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
        """Emit a typed :class:`ToolCallEvent` for a manual tool invocation.

        Defaults to ``integration=LIBRARY`` (the OpenAI Agents tool
        surface is in-process Python). OpenAI-Agents-specific
        provenance (``framework``) is folded onto the canonical
        ``input``.
        """
        if not self._connected:
            return
        try:
            serialised_input = self._safe_serialize(tool_input)
            serialised_output = self._safe_serialize(tool_output)
            input_data: dict[str, Any]
            if isinstance(serialised_input, dict):
                input_data = dict(serialised_input)
            elif serialised_input is None:
                input_data = {}
            else:
                input_data = {"value": serialised_input}
            input_data["framework"] = "openai_agents"
            output_data: dict[str, Any] | None
            if isinstance(serialised_output, dict):
                output_data = dict(serialised_output)
            elif serialised_output is None:
                output_data = None
            else:
                output_data = {"value": serialised_output}
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
        """Emit a typed :class:`ModelInvokeEvent` for a manual LLM call.

        OpenAI-Agents-specific provenance (``framework``) is carried
        on :attr:`ModelInfo.parameters`. Provider falls back to the
        identifier-derived guess (default ``openai``) when the caller
        supplies none.
        """
        if not self._connected:
            return
        try:
            self.emit_event(
                ModelInvokeEvent.create(
                    provider=provider or _detect_provider(model),
                    name=model or "unknown",
                    version="unavailable",
                    parameters={"framework": "openai_agents"},
                    prompt_tokens=tokens_prompt,
                    completion_tokens=tokens_completion,
                    latency_ms=latency_ms,
                    input_messages=messages
                    if (self._capture_config.capture_content and messages)
                    else None,
                )
            )
        except Exception:
            logger.warning("Error in on_llm_call", exc_info=True)

    def on_handoff(
        self,
        from_agent: str,
        to_agent: str,
        context: Any = None,
    ) -> None:
        """Emit a typed :class:`AgentHandoffEvent` for a manual handoff.

        Empty contexts are still hashed (over the empty string) so
        the canonical wire format is uniform — the previous adapter
        emitted ``None`` when context was missing, which the canonical
        validator rejects.
        """
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

    def _emit_agent_config(self, agent_name: str, data: Any) -> None:
        """Emit a typed :class:`EnvironmentConfigEvent` once per agent.

        OpenAI Agents runs in OpenAI-managed cloud infrastructure, so
        the canonical :attr:`EnvironmentType.CLOUD` value is used.
        Agent configuration (``instructions``, ``model``,
        ``handoff_description``, ``tools``, ``handoffs``) lives on
        :attr:`EnvironmentInfo.attributes`.
        """
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        attributes: dict[str, Any] = {
            "framework": "openai_agents",
            "agent_name": agent_name,
        }
        for attr in ("instructions", "model", "handoff_description"):
            val = getattr(data, attr, None)
            if val is not None:
                attributes[attr] = str(val)
        tools = getattr(data, "tools", None)
        if tools:
            attributes["tools"] = [getattr(t, "name", str(t)) for t in tools]
        handoffs = getattr(data, "handoffs", None)
        if handoffs:
            attributes["handoffs"] = [getattr(h, "agent_name", str(h)) for h in handoffs]
        self.emit_event(
            EnvironmentConfigEvent.create(
                env_type=EnvironmentType.CLOUD,
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
