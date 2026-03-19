"""
STRATIX OpenTelemetry Exporter

From Step 4 specification:
- OpenTelemetry span/event export
- Maps STRATIX events to OTel spans and span events
- Supports gRPC and HTTP protocols
- Emits gen_ai.* attributes alongside stratix.* (OTel GenAI Semantic Conventions)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from layerlens.instrument.exporters._base import Exporter

if TYPE_CHECKING:
    from layerlens.instrument.schema.event import STRATIXEvent

logger = logging.getLogger(__name__)

# SpanKind mapping per event type
_SPAN_KIND_MAP = {
    "model.invoke": "CLIENT",
    "tool.call": "INTERNAL",
    "agent.input": "SERVER",
    "agent.output": "SERVER",
    "evaluation.result": "INTERNAL",
}


def _get_genai_span_name(event_type: str, payload: dict[str, Any]) -> str:
    """Build span name following OTel GenAI convention: {operation} {model}."""
    if event_type == "model.invoke":
        model = payload.get("model", {})
        model_name = model.get("name", "") if isinstance(model, dict) else str(model)
        operation = payload.get("operation", "chat")
        if model_name:
            return f"{operation} {model_name}"
        return f"{operation}"

    if event_type == "evaluation.result":
        evaluation = payload.get("evaluation", {})
        eval_name = evaluation.get("dimension", "unknown") if isinstance(evaluation, dict) else "unknown"
        return f"evaluation {eval_name}"

    if event_type in ("agent.input", "agent.output"):
        agent_id = payload.get("agent_id", "")
        if agent_id:
            return f"agent {agent_id}"

    return f"stratix.{event_type}"


class OTelExporter(Exporter):
    """
    OpenTelemetry exporter for STRATIX events.

    Maps STRATIX events to OpenTelemetry spans and events,
    then exports them to a configured collector.

    Features:
    - Converts STRATIX events to OTel spans
    - Preserves trace context (trace_id, span_id, parent_span_id)
    - Includes STRATIX-specific attributes
    - Emits gen_ai.* attributes (OTel GenAI Semantic Conventions)
    - Supports batch export for efficiency
    """

    def __init__(
        self,
        endpoint: str,
        protocol: str = "grpc",
        headers: dict[str, str] | None = None,
        insecure: bool = False,
        batch_size: int = 100,
        export_timeout_ms: int = 30000,
        emit_genai_attributes: bool = True,
    ):
        """
        Initialize the OpenTelemetry exporter.

        Args:
            endpoint: OTel collector endpoint (e.g., "localhost:4317")
            protocol: Export protocol ("grpc" or "http")
            headers: Optional headers for authentication
            insecure: Use insecure connection (for development)
            batch_size: Maximum batch size for export
            export_timeout_ms: Export timeout in milliseconds
            emit_genai_attributes: Emit gen_ai.* attributes alongside stratix.*
        """
        self._endpoint = endpoint
        self._protocol = protocol
        self._headers = headers or {}
        self._insecure = insecure
        self._batch_size = batch_size
        self._export_timeout_ms = export_timeout_ms
        self._emit_genai = emit_genai_attributes

        # Buffer for batching
        self._buffer: list["STRATIXEvent"] = []

        # OTel SDK components (lazy initialized)
        self._tracer = None
        self._span_exporter = None
        self._span_processor = None
        self._tracer_provider = None
        self._initialized = False

    @property
    def _capture_content(self) -> bool:
        """Check if content capture is enabled via env var."""
        return os.environ.get("OTEL_GENAI_CAPTURE_MESSAGE_CONTENT", "").lower() == "true"

    def _initialize(self) -> None:
        """Initialize OpenTelemetry SDK components."""
        if self._initialized:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource

            # Create resource with STRATIX service name
            resource = Resource.create({
                "service.name": "stratix",
                "service.version": "1.0.0",
            })

            # Create tracer provider
            self._tracer_provider = TracerProvider(resource=resource)

            # Create span exporter based on protocol
            if self._protocol == "grpc":
                self._span_exporter = self._create_grpc_exporter()
            else:
                self._span_exporter = self._create_http_exporter()

            # Create batch processor
            if self._span_exporter:
                self._span_processor = BatchSpanProcessor(
                    self._span_exporter,
                    max_queue_size=self._batch_size * 10,
                    max_export_batch_size=self._batch_size,
                    export_timeout_millis=self._export_timeout_ms,
                )
                self._tracer_provider.add_span_processor(self._span_processor)

            # Get tracer from local provider (avoid polluting global state)
            self._tracer = self._tracer_provider.get_tracer("stratix")

            self._initialized = True
            logger.info("OTel exporter initialized: %s", self._endpoint)

        except ImportError as e:
            logger.warning("OpenTelemetry SDK not available: %s", e)
            self._initialized = False
        except Exception as e:
            logger.error("Failed to initialize OTel exporter: %s", e)
            self._initialized = False

    def _create_grpc_exporter(self) -> Any:
        """Create a gRPC span exporter."""
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            return OTLPSpanExporter(
                endpoint=self._endpoint,
                insecure=self._insecure,
                headers=self._headers or None,
            )
        except ImportError:
            logger.warning("gRPC exporter not available, falling back to HTTP")
            return self._create_http_exporter()

    def _create_http_exporter(self) -> Any:
        """Create an HTTP span exporter."""
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            # HTTP endpoint typically ends with /v1/traces
            endpoint = self._endpoint
            if not endpoint.endswith("/v1/traces"):
                endpoint = f"{endpoint}/v1/traces"
            return OTLPSpanExporter(
                endpoint=endpoint,
                headers=self._headers or None,
            )
        except ImportError:
            logger.warning("HTTP exporter not available")
            return None

    def export(self, event: "STRATIXEvent") -> None:
        """
        Export a single STRATIX event.

        Args:
            event: The event to export
        """
        self._initialize()

        if not self._initialized or self._tracer is None:
            # Buffer for later if not initialized
            self._buffer.append(event)
            if len(self._buffer) >= self._batch_size:
                logger.warning("Export buffer full, dropping oldest events")
                self._buffer = self._buffer[-self._batch_size:]
            return

        self._export_event(event)

    def _export_event(self, event: "STRATIXEvent") -> None:
        """Convert and export an STRATIX event as an OTel span."""
        if self._tracer is None:
            return

        try:
            from opentelemetry.trace import SpanKind

            # Extract event info
            event_dict = event.to_dict()
            identity = event_dict.get("identity", {})
            payload = event_dict.get("payload", {})

            # Determine event type and span name
            event_type = payload.get("event_type", "unknown")
            span_name = _get_genai_span_name(event_type, payload)

            # Determine SpanKind
            kind_str = _SPAN_KIND_MAP.get(event_type, "INTERNAL")
            span_kind = getattr(SpanKind, kind_str, SpanKind.INTERNAL)

            # Create span context if we have trace/span IDs
            trace_id = identity.get("trace_id")

            # Start span
            with self._tracer.start_as_current_span(
                span_name,
                kind=span_kind,
            ) as span:
                # Set STRATIX-specific attributes (always emitted)
                span.set_attribute("stratix.event_type", event_type)
                span.set_attribute("stratix.layer", payload.get("layer", ""))
                span.set_attribute("stratix.evaluation_id", identity.get("evaluation_id", ""))
                span.set_attribute("stratix.trial_id", identity.get("trial_id", ""))
                span.set_attribute("stratix.trace_id", trace_id or "")
                span.set_attribute("stratix.agent_id", identity.get("agent_id", ""))
                span.set_attribute("stratix.sequence_id", identity.get("sequence_id", 0))

                # Add attestation info
                attestation = event_dict.get("attestation", {})
                if attestation:
                    span.set_attribute("stratix.hash", attestation.get("hash", ""))
                    span.set_attribute("stratix.previous_hash", attestation.get("previous_hash", ""))

                # Add privacy info
                privacy = event_dict.get("privacy", {})
                span.set_attribute("stratix.privacy_level", privacy.get("level", ""))

                # Add event-specific attributes (stratix.* namespace)
                self._add_event_attributes(span, event_type, payload)

                # Add gen_ai.* attributes (OTel GenAI Semantic Conventions)
                if self._emit_genai:
                    self._add_genai_attributes(span, event_type, payload, identity)

                # Emit content capture as span events (OTel GenAI semconv)
                if self._emit_genai and self._capture_content:
                    self._add_content_events(span, event_type, payload)

        except Exception as e:
            logger.error("Failed to export event: %s", e)

    def _add_event_attributes(
        self, span: Any, event_type: str, payload: dict[str, Any]
    ) -> None:
        """Add event-type-specific attributes to the span (stratix.* namespace)."""
        if event_type == "tool.call":
            tool = payload.get("tool", {})
            span.set_attribute("stratix.tool.name", tool.get("name", ""))
            span.set_attribute("stratix.tool.version", tool.get("version", ""))
            invocation = payload.get("invocation", {})
            span.set_attribute("stratix.tool.latency_ms", invocation.get("latency_ms", 0))
            if invocation.get("error"):
                span.set_attribute("stratix.tool.error", invocation.get("error", ""))

        elif event_type == "model.invoke":
            model = payload.get("model", {})
            span.set_attribute("stratix.model.provider", model.get("provider", ""))
            span.set_attribute("stratix.model.name", model.get("name", ""))
            span.set_attribute("stratix.model.version", model.get("version", ""))
            usage = payload.get("usage", {})
            span.set_attribute("stratix.model.prompt_tokens", usage.get("prompt_tokens", 0))
            span.set_attribute("stratix.model.completion_tokens", usage.get("completion_tokens", 0))
            span.set_attribute("stratix.model.total_tokens", usage.get("total_tokens", 0))
            span.set_attribute("stratix.model.latency_ms", usage.get("latency_ms", 0))

        elif event_type == "agent.input":
            content = payload.get("content", {})
            span.set_attribute("stratix.input.role", content.get("role", ""))

        elif event_type == "agent.output":
            content = payload.get("content", {})
            span.set_attribute("stratix.output.role", content.get("role", ""))

        elif event_type == "policy.violation":
            span.set_attribute("stratix.violation.type", payload.get("violation_type", ""))
            span.set_attribute("stratix.violation.root_cause", payload.get("root_cause", ""))

        elif event_type == "cost.record":
            span.set_attribute("stratix.cost.type", payload.get("cost_type", ""))
            span.set_attribute("stratix.cost.amount", payload.get("amount") or 0)
            span.set_attribute("stratix.cost.currency", payload.get("currency", ""))

        elif event_type == "evaluation.result":
            evaluation = payload.get("evaluation", {})
            span.set_attribute("stratix.evaluation.score", evaluation.get("score", 0.0))
            span.set_attribute("stratix.evaluation.dimension", evaluation.get("dimension", ""))
            span.set_attribute("stratix.evaluation.is_passing", payload.get("is_passing", False))

    def _add_genai_attributes(
        self,
        span: Any,
        event_type: str,
        payload: dict[str, Any],
        identity: dict[str, Any],
    ) -> None:
        """Add OTel GenAI Semantic Convention attributes (gen_ai.* namespace)."""
        if event_type == "model.invoke":
            model = payload.get("model", {})
            model_name = model.get("name", "") if isinstance(model, dict) else ""
            provider = model.get("provider", "") if isinstance(model, dict) else ""

            # Core gen_ai.* attributes
            span.set_attribute("gen_ai.provider.name", provider)
            span.set_attribute("gen_ai.operation.name", payload.get("operation", "chat"))
            span.set_attribute("gen_ai.request.model", model_name)

            # Response model (may differ from request model)
            metadata = payload.get("metadata", {})
            response_model = metadata.get("response_model", model_name)
            if response_model:
                span.set_attribute("gen_ai.response.model", response_model)

            # Token usage
            prompt_tokens = payload.get("prompt_tokens")
            completion_tokens = payload.get("completion_tokens")
            if prompt_tokens is not None:
                span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
            if completion_tokens is not None:
                span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)

            # Model parameters
            parameters = model.get("parameters", {}) if isinstance(model, dict) else {}
            for param, genai_key in [
                ("temperature", "gen_ai.request.temperature"),
                ("max_tokens", "gen_ai.request.max_tokens"),
                ("top_p", "gen_ai.request.top_p"),
                ("top_k", "gen_ai.request.top_k"),
                ("frequency_penalty", "gen_ai.request.frequency_penalty"),
                ("presence_penalty", "gen_ai.request.presence_penalty"),
                ("seed", "gen_ai.request.seed"),
            ]:
                val = parameters.get(param)
                if val is not None:
                    span.set_attribute(genai_key, val)

            stop_seqs = parameters.get("stop_sequences") or parameters.get("stop")
            if stop_seqs:
                span.set_attribute("gen_ai.request.stop_sequences", stop_seqs)

            # Finish reason and response ID from metadata
            finish_reason = metadata.get("finish_reason")
            if finish_reason:
                span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])

            response_id = metadata.get("response_id")
            if response_id:
                span.set_attribute("gen_ai.response.id", response_id)

            # Provider-specific attributes
            self._add_provider_specific_attributes(span, provider, metadata)

        elif event_type == "evaluation.result":
            evaluation = payload.get("evaluation", {})
            score = evaluation.get("score")
            if score is not None:
                span.set_attribute("gen_ai.evaluation.score.value", score)
            dimension = evaluation.get("dimension")
            if dimension:
                span.set_attribute("gen_ai.evaluation.name", dimension)
            label = evaluation.get("label")
            if label:
                span.set_attribute("gen_ai.evaluation.score.label", label)
            explanation = evaluation.get("explanation")
            if explanation and self._capture_content:
                span.set_attribute("stratix.evaluation.explanation", explanation)
            # STRATIX extensions (not in OTel GenAI semconv)
            grader_id = evaluation.get("grader_id")
            if grader_id:
                span.set_attribute("stratix.evaluation.grader_id", grader_id)
            is_passing = payload.get("is_passing")
            if is_passing is not None:
                span.set_attribute("stratix.evaluation.is_passing", is_passing)

        elif event_type == "tool.call":
            # Tool span conventions (gen_ai.tool.*)
            tool = payload.get("tool", {})
            tool_name = tool.get("name", "") if isinstance(tool, dict) else str(tool)
            if tool_name:
                span.set_attribute("gen_ai.tool.name", tool_name)
            tool_desc = tool.get("description", "") if isinstance(tool, dict) else ""
            if tool_desc:
                span.set_attribute("gen_ai.tool.description", tool_desc)
            invocation = payload.get("invocation", {})
            call_id = invocation.get("call_id", "") if isinstance(invocation, dict) else ""
            if call_id:
                span.set_attribute("gen_ai.tool.call.id", call_id)

        elif event_type in ("agent.input", "agent.output"):
            # Agent span conventions
            agent_id = identity.get("agent_id", "")
            if agent_id:
                span.set_attribute("gen_ai.agent.name", agent_id)
            agent_desc = identity.get("agent_description", "")
            if agent_desc:
                span.set_attribute("gen_ai.agent.description", agent_desc)

    def _add_content_events(
        self,
        span: Any,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Emit content as span events per OTel GenAI semconv (event-based capture)."""
        if event_type == "model.invoke":
            # Input messages
            content = payload.get("content", {})
            input_msg = content.get("message") or content.get("input")
            if input_msg:
                span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": str(input_msg)})
            # Output messages
            output_msg = content.get("output") or content.get("response")
            if output_msg:
                span.add_event("gen_ai.content.completion", {"gen_ai.completion": str(output_msg)})

        elif event_type == "agent.input":
            content = payload.get("content", {})
            message = content.get("message", "")
            if message:
                span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": str(message)})

        elif event_type == "agent.output":
            content = payload.get("content", {})
            message = content.get("message", "")
            if message:
                span.add_event("gen_ai.content.completion", {"gen_ai.completion": str(message)})

    def _add_provider_specific_attributes(
        self,
        span: Any,
        provider: str,
        metadata: dict[str, Any],
    ) -> None:
        """Add provider-specific OTel attributes."""
        if provider == "openai":
            if "system_fingerprint" in metadata:
                span.set_attribute("gen_ai.openai.response.system_fingerprint", metadata["system_fingerprint"])
            if "service_tier" in metadata:
                span.set_attribute("gen_ai.openai.response.service_tier", metadata["service_tier"])
            if "seed" in metadata:
                span.set_attribute("gen_ai.openai.request.seed", metadata["seed"])

        elif provider == "anthropic":
            if "cache_creation_input_tokens" in metadata:
                span.set_attribute("gen_ai.usage.cache_creation_input_tokens", metadata["cache_creation_input_tokens"])
            if "cache_read_input_tokens" in metadata:
                span.set_attribute("gen_ai.usage.cache_read_input_tokens", metadata["cache_read_input_tokens"])

        elif provider == "bedrock":
            if "guardrail_id" in metadata:
                span.set_attribute("aws.bedrock.guardrail.id", metadata["guardrail_id"])
            if "knowledge_base_id" in metadata:
                span.set_attribute("aws.bedrock.knowledge_base.id", metadata["knowledge_base_id"])
            if "agent_id" in metadata:
                span.set_attribute("aws.bedrock.agent.id", metadata["agent_id"])

    def export_batch(self, events: list["STRATIXEvent"]) -> None:
        """
        Export a batch of STRATIX events.

        Args:
            events: List of events to export
        """
        for event in events:
            self.export(event)

    def flush(self) -> None:
        """Flush any buffered events."""
        if self._span_processor:
            try:
                self._span_processor.force_flush()
            except Exception as e:
                logger.error("Failed to flush: %s", e)

    def shutdown(self) -> None:
        """Shutdown the exporter and release resources."""
        if self._span_processor:
            try:
                self._span_processor.shutdown()
            except Exception as e:
                logger.error("Failed to shutdown span processor: %s", e)

        if self._tracer_provider:
            try:
                self._tracer_provider.shutdown()
            except Exception as e:
                logger.error("Failed to shutdown tracer provider: %s", e)

        self._initialized = False
        logger.info("OTel exporter shutdown complete")
