"""OpenTelemetry integration with STRATIX and W3C trace-context propagation.

Demonstrates:
- Setting up an OTel TracerProvider with an OTLP exporter
- Integrating STRATIX's OTelExporter so STRATIX events become OTel spans
- Creating a parent OTel span and nesting STRATIX instrumentation inside it
- Inspecting span attributes and W3C traceparent headers
- Flushing and shutting down exporters cleanly

Requires:
    OPENAI_API_KEY                           - OpenAI API key (for the LLM call)
    OTEL_EXPORTER_OTLP_ENDPOINT (optional)   - Collector endpoint (default: localhost:4317)

Install OTel dependencies:
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from layerlens.instrument import STRATIX, emit_input, emit_output, emit_model_invoke
from layerlens.instrument.exporters import OTelExporter


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def setup_otel(endpoint: str, insecure: bool):
    """Initialize an OTel TracerProvider with OTLP gRPC exporter.

    Returns the tracer and provider so we can shut them down later.
    """
    try:
        from opentelemetry import trace, context as otel_context
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        print(
            "ERROR: opentelemetry-sdk is required. Install with:\n"
            "  pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc",
            file=sys.stderr,
        )
        sys.exit(1)

    resource = Resource.create({
        "service.name": "stratix-otel-sample",
        "service.version": "1.0.0",
    })
    provider = TracerProvider(resource=resource)

    # Add console exporter so we can see spans in stdout
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Optionally add OTLP gRPC exporter
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        print(f"[otel]    OTLP gRPC exporter -> {endpoint}")
    except ImportError:
        print("[otel]    OTLP gRPC exporter not available; using console only.")

    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("stratix-sample")
    print("[otel]    TracerProvider initialized.")
    return tracer, provider


def extract_traceparent(span) -> str:
    """Build a W3C traceparent header from the current span context."""
    ctx = span.get_span_context()
    trace_id = format(ctx.trace_id, "032x")
    span_id = format(ctx.span_id, "016x")
    flags = format(ctx.trace_flags, "02x")
    return f"00-{trace_id}-{span_id}-{flags}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STRATIX + OpenTelemetry integration with W3C context propagation."
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317"),
        help="OTel collector endpoint (default: localhost:4317).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=True,
        help="Use insecure gRPC connection (default: True).",
    )
    parser.add_argument(
        "--prompt",
        default="What are the three laws of thermodynamics?",
        help="Prompt to send to OpenAI.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--policy-ref",
        default="stratix-policy-otel-v1@1.0.0",
        help="STRATIX policy reference.",
    )
    args = parser.parse_args()

    openai_key = _require_env("OPENAI_API_KEY")

    # -- 1. Set up OTel --
    tracer, provider = setup_otel(args.endpoint, args.insecure)

    # -- 2. Initialize STRATIX with the OTel exporter --
    stratix_otel = OTelExporter(
        endpoint=args.endpoint,
        protocol="grpc",
        insecure=args.insecure,
        emit_genai_attributes=True,
    )
    stratix = STRATIX(
        policy_ref=args.policy_ref,
        agent_id="otel_sample_agent",
        framework="openai",
        exporter="otel",
        endpoint=args.endpoint,
    )
    print(f"[stratix] STRATIX initialized with OTel exporter -> {args.endpoint}")

    # -- 3. Create an OTel parent span, nest STRATIX inside --
    from opentelemetry import trace as otel_trace

    with tracer.start_as_current_span("sample.workflow") as parent_span:
        traceparent = extract_traceparent(parent_span)
        print(f"[otel]    W3C traceparent: {traceparent}")

        # Set custom attributes on the parent span
        parent_span.set_attribute("stratix.policy_ref", args.policy_ref)
        parent_span.set_attribute("stratix.agent_id", "otel_sample_agent")

        # Start a STRATIX trial inside the OTel span
        ctx = stratix.start_trial()
        print(f"[stratix] Trial started  trace_id={ctx.trace_id}")

        with stratix.context():
            # Record input
            emit_input(args.prompt, role="human")

            # Nested OTel span for the LLM call
            with tracer.start_as_current_span("llm.call") as llm_span:
                llm_span.set_attribute("gen_ai.request.model", args.model)
                llm_span.set_attribute("gen_ai.operation.name", "chat")

                # Call OpenAI
                from openai import OpenAI

                oai = OpenAI(api_key=openai_key)
                t0 = time.perf_counter()
                response = oai.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": args.prompt}],
                    max_tokens=256,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                usage = response.usage
                answer = response.choices[0].message.content or ""

                # Set OTel span attributes
                if usage:
                    llm_span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
                    llm_span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
                llm_span.set_attribute("gen_ai.response.model", args.model)

                child_traceparent = extract_traceparent(llm_span)
                print(f"[otel]    LLM span traceparent: {child_traceparent}")

            # Emit STRATIX model invocation event (also exported as OTel span)
            emit_model_invoke(
                provider="openai",
                name=args.model,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                latency_ms=latency_ms,
            )
            print(f"[stratix] Model event emitted: {args.model} ({latency_ms:.0f}ms)")

            emit_output(answer)
            print(f"[stratix] Output: {answer[:80]}...")

        # End trial
        summary = stratix.end_trial()
        print(f"[stratix] Trial ended  status={summary.get('status')}  events={summary.get('events')}")

    # -- 4. Flush and shutdown --
    stratix_otel.flush()
    stratix_otel.shutdown()
    provider.force_flush()
    provider.shutdown()
    print("[otel]    All exporters flushed and shut down.")


if __name__ == "__main__":
    main()
