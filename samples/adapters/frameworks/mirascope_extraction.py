#!/usr/bin/env python3
"""
Mirascope Entity Extraction with STRATIX Instrumentation

Demonstrates Mirascope's decorator-based LLM calls for structured entity
extraction, with STRATIX tracing of each extraction step.

Requirements:
    pip install mirascope[openai]

Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import mirascope
    from mirascope.core import openai
except ImportError:
    sys.exit(
        "This sample requires mirascope. Install with:\n"
        "  pip install mirascope[openai]"
    )

from pydantic import BaseModel, Field
from layerlens.instrument import STRATIX


# ---------------------------------------------------------------------------
# Extraction models
# ---------------------------------------------------------------------------

class TechEntity(BaseModel):
    """A technology entity extracted from text."""
    name: str = Field(description="Name of the technology, tool, or framework")
    category: str = Field(description="Category: language, framework, platform, tool, or service")
    relevance: str = Field(description="Why this entity matters in context")


class TechAnalysis(BaseModel):
    """Structured analysis of technology mentions in text."""
    entities: list[TechEntity] = Field(description="All technology entities found")
    primary_domain: str = Field(description="Primary technology domain discussed")
    complexity_level: str = Field(description="Complexity: beginner, intermediate, or advanced")
    summary: str = Field(description="Brief summary of the technology landscape described")


# ---------------------------------------------------------------------------
# Mirascope-decorated extraction functions
# ---------------------------------------------------------------------------

@openai.call(model="gpt-4o-mini", response_model=TechAnalysis)
def extract_tech_entities(text: str) -> str:
    return (
        f"Analyze the following text and extract all technology entities. "
        f"Categorize each entity and assess the overall technology landscape.\n\n"
        f"Text: {text}"
    )


@openai.call(model="gpt-4o-mini")
def generate_recommendation(analysis: str) -> str:
    return (
        f"Based on this technology analysis, provide a brief recommendation "
        f"for someone wanting to learn these technologies. Be specific and "
        f"actionable.\n\nAnalysis: {analysis}"
    )


# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = {
    "stack": (
        "Our production stack runs Python 3.12 with FastAPI for the backend, "
        "deployed on AWS ECS with Terraform. The ML pipeline uses PyTorch for "
        "training and ONNX Runtime for inference. We monitor with Datadog and "
        "use LangGraph for our agent orchestration, instrumented with STRATIX."
    ),
    "migration": (
        "We're migrating from a monolithic Django application to microservices "
        "using Go and gRPC. The frontend is moving from React to Next.js 14 "
        "with server components. PostgreSQL remains our primary database, but "
        "we're adding Redis for caching and Kafka for event streaming."
    ),
    "aiops": (
        "The AI ops team uses Kubernetes for container orchestration, ArgoCD "
        "for GitOps deployments, and Prometheus plus Grafana for monitoring. "
        "Model serving is handled by vLLM behind an Envoy proxy. Vector search "
        "runs on Pinecone, with LangChain managing the RAG pipeline."
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Mirascope entity extraction with STRATIX tracing")
    parser.add_argument("--text", choices=list(SAMPLE_TEXTS.keys()), default="stack",
                        help="Sample text to analyze")
    parser.add_argument("--custom-text", default=None, help="Custom text (overrides --text)")
    parser.add_argument("--agent-id", default="mirascope-extraction-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable before running this sample.")

    text = args.custom_text or SAMPLE_TEXTS[args.text]

    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="openai",
    )
    ctx = stratix.start_trial()
    stratix.emit_input(f"Analyze: {text[:100]}...")

    print(f"Input text:\n  {text}\n")

    # Step 1: Extract entities
    analysis = extract_tech_entities(text)
    print(f"Domain: {analysis.primary_domain}")
    print(f"Complexity: {analysis.complexity_level}")
    print(f"Summary: {analysis.summary}\n")
    print(f"Entities ({len(analysis.entities)}):")
    for entity in analysis.entities:
        print(f"  [{entity.category}] {entity.name} - {entity.relevance}")

    # Step 2: Generate recommendation
    print("\nGenerating recommendation...\n")
    recommendation = generate_recommendation(analysis.summary)
    print(f"Recommendation:\n  {recommendation.content}\n")

    stratix.emit_output(f"Found {len(analysis.entities)} entities. {analysis.summary}")

    # Summary
    summary = stratix.end_trial()
    events = stratix.get_events()
    print(f"\n--- STRATIX Trace Summary ---")
    print(f"Status: {summary.get('status')}")
    print(f"Captured {len(events)} events:")
    for e in events:
        print(f"  {e.get_event_type()}: {str(e.payload)[:80]}")


if __name__ == "__main__":
    main()
