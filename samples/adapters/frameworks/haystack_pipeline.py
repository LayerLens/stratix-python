#!/usr/bin/env python3
"""
Haystack Document Pipeline with STRATIX Instrumentation

Demonstrates a Haystack indexing and query pipeline traced by STRATIX,
capturing document processing steps and retrieval operations.

Requirements:
    pip install haystack-ai

Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from haystack import Document, Pipeline
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore
except ImportError:
    sys.exit(
        "This sample requires haystack-ai. Install with:\n"
        "  pip install haystack-ai"
    )

from layerlens.instrument import STRATIX


def build_document_store() -> InMemoryDocumentStore:
    """Create and populate an in-memory document store."""
    store = InMemoryDocumentStore()
    documents = [
        Document(content="STRATIX provides tamper-evident instrumentation for AI agents.", meta={"source": "stratix_overview"}),
        Document(content="Hash-chain attestation ensures every event is cryptographically linked.", meta={"source": "attestation_docs"}),
        Document(content="Framework adapters auto-detect LangChain, LangGraph, CrewAI, and more.", meta={"source": "adapter_guide"}),
        Document(content="Policy enforcement prevents agents from exceeding defined guardrails.", meta={"source": "policy_docs"}),
        Document(content="OpenTelemetry export sends traces to any compatible collector.", meta={"source": "otel_guide"}),
        Document(content="Haystack pipelines can be instrumented with STRATIX for full observability.", meta={"source": "haystack_guide"}),
    ]
    store.write_documents(documents)
    return store


def build_query_pipeline(store: InMemoryDocumentStore, model: str) -> Pipeline:
    """Construct a Haystack RAG query pipeline."""
    retriever = InMemoryBM25Retriever(document_store=store, top_k=3)

    prompt_template = """
    Answer the question based on the provided context documents.

    Context:
    {% for doc in documents %}
    - {{ doc.content }}
    {% endfor %}

    Question: {{ question }}
    Answer:
    """
    prompt_builder = PromptBuilder(template=prompt_template)
    generator = OpenAIGenerator(model=model)

    pipeline = Pipeline()
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "generator")

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Haystack pipeline with STRATIX tracing")
    parser.add_argument("--query", default="How does STRATIX ensure trace integrity?")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--agent-id", default="haystack-pipeline-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable before running this sample.")

    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="haystack",
    )
    ctx = stratix.start_trial()
    stratix.emit_input(args.query)

    # Build pipeline
    store = build_document_store()
    pipeline = build_query_pipeline(store, args.model)

    print(f"Query: {args.query}\n")

    # Run
    result = pipeline.run({
        "retriever": {"query": args.query},
        "prompt_builder": {"question": args.query},
    })

    answer = result["generator"]["replies"][0]
    print(f"Answer: {answer}\n")
    stratix.emit_output(answer)

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
