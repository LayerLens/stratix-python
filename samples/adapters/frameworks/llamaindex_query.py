#!/usr/bin/env python3
"""
LlamaIndex Query Engine with STRATIX Instrumentation

Demonstrates a LlamaIndex document index and query engine traced by STRATIX,
capturing indexing, retrieval, and synthesis steps.

Requirements:
    pip install llama-index llama-index-llms-openai llama-index-embeddings-openai

Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    sys.exit(
        "This sample requires llama-index. Install with:\n"
        "  pip install llama-index llama-index-llms-openai llama-index-embeddings-openai"
    )

from layerlens.instrument import STRATIX


def build_documents() -> list[Document]:
    """Create sample documents for indexing."""
    return [
        Document(text="STRATIX is the instrumentation SDK within the LayerLens platform."),
        Document(text="Each STRATIX event is linked via a hash chain for tamper evidence."),
        Document(text="Adapters provide zero-config tracing for popular agent frameworks."),
        Document(text="Policy enforcement ensures agents operate within defined guardrails."),
        Document(text="Events are exported via OpenTelemetry to any compatible backend."),
        Document(text="The SDK supports LangGraph, LangChain, CrewAI, AutoGen, and many more."),
        Document(text="Cost tracking records token usage and estimated dollar costs per call."),
        Document(text="Privacy levels control what data is stored: cleartext, redacted, or encrypted."),
    ]


def main():
    parser = argparse.ArgumentParser(description="LlamaIndex query engine with STRATIX tracing")
    parser.add_argument("--query", default="How does STRATIX handle privacy and cost tracking?")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--agent-id", default="llamaindex-query-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable before running this sample.")

    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="llama_index",
    )
    ctx = stratix.start_trial()
    stratix.emit_input(args.query)

    # Configure LlamaIndex settings
    Settings.llm = LlamaOpenAI(model=args.model, temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Build index
    documents = build_documents()
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine
    query_engine = index.as_query_engine(similarity_top_k=3)

    print(f"Query: {args.query}\n")

    # Run query
    response = query_engine.query(args.query)
    answer = str(response)
    print(f"Answer: {answer}\n")

    # Show source nodes
    if response.source_nodes:
        print("Sources:")
        for node in response.source_nodes:
            print(f"  [{node.score:.3f}] {node.text[:80]}...")
        print()

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
