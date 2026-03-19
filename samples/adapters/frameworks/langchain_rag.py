#!/usr/bin/env python3
"""
LangChain RAG Pipeline with STRATIX Instrumentation

Demonstrates a retrieval-augmented generation pipeline using LangChain
with full STRATIX tracing of retrieval spans, document metadata, and
LLM invocations.

Requirements:
    pip install langchain langchain-openai langchain-community

Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
except ImportError:
    sys.exit(
        "This sample requires langchain + langchain-openai. Install with:\n"
        "  pip install langchain langchain-openai langchain-community"
    )

from layerlens.instrument import STRATIX, emit_input, emit_output, emit_model_invoke, emit_tool_call


# ---------------------------------------------------------------------------
# Simple in-memory retriever (no FAISS dependency)
# ---------------------------------------------------------------------------

class SimpleRetriever:
    """Cosine-similarity retriever backed by a plain list of documents."""

    def __init__(self, documents: list[Document], embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self._doc_vectors: list[list[float]] | None = None

    def _ensure_indexed(self):
        if self._doc_vectors is None:
            texts = [d.page_content for d in self.documents]
            self._doc_vectors = self.embeddings.embed_documents(texts)

    def invoke(self, query: str) -> list[Document]:
        self._ensure_indexed()
        q_vec = self.embeddings.embed_query(query)
        scored = []
        for i, dv in enumerate(self._doc_vectors):
            dot = sum(a * b for a, b in zip(q_vec, dv))
            scored.append((dot, i))
        scored.sort(reverse=True)
        return [self.documents[i] for _, i in scored[:3]]


def build_knowledge_base() -> list[Document]:
    """Return sample documents for the RAG pipeline."""
    return [
        Document(page_content="LayerLens provides observability for AI agents.", metadata={"source": "docs/overview.md"}),
        Document(page_content="STRATIX is the instrumentation SDK within LayerLens.", metadata={"source": "docs/stratix.md"}),
        Document(page_content="Adapters connect STRATIX to frameworks like LangChain.", metadata={"source": "docs/adapters.md"}),
        Document(page_content="Policy enforcement ensures agents stay within guardrails.", metadata={"source": "docs/policy.md"}),
        Document(page_content="Hash-chain attestation provides tamper-evident audit trails.", metadata={"source": "docs/attestation.md"}),
    ]


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def main():
    parser = argparse.ArgumentParser(description="LangChain RAG with STRATIX tracing")
    parser.add_argument("--query", default="What is STRATIX and how does it work?")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--agent-id", default="langchain-rag-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable before running this sample.")

    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="langchain",
    )
    ctx = stratix.start_trial()
    print(f"[stratix] Trial started  trace_id={ctx.trace_id}")

    # Build retriever
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = build_knowledge_base()
    retriever = SimpleRetriever(docs, embeddings)

    # Build RAG chain
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    llm = ChatOpenAI(model=args.model, temperature=0)

    # Instrument retrieval
    with stratix.context():
        emit_input(args.query, role="human")
        print(f"Query: {args.query}\n")

        # Retrieval step
        t0 = time.perf_counter()
        retrieved = retriever.invoke(args.query)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        emit_tool_call(
            name="retriever", version="1.0.0",
            input_data={"query": args.query},
            output_data={"num_results": len(retrieved), "sources": [d.metadata.get("source", "") for d in retrieved]},
            latency_ms=round(retrieval_ms),
        )
        print(f"[retrieval] Found {len(retrieved)} documents ({retrieval_ms:.0f}ms)")
        for d in retrieved:
            print(f"  - {d.metadata.get('source', 'unknown')}: {d.page_content[:60]}...")

        # Generation step
        context_text = format_docs(retrieved)
        rag_chain = (
            {"context": lambda q: context_text, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        t0 = time.perf_counter()
        answer = rag_chain.invoke(args.query)
        gen_ms = (time.perf_counter() - t0) * 1000

        emit_model_invoke(provider="openai", name=args.model, latency_ms=round(gen_ms))
        emit_output(answer)

    print(f"\nAnswer: {answer}")
    print(f"[generation] {gen_ms:.0f}ms")

    # Summary
    summary = stratix.end_trial()
    events = stratix.get_events()
    print(f"\n--- STRATIX Trace Summary ---")
    print(f"Status: {summary.get('status')}  |  Events: {len(events)}")
    for e in events:
        print(f"  {e.get_event_type()}: {str(e.payload)[:80]}")


if __name__ == "__main__":
    main()
