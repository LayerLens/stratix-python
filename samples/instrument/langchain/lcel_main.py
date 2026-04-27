"""Sample: LCEL (LangChain Expression Language) instrumentation walkthrough.

Demonstrates the LCEL tracing capability added to the LayerLens
LangChain adapter (spec 04b §4). The sample builds a canonical RAG-
style LCEL pipeline:

    {"context": retriever, "question": passthrough} | prompt | llm | parser

then invokes it with the LayerLens callback handler installed and
prints the events that the adapter emitted — including the synthetic
``chain.composition`` snapshot produced at root completion.

The sample runs **offline**. The "LLM" and "retriever" are local
``RunnableLambda`` stand-ins that return deterministic strings; no
network access or API key is required. To wire it into a real model,
swap the ``fake_llm`` / ``fake_retriever`` / ``fake_parser`` lambdas
for ``ChatOpenAI``, ``VectorStoreRetriever``, and ``StrOutputParser``
respectively.

Run::

    pip install 'layerlens[langchain]'
    python -m samples.instrument.langchain.lcel_main
"""

from __future__ import annotations

import sys
import json
from typing import Any

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.frameworks.langchain import LayerLensCallbackHandler


def main() -> int:
    try:
        from langchain_core.runnables import (
            RunnableLambda,
            RunnableBranch,
            RunnableParallel,
            RunnablePassthrough,
        )
    except ImportError:
        print(
            "langchain-core is not installed. Install with:\n"
            "    pip install 'layerlens[langchain]'",
            file=sys.stderr,
        )
        return 2

    # ------------------------------------------------------------------
    # Build a representative LCEL pipeline that exercises every
    # Runnable primitive the adapter knows about. The shape mirrors the
    # canonical RAG pattern from langchain.com/docs/concepts/lcel.
    # ------------------------------------------------------------------

    def fake_retriever(question: str) -> str:
        # Deterministic stand-in for a vector store retriever.
        return f"Context: facts relevant to '{question}'."

    def fake_llm(prompt_input: dict[str, str]) -> str:
        return f"Answer using {prompt_input['context']!r} for: {prompt_input['question']!r}"

    def fake_parser(response: str) -> str:
        return response.strip()

    def is_short_question(q: str) -> bool:
        return len(q) < 20

    short_branch = RunnableLambda(lambda q: f"Short answer for: {q}")
    long_branch = RunnableLambda(lambda q: f"Detailed answer for: {q}")

    pipeline = (
        RunnableParallel(
            context=RunnableLambda(fake_retriever),
            question=RunnablePassthrough(),
        )
        | RunnableLambda(fake_llm)
        | RunnableLambda(fake_parser)
        | RunnableBranch(
            (is_short_question, short_branch),
            long_branch,
        )
    )

    # ------------------------------------------------------------------
    # Instrument with LayerLens. ``CaptureConfig.full()`` enables L1
    # (agent.input/output) AND L2 (agent.code + chain.composition) so
    # the printout below shows the entire LCEL signal surface.
    # ------------------------------------------------------------------

    handler = LayerLensCallbackHandler(capture_config=CaptureConfig.full())
    handler.connect()

    try:
        result = pipeline.invoke("What is LCEL?", config={"callbacks": [handler]})
    finally:
        events = list(handler.get_events())
        handler.disconnect()

    print(f"Pipeline output: {result}")
    print(f"Total events captured: {len(events)}")
    print()

    # ------------------------------------------------------------------
    # Walk through the events to highlight what the adapter saw. We
    # print one line per LCEL event type so the output stays readable.
    # ------------------------------------------------------------------

    runnable_inputs = [
        e
        for e in events
        if e["type"] == "agent.input" and "runnable" in (e.get("payload") or {})
    ]
    runnable_codes = [
        e
        for e in events
        if e["type"] == "agent.code" and (e.get("payload") or {}).get("kind") != "chain.composition"
    ]
    composition_events = [
        e
        for e in events
        if e["type"] == "agent.code" and (e.get("payload") or {}).get("kind") == "chain.composition"
    ]

    print(f"== LCEL agent.input events ({len(runnable_inputs)}) ==")
    for e in runnable_inputs:
        runnable = e["payload"]["runnable"]
        position = runnable.get("position")
        loc = (
            f" [{position['parent_kind']}.{position['role']}={position['label']}]"
            if position
            else ""
        )
        depth_indent = "  " * runnable["depth"]
        print(f"{depth_indent}- {runnable['kind']}: {runnable['name']}{loc}")

    print()
    print(f"== LCEL agent.code events ({len(runnable_codes)}) ==")
    for e in runnable_codes:
        payload = e["payload"]
        depth_indent = "  " * payload["depth"]
        marker = ""
        if payload.get("passthrough"):
            marker = "  (passthrough)"
        elif payload.get("fingerprint"):
            marker = f"  fp={payload['fingerprint']}"
        duration = payload.get("duration_ns")
        dur_ms = f" {duration / 1e6:.2f}ms" if duration is not None else ""
        print(f"{depth_indent}- {payload['kind']}: {payload['name']}{marker}{dur_ms}")

    print()
    print(f"== chain.composition snapshot ({len(composition_events)}) ==")
    for e in composition_events:
        comp = e["payload"]["composition"]
        print(
            f"  root={comp['root_kind']} ({comp['root_name']!r})  "
            f"nodes={comp['node_count']}  max_depth={comp['max_depth']}  "
            f"status={comp['status']}"
        )
        print(f"  kind_counts: {comp['kind_counts']}")
        # First few nodes for visibility.
        print("  nodes (first 6):")
        for node in comp["nodes"][:6]:
            label = ""
            if node.get("position"):
                label = (
                    f" [{node['position']['parent_kind']}."
                    f"{node['position']['role']}={node['position']['label']}]"
                )
            indent = "    " + "  " * node["depth"]
            print(f"{indent}- depth={node['depth']} {node['kind']}: {node['name']}{label}")

    print()
    print("Sample complete. Verify the events match the executed pipeline:")
    print("  RunnableSequence -> RunnableParallel(context, question) -> RunnableLambda")
    print("                  -> RunnableLambda -> RunnableBranch -> (short OR long)")
    return 0


def _serialize(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except (TypeError, ValueError):
        return str(obj)


if __name__ == "__main__":
    raise SystemExit(main())
