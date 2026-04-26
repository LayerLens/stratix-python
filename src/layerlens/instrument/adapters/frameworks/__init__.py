"""Framework adapters for the LayerLens Instrument layer.

Each framework adapter wraps an agent / chain framework's lifecycle to
intercept agent runs, model invocations, tool calls, state changes, and
handoffs, emitting events through the LayerLens telemetry pipeline.

Adapters available (loaded on demand via :class:`AdapterRegistry`):

* ``autogen`` — Microsoft AutoGen (group chat + lifecycle)

Other framework adapters (LangChain, LangGraph, CrewAI, Agentforce,
Langfuse, Semantic Kernel, OpenAI Agents, etc.) ship in sibling M2
fan-out PRs.

Importing this package does NOT import any framework SDK.
"""

from __future__ import annotations
