"""Framework adapters for the LayerLens Instrument layer.

Each framework adapter wraps an agent / chain framework's lifecycle to
intercept agent runs, model invocations, tool calls, state changes, and
handoffs, emitting events through the LayerLens telemetry pipeline.

Adapters available (loaded on demand via :class:`AdapterRegistry`):

* ``crewai`` — CrewAI (delegation + team metadata)

Sibling framework adapters (LangChain, LangGraph, AutoGen, Agentforce,
Semantic Kernel, etc.) ship in their own M2 fan-out PRs and live under
this same package.

Importing this package does NOT import any framework SDK.
"""

from __future__ import annotations
