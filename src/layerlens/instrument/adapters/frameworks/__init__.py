"""Framework adapters for the LayerLens Instrument layer.

Each framework adapter wraps an agent / chain framework's lifecycle to
intercept agent runs, model invocations, tool calls, state changes, and
handoffs, emitting events through the LayerLens telemetry pipeline.

Adapters available (loaded on demand via :class:`AdapterRegistry` —
importing this package does NOT import any framework SDK):

* ``langchain`` — LangChain (callbacks + agent + chain + memory)
* ``langgraph`` — LangGraph (graph hooks + handoff detection + state)
* ``crewai`` — CrewAI (delegation + team metadata)
* ``autogen`` — AutoGen (group chat + lifecycle)
* ``agentforce`` — Salesforce Agentforce (auth, client, event mapping)
* ``semantic_kernel`` — Microsoft Semantic Kernel (filters + lifecycle)
* ``langfuse_importer`` — Langfuse trace import / export
* ``embedding`` — Embedding + vector store instrumentation
* ``openai_agents`` — OpenAI Agents SDK lifecycle
* ``ms_agent_framework`` — MS Agent Framework lifecycle
* ``agno`` — Agno lifecycle
* ``bedrock_agents`` — AWS Bedrock Agents lifecycle
* ``llama_index`` — LlamaIndex lifecycle
* ``google_adk`` — Google ADK lifecycle
* ``strands`` — Strands lifecycle
* ``benchmark_import`` — Benchmark replay-based ingestion
* ``pydantic_ai`` — Pydantic-AI lifecycle
* ``smolagents`` — SmolAgents (HuggingFace) lifecycle
* ``browser_use`` — Browser-Use lifecycle (placeholder; ported in M7)

Usage::

    # Lazy import — does not pull in framework dependencies until used.
    from layerlens.instrument.adapters.frameworks.agentforce import (
        AgentForceAdapter,
        SalesforceCredentials,
    )

Each per-framework subpackage handles its own optional dependency surface,
so missing SDKs do not break ``import layerlens.instrument.adapters.frameworks``.
"""

from __future__ import annotations
