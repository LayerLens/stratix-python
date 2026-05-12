"""Framework adapters for the LayerLens Instrument layer.

Each framework adapter wraps an agent / chain framework's lifecycle to
intercept agent runs, model invocations, tool calls, state changes, and
handoffs, emitting events through the LayerLens telemetry pipeline.

Adapters available (loaded on demand via :class:`AdapterRegistry`):

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

Importing this package does NOT import any framework SDK.
"""

from __future__ import annotations
