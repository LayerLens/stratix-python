"""Canonical live workloads per framework adapter.

Each ``run_*`` instruments a real framework with its Stratix adapter, drives a
minimal real workload (a paid tool/chat call where the framework needs one),
and uninstruments. It must run inside the harness's active ``TraceCollector``
(see ``_framework_harness``). SDK imports are lazy so collection never requires
a framework package to be installed — the test ``importorskip``s instead.

Redaction note: framework adapters gate message/IO content on **their own**
``capture_config`` (via ``_set_if_capturing``), independently of the collector.
So the redaction flow must construct the adapter with ``capture_content=False``
too — exactly as ``layerlens.instrument.auto(client, capture_config=...)`` does.
``_cfg(flow)`` does this; every prompt embeds ``SENTINEL`` for the redaction check.
"""

from __future__ import annotations

import os

from layerlens.instrument._capture_config import CaptureConfig

from ._scenarios import SENTINEL

_OPENAI_MODEL = os.environ.get("LL_OPENAI_MODEL", "gpt-4o-mini")
_PROMPT = f"Name two oceans in a few words. {SENTINEL}"


def _cfg(flow: str) -> CaptureConfig:
    """Capture config matching the variant (redaction => capture_content off)."""
    return CaptureConfig(capture_content=False) if flow == "redaction" else CaptureConfig.standard()


# --------------------------------------------------------------------------- #
# LangChain — callback handler on a ChatOpenAI call
# --------------------------------------------------------------------------- #
def run_langchain(flow: str) -> None:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    from layerlens.instrument.adapters.frameworks.langchain import LangChainCallbackHandler

    # client=None -> emit to the harness's ambient collector. The sample's no-arg
    # call is a bug — __init__ requires the positional client (see report C2).
    handler = LangChainCallbackHandler(None, capture_config=_cfg(flow))
    llm = ChatOpenAI(model=_OPENAI_MODEL, max_tokens=32, callbacks=[handler])
    llm.invoke([HumanMessage(content=_PROMPT)])


# --------------------------------------------------------------------------- #
# LangGraph — a one-node graph whose node makes a real LLM call
# --------------------------------------------------------------------------- #
def run_langgraph(flow: str) -> None:
    from langgraph.graph import END, StateGraph
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    from layerlens.instrument.adapters.frameworks.langgraph import LangGraphCallbackHandler

    handler = LangGraphCallbackHandler(None, capture_config=_cfg(flow))
    llm = ChatOpenAI(model=_OPENAI_MODEL, max_tokens=32)

    def respond(state: dict) -> dict:
        state["reply"] = llm.invoke([HumanMessage(content=state["text"])], config={"callbacks": [handler]}).content
        return state

    builder = StateGraph(dict)
    builder.add_node("respond", respond)
    builder.set_entry_point("respond")
    builder.add_edge("respond", END)
    graph = builder.compile()

    graph.invoke({"text": _PROMPT}, config={"callbacks": [handler]})


# --------------------------------------------------------------------------- #
# PydanticAI — typed agent with a plain tool
# --------------------------------------------------------------------------- #
def run_pydantic_ai(flow: str) -> None:
    from pydantic_ai import Agent

    from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter

    agent = Agent(f"openai:{_OPENAI_MODEL}", system_prompt="Reply in one short sentence.")

    @agent.tool_plain
    def text_length(text: str) -> int:
        return len(text)

    adapter = PydanticAIAdapter(None, capture_config=_cfg(flow))
    adapter.connect(agent)
    try:
        agent.run_sync(_PROMPT)
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# OpenAI Agents SDK — NOTE: this adapter is a global TracingProcessor that
# manages its own per-trace collectors and *flushes* (uploads) them itself, so
# it is driven by ``_framework_harness.run_openai_agents_case`` (not the ambient
# collector path). ``client`` must be the real Stratix client for the flush.
# --------------------------------------------------------------------------- #
def run_openai_agents(flow: str, client: object) -> None:
    from agents import Agent, Runner

    from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter

    agent = Agent(name="live-audit", instructions="Answer in one short sentence.")
    adapter = OpenAIAgentsAdapter(client=client, capture_config=_cfg(flow))
    adapter.connect()
    try:
        Runner.run_sync(agent, _PROMPT)
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# Embedding (cross-cutting) — wrap OpenAI client.embeddings.create
# --------------------------------------------------------------------------- #
def run_embedding(flow: str) -> None:
    import openai
    from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter

    client = openai.OpenAI()
    adapter = EmbeddingAdapter(None, capture_config=_cfg(flow))
    adapter.connect(client)
    try:
        client.embeddings.create(
            model=os.environ.get("LL_EMBED_MODEL", "text-embedding-3-small"),
            input=f"vector-store audit ocean river lake {SENTINEL}",
        )
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# Vector store (cross-cutting) — wrap an in-process Chroma collection query
# --------------------------------------------------------------------------- #
def run_vector_store(flow: str) -> None:  # noqa: ARG001
    import chromadb

    from layerlens.instrument.adapters.frameworks.vector_store import VectorStoreAdapter

    cc = chromadb.EphemeralClient()
    coll = cc.get_or_create_collection("audit")
    # Explicit vectors (no default embedding function) so the workload stays
    # offline + fast (no ONNX model download).
    coll.add(
        ids=["a", "b", "c"],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        documents=["alpha ocean", "beta river", "gamma lake"],
    )
    adapter = VectorStoreAdapter(None)
    adapter.connect(coll)
    try:
        coll.query(query_embeddings=[[0.15, 0.25, 0.35]], n_results=2)
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# CrewAI (Python >=3.10) — single-agent crew with one task
# --------------------------------------------------------------------------- #
def run_crewai(flow: str, client: object) -> None:
    # CrewAIAdapter is self-flushing: it listens on crewai's event bus, builds
    # its own collector, and flushes (uploads via ``client``) on crew completion.
    from crewai import Crew, Task, Agent

    from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter

    researcher = Agent(
        role="researcher",
        goal=f"name an ocean {SENTINEL}",
        backstory="curious and terse",
        allow_delegation=False,
    )
    task = Task(
        description=f"Name one ocean in a few words. {SENTINEL}",
        agent=researcher,
        expected_output="a few words",
    )
    crew = Crew(agents=[researcher], tasks=[task])

    adapter = CrewAIAdapter(client, capture_config=_cfg(flow))
    adapter.connect(crew)
    try:
        crew.kickoff()
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# Semantic Kernel (Python >=3.10) — prompt function invocation
# --------------------------------------------------------------------------- #
def run_semantic_kernel(flow: str) -> None:
    import asyncio

    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    from layerlens.instrument.adapters.frameworks.semantic_kernel import SemanticKernelAdapter

    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id=_OPENAI_MODEL))
    fn = kernel.add_function(
        plugin_name="demo", function_name="greet", prompt=f"Name two oceans in a few words. {SENTINEL}"
    )
    adapter = SemanticKernelAdapter(None, capture_config=_cfg(flow))
    adapter.connect(kernel)
    try:
        asyncio.run(kernel.invoke(fn))
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# LlamaIndex (no extra: `pip install llama-index`) — RAG query over one doc
# --------------------------------------------------------------------------- #
def run_llamaindex(flow: str, client: object) -> None:
    # LlamaIndexAdapter is self-flushing: it registers handlers on llama_index's
    # root dispatcher, builds its own per-span collectors, and flushes them
    # (uploads via ``client``) on disconnect.
    from llama_index.core import Document, VectorStoreIndex

    from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter

    index = VectorStoreIndex.from_documents([Document(text=f"The Pacific and Atlantic are oceans. {SENTINEL}")])
    adapter = LlamaIndexAdapter(client, capture_config=_cfg(flow))
    adapter.connect(index)
    try:
        index.as_query_engine().query(f"Name an ocean. {SENTINEL}")
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# Haystack (no extra: `pip install haystack-ai`) — LLM-free BM25 retrieval
# --------------------------------------------------------------------------- #
def run_haystack(flow: str) -> None:
    from haystack import Document, Pipeline
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

    from layerlens.instrument.adapters.frameworks.haystack import HaystackAdapter

    store = InMemoryDocumentStore()
    store.write_documents([Document(content=f"Oceans: Pacific and Atlantic. {SENTINEL}")])
    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
    adapter = HaystackAdapter(None, capture_config=_cfg(flow))
    adapter.connect(pipeline)
    try:
        pipeline.run({"retriever": {"query": "name an ocean", "top_k": 1}})
    finally:
        adapter.disconnect()
