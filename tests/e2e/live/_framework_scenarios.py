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

Error note: frameworks whose registry entry sets ``supports_error=True`` also
take ``flow == "error"``: the runner swaps in an invalid model id (the same
``_BAD_MODEL`` the provider suite uses), swallows the expected provider
exception, and the harness asserts an ``agent.error`` event landed in the trace.
"""

from __future__ import annotations

import os

from layerlens.instrument._capture_config import CaptureConfig

from ._scenarios import SENTINEL, _BAD_MODEL

_OPENAI_MODEL = os.environ.get("LL_OPENAI_MODEL", "gpt-4o-mini")
_PROMPT = f"Name two oceans in a few words. {SENTINEL}"


def _model_for(flow: str) -> str:
    """The OpenAI model for the flow — the error flow injects an invalid id."""
    return _BAD_MODEL if flow == "error" else _OPENAI_MODEL


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
    llm = ChatOpenAI(model=_model_for(flow), max_tokens=32, callbacks=[handler])
    try:
        llm.invoke([HumanMessage(content=_PROMPT)])
    except Exception:
        # error flow: the provider rejects _BAD_MODEL inside the LLM callback
        # path -> on_llm_error emits agent.error; the exception is expected.
        if flow != "error":
            raise


# --------------------------------------------------------------------------- #
# LangGraph — a one-node graph whose node makes a real LLM call
# --------------------------------------------------------------------------- #
def run_langgraph(flow: str) -> None:
    from langgraph.graph import END, StateGraph
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    from layerlens.instrument.adapters.frameworks.langgraph import LangGraphCallbackHandler

    handler = LangGraphCallbackHandler(None, capture_config=_cfg(flow))
    llm = ChatOpenAI(model=_model_for(flow), max_tokens=32)

    def respond(state: dict) -> dict:
        state["reply"] = llm.invoke([HumanMessage(content=state["text"])], config={"callbacks": [handler]}).content
        return state

    builder = StateGraph(dict)
    builder.add_node("respond", respond)
    builder.set_entry_point("respond")
    builder.add_edge("respond", END)
    graph = builder.compile()

    try:
        graph.invoke({"text": _PROMPT}, config={"callbacks": [handler]})
    except Exception:
        # error flow: the node's LLM call fails on _BAD_MODEL -> on_llm_error
        # (and the graph's on_chain_error) emit agent.error; expected.
        if flow != "error":
            raise


# --------------------------------------------------------------------------- #
# PydanticAI — typed agent with a plain tool
# --------------------------------------------------------------------------- #
def run_pydantic_ai(flow: str) -> None:
    from pydantic_ai import Agent

    from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter

    agent = Agent(f"openai:{_model_for(flow)}", system_prompt="Reply in one short sentence.")

    @agent.tool_plain
    def text_length(text: str) -> int:
        return len(text)

    adapter = PydanticAIAdapter(None, capture_config=_cfg(flow))
    adapter.connect(agent)
    try:
        agent.run_sync(_PROMPT)
    except Exception:
        # error flow: the instrumented model raises on _BAD_MODEL ->
        # _emit_model_error / _finish_run_error emit agent.error; expected.
        if flow != "error":
            raise
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
    kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id=_model_for(flow)))
    fn = kernel.add_function(
        plugin_name="demo", function_name="greet", prompt=f"Name two oceans in a few words. {SENTINEL}"
    )
    adapter = SemanticKernelAdapter(None, capture_config=_cfg(flow))
    adapter.connect(kernel)
    try:
        asyncio.run(kernel.invoke(fn))
    except Exception:
        # error flow: the chat service fails on _BAD_MODEL inside the adapter's
        # invocation filter, which emits agent.error and re-raises; expected.
        if flow != "error":
            raise
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


# --------------------------------------------------------------------------- #
# SmolAgents (no extra: `pip install smolagents openai`) — tool-calling agent.
# Self-flushing: the adapter creates its own collector per run and uploads via
# the client it was constructed with.
# --------------------------------------------------------------------------- #
def run_smolagents(flow: str, client: object) -> None:
    from smolagents import ToolCallingAgent, OpenAIServerModel, tool

    from layerlens.instrument.adapters.frameworks.smolagents import SmolAgentsAdapter

    @tool
    def ocean_count() -> int:
        """Return the number of oceans on Earth."""
        return 5

    model = OpenAIServerModel(model_id=_OPENAI_MODEL)
    agent = ToolCallingAgent(tools=[ocean_count], model=model, max_steps=3)
    adapter = SmolAgentsAdapter(client, capture_config=_cfg(flow))
    adapter.connect(target=agent)
    try:
        agent.run(f"Use the ocean_count tool, then answer in one short sentence. {SENTINEL}")
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# Agno (no extra: `pip install agno openai`) — wrapped agent.run()
# --------------------------------------------------------------------------- #
def run_agno(flow: str) -> None:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

    agent = Agent(model=OpenAIChat(id=_OPENAI_MODEL), markdown=False)
    adapter = AgnoAdapter(None, capture_config=_cfg(flow))
    adapter.connect(target=agent)
    try:
        agent.run(_PROMPT)
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# AWS Strands Agents (no extra: `pip install 'strands-agents[openai]'`) —
# adapter is a native HookProvider. Self-flushing (own collector per run).
# --------------------------------------------------------------------------- #
def run_strands(flow: str, client: object) -> None:
    from strands import Agent
    from strands.models.openai import OpenAIModel

    from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter

    adapter = StrandsAdapter(client, capture_config=_cfg(flow))
    adapter.connect()
    model = OpenAIModel(model_id=_OPENAI_MODEL, params={"max_tokens": 64})
    agent = Agent(model=model, hooks=[adapter], callback_handler=None)
    try:
        agent(_PROMPT)
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# Google ADK (no extra: `pip install google-adk`) — plugin on the Runner,
# Gemini via API key (maps GEMINI_API_KEY -> GOOGLE_API_KEY if needed).
# Self-flushing (own collector per run).
# --------------------------------------------------------------------------- #
def run_google_adk(flow: str, client: object) -> None:
    import asyncio

    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    from google.genai import types
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter

    agent = Agent(
        name="live_audit",
        model=os.environ.get("LL_GEMINI_MODEL", "gemini-2.5-flash"),
        instruction="Answer in one short sentence.",
    )
    adapter = GoogleADKAdapter(client, capture_config=_cfg(flow))
    adapter.connect()
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="layerlens-live",
        agent=agent,
        session_service=session_service,
        plugins=[adapter.plugin],
    )

    async def _run() -> None:
        session = await session_service.create_session(app_name="layerlens-live", user_id="auditor")
        message = types.Content(role="user", parts=[types.Part(text=_PROMPT)])
        async for _event in runner.run_async(user_id="auditor", session_id=session.id, new_message=message):
            pass

    try:
        asyncio.run(_run())
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# AutoGen (autogen-agentchat >= 0.4, the package the `autogen` extra installs).
# The adapter taps autogen_core's event logger and manages its own collector;
# it flushes on disconnect — self-flushing.
# --------------------------------------------------------------------------- #
def run_autogen(flow: str, client: object) -> None:
    import asyncio

    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    from layerlens.instrument.adapters.frameworks.autogen import AutoGenAdapter

    adapter = AutoGenAdapter(client, capture_config=_cfg(flow))
    adapter.connect()
    try:
        model_client = OpenAIChatCompletionClient(model=_OPENAI_MODEL)
        assistant = AssistantAgent("assistant", model_client=model_client)
        asyncio.run(assistant.run(task=_PROMPT))
    finally:
        adapter.disconnect()  # flushes the adapter-managed collector


# --------------------------------------------------------------------------- #
# MS Agent Framework (semantic-kernel agents surface — AgentGroupChat).
# NOTE: the adapter instruments semantic_kernel.agents chats, NOT Microsoft's
# separate `agent-framework` package (registry shares the SK detection key).
# --------------------------------------------------------------------------- #
def run_ms_agent_framework(flow: str) -> None:
    import asyncio

    from semantic_kernel import Kernel
    from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    from layerlens.instrument.adapters.frameworks.ms_agent_framework import MSAgentFrameworkAdapter

    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(ai_model_id=_OPENAI_MODEL, service_id="chat"))
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="auditor",
        instructions="Answer in one short sentence.",
    )
    chat = AgentGroupChat(agents=[agent])

    adapter = MSAgentFrameworkAdapter(None, capture_config=_cfg(flow))
    adapter.connect()
    adapter.instrument_chat(chat)

    async def _run() -> None:
        await chat.add_chat_message(message=_PROMPT)
        async for _message in chat.invoke():
            pass

    try:
        asyncio.run(_run())
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# Langfuse — bidirectional batch sync, exercised as an export -> import
# round-trip against a real Langfuse instance (LLM-free). The adapter is
# self-flushing on import: each imported Langfuse trace gets its own collector,
# uploaded via ``client``. Deep mapping assertions live in the unit suite;
# this verifies the real REST API contract end-to-end.
# --------------------------------------------------------------------------- #
def run_langfuse(flow: str, client: object) -> None:
    import time
    from datetime import datetime, timezone

    from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter

    adapter = LangfuseAdapter(client, capture_config=_cfg(flow))
    adapter.connect(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL"),
    )
    try:
        since = datetime.now(timezone.utc).isoformat()

        # Export a small synthetic LayerLens trace into Langfuse
        events = [
            {
                "event_type": "agent.input",
                "span_id": "span-root",
                "span_name": "sdk-audit-langfuse",
                "payload": {"name": "sdk-audit-langfuse", "content": _PROMPT},
            },
            {
                "event_type": "model.invoke",
                "span_id": "span-llm",
                "span_name": "llm",
                "payload": {
                    "model": "gpt-4o-mini",
                    "messages": _PROMPT,
                    "output_message": "Pacific and Atlantic.",
                    "tokens_prompt": 12,
                    "tokens_completion": 6,
                    "tokens_total": 18,
                },
            },
            {
                "event_type": "tool.call",
                "span_id": "span-tool",
                "span_name": "lookup",
                "payload": {"tool_name": "lookup", "input": "oceans", "output": "5 oceans"},
            },
            {
                "event_type": "agent.output",
                "span_id": "span-root",
                "payload": {"content": "Pacific and Atlantic."},
            },
        ]
        exported = adapter.export_traces(events_by_trace={"ll-audit-roundtrip": events})
        assert exported == 1, f"langfuse export failed (exported={exported})"

        # Langfuse ingestion is async and item-by-item — poll until the
        # exported trace AND its observations (generation + span) are visible,
        # so the import below round-trips the full batch.
        deadline = time.time() + 60
        ready = False
        while time.time() < deadline:
            resp = adapter._http.get("/api/public/traces", params={"fromTimestamp": since, "limit": 10})
            data = resp.json().get("data", []) if resp.status_code == 200 else []
            if data:
                detail = adapter._http.get(f"/api/public/traces/{data[0]['id']}").json()
                if len(detail.get("observations", [])) >= 2:
                    ready = True
                    break
            time.sleep(2)
        assert ready, "exported trace (with observations) never became visible in Langfuse"

        imported = adapter.import_traces(since=since, limit=10)
        assert imported >= 1, f"langfuse import returned {imported}"
    finally:
        adapter.disconnect()


def run_agentforce(flow: str, client: object) -> None:
    import pytest

    from layerlens.instrument.adapters.frameworks.agentforce import AgentforceAdapter

    adapter = AgentforceAdapter(client, capture_config=_cfg(flow))
    adapter.connect(
        credentials={
            "client_id": os.environ["SF_CLIENT_ID"],
            "client_secret": os.environ["SF_CLIENT_SECRET"],
            "instance_url": os.environ["SF_INSTANCE_URL"],
        }
    )
    try:
        # Agentforce is a read-only importer of the Salesforce Session Tracing
        # Data Model — it cannot create sessions, so this check depends on the
        # org having ingested conversation data. The STDM DMOs also ingest on
        # staggered streams, so skip cleanly when nothing is present yet.
        summary = adapter.import_sessions(limit=10)
        if summary["sessions_imported"] == 0:
            pytest.skip("no Agentforce sessions ingested in the org to import")
        assert summary["errors"] == 0, f"agentforce import reported errors: {summary}"
        assert summary["events_emitted"] > 0, f"agentforce produced no events: {summary}"
    finally:
        adapter.disconnect()
