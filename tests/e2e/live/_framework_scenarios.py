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
import glob
from typing import Optional

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
        # Assert the live wire actually labelled this as Chroma. connect() routes
        # via _auto_wrap, which used to mis-route a real Chroma Collection to the
        # Pinecone wrapper -> provider='pinecone' / result_count=0, undetected
        # (LAY-3616). Pin the corrected routing on the real backend.
        from layerlens.instrument._context import _current_collector

        collector = _current_collector.get()
        retrieval = [e for e in (collector.events if collector else []) if e["event_type"] == "retrieval.query"]
        assert retrieval, "vector_store emitted no retrieval.query event"
        payload = retrieval[-1]["payload"]
        assert payload["provider"] == "chroma", f"expected provider='chroma', got {payload.get('provider')!r}"
        assert payload.get("result_count", 0) > 0, f"expected result_count>0, got {payload.get('result_count')!r}"
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


# --------------------------------------------------------------------------- #
# AWS Bedrock Agents — boto3 ``invoke_agent`` against a provisioned Agent.
# The adapter hooks the bedrock-agent-runtime client and observes the
# ``completion`` EventStream as the customer drains it; it is self-flushing
# (one collector per invoke, flushed when the stream is drained). The live
# Agent (Nova, no action groups / KBs) exercises the model.invoke + output +
# proxy-transparency paths; tool/KB/handoff are covered by the unit doubles.
# --------------------------------------------------------------------------- #
def run_bedrock_agents(flow: str, client: object) -> None:
    import uuid

    import boto3

    from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter

    rt = boto3.client("bedrock-agent-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    adapter = BedrockAgentsAdapter(client, capture_config=_cfg(flow))
    adapter.connect(target=rt)
    try:
        response = rt.invoke_agent(
            agentId=os.environ["BEDROCK_AGENT_ID"],
            agentAliasId=os.environ["BEDROCK_AGENT_ALIAS_ID"],
            sessionId="ll-audit-" + uuid.uuid4().hex[:12],
            inputText=f"What is 2+2? Reply with only the number. {SENTINEL}",
            enableTrace=True,
        )
        # Drain the completion stream exactly as a customer would — this is what
        # drives the adapter's per-trace emission and the final flush.
        events = list(response["completion"])
        assert events, "bedrock_agents completion stream was empty"
    finally:
        adapter.disconnect()


def run_bedrock_agents_features(client: object) -> dict:
    """Live verification of the LAY-3606 trace-completeness members against a
    FEATURE-CONFIGURED Bedrock agent (guardrail + RETURN_CONTROL action group +
    code interpreter + user input).

    Opt-in / not part of the default live run: the default ``BEDROCK_AGENT_ALIAS_ID``
    points at a vanilla version, so set ``BEDROCK_FEATURES_ALIAS_ID`` to an alias whose
    version/DRAFT has the features (and grant the agent role ``bedrock:ApplyGuardrail``).
    Drives one trigger prompt per member through the REAL adapter (``CaptureConfig.full()``
    so ``agent.code`` is enabled) into the REAL client, and returns which canonical
    members were emitted + the uploaded trace ids.
    """
    import time as _time
    import uuid

    import boto3

    import layerlens.instrument._upload as _upload
    from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter

    rt = boto3.client("bedrock-agent-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    agent_id = os.environ["BEDROCK_AGENT_ID"]
    alias = os.environ.get("BEDROCK_FEATURES_ALIAS_ID", os.environ["BEDROCK_AGENT_ALIAS_ID"])
    prompts = (
        "Should I invest in gold or bonds to maximize my returns?",  # guardrail -> policy.violation
        "Use the getWeather tool to look up the current weather in Tokyo, Japan.",  # returnControl -> tool.call
        "Using Python, compute the first 20 Fibonacci numbers and save them to a CSV named fib.csv and return it.",  # code+files -> agent.code
        "Book me a flight.",  # ASK_USER -> agent.step
    )

    seen_types = set()
    trace_ids = []
    traces_res = client.traces  # type: ignore[attr-defined]
    orig_upload = traces_res.upload
    captured = []

    def _wrap(path, **kw):  # type: ignore[no-untyped-def]
        result = orig_upload(path, **kw)
        if result is not None and getattr(result, "trace_ids", None):
            captured.extend(result.trace_ids)
        return result

    traces_res.upload = _wrap  # type: ignore[method-assign]
    try:
        for prompt in prompts:
            captured.clear()
            adapter = BedrockAgentsAdapter(client, capture_config=CaptureConfig.full())
            adapter.connect(target=rt)
            try:
                resp = rt.invoke_agent(
                    agentId=agent_id,
                    agentAliasId=alias,
                    sessionId="ll-feat-" + uuid.uuid4().hex[:12],
                    inputText=f"{prompt} {SENTINEL}",
                    enableTrace=True,
                )
                list(resp["completion"])
            finally:
                adapter.disconnect()
            deadline = _time.time() + 30
            while not captured and _time.time() < deadline:
                _time.sleep(1.0)
                _upload.shutdown_uploads(timeout=10)
            if captured:
                tid = captured[0]
                trace_ids.append(tid)
                trace = client.traces.get(tid)  # type: ignore[attr-defined]
                evs = trace.data.get("events", []) if trace is not None and isinstance(trace.data, dict) else []
                for ev in evs:
                    seen_types.add(ev["event_type"])
    finally:
        traces_res.upload = orig_upload  # type: ignore[method-assign]

    return {"seen_types": sorted(seen_types), "trace_ids": trace_ids}


# =========================================================================== #
# ADP-PORT lanes — the seven adapters ported from the ateam reference SDK.
#
# Model choice per lane is decided by what the workload HONESTLY needs, not by
# what is cheapest: the local ollama ``llama3:8b`` does not support tool-calling
# (the server returns a real 400 "does not support tools"), so any lane whose
# framework drives the model through a forced-tool / structured-output schema
# must use a real paid provider. The lanes that only need a plain completion
# (dspy, mirascope-in-json-mode, openinference's ingestion workload) run FREE
# against the local model.
# =========================================================================== #

_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

#: litellm (which dspy calls through) addresses a local ollama via its
#: ``ollama_chat/`` provider prefix; the adapter's ``_split_model_id`` splits it
#: back into model + provider.
_DSPY_MODEL_ID = "ollama_chat/%s" % _OLLAMA_MODEL

#: mirascope v2 addresses the local ollama through its own ``ollama/`` scope.
_MIRASCOPE_MODEL_ID = "ollama/%s" % _OLLAMA_MODEL


# --------------------------------------------------------------------------- #
# DSPy (`pip install dspy`) — a real dspy.Module (ChainOfThought) over the FREE
# local ollama. dspy needs no tool-calling for a plain predict, so no paid
# provider is used. Ambient: the adapter's callback bus emits into the harness's
# collector (DSPyAdapter reuses an ambient collector via _begin_run).
# --------------------------------------------------------------------------- #
def run_dspy(flow: str) -> None:
    import dspy

    from layerlens.instrument.adapters.frameworks.dspy import DSPyAdapter

    class OceanAnswer(dspy.Signature):
        """Answer a short factual question about Earth's oceans in one sentence."""

        question: str = dspy.InputField(desc="The question to answer.")
        answer: str = dspy.OutputField(desc="One short sentence.")

    lm = dspy.LM(
        _DSPY_MODEL_ID,
        api_base=_OLLAMA_HOST,
        api_key="",
        # No cache: a cached hit would emit no real model.invoke, so the lane
        # would prove nothing about the live path.
        cache=False,
        max_tokens=200,
    )
    dspy.configure(lm=lm)
    program = dspy.ChainOfThought(OceanAnswer)

    adapter = DSPyAdapter(None, capture_config=_cfg(flow))
    adapter.connect()
    try:
        program(question=_PROMPT)
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# Instructor (`pip install instructor openai`) — a real instructor-patched
# client extracting into a real Pydantic response_model. Instructor's default
# Mode.TOOLS drives the provider's function-calling schema, which the local
# llama3:8b genuinely cannot do -> this lane needs OPENAI_API_KEY.
# --------------------------------------------------------------------------- #
def run_instructor(flow: str) -> None:
    import instructor
    from pydantic import Field, BaseModel

    import openai
    from layerlens.instrument.adapters.frameworks.instructor import InstructorAdapter

    class OceanFacts(BaseModel):
        """A structured record of an ocean fact."""

        ocean_name: str = Field(description="The name of one of Earth's oceans.")
        is_largest: bool = Field(description="True if this is the largest ocean on Earth.")

    patched = instructor.from_openai(openai.OpenAI())
    adapter = InstructorAdapter(None, capture_config=_cfg(flow))
    # agent_name is the ONLY honest identity source for instructor (it declares
    # none of its own), so the caller declares it — same seam a customer uses.
    adapter.connect(target=patched, agent_name="ocean-facts-extractor")
    try:
        patched.chat.completions.create(
            model=_OPENAI_MODEL,
            response_model=OceanFacts,
            max_retries=2,
            temperature=0,
            max_tokens=64,
            messages=[{"role": "user", "content": f"Name the largest ocean on Earth. {SENTINEL}"}],
        )
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# Marvin (`pip install marvin`) — a real marvin primitive through a real, NAMED
# marvin.Agent. Marvin 3.x rides pydantic-ai, whose structured output is an
# output TOOL — llama3:8b cannot hold it, so this lane needs OPENAI_API_KEY.
# --------------------------------------------------------------------------- #
def run_marvin(flow: str) -> None:
    # ``import marvin`` calls ensure_db_tables_exist() at MODULE scope, creating
    # a SQLite database as an import side effect. Point it at a throwaway file
    # BEFORE the import so a live run never touches a real Marvin database.
    if "MARVIN_DATABASE_URL" not in os.environ:
        import tempfile

        os.environ["MARVIN_DATABASE_URL"] = "sqlite+aiosqlite:///" + os.path.join(
            tempfile.mkdtemp(prefix="layerlens-marvin-live-"), "marvin.db"
        )

    import marvin
    from pydantic_ai.models.openai import OpenAIChatModel

    from layerlens.instrument.adapters.frameworks.marvin import MarvinAdapter

    # Marvin's rich console handler renders a live panel per call — pure noise
    # under pytest.
    marvin.settings.enable_default_print_handler = False

    # ``model=`` on the Agent is the developer's explicit, real configuration —
    # the adapter reads it off ``Agent.model`` (no Marvin 3.x primitive takes a
    # ``model=`` kwarg, so this is the only honest per-call seam).
    agent = marvin.Agent(
        name="ocean-classifier",
        model=OpenAIChatModel(_OPENAI_MODEL),
        instructions="You answer questions about Earth's oceans. Be terse and factual.",
    )

    adapter = MarvinAdapter(None, capture_config=_cfg(flow))
    adapter.connect(target=marvin)
    try:
        marvin.cast(
            f"The largest ocean on Earth. {SENTINEL}",
            target=str,
            instructions="Return only the ocean's name.",
            agent=agent,
        )
    finally:
        adapter.disconnect()


# --------------------------------------------------------------------------- #
# Mirascope (`pip install mirascope`) — a real ``mirascope.llm.call`` on the v2
# API (``mirascope.core`` does NOT exist on v2). A plain call needs no tools, so
# this lane runs FREE against the local ollama through mirascope's own
# OllamaProvider.
# --------------------------------------------------------------------------- #
def run_mirascope(flow: str) -> None:
    import mirascope.llm as llm
    from mirascope.llm.providers.ollama import OllamaProvider
    from mirascope.llm.providers.provider_registry import (
        PROVIDER_REGISTRY,
        provider_singleton,
        reset_provider_registry,
    )

    from openai import OpenAI
    from layerlens.instrument.adapters.frameworks.mirascope import MirascopeAdapter

    # A REAL mirascope OllamaProvider pointed at the real local server. The
    # OpenAI-compatible client is how mirascope's ollama provider really talks.
    provider = OllamaProvider()
    provider.client = OpenAI(api_key="ollama", base_url=_OLLAMA_HOST.rstrip("/") + "/v1/")
    saved = dict(PROVIDER_REGISTRY)
    llm.register_provider(provider, scope="ollama/")

    adapter = MirascopeAdapter(None, capture_config=_cfg(flow))
    adapter.connect()
    try:

        @llm.call(_MIRASCOPE_MODEL_ID)
        def ocean_agent(question: str) -> str:
            return f"Answer in one short sentence. {question}"

        ocean_agent(_PROMPT)
    finally:
        adapter.disconnect()
        reset_provider_registry()
        PROVIDER_REGISTRY.update(saved)
        provider_singleton.cache_clear()


# --------------------------------------------------------------------------- #
# BrowserUse (`pip install browser-use`) — a REAL headless-Chromium browse.
#
# The page is served over REAL HTTP on loopback for the duration of the lane
# rather than fetched from a public site: a live lane that depends on a third
# party's DOM fails for reasons that have nothing to do with the adapter. The
# browse itself is entirely real — real Chromium over CDP, real navigation,
# real DOM serialization, real model-chosen actions, real token usage.
#
# browser-use drives the browser by making the model emit a STRICT JSON action
# schema per step against a large serialized-DOM prompt; llama3:8b does not hold
# that schema (it flails until it runs out of steps), so this lane needs
# OPENAI_API_KEY. It is kept to a couple of steps — ``initial_actions``
# navigates deterministically instead of paying the model to find the URL.
# --------------------------------------------------------------------------- #
_BOARD_HTML = """<!doctype html>
<html><head><title>Ocean Facts Board</title></head>
<body>
  <h1>Ocean Facts Board</h1>
  <table>
    <tr><th>Ocean</th><th>Area (million km2)</th></tr>
    <tr><td>Pacific</td><td>165.2</td></tr>
    <tr><td>Atlantic</td><td>85.1</td></tr>
    <tr><td>Indian</td><td>70.6</td></tr>
  </table>
</body></html>
"""

#: Playwright's cached chromium builds name their binary differently across
#: revisions/platforms ("Google Chrome for Testing" on the mac-arm64 full build,
#: ``chrome-headless-shell`` on the shell build, plain ``chrome`` on linux), so
#: resolve rather than hardcode one machine's layout. BROWSER_USE_SAMPLE_CHROME
#: overrides everything; the lane skips honestly if nothing resolves.
_CHROME_GLOBS = (
    "~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/*",
    "~/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-mac*/chrome-headless-shell",
    "~/.cache/ms-playwright/chromium-*/chrome-linux/chrome",
    "~/.cache/ms-playwright/chromium_headless_shell-*/chrome-linux/chrome-headless-shell",
)


def _resolve_chrome() -> Optional[str]:
    """The first executable chromium in the playwright cache, or None."""
    explicit = os.environ.get("BROWSER_USE_SAMPLE_CHROME")
    if explicit:
        return explicit if os.access(explicit, os.X_OK) else None
    for pattern in _CHROME_GLOBS:
        for path in sorted(glob.glob(os.path.expanduser(pattern)), reverse=True):
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    return None


def _serve_board():
    """Serve the facts board over REAL HTTP on loopback, for this lane only."""
    import tempfile
    import functools
    import threading
    import http.server
    import socketserver

    directory = tempfile.mkdtemp(prefix="ll-live-board-")
    with open(os.path.join(directory, "index.html"), "w") as f:
        f.write(_BOARD_HTML)

    class _Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
            pass

    handler = functools.partial(_Quiet, directory=directory)
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, "http://127.0.0.1:%d/index.html" % server.server_address[1]


def run_browser_use(flow: str) -> None:
    import asyncio

    from browser_use import Agent
    from browser_use.llm import ChatOpenAI
    from browser_use.browser.profile import BrowserProfile

    from layerlens.instrument.adapters.frameworks.browser_use import BrowserUseAdapter

    chrome = _resolve_chrome()
    if chrome is None:
        raise RuntimeError(
            "browser_use live lane: no executable chromium found in the playwright cache "
            "(set BROWSER_USE_SAMPLE_CHROME, or run `playwright install chromium`). "
            "Refusing to fake a browse."
        )

    server, url = _serve_board()
    agent = Agent(
        task=(
            "Read the Ocean Facts Board that is already open in the browser and report "
            "which ocean has the largest area, with its area. %s" % SENTINEL
        ),
        llm=ChatOpenAI(model=_OPENAI_MODEL),
        browser_profile=BrowserProfile(headless=True, executable_path=chrome),
        initial_actions=[{"navigate": {"url": url, "new_tab": False}}],
        use_vision=False,
        enable_planning=False,
        use_judge=False,
        max_actions_per_step=2,
    )
    adapter = BrowserUseAdapter(None, capture_config=_cfg(flow))
    # A real browser_use Agent declares no name of its own, so the honest agent
    # identity is the one the caller declares — the same seam a customer uses
    # via ``instrument_browser_use(agent, agent_name=...)``.
    adapter.connect(target=agent, agent_name="ocean-board-reader")

    async def _drive():
        try:
            return await agent.run(3)
        finally:
            await agent.close()

    try:
        asyncio.run(_drive())
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- #
# OpenInference (`pip install openinference-instrumentation-openai
# opentelemetry-sdk`) — INGESTION. The adapter makes no LLM call of its own; it
# is an OTel SpanProcessor that normalizes OpenInference spans into layerlens
# events. So the honest live lane is a REAL OTel workload: the REAL
# OpenInference auto-instrumentor for the openai SDK, on a REAL TracerProvider,
# producing REAL OpenInference spans off a REAL chat completion — which the
# adapter ingests, seals into a trace, and uploads.
#
# The completion is a plain (tool-free) call, so it is served credlessly by the
# local ollama through its OpenAI-COMPATIBLE endpoint. The provider really is
# ollama and the trace says so; only the wire protocol is OpenAI's, which is
# exactly what makes ``openinference-instrumentation-openai`` the real, correct
# instrumentor for this call.
#
# Self-flushing: the adapter owns one TraceCollector per SOURCE OTel trace id
# and uploads them itself on flush()/disconnect().
# --------------------------------------------------------------------------- #
def run_openinference(flow: str, client: object) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from openinference.instrumentation.openai import OpenAIInstrumentor

    from openai import OpenAI
    from layerlens.instrument.adapters.frameworks.openinference import OpenInferenceAdapter

    provider = TracerProvider()
    instrumentor = OpenAIInstrumentor()
    adapter = OpenInferenceAdapter(client, capture_config=_cfg(flow))
    adapter.connect()
    # The adapter's LIVE production wiring: it is an OTel SpanProcessor.
    provider.add_span_processor(adapter.span_processor())
    try:
        instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
        openai_client = OpenAI(api_key="ollama", base_url=_OLLAMA_HOST.rstrip("/") + "/v1")
        openai_client.chat.completions.create(
            model=_OLLAMA_MODEL,
            messages=[{"role": "user", "content": _PROMPT}],
            max_tokens=64,
        )
    finally:
        try:
            instrumentor.uninstrument()
        except Exception:
            pass
        # Seals + flushes (and uploads) every open collector.
        adapter.disconnect()
