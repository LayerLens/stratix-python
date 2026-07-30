"""Per-framework registry: import package, workload runner, expected-event contract.

Mirrors ``_registry.PROVIDERS`` but for framework adapters. Contracts are loose
by design: frameworks emit varied event shapes, so we assert a minimum event
count + a small set of event types that must be present, and record the rest.
"""

from __future__ import annotations

import os
from typing import Tuple, Callable, Optional
from dataclasses import dataclass

from . import _framework_scenarios as fs


@dataclass(frozen=True)
class FrameworkCase:
    id: str
    import_name: str  # module passed to pytest.importorskip
    runner: Callable[[str], None]
    required_env: Tuple[str, ...] = ()  # all must be present
    any_of_env: Tuple[str, ...] = ()  # at least one must be present
    min_events: int = 1
    expected_types: Tuple[str, ...] = ()  # every type here must appear in the trace
    supports_redaction: bool = True  # also run a capture_content=False variant
    # Also run an "error" variant: the scenario injects a failing model id, swallows
    # the expected exception, and the harness asserts an agent.error event landed.
    # Only set where the adapter emits agent.error AND the scenario helper can
    # inject the bad model in one line (langchain-family + pydantic_ai + SK filter).
    supports_error: bool = False
    extra_imports: Tuple[str, ...] = ()  # additional packages the workload needs
    self_flushing: bool = False  # adapter manages+uploads its own trace (e.g. openai_agents)
    # Additional depth variants beyond default/redaction/error (ADP-partials
    # Cluster F). Each name is passed to the runner as its ``flow`` and the
    # harness asserts the depth it implies: "tool" -> a tool.call, "multi" -> an
    # agent.handoff, "streaming" -> a streamed model.invoke, "async" -> the async
    # path still emits. The runner must handle each name it declares here.
    extra_variants: Tuple[str, ...] = ()
    install_hint: str = ""


# Base env (Python 3.9) frameworks. The >=3.10 and no-extra frameworks are added
# in their own venvs during the broad run; this tuple is filtered by what imports.
FRAMEWORKS: Tuple[FrameworkCase, ...] = (
    FrameworkCase(
        id="langchain",
        import_name="langchain_openai",
        runner=fs.run_langchain,
        required_env=("OPENAI_API_KEY",),
        expected_types=("model.invoke",),
        supports_error=True,  # on_llm_error -> agent.error
        # tool: a real prompt|llm chain (with run_name -> agent_name) + a bound
        # tool -> tool.call. streaming: chain.stream -> streamed model.invoke.
        # async: chain.ainvoke. Closes "bare invoke; chain/tool/agent_name/
        # streaming never on the real wire".
        extra_variants=("tool", "streaming", "async"),
        extra_imports=("langchain_core",),
        install_hint="layerlens[langchain] langchain-openai",
    ),
    FrameworkCase(
        id="langgraph",
        import_name="langgraph",
        runner=fs.run_langgraph,
        required_env=("OPENAI_API_KEY",),
        expected_types=("model.invoke",),
        supports_error=True,  # inherits the langchain handler's error callbacks
        extra_imports=("langchain_openai", "langchain_core"),
        install_hint="layerlens[langgraph] langchain-openai",
    ),
    FrameworkCase(
        id="pydantic_ai",
        import_name="pydantic_ai",
        runner=fs.run_pydantic_ai,
        required_env=("OPENAI_API_KEY",),
        expected_types=("model.invoke",),
        supports_error=True,  # _emit_model_error + _finish_run_error -> agent.error
        install_hint="layerlens[pydantic-ai]",
    ),
    FrameworkCase(
        id="openai_agents",
        import_name="agents",
        runner=fs.run_openai_agents,
        required_env=("OPENAI_API_KEY",),
        expected_types=(),  # refined from observed output
        supports_redaction=False,
        self_flushing=True,  # global TracingProcessor: manages + uploads its own trace
        install_hint="layerlens[openai-agents]",
    ),
    FrameworkCase(
        id="crewai",
        import_name="crewai",
        runner=fs.run_crewai,
        required_env=("OPENAI_API_KEY",),
        supports_redaction=False,
        self_flushing=True,  # listens on crewai event bus, uploads its own trace
        # multi: a 2-agent crew where a manager delegates to a coworker -> real
        # agent.handoff. Closes "DEFAULT only; no multi-agent/hierarchical".
        extra_variants=("multi",),
        install_hint="layerlens[crewai] (py>=3.10)",
    ),
    FrameworkCase(
        id="semantic_kernel",
        import_name="semantic_kernel",
        runner=fs.run_semantic_kernel,
        required_env=("OPENAI_API_KEY",),
        supports_error=True,  # invocation-filter except path -> agent.error
        install_hint="layerlens[semantic-kernel] (py>=3.10)",
    ),
    FrameworkCase(
        id="llamaindex",
        import_name="llama_index",
        runner=fs.run_llamaindex,
        required_env=("OPENAI_API_KEY",),
        supports_redaction=False,
        self_flushing=True,  # root-dispatcher handlers + own collectors, flush on disconnect
        # multi: a real AgentWorkflow where a coordinator hands off to a
        # researcher FunctionAgent -> agent.handoff. Closes "RAG default only;
        # no multi-agent lane".
        extra_variants=("multi",),
        install_hint="llama-index (no extra)",
    ),
    FrameworkCase(
        id="haystack",
        import_name="haystack",
        runner=fs.run_haystack,
        supports_redaction=False,  # LLM-free BM25 retrieval workload
        install_hint="haystack-ai (no extra)",
    ),
    FrameworkCase(
        id="embedding",
        import_name="openai",
        runner=fs.run_embedding,
        required_env=("OPENAI_API_KEY",),
        expected_types=("embedding.create",),
        install_hint="openai (or cohere / sentence-transformers)",
    ),
    FrameworkCase(
        id="vector_store",
        import_name="chromadb",
        runner=fs.run_vector_store,
        expected_types=("retrieval.query",),
        supports_redaction=False,
        install_hint="chromadb (in-proc); Pinecone/Weaviate need a remote service",
    ),
    FrameworkCase(
        id="smolagents",
        import_name="smolagents",
        runner=fs.run_smolagents,
        required_env=("OPENAI_API_KEY",),
        expected_types=("model.invoke",),
        supports_redaction=False,
        self_flushing=True,  # per-run collector created by the adapter, uploaded via client
        install_hint="smolagents openai (no extra)",
    ),
    FrameworkCase(
        id="agno",
        import_name="agno",
        runner=fs.run_agno,
        required_env=("OPENAI_API_KEY",),
        expected_types=("agent.input", "agent.output"),
        install_hint="agno openai (no extra)",
    ),
    FrameworkCase(
        id="strands",
        import_name="strands",
        runner=fs.run_strands,
        required_env=("OPENAI_API_KEY",),
        supports_redaction=False,
        self_flushing=True,  # per-run collector created by the adapter, uploaded via client
        install_hint="strands-agents[openai] (no extra)",
    ),
    FrameworkCase(
        id="google_adk",
        import_name="google.adk",
        runner=fs.run_google_adk,
        any_of_env=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        supports_redaction=False,
        self_flushing=True,  # per-run collector created by the adapter, uploaded via client
        install_hint="google-adk (no extra; Gemini API key)",
    ),
    FrameworkCase(
        id="autogen",
        import_name="autogen_agentchat",
        runner=fs.run_autogen,
        required_env=("OPENAI_API_KEY",),
        supports_redaction=False,
        self_flushing=True,  # adapter-managed collector, flushed on disconnect
        extra_imports=("autogen_ext",),
        install_hint="layerlens[autogen] 'autogen-ext[openai]' (py>=3.10)",
    ),
    FrameworkCase(
        id="ms_agent_framework",
        import_name="semantic_kernel.agents",
        runner=fs.run_ms_agent_framework,
        required_env=("OPENAI_API_KEY",),
        install_hint="layerlens[semantic-kernel] (py>=3.10); instruments SK AgentGroupChat",
    ),
    FrameworkCase(
        id="bedrock_agents",
        import_name="boto3",
        runner=fs.run_bedrock_agents,
        required_env=("BEDROCK_AGENT_ID", "BEDROCK_AGENT_ALIAS_ID"),
        any_of_env=("AWS_ACCESS_KEY_ID", "AWS_PROFILE"),
        expected_types=("model.invoke", "agent.output"),
        supports_redaction=False,  # content gating covered deterministically in unit doubles
        self_flushing=True,  # one collector per invoke_agent, flushed when the stream drains
        install_hint=(
            "boto3 (core dep); AWS creds + BEDROCK_AGENT_ID/BEDROCK_AGENT_ALIAS_ID "
            "for a PREPARED Bedrock Agent with enableTrace"
        ),
    ),
    # ----------------------------------------------------------------- #
    # ADP-PORT: the seven adapters ported from the ateam reference SDK.
    # ----------------------------------------------------------------- #
    FrameworkCase(
        id="dspy",
        import_name="dspy",
        runner=fs.run_dspy,
        required_env=("OLLAMA_HOST",),  # FREE: a plain predict needs no tools
        expected_types=("model.invoke",),
        install_hint="dspy (py>=3.10); a local ollama + llama3:8b",
    ),
    FrameworkCase(
        id="instructor",
        import_name="instructor",
        runner=fs.run_instructor,
        # Mode.TOOLS drives the provider's function-calling schema: llama3:8b
        # genuinely cannot ("does not support tools"), so this needs OpenAI.
        required_env=("OPENAI_API_KEY",),
        expected_types=("model.invoke",),
        install_hint="instructor openai (no extra)",
    ),
    FrameworkCase(
        id="marvin",
        import_name="marvin",
        runner=fs.run_marvin,
        # Marvin 3.x rides pydantic-ai, whose structured output is an output
        # TOOL — ollama cannot hold it. Venv pin: pydantic-ai must stay <1.95.
        required_env=("OPENAI_API_KEY",),
        expected_types=("model.invoke",),
        extra_imports=("pydantic_ai",),
        install_hint="marvin (py>=3.10; pydantic-ai<1.95)",
    ),
    FrameworkCase(
        id="mirascope",
        import_name="mirascope",
        runner=fs.run_mirascope,
        required_env=("OLLAMA_HOST",),  # FREE: a plain llm.call needs no tools
        expected_types=("model.invoke",),
        install_hint="mirascope>=2 openai (v2 API: mirascope.llm, not mirascope.core)",
    ),
    FrameworkCase(
        id="browser_use",
        import_name="browser_use",
        runner=fs.run_browser_use,
        # browser-use makes the model emit a strict JSON action schema per step
        # against a serialized DOM; llama3:8b does not hold it.
        required_env=("OPENAI_API_KEY",),
        expected_types=("tool.call",),
        install_hint="browser-use (py>=3.11) + a playwright-cached chromium",
    ),
    FrameworkCase(
        id="openinference",
        import_name="openinference.instrumentation.openai",
        runner=fs.run_openinference,
        # Ingestion adapter: makes no LLM call of its own. The workload is a real
        # OTel-instrumented completion, served FREE by the local ollama over its
        # OpenAI-compatible endpoint.
        required_env=("OLLAMA_HOST",),
        supports_redaction=False,  # content gating covered by the unit doubles
        self_flushing=True,  # owns one collector per SOURCE OTel trace id
        extra_imports=("opentelemetry.sdk",),
        install_hint="openinference-instrumentation-openai opentelemetry-sdk",
    ),
    FrameworkCase(
        id="langfuse",
        import_name="httpx",
        runner=fs.run_langfuse,
        required_env=("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"),
        supports_redaction=False,  # content gating covered deterministically in unit tests
        self_flushing=True,  # one collector per imported Langfuse trace, uploaded via client
        install_hint="httpx (core dep); LANGFUSE_PUBLIC_KEY/SECRET_KEY + LANGFUSE_HOST",
    ),
    FrameworkCase(
        id="agentforce",
        import_name="httpx",
        runner=fs.run_agentforce,
        required_env=("SF_CLIENT_ID", "SF_CLIENT_SECRET", "SF_INSTANCE_URL"),
        supports_redaction=False,  # content gating covered deterministically in unit tests
        self_flushing=True,  # one collector per imported session, uploaded via client
        install_hint=(
            "httpx (core dep); SF_CLIENT_ID/SF_CLIENT_SECRET/SF_INSTANCE_URL "
            "for a provisioned Agentforce + Data Cloud org with Session Tracing"
        ),
    ),
)


def missing_credentials(case: FrameworkCase) -> Optional[str]:
    missing = [k for k in case.required_env if not os.environ.get(k)]
    if missing:
        return f"missing env: {', '.join(missing)}"
    if case.any_of_env and not any(os.environ.get(k) for k in case.any_of_env):
        return f"none of {', '.join(case.any_of_env)} set"
    return None
