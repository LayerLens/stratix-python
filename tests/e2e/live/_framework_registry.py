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
    extra_imports: Tuple[str, ...] = ()  # additional packages the workload needs
    self_flushing: bool = False  # adapter manages+uploads its own trace (e.g. openai_agents)
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
        extra_imports=("langchain_core",),
        install_hint="layerlens[langchain] langchain-openai",
    ),
    FrameworkCase(
        id="langgraph",
        import_name="langgraph",
        runner=fs.run_langgraph,
        required_env=("OPENAI_API_KEY",),
        expected_types=("model.invoke",),
        extra_imports=("langchain_openai", "langchain_core"),
        install_hint="layerlens[langgraph] langchain-openai",
    ),
    FrameworkCase(
        id="pydantic_ai",
        import_name="pydantic_ai",
        runner=fs.run_pydantic_ai,
        required_env=("OPENAI_API_KEY",),
        expected_types=("model.invoke",),
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
        install_hint="layerlens[crewai] (py>=3.10)",
    ),
    FrameworkCase(
        id="semantic_kernel",
        import_name="semantic_kernel",
        runner=fs.run_semantic_kernel,
        required_env=("OPENAI_API_KEY",),
        install_hint="layerlens[semantic-kernel] (py>=3.10)",
    ),
    FrameworkCase(
        id="llamaindex",
        import_name="llama_index",
        runner=fs.run_llamaindex,
        required_env=("OPENAI_API_KEY",),
        supports_redaction=False,
        self_flushing=True,  # root-dispatcher handlers + own collectors, flush on disconnect
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
)


def missing_credentials(case: FrameworkCase) -> Optional[str]:
    missing = [k for k in case.required_env if not os.environ.get(k)]
    if missing:
        return f"missing env: {', '.join(missing)}"
    if case.any_of_env and not any(os.environ.get(k) for k in case.any_of_env):
        return f"none of {', '.join(case.any_of_env)} set"
    return None
