"""Capability declaration consistency lint guard.

A framework adapter MUST declare each capability it actually implements:

* If the adapter implements ``serialize_for_replay()`` (with a body — not
  just inherited / pass), it MUST declare ``AdapterCapability.REPLAY``.
* If the adapter wraps a streaming method (``arun``, ``invoke_stream``,
  ``run_stream``, async-iter response wrappers, framework streaming
  callbacks), it MUST declare ``AdapterCapability.STREAMING``.

Without this guard, capability lists drift from reality and the
``atlas-app`` adapter catalog UI surfaces incorrect feature support to
customers — they think they can replay traces from an adapter that
declares no REPLAY, or that they cannot stream from one that wraps every
streaming entry-point.

This file is the in-tree counterpart of the upstream
``manifest_consistency`` lint guard (shipping in the manifest emitter
PR); it runs at unit-test time so regressions are caught before merge.
"""

from __future__ import annotations

import inspect
import importlib
from typing import Type

import pytest

from layerlens.instrument.adapters._base.adapter import (
    BaseAdapter,
    AdapterCapability,
)

# ---------------------------------------------------------------------------
# Adapter discovery
# ---------------------------------------------------------------------------


# Adapters whose source wraps at least one streaming entry-point. Each is
# documented at the call-site grep that proves the wrap exists. When a new
# streaming adapter ships, add it here AND make its ``get_adapter_info``
# declare ``STREAMING`` — both lists stay in lock-step.
_STREAMING_ADAPTERS: list[tuple[str, str, str]] = [
    # (display_name, dotted_module, attribute) — wraps Agent.arun
    ("agno", "layerlens.instrument.adapters.frameworks.agno", "AgnoAdapter"),
    # wraps ChatCompletionAgent.invoke_stream
    (
        "ms_agent_framework",
        "layerlens.instrument.adapters.frameworks.ms_agent_framework",
        "MSAgentAdapter",
    ),
    # TraceProcessor receives GenerationSpanData per chunk
    (
        "openai_agents",
        "layerlens.instrument.adapters.frameworks.openai_agents",
        "OpenAIAgentsAdapter",
    ),
    # BeforeModelCallback / AfterModelCallback fire per chunk
    (
        "google_adk",
        "layerlens.instrument.adapters.frameworks.google_adk",
        "GoogleADKAdapter",
    ),
    # LLMChatStartEvent / LLMChatEndEvent emitted per chunk via Instrumentation
    (
        "llama_index",
        "layerlens.instrument.adapters.frameworks.llama_index",
        "LlamaIndexAdapter",
    ),
    # invoke_agent returns an EventStream completion
    (
        "bedrock_agents",
        "layerlens.instrument.adapters.frameworks.bedrock_agents",
        "BedrockAgentsAdapter",
    ),
]


def _all_adapter_classes() -> list[tuple[str, Type[BaseAdapter]]]:
    """Discover every framework adapter present in this branch.

    Mirrors ``test_bulk_ported_smoke._adapter_classes`` but only returns
    ``BaseAdapter`` subclasses (skips standalone helpers like
    ``BenchmarkImportAdapter`` and callback handlers that are not adapters).
    """
    candidates: list[tuple[str, str, str]] = [
        ("agno", "layerlens.instrument.adapters.frameworks.agno", "AgnoAdapter"),
        (
            "bedrock_agents",
            "layerlens.instrument.adapters.frameworks.bedrock_agents",
            "BedrockAgentsAdapter",
        ),
        (
            "google_adk",
            "layerlens.instrument.adapters.frameworks.google_adk",
            "GoogleADKAdapter",
        ),
        (
            "llama_index",
            "layerlens.instrument.adapters.frameworks.llama_index",
            "LlamaIndexAdapter",
        ),
        (
            "pydantic_ai",
            "layerlens.instrument.adapters.frameworks.pydantic_ai",
            "PydanticAIAdapter",
        ),
        ("strands", "layerlens.instrument.adapters.frameworks.strands", "StrandsAdapter"),
        (
            "openai_agents",
            "layerlens.instrument.adapters.frameworks.openai_agents",
            "OpenAIAgentsAdapter",
        ),
        (
            "ms_agent_framework",
            "layerlens.instrument.adapters.frameworks.ms_agent_framework",
            "MSAgentAdapter",
        ),
        (
            "embedding",
            "layerlens.instrument.adapters.frameworks.embedding",
            "EmbeddingAdapter",
        ),
        (
            "embedding_vector_store",
            "layerlens.instrument.adapters.frameworks.embedding.vector_store_adapter",
            "VectorStoreAdapter",
        ),
        (
            "semantic_kernel",
            "layerlens.instrument.adapters.frameworks.semantic_kernel",
            "SemanticKernelAdapter",
        ),
        ("crewai", "layerlens.instrument.adapters.frameworks.crewai", "CrewAIAdapter"),
        ("autogen", "layerlens.instrument.adapters.frameworks.autogen", "AutoGenAdapter"),
        (
            "langgraph",
            "layerlens.instrument.adapters.frameworks.langgraph",
            "LayerLensLangGraphAdapter",
        ),
        (
            "langfuse",
            "layerlens.instrument.adapters.frameworks.langfuse",
            "LangfuseAdapter",
        ),
        (
            "smolagents",
            "layerlens.instrument.adapters.frameworks.smolagents",
            "SmolAgentsAdapter",
        ),
        (
            "agentforce",
            "layerlens.instrument.adapters.frameworks.agentforce",
            "AgentForceAdapter",
        ),
    ]

    discovered: list[tuple[str, Type[BaseAdapter]]] = []
    for display_name, dotted, attr in candidates:
        try:
            module = importlib.import_module(dotted)
        except ImportError:
            # Adapter package not present in this branch.
            continue
        cls = getattr(module, attr, None)
        if cls is None or not isinstance(cls, type):
            continue
        if not issubclass(cls, BaseAdapter):
            continue
        discovered.append((display_name, cls))
    return discovered


def _has_own_serialize_for_replay(cls: Type[BaseAdapter]) -> bool:
    """Return True if the adapter defines its own ``serialize_for_replay``.

    Inherited stubs / pass-through definitions on the base class do not
    count — only an override that returns a populated ``ReplayableTrace``
    qualifies as "implements REPLAY".
    """
    own = cls.__dict__.get("serialize_for_replay")
    if own is None:
        return False
    if not callable(own):
        return False
    # Reject trivial stubs (one-line ``pass`` or ``raise NotImplementedError``).
    try:
        source = inspect.getsource(own)
    except (OSError, TypeError):
        return True  # cannot read source — be conservative, treat as implemented
    body_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith(("def ", '"""', "#"))
    ]
    if not body_lines:
        return False
    if all(line in ("pass", "raise NotImplementedError", "...") for line in body_lines):
        return False
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,cls",
    _all_adapter_classes(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_replay_capability_matches_serialize_for_replay(
    name: str,
    cls: Type[BaseAdapter],
) -> None:
    """REPLAY must be declared iff ``serialize_for_replay`` is implemented.

    Drift in either direction is a customer-facing bug:

    * Implements ``serialize_for_replay`` but does NOT declare REPLAY:
      replay UI will refuse to offer the feature for traces produced by
      this adapter, even though the adapter supports it.
    * Declares REPLAY but does NOT implement ``serialize_for_replay``:
      replay UI will offer the feature, then crash with
      ``NotImplementedError`` when the user clicks it.
    """
    info = cls().get_adapter_info()
    declared = AdapterCapability.REPLAY in info.capabilities
    implemented = _has_own_serialize_for_replay(cls)

    assert declared == implemented, (
        f"{cls.__name__} REPLAY capability declaration mismatches "
        f"serialize_for_replay implementation: declared={declared}, "
        f"implemented={implemented}. Either add REPLAY to "
        f"capabilities or remove it (whichever matches reality)."
    )


@pytest.mark.parametrize(
    "name,dotted,attr",
    _STREAMING_ADAPTERS,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_streaming_capability_declared_for_streaming_adapters(
    name: str,
    dotted: str,
    attr: str,
) -> None:
    """Adapters that wrap a streaming entry-point MUST declare STREAMING.

    The fixed list above is intentional: when a new streaming adapter
    ships it must be added here AND declare STREAMING in
    ``get_adapter_info``. Both lists stay in sync — this is the lint
    guard that enforces it.
    """
    try:
        module = importlib.import_module(dotted)
    except ImportError:
        pytest.skip(f"{name} adapter not present in this branch")

    cls = getattr(module, attr, None)
    assert cls is not None, f"{attr} not exported from {dotted}"

    info = cls().get_adapter_info()
    assert AdapterCapability.STREAMING in info.capabilities, (
        f"{cls.__name__} wraps a streaming entry-point but does not "
        f"declare AdapterCapability.STREAMING. Add it to the "
        f"capabilities list in get_adapter_info()."
    )
