"""Per-adapter org_id propagation smoke tests.

Companion to ``tests/instrument/adapters/_base/test_org_id_propagation.py``.
That suite proves the contract at the BaseAdapter level. This suite
proves every concrete framework adapter has wired the contract through
its constructor — i.e. ``org_id`` is accepted, the bound value reaches
``self._org_id``, and the property reports it.

These tests deliberately do NOT exercise framework-specific event
emission paths (those are covered in each adapter's own test file).
The coverage here is the cross-cutting "every adapter accepts org_id"
contract, parametrized over all 17 framework adapters.
"""

from __future__ import annotations

from typing import Type

import pytest

from layerlens.instrument.adapters._base import BaseAdapter


def _all_adapter_classes() -> list[tuple[str, Type[BaseAdapter]]]:
    """Import each shipped framework adapter class.

    Tracks the cross-cutting cardinality cited by the audit
    (``A:/tmp/adapter-depth-audit.md`` — 17 framework adapters).
    """
    cases: list[tuple[str, Type[BaseAdapter]]] = []

    from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

    cases.append(("agno", AgnoAdapter))

    from layerlens.instrument.adapters.frameworks.autogen import AutoGenAdapter

    cases.append(("autogen", AutoGenAdapter))

    from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter

    cases.append(("bedrock_agents", BedrockAgentsAdapter))

    from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter

    cases.append(("crewai", CrewAIAdapter))

    from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter

    cases.append(("embedding", EmbeddingAdapter))

    from layerlens.instrument.adapters.frameworks.embedding.vector_store_adapter import (
        VectorStoreAdapter,
    )

    cases.append(("embedding_vector_store", VectorStoreAdapter))

    from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter

    cases.append(("google_adk", GoogleADKAdapter))

    from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter

    cases.append(("langfuse", LangfuseAdapter))

    from layerlens.instrument.adapters.frameworks.langgraph import (
        LayerLensLangGraphAdapter,
    )

    cases.append(("langgraph", LayerLensLangGraphAdapter))

    from layerlens.instrument.adapters.frameworks.langchain import (
        LayerLensCallbackHandler,
    )

    cases.append(("langchain", LayerLensCallbackHandler))

    from layerlens.instrument.adapters.frameworks.llama_index import LlamaIndexAdapter

    cases.append(("llama_index", LlamaIndexAdapter))

    from layerlens.instrument.adapters.frameworks.ms_agent_framework import MSAgentAdapter

    cases.append(("ms_agent_framework", MSAgentAdapter))

    from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter

    cases.append(("openai_agents", OpenAIAgentsAdapter))

    from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter

    cases.append(("pydantic_ai", PydanticAIAdapter))

    from layerlens.instrument.adapters.frameworks.semantic_kernel import (
        SemanticKernelAdapter,
    )

    cases.append(("semantic_kernel", SemanticKernelAdapter))

    from layerlens.instrument.adapters.frameworks.smolagents import SmolAgentsAdapter

    cases.append(("smolagents", SmolAgentsAdapter))

    from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter

    cases.append(("strands", StrandsAdapter))

    from layerlens.instrument.adapters.frameworks.agentforce import AgentForceAdapter

    cases.append(("agentforce", AgentForceAdapter))

    return cases


_PARAMS = _all_adapter_classes()


@pytest.mark.parametrize("name,cls", _PARAMS, ids=lambda v: v if isinstance(v, str) else "")
def test_adapter_constructor_accepts_org_id(name: str, cls: Type[BaseAdapter]) -> None:
    """Every adapter accepts ``org_id`` and exposes it on the bound property.

    Asserted at construction (no connect/emit needed). Cross-cutting
    multi-tenancy contract — see ``A:/tmp/adapter-depth-audit.md``.
    """
    adapter = cls(org_id=f"org-{name}")
    assert adapter.org_id == f"org-{name}"


@pytest.mark.parametrize("name,cls", _PARAMS, ids=lambda v: v if isinstance(v, str) else "")
def test_adapter_constructor_fails_without_org_id(name: str, cls: Type[BaseAdapter]) -> None:
    """Every adapter constructed without org_id (and without a stratix
    client carrying one) raises ``ValueError`` — fail-fast per CLAUDE.md."""
    with pytest.raises(ValueError, match="non-empty org_id"):
        cls()


def test_adapter_count_matches_audit() -> None:
    """The audit cites 17 framework adapters; this fixture lists them.

    Embedding ships two adapters from the same package (embedding +
    vector_store) so the parametrized list is 18; the canonical
    framework-package count is 17. Anything diverging from that is a
    silent regression.
    """
    package_names = {name.split("_")[0] for name, _ in _PARAMS}
    # 17 framework packages: agno, autogen, bedrock (agents), crewai,
    # embedding, google (adk), langfuse, langgraph, langchain,
    # llama (index), ms (agent framework), openai (agents),
    # pydantic (ai), semantic (kernel), smolagents, strands, agentforce.
    assert len(package_names) == 17, sorted(package_names)
