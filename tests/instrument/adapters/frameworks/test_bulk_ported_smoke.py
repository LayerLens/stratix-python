"""Smoke tests for the 9 bulk-ported framework adapters.

These tests verify the **mechanical port** worked: each adapter imports
cleanly, instantiates, completes the connect → health_check →
get_adapter_info → serialize_for_replay → disconnect cycle without
raising, and exposes ``ADAPTER_CLASS`` for registry lazy-loading.

Deeper per-adapter tests (event emission, capture-config gating, etc.)
follow the SmolAgents test pattern — see
``test_smolagents_adapter.py``. Each adapter gets that level of coverage
in a follow-up PR; this smoke suite is the entry-criteria for the bulk
port itself.
"""

from __future__ import annotations

from typing import Any, Type

import pytest

from layerlens.instrument.adapters._base import (
    BaseAdapter,
    AdapterStatus,
    CaptureConfig,
)


def _adapter_classes() -> list[tuple[str, Type[BaseAdapter]]]:
    """Import each ported adapter and return ``(name, class)`` tuples."""
    cases: list[tuple[str, Type[BaseAdapter]]] = []

    from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

    cases.append(("agno", AgnoAdapter))

    from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter

    cases.append(("bedrock_agents", BedrockAgentsAdapter))

    from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter

    cases.append(("google_adk", GoogleADKAdapter))

    from layerlens.instrument.adapters.frameworks.llama_index import LlamaIndexAdapter

    cases.append(("llama_index", LlamaIndexAdapter))

    from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter

    cases.append(("pydantic_ai", PydanticAIAdapter))

    from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter

    cases.append(("strands", StrandsAdapter))

    from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter

    cases.append(("openai_agents", OpenAIAgentsAdapter))

    from layerlens.instrument.adapters.frameworks.ms_agent_framework import MSAgentAdapter

    cases.append(("ms_agent_framework", MSAgentAdapter))

    # Multi-file framework adapters.
    from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter

    cases.append(("embedding", EmbeddingAdapter))

    from layerlens.instrument.adapters.frameworks.semantic_kernel import (
        SemanticKernelAdapter,
    )

    cases.append(("semantic_kernel", SemanticKernelAdapter))

    from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter

    cases.append(("crewai", CrewAIAdapter))

    from layerlens.instrument.adapters.frameworks.autogen import AutoGenAdapter

    cases.append(("autogen", AutoGenAdapter))

    from layerlens.instrument.adapters.frameworks.langchain import (
        LayerLensCallbackHandler,
    )

    cases.append(("langchain", LayerLensCallbackHandler))

    from layerlens.instrument.adapters.frameworks.langgraph import (
        LayerLensLangGraphAdapter,
    )

    cases.append(("langgraph", LayerLensLangGraphAdapter))

    from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter

    cases.append(("langfuse", LangfuseAdapter))

    from layerlens.instrument.adapters.frameworks.agentforce import AgentForceAdapter

    # Note: package directory is ``agentforce`` but the adapter declares
    # ``FRAMEWORK = "salesforce_agentforce"``. Test ID uses the package
    # name; the metadata test handles the mismatch.
    cases.append(("agentforce", AgentForceAdapter))

    return cases


# Map package name → expected FRAMEWORK string (most are identical;
# Agentforce is the only mismatch).
_PKG_TO_FRAMEWORK = {
    "agentforce": "salesforce_agentforce",
}


@pytest.mark.parametrize("name,cls", _adapter_classes(), ids=lambda v: v if isinstance(v, str) else "")
def test_adapter_metadata(name: str, cls: Type[BaseAdapter]) -> None:
    """Every adapter has a ``FRAMEWORK`` and ``VERSION``."""
    expected = _PKG_TO_FRAMEWORK.get(name, name)
    assert cls.FRAMEWORK == expected
    assert cls.VERSION


@pytest.mark.parametrize("name,cls", _adapter_classes(), ids=lambda v: v if isinstance(v, str) else "")
def test_lifecycle(name: str, cls: Type[BaseAdapter]) -> None:
    """connect → healthy → disconnect → disconnected."""
    if name == "agentforce":
        # AgentForceAdapter.connect() requires Salesforce credentials —
        # not a property of the base lifecycle. Lifecycle exercise for
        # this adapter happens in its own integration test (gated by
        # SALESFORCE_* env vars), not in the bulk smoke suite.
        pytest.skip("agentforce.connect() requires Salesforce credentials")
    # Multi-tenant: every adapter requires an org_id at construction
    # — the smoke suite uses a fixed test value.
    adapter = cls(org_id="test-org")
    adapter.connect()
    assert adapter.is_connected is True
    assert adapter.status == AdapterStatus.HEALTHY

    health = adapter.health_check()
    assert health.framework_name == cls.FRAMEWORK

    info = adapter.get_adapter_info()
    assert info.framework == cls.FRAMEWORK

    rt = adapter.serialize_for_replay()
    assert rt.framework == cls.FRAMEWORK

    adapter.disconnect()
    assert adapter.is_connected is False
    assert adapter.status == AdapterStatus.DISCONNECTED


@pytest.mark.parametrize("name,cls", _adapter_classes(), ids=lambda v: v if isinstance(v, str) else "")
def test_adapter_class_registered(name: str, cls: Type[BaseAdapter]) -> None:
    """The package exports ``ADAPTER_CLASS`` for registry lazy-loading."""
    import importlib

    module = importlib.import_module(f"layerlens.instrument.adapters.frameworks.{name}")
    assert getattr(module, "ADAPTER_CLASS", None) is cls


@pytest.mark.parametrize("name,cls", _adapter_classes(), ids=lambda v: v if isinstance(v, str) else "")
def test_constructor_accepts_capture_config(name: str, cls: Type[BaseAdapter]) -> None:
    """Adapters accept the standard ``capture_config`` constructor arg."""
    adapter = cls(capture_config=CaptureConfig.standard(), org_id="test-org")
    assert adapter.capture_config.l1_agent_io is True


def test_benchmark_import_adapter_independent() -> None:
    """benchmark_import does NOT extend BaseAdapter (it's a data importer).

    Verify it's importable and its public dataclasses construct correctly.
    """
    from layerlens.instrument.adapters.frameworks.benchmark_import import (
        ImportResult,
        BenchmarkMetadata,
        BenchmarkImportAdapter,
    )

    meta = BenchmarkMetadata(name="test", source="csv")
    assert meta.benchmark_id.startswith("bench-")

    result = ImportResult(success=True, benchmark_id=meta.benchmark_id)
    assert result.success is True

    adapter: Any = BenchmarkImportAdapter()
    # No connect/disconnect — different shape than BaseAdapter subclasses.
    assert adapter is not None
