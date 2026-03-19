"""Tests for STRATIX Adapter Registry with CrewAI, AutoGen, and LLM Providers."""

import pytest

from layerlens.instrument.adapters._registry import (
    AdapterRegistry,
    _ADAPTER_MODULES,
    _FRAMEWORK_PACKAGES,
)
from layerlens.instrument.adapters._base import BaseAdapter


class TestRegistryNewAdapters:
    """Tests for registry inclusion of CrewAI and AutoGen adapters."""

    def setup_method(self):
        """Reset registry before each test."""
        AdapterRegistry.reset()

    def test_crewai_in_adapter_modules(self):
        """Test crewai is registered in adapter modules."""
        assert "crewai" in _ADAPTER_MODULES
        assert _ADAPTER_MODULES["crewai"] == "layerlens.instrument.adapters.crewai"

    def test_autogen_in_adapter_modules(self):
        """Test autogen is registered in adapter modules."""
        assert "autogen" in _ADAPTER_MODULES
        assert _ADAPTER_MODULES["autogen"] == "layerlens.instrument.adapters.autogen"

    def test_crewai_in_framework_packages(self):
        """Test crewai is in framework packages."""
        assert "crewai" in _FRAMEWORK_PACKAGES
        assert _FRAMEWORK_PACKAGES["crewai"] == "crewai"

    def test_autogen_in_framework_packages(self):
        """Test autogen is in framework packages."""
        assert "autogen" in _FRAMEWORK_PACKAGES
        assert _FRAMEWORK_PACKAGES["autogen"] == "autogen"

    def test_lazy_load_crewai(self):
        """Test registry lazy-loads CrewAI adapter."""
        registry = AdapterRegistry()
        adapter = registry.get("crewai")

        assert adapter is not None
        assert adapter.FRAMEWORK == "crewai"
        assert adapter.is_connected

    def test_lazy_load_autogen(self):
        """Test registry lazy-loads AutoGen adapter."""
        registry = AdapterRegistry()
        adapter = registry.get("autogen")

        assert adapter is not None
        assert adapter.FRAMEWORK == "autogen"
        assert adapter.is_connected

    def test_all_frameworks_in_modules(self):
        """Test all frameworks are in adapter modules."""
        expected = {
            "langgraph", "langchain", "crewai", "autogen",
            "openai", "anthropic", "azure_openai", "google_vertex",
            "aws_bedrock", "ollama", "litellm",
        }
        assert expected.issubset(set(_ADAPTER_MODULES.keys()))

    def test_crewai_adapter_info_via_registry(self):
        """Test CrewAI adapter info accessible via registry."""
        registry = AdapterRegistry()
        # Force load
        registry.get("crewai")

        available = registry.list_available()
        crewai_infos = [a for a in available if a.framework == "crewai"]
        assert len(crewai_infos) == 1
        assert crewai_infos[0].name == "CrewAIAdapter"

    def test_autogen_adapter_info_via_registry(self):
        """Test AutoGen adapter info accessible via registry."""
        registry = AdapterRegistry()
        # Force load
        registry.get("autogen")

        available = registry.list_available()
        autogen_infos = [a for a in available if a.framework == "autogen"]
        assert len(autogen_infos) == 1
        assert autogen_infos[0].name == "AutoGenAdapter"


class TestRegistryLLMProviders:
    """Tests for registry inclusion of LLM provider adapters."""

    def setup_method(self):
        AdapterRegistry.reset()

    def test_openai_in_adapter_modules(self):
        assert "openai" in _ADAPTER_MODULES
        assert "openai_adapter" in _ADAPTER_MODULES["openai"]

    def test_anthropic_in_adapter_modules(self):
        assert "anthropic" in _ADAPTER_MODULES
        assert "anthropic_adapter" in _ADAPTER_MODULES["anthropic"]

    def test_azure_openai_in_adapter_modules(self):
        assert "azure_openai" in _ADAPTER_MODULES
        assert "azure_openai_adapter" in _ADAPTER_MODULES["azure_openai"]

    def test_google_vertex_in_adapter_modules(self):
        assert "google_vertex" in _ADAPTER_MODULES
        assert "google_vertex_adapter" in _ADAPTER_MODULES["google_vertex"]

    def test_aws_bedrock_in_adapter_modules(self):
        assert "aws_bedrock" in _ADAPTER_MODULES
        assert "bedrock_adapter" in _ADAPTER_MODULES["aws_bedrock"]

    def test_ollama_in_adapter_modules(self):
        assert "ollama" in _ADAPTER_MODULES
        assert "ollama_adapter" in _ADAPTER_MODULES["ollama"]

    def test_litellm_in_adapter_modules(self):
        assert "litellm" in _ADAPTER_MODULES
        assert "litellm_adapter" in _ADAPTER_MODULES["litellm"]

    def test_openai_in_framework_packages(self):
        assert "openai" in _FRAMEWORK_PACKAGES
        assert _FRAMEWORK_PACKAGES["openai"] == "openai"

    def test_anthropic_in_framework_packages(self):
        assert "anthropic" in _FRAMEWORK_PACKAGES
        assert _FRAMEWORK_PACKAGES["anthropic"] == "anthropic"

    def test_azure_uses_openai_package(self):
        assert _FRAMEWORK_PACKAGES["azure_openai"] == "openai"

    def test_google_vertex_package(self):
        assert _FRAMEWORK_PACKAGES["google_vertex"] == "google.cloud.aiplatform"

    def test_aws_bedrock_package(self):
        assert _FRAMEWORK_PACKAGES["aws_bedrock"] == "boto3"

    def test_ollama_package(self):
        assert _FRAMEWORK_PACKAGES["ollama"] == "ollama"

    def test_litellm_package(self):
        assert _FRAMEWORK_PACKAGES["litellm"] == "litellm"

    def test_lazy_load_openai(self):
        registry = AdapterRegistry()
        adapter = registry.get("openai")
        assert adapter is not None
        assert adapter.FRAMEWORK == "openai"
        assert adapter.is_connected

    def test_lazy_load_anthropic(self):
        registry = AdapterRegistry()
        adapter = registry.get("anthropic")
        assert adapter is not None
        assert adapter.FRAMEWORK == "anthropic"
        assert adapter.is_connected

    def test_lazy_load_azure_openai(self):
        registry = AdapterRegistry()
        adapter = registry.get("azure_openai")
        assert adapter is not None
        assert adapter.FRAMEWORK == "azure_openai"

    def test_lazy_load_google_vertex(self):
        registry = AdapterRegistry()
        adapter = registry.get("google_vertex")
        assert adapter is not None
        assert adapter.FRAMEWORK == "google_vertex"

    def test_lazy_load_aws_bedrock(self):
        registry = AdapterRegistry()
        adapter = registry.get("aws_bedrock")
        assert adapter is not None
        assert adapter.FRAMEWORK == "aws_bedrock"

    def test_lazy_load_ollama(self):
        registry = AdapterRegistry()
        adapter = registry.get("ollama")
        assert adapter is not None
        assert adapter.FRAMEWORK == "ollama"

    def test_lazy_load_litellm(self):
        registry = AdapterRegistry()
        adapter = registry.get("litellm")
        assert adapter is not None
        assert adapter.FRAMEWORK == "litellm"

    def test_eleven_frameworks_total(self):
        """Verify all 11 frameworks registered."""
        expected = {
            "langgraph", "langchain", "crewai", "autogen",
            "openai", "anthropic", "azure_openai", "google_vertex",
            "aws_bedrock", "ollama", "litellm",
        }
        assert expected.issubset(set(_ADAPTER_MODULES.keys()))
