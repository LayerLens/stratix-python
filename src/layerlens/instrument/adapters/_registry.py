"""
STRATIX Adapter Registry

Singleton registry that maps framework names to adapter classes, supports
auto-detection of installed frameworks, and provides lazy instantiation.
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any, Type

from layerlens.instrument.adapters._base import AdapterInfo, BaseAdapter
from layerlens.instrument.adapters._capture import CaptureConfig

logger = logging.getLogger(__name__)


# Module path for each framework adapter package
_ADAPTER_MODULES: dict[str, str] = {
    "langgraph": "layerlens.instrument.adapters.langgraph",
    "langchain": "layerlens.instrument.adapters.langchain",
    "crewai": "layerlens.instrument.adapters.crewai",
    "autogen": "layerlens.instrument.adapters.autogen",
    "openai": "layerlens.instrument.adapters.llm_providers.openai_adapter",
    "anthropic": "layerlens.instrument.adapters.llm_providers.anthropic_adapter",
    "azure_openai": "layerlens.instrument.adapters.llm_providers.azure_openai_adapter",
    "google_vertex": "layerlens.instrument.adapters.llm_providers.google_vertex_adapter",
    "aws_bedrock": "layerlens.instrument.adapters.llm_providers.bedrock_adapter",
    "ollama": "layerlens.instrument.adapters.llm_providers.ollama_adapter",
    "litellm": "layerlens.instrument.adapters.llm_providers.litellm_adapter",
    "semantic_kernel": "layerlens.instrument.adapters.semantic_kernel",
    "langfuse": "layerlens.instrument.adapters.langfuse",
    "openai_agents": "layerlens.instrument.adapters.openai_agents",
    "google_adk": "layerlens.instrument.adapters.google_adk",
    "bedrock_agents": "layerlens.instrument.adapters.bedrock_agents",
    "pydantic_ai": "layerlens.instrument.adapters.pydantic_ai",
    "llama_index": "layerlens.instrument.adapters.llama_index",
    "smolagents": "layerlens.instrument.adapters.smolagents",
    # Protocol adapters (Epic 23)
    "a2a": "layerlens.instrument.adapters.protocols.a2a",
    "agui": "layerlens.instrument.adapters.protocols.agui",
    "mcp_extensions": "layerlens.instrument.adapters.protocols.mcp",
}

# Pip-installable package name used to probe whether the framework is
# available in the current environment.
_FRAMEWORK_PACKAGES: dict[str, str] = {
    "langgraph": "langgraph",
    "langchain": "langchain",
    "crewai": "crewai",
    "autogen": "autogen",
    "openai": "openai",
    "anthropic": "anthropic",
    "azure_openai": "openai",
    "google_vertex": "google.cloud.aiplatform",
    "aws_bedrock": "boto3",
    "ollama": "ollama",
    "litellm": "litellm",
    "semantic_kernel": "semantic_kernel",
    "openai_agents": "agents",
    "google_adk": "google.adk",
    "bedrock_agents": "boto3",
    "pydantic_ai": "pydantic_ai",
    "llama_index": "llama_index",
    "smolagents": "smolagents",
    # langfuse has no SDK dependency — adapter uses stdlib urllib
    # Protocol adapters (Epic 23)
    "a2a": "layerlens.instrument.adapters.protocols.a2a",
    "agui": "ag_ui",
    "mcp_extensions": "mcp",
}


class AdapterRegistry:
    """
    Singleton registry of STRATIX framework adapters.

    Usage:
        registry = AdapterRegistry()
        registry.register(MyCustomAdapter)
        adapter = registry.get("langgraph", stratix=stratix_instance)
    """

    _instance: AdapterRegistry | None = None
    _lock: threading.Lock = threading.Lock()
    _registry: dict[str, Type[BaseAdapter]]

    def __new__(cls) -> AdapterRegistry:
        if cls._instance is None:
            with cls._lock:
                # Double-check after acquiring lock
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._registry = {}
                    cls._instance = inst
        return cls._instance

    # --- Public API ---

    def register(self, adapter_class: Type[BaseAdapter]) -> None:
        """
        Register an adapter class.

        The class must define a ``FRAMEWORK`` class attribute.

        Args:
            adapter_class: A subclass of BaseAdapter
        """
        framework = getattr(adapter_class, "FRAMEWORK", None)
        if not framework:
            raise ValueError(
                f"{adapter_class.__name__} does not define a FRAMEWORK class attribute"
            )
        self._registry[framework] = adapter_class
        logger.debug("Registered adapter %s for framework '%s'", adapter_class.__name__, framework)

    def auto_detect(self) -> list[str]:
        """
        Return a list of frameworks whose packages are importable in the
        current environment.
        """
        available: list[str] = []
        for framework, package in _FRAMEWORK_PACKAGES.items():
            try:
                importlib.import_module(package)
                available.append(framework)
            except ImportError:
                pass
        return available

    def get(
        self,
        framework: str,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
    ) -> BaseAdapter:
        """
        Retrieve (lazy-loading if necessary), instantiate, and connect an
        adapter for the given framework.

        Args:
            framework: Framework name (e.g., "langgraph", "langchain")
            stratix: STRATIX SDK instance
            capture_config: CaptureConfig to use

        Returns:
            Connected BaseAdapter instance

        Raises:
            KeyError: If the framework has no registered adapter and cannot
                      be lazy-loaded.
        """
        if framework not in self._registry:
            self._lazy_load(framework)

        adapter_cls = self._registry.get(framework)
        if adapter_cls is None:
            raise KeyError(
                f"No adapter registered for framework '{framework}'. "
                f"Available: {list(self._registry.keys())}"
            )

        adapter = adapter_cls(stratix=stratix, capture_config=capture_config)
        adapter.connect()
        return adapter

    def list_available(self) -> list[AdapterInfo]:
        """
        Return AdapterInfo for every registered adapter.
        """
        results: list[AdapterInfo] = []
        for framework in list(self._registry.keys()):
            cls = self._registry[framework]
            # Instantiate temporarily to get info (no STRATIX needed)
            try:
                tmp = cls()
                results.append(tmp.get_adapter_info())
            except Exception:
                results.append(AdapterInfo(
                    name=cls.__name__,
                    version=getattr(cls, "VERSION", "0.0.0"),
                    framework=framework,
                ))
        return results

    # --- Internal ---

    def _lazy_load(self, framework: str) -> None:
        """
        Attempt to import the adapter module for *framework* and look for
        an ``ADAPTER_CLASS`` attribute at the module level.
        """
        module_path = _ADAPTER_MODULES.get(framework)
        if module_path is None:
            return

        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            logger.debug("Could not import adapter module %s", module_path)
            return

        adapter_cls = getattr(mod, "ADAPTER_CLASS", None)
        if adapter_cls is not None and issubclass(adapter_cls, BaseAdapter):
            self._registry[framework] = adapter_cls
            logger.debug("Lazy-loaded adapter %s from %s", adapter_cls.__name__, module_path)

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton — primarily for test isolation.
        """
        if cls._instance is not None:
            cls._instance._registry.clear()
        cls._instance = None
