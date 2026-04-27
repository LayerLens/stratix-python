"""LayerLens Adapter Registry.

Singleton registry that maps framework names to adapter classes,
supports auto-detection of installed frameworks, and provides lazy
instantiation.

Ported from ``ateam/stratix/sdk/python/adapters/registry.py``. Module
paths are remapped from ``stratix.sdk.python.adapters.*`` to
``layerlens.instrument.adapters.*``. Lazy loading still uses
``importlib.import_module`` so unused adapter modules do not pull their
optional framework dependencies until first use.
"""

from __future__ import annotations

import logging
import importlib
import threading
from typing import Any, Dict, List, Type, Optional

from layerlens.instrument.adapters._base.adapter import AdapterInfo, BaseAdapter
from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

logger = logging.getLogger(__name__)


# Module path for each framework adapter package.
#
# These point at the ``stratix-python`` SDK locations after the port.
# A module is registered here if its ``__init__.py`` (or the explicit
# leaf module named below) defines an ``ADAPTER_CLASS`` attribute that
# subclasses :class:`BaseAdapter`. Importing a module that requires an
# unavailable optional dependency raises :class:`ImportError`, which
# :meth:`AdapterRegistry._lazy_load` swallows and logs.
_ADAPTER_MODULES: Dict[str, str] = {
    # Framework adapters
    "langgraph": "layerlens.instrument.adapters.frameworks.langgraph",
    "langchain": "layerlens.instrument.adapters.frameworks.langchain",
    "crewai": "layerlens.instrument.adapters.frameworks.crewai",
    "autogen": "layerlens.instrument.adapters.frameworks.autogen",
    "semantic_kernel": "layerlens.instrument.adapters.frameworks.semantic_kernel",
    "langfuse": "layerlens.instrument.adapters.frameworks.langfuse",
    "openai_agents": "layerlens.instrument.adapters.frameworks.openai_agents",
    "google_adk": "layerlens.instrument.adapters.frameworks.google_adk",
    "bedrock_agents": "layerlens.instrument.adapters.frameworks.bedrock_agents",
    "pydantic_ai": "layerlens.instrument.adapters.frameworks.pydantic_ai",
    "llama_index": "layerlens.instrument.adapters.frameworks.llama_index",
    "smolagents": "layerlens.instrument.adapters.frameworks.smolagents",
    "agno": "layerlens.instrument.adapters.frameworks.agno",
    "strands": "layerlens.instrument.adapters.frameworks.strands",
    "ms_agent_framework": "layerlens.instrument.adapters.frameworks.ms_agent_framework",
    "salesforce_agentforce": "layerlens.instrument.adapters.frameworks.agentforce",
    "embedding": "layerlens.instrument.adapters.frameworks.embedding",
    "browser_use": "layerlens.instrument.adapters.frameworks.browser_use",
    "benchmark_import": "layerlens.instrument.adapters.frameworks.benchmark_import",
    # LLM provider adapters
    "openai": "layerlens.instrument.adapters.providers.openai_adapter",
    "anthropic": "layerlens.instrument.adapters.providers.anthropic_adapter",
    "azure_openai": "layerlens.instrument.adapters.providers.azure_openai_adapter",
    "google_vertex": "layerlens.instrument.adapters.providers.google_vertex_adapter",
    "aws_bedrock": "layerlens.instrument.adapters.providers.bedrock_adapter",
    "ollama": "layerlens.instrument.adapters.providers.ollama_adapter",
    "litellm": "layerlens.instrument.adapters.providers.litellm_adapter",
    "cohere": "layerlens.instrument.adapters.providers.cohere_adapter",
    "mistral": "layerlens.instrument.adapters.providers.mistral_adapter",
    # Protocol adapters
    "a2a": "layerlens.instrument.adapters.protocols.a2a",
    "agui": "layerlens.instrument.adapters.protocols.agui",
    "mcp_extensions": "layerlens.instrument.adapters.protocols.mcp",
    "ap2": "layerlens.instrument.adapters.protocols.ap2",
    "a2ui": "layerlens.instrument.adapters.protocols.a2ui",
    "ucp": "layerlens.instrument.adapters.protocols.ucp",
}

# Pip-installable package name used to probe whether the framework is
# available in the current environment. Used by :meth:`auto_detect`.
_FRAMEWORK_PACKAGES: Dict[str, str] = {
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
    "cohere": "cohere",
    "mistral": "mistralai",
    "semantic_kernel": "semantic_kernel",
    "openai_agents": "agents",
    "google_adk": "google.adk",
    "bedrock_agents": "boto3",
    "pydantic_ai": "pydantic_ai",
    "llama_index": "llama_index",
    "smolagents": "smolagents",
    "agno": "agno",
    "strands": "strands",
    "ms_agent_framework": "semantic_kernel",
    "salesforce_agentforce": "requests",
    "embedding": "layerlens.instrument.adapters.frameworks.embedding",
    "browser_use": "browser_use",
    "benchmark_import": "layerlens.instrument.adapters.frameworks.benchmark_import",
    "langfuse": "layerlens.instrument.adapters.frameworks.langfuse",
    "a2a": "layerlens.instrument.adapters.protocols.a2a",
    "agui": "ag_ui",
    "mcp_extensions": "mcp",
    "ap2": "layerlens.instrument.adapters.protocols.ap2",
    "a2ui": "layerlens.instrument.adapters.protocols.a2ui",
    "ucp": "layerlens.instrument.adapters.protocols.ucp",
}


class AdapterRegistry:
    """Singleton registry of LayerLens framework adapters.

    Usage::

        registry = AdapterRegistry()
        registry.register(MyCustomAdapter)
        adapter = registry.get("langgraph", stratix=client)
    """

    _instance: Optional["AdapterRegistry"] = None
    _lock: threading.Lock = threading.Lock()
    _registry: Dict[str, Type[BaseAdapter]]

    def __new__(cls) -> "AdapterRegistry":
        if cls._instance is None:
            with cls._lock:
                # Double-check after acquiring lock.
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._registry = {}
                    cls._instance = inst
        return cls._instance

    # --- Public API ---

    def register(self, adapter_class: Type[BaseAdapter]) -> None:
        """Register an adapter class.

        The class must define a ``FRAMEWORK`` class attribute.

        Args:
            adapter_class: A subclass of :class:`BaseAdapter`.

        Raises:
            ValueError: If the class does not define ``FRAMEWORK``.
        """
        framework = getattr(adapter_class, "FRAMEWORK", None)
        if not framework:
            raise ValueError(
                f"{adapter_class.__name__} does not define a FRAMEWORK class attribute"
            )
        self._registry[framework] = adapter_class
        logger.debug(
            "Registered adapter %s for framework '%s'",
            adapter_class.__name__,
            framework,
        )

    def auto_detect(self) -> List[str]:
        """Return a list of frameworks whose packages are importable."""
        available: List[str] = []
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
        stratix: Any = None,
        capture_config: Optional[CaptureConfig] = None,
    ) -> BaseAdapter:
        """Retrieve, instantiate, and connect an adapter.

        Lazy-loads the adapter module on first use so framework
        dependencies are never imported by ``import layerlens`` alone.

        Args:
            framework: Framework name (e.g., ``"langgraph"``,
                ``"langchain"``).
            stratix: LayerLens client instance.
            capture_config: :class:`CaptureConfig` to use.

        Returns:
            Connected :class:`BaseAdapter` instance.

        Raises:
            KeyError: If the framework has no registered adapter and
                cannot be lazy-loaded.
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

    def list_available(self) -> List[AdapterInfo]:
        """Return :class:`AdapterInfo` for every registered adapter.

        Uses :meth:`BaseAdapter.info` so the class-level
        ``requires_pydantic`` declaration is applied even if the subclass
        omits it from its :meth:`get_adapter_info` constructor call.
        """
        results: List[AdapterInfo] = []
        for framework in list(self._registry.keys()):
            cls = self._registry[framework]
            try:
                tmp = cls()
                results.append(tmp.info())
            except Exception:
                results.append(
                    AdapterInfo(
                        name=cls.__name__,
                        version=getattr(cls, "VERSION", "0.0.0"),
                        framework=framework,
                        requires_pydantic=getattr(cls, "requires_pydantic", PydanticCompat.V1_OR_V2),
                    )
                )
        return results

    # --- Internal ---

    def _lazy_load(self, framework: str) -> None:
        """Import the adapter module for *framework* and pull ``ADAPTER_CLASS``."""
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
            logger.debug(
                "Lazy-loaded adapter %s from %s",
                adapter_cls.__name__,
                module_path,
            )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton — primarily for test isolation."""
        if cls._instance is not None:
            cls._instance._registry.clear()
        cls._instance = None
