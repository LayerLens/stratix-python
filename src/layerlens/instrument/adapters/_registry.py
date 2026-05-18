from __future__ import annotations

import logging
import importlib
import importlib.util
from typing import Any, Dict, List, Tuple, Optional

from ._base import AdapterInfo, BaseAdapter

log: logging.Logger = logging.getLogger(__name__)

_adapters: Dict[str, BaseAdapter] = {}


# Map adapter name -> import package name. We probe these with
# ``importlib.util.find_spec`` (no actual import) so detection is cheap and
# free of side effects. Adapters that need credentials at connect time
# (agentforce, langfuse) are intentionally excluded from auto-wiring; users
# instantiate those explicitly.
_FRAMEWORK_PACKAGES: Dict[str, str] = {
    "langchain": "langchain_core",
    "langgraph": "langgraph",
    "crewai": "crewai",
    "openai_agents": "agents",
    "semantic_kernel": "semantic_kernel",
    "pydantic_ai": "pydantic_ai",
    "google_adk": "google.adk",
    "strands": "strands",
    "smolagents": "smolagents",
    "llamaindex": "llama_index",
    "haystack": "haystack",
    "autogen": "autogen",
    "agno": "agno",
    "bedrock_agents": "boto3",
    # MS Agent Framework ships as part of semantic-kernel; we share the
    # detection key. Both adapters can coexist — they instrument different
    # surface areas (filters vs AgentChat wrapping).
    "ms_agent_framework": "semantic_kernel",
}

_PROVIDER_PACKAGES: Dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "azure_openai": "openai",
    "google_vertex": "vertexai",
    "bedrock": "boto3",
    "ollama": "ollama",
    "litellm": "litellm",
}

# Map adapter name -> (module path, class name) for ``auto()`` instantiation.
# Only frameworks that can connect with just a layerlens client are listed.
_FRAMEWORK_ADAPTERS: Dict[str, Tuple[str, str]] = {
    "langchain": ("layerlens.instrument.adapters.frameworks.langchain", "LangChainCallbackHandler"),
    "langgraph": ("layerlens.instrument.adapters.frameworks.langgraph", "LangGraphCallbackHandler"),
    "crewai": ("layerlens.instrument.adapters.frameworks.crewai", "CrewAIAdapter"),
    "openai_agents": ("layerlens.instrument.adapters.frameworks.openai_agents", "OpenAIAgentsAdapter"),
    "semantic_kernel": ("layerlens.instrument.adapters.frameworks.semantic_kernel", "SemanticKernelAdapter"),
    "pydantic_ai": ("layerlens.instrument.adapters.frameworks.pydantic_ai", "PydanticAIAdapter"),
    "google_adk": ("layerlens.instrument.adapters.frameworks.google_adk", "GoogleADKAdapter"),
    "strands": ("layerlens.instrument.adapters.frameworks.strands", "StrandsAdapter"),
    "smolagents": ("layerlens.instrument.adapters.frameworks.smolagents", "SmolAgentsAdapter"),
    "llamaindex": ("layerlens.instrument.adapters.frameworks.llamaindex", "LlamaIndexAdapter"),
    "haystack": ("layerlens.instrument.adapters.frameworks.haystack", "HaystackAdapter"),
    "autogen": ("layerlens.instrument.adapters.frameworks.autogen", "AutoGenAdapter"),
    "agno": ("layerlens.instrument.adapters.frameworks.agno", "AgnoAdapter"),
    "bedrock_agents": ("layerlens.instrument.adapters.frameworks.bedrock_agents", "BedrockAgentsAdapter"),
    "ms_agent_framework": (
        "layerlens.instrument.adapters.frameworks.ms_agent_framework",
        "MSAgentFrameworkAdapter",
    ),
}


def register(name: str, adapter: BaseAdapter) -> None:
    """Register an adapter. Disconnects any existing adapter with the same name."""
    existing = _adapters.get(name)
    if existing is not None and existing.is_connected:
        existing.disconnect()
    _adapters[name] = adapter


def unregister(name: str) -> Optional[BaseAdapter]:
    """Remove and disconnect an adapter. Returns the adapter or None."""
    adapter = _adapters.pop(name, None)
    if adapter is not None and adapter.is_connected:
        adapter.disconnect()
    return adapter


def get(name: str) -> Optional[BaseAdapter]:
    """Look up an adapter by name."""
    return _adapters.get(name)


def list_adapters() -> List[AdapterInfo]:
    """Return info for all registered adapters."""
    return [a.adapter_info() for a in _adapters.values()]


def disconnect_all() -> None:
    """Disconnect and remove all adapters."""
    for adapter in _adapters.values():
        try:
            adapter.disconnect()
        except Exception:
            log.warning("Error disconnecting adapter %s", adapter, exc_info=True)
    _adapters.clear()


def _is_installed(package: str) -> bool:
    """Cheap, side-effect-free check whether *package* is importable."""
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def discover_installed() -> Dict[str, List[str]]:
    """Return adapter names whose underlying SDK packages are importable.

    Result shape::

        {"frameworks": ["langchain", "crewai", ...], "providers": ["openai", "anthropic", ...]}

    Use this to inspect what `auto()` would wire up without actually
    connecting anything.
    """
    return {
        "frameworks": sorted(name for name, pkg in _FRAMEWORK_PACKAGES.items() if _is_installed(pkg)),
        "providers": sorted(name for name, pkg in _PROVIDER_PACKAGES.items() if _is_installed(pkg)),
    }


def auto(
    client: Any,
    *,
    capture_config: Any = None,
    skip: Optional[List[str]] = None,
) -> Dict[str, BaseAdapter]:
    """Detect installed frameworks and register a connected adapter for each.

    Only frameworks that can connect with just a layerlens client are wired
    here. Adapters that need credentials at connect time (agentforce,
    langfuse) must be instantiated explicitly. Providers also need explicit
    setup with the user's SDK client — use ``instrument_openai(client)``
    etc. for those.

    Args:
        client: The ``layerlens.Stratix`` instance to attach.
        capture_config: Optional ``CaptureConfig`` shared by every adapter.
        skip: Adapter names to leave un-wired even if installed.

    Returns:
        A dict of ``{adapter_name: connected_adapter}`` for the adapters
        that were successfully connected. Adapters that fail to import or
        connect are logged at WARNING level and omitted from the result.
    """
    skip_set = set(skip or ())
    connected: Dict[str, BaseAdapter] = {}

    for name, package in _FRAMEWORK_PACKAGES.items():
        if name in skip_set:
            continue
        if not _is_installed(package):
            continue
        spec = _FRAMEWORK_ADAPTERS.get(name)
        if spec is None:
            continue
        module_path, class_name = spec
        try:
            module = importlib.import_module(module_path)
            adapter_cls = getattr(module, class_name)
            adapter = (
                adapter_cls(client, capture_config=capture_config)
                if capture_config is not None
                else adapter_cls(client)
            )
            adapter.connect()
        except Exception:
            log.warning("layerlens.instrument.auto: could not wire %s adapter", name, exc_info=True)
            continue
        register(name, adapter)
        connected[name] = adapter

    return connected
