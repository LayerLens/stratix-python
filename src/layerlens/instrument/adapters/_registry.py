from __future__ import annotations

import logging
import importlib
import importlib.util
from typing import Any, Dict, List, Tuple, Optional, FrozenSet

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
    # The ``autogen`` extra installs autogen-agentchat (AutoGen 0.4+), whose
    # modules are ``autogen_core``/``autogen_agentchat`` — there is no
    # top-level ``autogen`` module. Probe what the adapter imports.
    "autogen": "autogen_core",
    "agno": "agno",
    "bedrock_agents": "boto3",
    # MS Agent Framework ships as part of semantic-kernel; we share the
    # detection key. Both adapters can coexist — they instrument different
    # surface areas (filters vs AgentChat wrapping).
    "ms_agent_framework": "semantic_kernel",
    "dspy": "dspy",
    "instructor": "instructor",
    "marvin": "marvin",
    "mirascope": "mirascope",
    # pip dist is hyphenated ``browser-use``; the import name is underscored.
    "browser_use": "browser_use",
    # Ingestion tier: consumes OpenInference-convention OTel spans and patches
    # nothing. Probing ``opentelemetry`` would fire for nearly every user (it is
    # a common transitive dep — it is already in this repo's own base venv), so
    # detection keys off the OpenInference semantic conventions, which are
    # present only when the producer genuinely emits that convention. The
    # adapter itself has no hard dependency on the package: the conventions are
    # attribute-key strings, so an explicitly-wired importer still ingests plain
    # span dicts without it.
    "openinference": "openinference.semconv",
}

_PROVIDER_PACKAGES: Dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "azure_openai": "openai",
    "google_vertex": "vertexai",
    "bedrock": "boto3",
    "ollama": "ollama",
    "litellm": "litellm",
    # OpenRouter has no package of its own — it is an OpenAI-compatible gateway
    # reached through the ``openai`` SDK against openrouter.ai/api/v1.
    "openrouter": "openai",
}

# Map adapter name -> (module path, class name) for ``auto()`` instantiation.
# Only frameworks that can connect with just a layerlens client are listed.
_FRAMEWORK_ADAPTERS: Dict[str, Tuple[str, str]] = {
    "langchain": (
        "layerlens.instrument.adapters.frameworks.langchain",
        "LangChainCallbackHandler",
    ),
    "langgraph": (
        "layerlens.instrument.adapters.frameworks.langgraph",
        "LangGraphCallbackHandler",
    ),
    "crewai": ("layerlens.instrument.adapters.frameworks.crewai", "CrewAIAdapter"),
    "openai_agents": (
        "layerlens.instrument.adapters.frameworks.openai_agents",
        "OpenAIAgentsAdapter",
    ),
    "semantic_kernel": (
        "layerlens.instrument.adapters.frameworks.semantic_kernel",
        "SemanticKernelAdapter",
    ),
    "pydantic_ai": (
        "layerlens.instrument.adapters.frameworks.pydantic_ai",
        "PydanticAIAdapter",
    ),
    "google_adk": (
        "layerlens.instrument.adapters.frameworks.google_adk",
        "GoogleADKAdapter",
    ),
    "strands": ("layerlens.instrument.adapters.frameworks.strands", "StrandsAdapter"),
    "smolagents": (
        "layerlens.instrument.adapters.frameworks.smolagents",
        "SmolAgentsAdapter",
    ),
    "llamaindex": (
        "layerlens.instrument.adapters.frameworks.llamaindex",
        "LlamaIndexAdapter",
    ),
    "haystack": (
        "layerlens.instrument.adapters.frameworks.haystack",
        "HaystackAdapter",
    ),
    "autogen": ("layerlens.instrument.adapters.frameworks.autogen", "AutoGenAdapter"),
    "agno": ("layerlens.instrument.adapters.frameworks.agno", "AgnoAdapter"),
    "bedrock_agents": (
        "layerlens.instrument.adapters.frameworks.bedrock_agents",
        "BedrockAgentsAdapter",
    ),
    "ms_agent_framework": (
        "layerlens.instrument.adapters.frameworks.ms_agent_framework",
        "MSAgentFrameworkAdapter",
    ),
    "dspy": ("layerlens.instrument.adapters.frameworks.dspy", "DSPyAdapter"),
    "instructor": (
        "layerlens.instrument.adapters.frameworks.instructor",
        "InstructorAdapter",
    ),
    "marvin": ("layerlens.instrument.adapters.frameworks.marvin", "MarvinAdapter"),
    "mirascope": (
        "layerlens.instrument.adapters.frameworks.mirascope",
        "MirascopeAdapter",
    ),
    "browser_use": (
        "layerlens.instrument.adapters.frameworks.browser_use",
        "BrowserUseAdapter",
    ),
    "openinference": (
        "layerlens.instrument.adapters.frameworks.openinference",
        "OpenInferenceAdapter",
    ),
}

# Framework adapters that ARE detected (so ``discover_installed()`` reports
# them) and instantiable, but whose ``connect()`` requires an explicit target
# — a specific agent / runtime-client object to wrap — and so cannot be wired
# from a bare layerlens client. ``auto()`` skips these silently, the same way
# the credential-only adapters (agentforce, langfuse) are simply never listed
# in ``_FRAMEWORK_ADAPTERS``; users instantiate them explicitly with a target.
# (bedrock_agents is probed via bare ``boto3``, so attempting it on every
# ``auto()`` call previously logged a WARNING + traceback for every AWS user.)
_TARGET_REQUIRED_ADAPTERS: FrozenSet[str] = frozenset({"pydantic_ai", "bedrock_agents", "instructor", "browser_use"})


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
    here. Adapters that need credentials (agentforce, langfuse) or an explicit
    target object to wrap (pydantic_ai, bedrock_agents) at connect time must
    be instantiated explicitly. Providers also need explicit setup with the
    user's SDK client — use ``instrument_openai(client)`` etc. for those.

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
        if name in _TARGET_REQUIRED_ADAPTERS:
            # Explicit-wire-only: connect() needs a target object to wrap, so
            # it can't be auto-wired from just a client. Skip quietly rather
            # than attempt a connect() that always raises.
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
            log.warning(
                "layerlens.instrument.auto: could not wire %s adapter",
                name,
                exc_info=True,
            )
            continue
        register(name, adapter)
        connected[name] = adapter

    return connected
