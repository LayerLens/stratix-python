#!/usr/bin/env python3
"""Emit ``adapter_catalog/manifest.json`` from the SDK registry.

Used to keep the atlas-app adapter catalog in sync with what
``stratix-python`` actually ships. Run this in CI on every release;
the output is opened as a PR against
``apps/backend/internal/adapter_catalog/manifest.json`` in atlas-app.

Manifest schema (each entry):

::

    {
        "key": "openai",  # registry framework name
        "category": "provider" | "framework" | "protocol",
        "language": "python",
        "package": "layerlens.instrument.adapters.providers.openai_adapter",
        "class_name": "OpenAIAdapter",
        "version": "0.1.0",
        "framework_pip_package": "openai",  # what to ``pip install`` (None for adapters whose runtime is the SDK itself)
        "extras": ["providers-openai"],  # pyproject extra(s) that pull the runtime
        "maturity": "mature" | "lifecycle_preview" | "smoke_only",
        "requires_pydantic": "v1_only" | "v2_only" | "v1_or_v2",
        "capabilities": ["trace_models", "trace_tools"],
        "description": "...",
    }

Maturity tier rules:

* ``mature`` — has dedicated unit-test file in ``tests/instrument/`` AND a
  reference doc in ``docs/adapters/``.
* ``smoke_only`` — only covered by the bulk smoke-test suite.
* ``lifecycle_preview`` — adapter exists but its runtime hooks are
  intentionally minimal (e.g., the source `ateam` lifecycle.py is < 100
  LOC and only wraps lifecycle, no deep instrumentation). None apply
  today — all 33 ported adapters have at least lifecycle-shape tests.

Usage::

    python scripts/emit_adapter_manifest.py [--out PATH]

Default output: ``apps/backend/internal/adapter_catalog/manifest.json``
relative to the *atlas-app* sibling repo (``../atlas-app``). Override
with ``--out`` for CI flows that need a custom path.
"""

from __future__ import annotations

import sys
import json
import argparse
import importlib
from typing import Any, Dict, List, Optional
from pathlib import Path

# -------------------- Static manifest metadata --------------------
#
# The values here are NOT discoverable from the registry alone — they
# come from this module's fixed knowledge of the port: which extra pulls
# which framework, which adapters have full unit-test coverage, etc.
# When you ship a new adapter, update both the registry AND the entry
# here.

_CATEGORY: Dict[str, str] = {
    # Frameworks
    "langgraph": "framework",
    "langchain": "framework",
    "crewai": "framework",
    "autogen": "framework",
    "semantic_kernel": "framework",
    "langfuse": "framework",
    "openai_agents": "framework",
    "google_adk": "framework",
    "bedrock_agents": "framework",
    "pydantic_ai": "framework",
    "llama_index": "framework",
    "smolagents": "framework",
    "agno": "framework",
    "strands": "framework",
    "ms_agent_framework": "framework",
    "salesforce_agentforce": "framework",
    "embedding": "framework",
    "browser_use": "framework",
    "benchmark_import": "framework",
    # Providers
    "openai": "provider",
    "anthropic": "provider",
    "azure_openai": "provider",
    "google_vertex": "provider",
    "aws_bedrock": "provider",
    "ollama": "provider",
    "litellm": "provider",
    "cohere": "provider",
    "mistral": "provider",
    # Protocols
    "a2a": "protocol",
    "agui": "protocol",
    "mcp_extensions": "protocol",
    "ap2": "protocol",
    "a2ui": "protocol",
    "ucp": "protocol",
}

# Map registry key → pyproject extra group(s). ``None`` means no extra
# is needed (e.g., browser_use is a placeholder).
_EXTRAS: Dict[str, List[str]] = {
    "langchain": ["langchain"],
    "langgraph": ["langgraph"],
    "crewai": ["crewai"],
    "autogen": ["autogen"],
    "semantic_kernel": ["semantic-kernel"],
    "langfuse": ["langfuse-importer"],
    "openai_agents": ["openai-agents"],
    "google_adk": ["google-adk"],
    "bedrock_agents": ["bedrock-agents"],
    "pydantic_ai": ["pydantic-ai"],
    "llama_index": ["llama-index"],
    "smolagents": ["smolagents"],
    "agno": ["agno"],
    "strands": ["strands"],
    "ms_agent_framework": ["ms-agent-framework"],
    "salesforce_agentforce": ["agentforce"],
    "embedding": ["embedding"],
    "browser_use": ["browser-use"],
    "benchmark_import": ["benchmark-import"],
    "openai": ["providers-openai"],
    "anthropic": ["providers-anthropic"],
    "azure_openai": ["providers-azure-openai"],
    "google_vertex": ["providers-vertex"],
    "aws_bedrock": ["providers-bedrock"],
    "ollama": ["providers-ollama"],
    "litellm": ["providers-litellm"],
    "cohere": ["providers-cohere"],
    "mistral": ["providers-mistral"],
    "a2a": ["protocols-a2a"],
    "agui": ["protocols-agui"],
    "mcp_extensions": ["protocols-mcp"],
    "ap2": ["protocols-ap2"],
    "a2ui": ["protocols-a2ui"],
    "ucp": ["protocols-ucp"],
}

# Adapters with dedicated unit-test files + reference docs (full coverage).
# All others fall back to ``smoke_only`` (bulk smoke-test coverage only).
# Updated as more adapters reach full-coverage status in the M7 track.
_MATURE: set = {
    "openai",
    "anthropic",
    "azure_openai",
    "aws_bedrock",
    "google_vertex",
    "ollama",
    "litellm",
    "cohere",
    "mistral",
    "smolagents",
    # Frameworks promoted from ``lifecycle_preview`` once they ship a
    # dedicated unit-test suite, a reference doc in ``docs/adapters/``,
    # and a runnable mocked sample under ``samples/instrument/``.
    "langfuse",
}


def _load_registry_modules() -> Dict[str, str]:
    """Import the registry to get the canonical ``key → module path`` map."""
    from layerlens.instrument.adapters._base.registry import _ADAPTER_MODULES

    return dict(_ADAPTER_MODULES)


def _load_framework_packages() -> Dict[str, str]:
    from layerlens.instrument.adapters._base.registry import _FRAMEWORK_PACKAGES

    return dict(_FRAMEWORK_PACKAGES)


def _resolve_adapter_class(module_path: str) -> Optional[type]:
    """Import the module and return its ``ADAPTER_CLASS`` attribute, if any.

    Returns ``None`` for modules that fail to import (e.g., because their
    runtime SDK isn't installed in the manifest-emitter's environment).
    The manifest still includes such entries with whatever metadata is
    statically known.
    """
    try:
        module = importlib.import_module(module_path)
    except Exception:
        return None
    cls = getattr(module, "ADAPTER_CLASS", None)
    return cls if isinstance(cls, type) else None


def _entry(key: str, module_path: str) -> Dict[str, Any]:
    cls = _resolve_adapter_class(module_path)
    pkg = _load_framework_packages().get(key)
    capabilities: List[str] = []
    framework_string: Optional[str] = None
    version = "0.1.0"
    description = ""
    class_name: Optional[str] = None
    # Default to V1_OR_V2 — the BaseAdapter default. Round-2 item 20:
    # surface the per-adapter Pydantic compat in the manifest so the
    # atlas-app catalog UI can warn customers before they pin an
    # incompatible runtime.
    requires_pydantic_value = "v1_or_v2"
    if cls is not None:
        class_name = cls.__name__
        framework_string = getattr(cls, "FRAMEWORK", None)
        version = str(getattr(cls, "VERSION", "0.1.0"))
        compat = getattr(cls, "requires_pydantic", None)
        if compat is not None:
            requires_pydantic_value = compat.value if hasattr(compat, "value") else str(compat)
        try:
            tmp = cls()  # type: ignore[call-arg]
            # ``info()`` overlays the class-level ``requires_pydantic``
            # onto whatever the subclass returned from
            # ``get_adapter_info`` so the manifest stays in sync with the
            # class attribute even if the constructor call omits the field.
            info_obj = tmp.info() if hasattr(tmp, "info") else tmp.get_adapter_info()
            capabilities = [c.value if hasattr(c, "value") else str(c) for c in info_obj.capabilities]
            description = info_obj.description or ""
            info_compat = getattr(info_obj, "requires_pydantic", None)
            if info_compat is not None:
                requires_pydantic_value = info_compat.value if hasattr(info_compat, "value") else str(info_compat)
        except Exception:
            pass

    return {
        "key": key,
        "framework": framework_string or key,
        "category": _CATEGORY.get(key, "framework"),
        "language": "python",
        "package": module_path,
        "class_name": class_name,
        "version": version,
        "framework_pip_package": pkg,
        "extras": _EXTRAS.get(key, []),
        "maturity": "mature" if key in _MATURE else "smoke_only",
        "requires_pydantic": requires_pydantic_value,
        "capabilities": capabilities,
        "description": description,
    }


def build_manifest() -> Dict[str, Any]:
    modules = _load_registry_modules()
    entries = [_entry(key, path) for key, path in sorted(modules.items())]
    return {
        "schema_version": "1.0.0",
        "source": "layerlens",
        "adapter_count": len(entries),
        "by_category": {
            cat: sum(1 for e in entries if e["category"] == cat) for cat in ("framework", "provider", "protocol")
        },
        "adapters": entries,
    }


def _default_output_path() -> Path:
    """``../atlas-app/apps/backend/internal/adapter_catalog/manifest.json``."""
    here = Path(__file__).resolve().parents[1]
    candidate = here.parent / "atlas-app" / "apps" / "backend" / "internal" / "adapter_catalog" / "manifest.json"
    return candidate


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=_default_output_path(),
        help="Output path for manifest.json. Default: atlas-app sibling repo.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of writing to a file.",
    )
    args = parser.parse_args(argv)

    manifest = build_manifest()
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    if args.stdout:
        sys.stdout.write(text)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(
        f"Wrote {len(manifest['adapters'])} adapter entries to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
