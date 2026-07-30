"""Guard against silently-dead test modules.

A module-level ``pytest.importorskip("X")`` where ``X`` never exists in *any*
environment doesn't skip a test — it permanently deletes the whole module
from every run while CI stays green. That exact failure mode hid a broken
adapter for months (the pydantic_ai suite skipped itself on the fictional
``pydantic_ai.capabilities.hooks``, LAY-3567 B2) and killed the suite's only
concurrency guard the same way.

Every constant ``importorskip`` target must therefore be a known, real
package listed below. Adding a new optional test dependency is fine — add it
here consciously, so a typo'd or imaginary module can never silently bury a
test module again. (Dynamic targets, e.g. the live suite's per-case
``importorskip(case.import_name)``, are registry-driven and not checked here.)
"""

from __future__ import annotations

import os
import ast
from typing import Dict, List

TESTS_DIR = os.path.dirname(__file__)

#: Real, installable modules the test tree may gate on.
KNOWN_SKIP_TARGETS = {
    # core / always-present (gates only on version or odd environments)
    "httpx",
    "pydantic",
    "boto3",
    "fastapi",
    "litellm",
    "ollama",
    "openai",
    "vertexai.generative_models",
    # framework adapters (installed per-venv; absent on the base env by design)
    "pydantic_ai",
    "langchain",
    "langchain.memory",
    "langchain_core",
    "langchain_openai",
    "langgraph.graph",
    "crewai",
    "crewai.events",
    "crewai.tasks.task_output",
    "semantic_kernel",
    "semantic_kernel.agents",
    "semantic_kernel.contents",
    "llama_index",
    "llama_index.core",
    "llama_index.llms.openai",
    "haystack",
    "chromadb",
    "smolagents",
    "strands",
    "agno",
    "google.adk",
    "google.genai",
    "dspy",
    "instructor",
    # instructor's retry surface — the validation-retry lane drives a REAL
    # tenacity RetryError rather than a hand-built stand-in.
    "tenacity",
    "marvin",
    # Mirascope v2's call surface. v1's ``mirascope.core`` does NOT exist on the
    # v2 line the adapter targets, so gate on what is actually imported.
    "mirascope.llm",
    "browser_use",
    # The openinference adapter ingests OTel spans; the suite builds real
    # ReadableSpans from the OTel SDK.
    "opentelemetry.sdk",
    # protocol / integration extras
    "copilotkit",
    "ag_ui_langgraph",
    # protocol SDKs the rewritten adapters test against (installed per-venv:
    # mcp + a2a in the sk venv, ap2 pinned in the ap2 venv; absent on base by design)
    "mcp",
    "a2a",
    "ap2",
}


def _iter_test_files() -> List[str]:
    files = []
    for root, dirs, names in os.walk(TESTS_DIR):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".pytest_cache"}]
        for name in names:
            if name.endswith(".py"):
                files.append(os.path.join(root, name))
    return sorted(files)


def _importorskip_targets(path: str) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        try:
            tree = ast.parse(fh.read(), filename=path)
        except SyntaxError:
            return []
    targets = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "importorskip" or not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            targets.append(arg.value)
    return targets


def test_importorskip_targets_are_known():
    offenders: Dict[str, List[str]] = {}
    for path in _iter_test_files():
        if os.path.abspath(path) == os.path.abspath(__file__):
            continue
        for target in _importorskip_targets(path):
            if target not in KNOWN_SKIP_TARGETS:
                rel = os.path.relpath(path, os.path.dirname(TESTS_DIR))
                offenders.setdefault(target, []).append(rel)

    assert not offenders, (
        "importorskip targets not in the known-module allowlist "
        "(a fictional/typo'd target permanently disables the whole test module "
        "while CI stays green): "
        + "; ".join(f"{t!r} in {', '.join(files)}" for t, files in sorted(offenders.items()))
        + ". If the module is real, add it to KNOWN_SKIP_TARGETS in tests/test_skip_hygiene.py."
    )
