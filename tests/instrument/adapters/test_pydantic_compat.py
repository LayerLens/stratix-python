"""Tests for the per-adapter Pydantic v1/v2 compatibility matrix.

Round-2 deliberation item 20. Three behavioral guarantees:

1. Every framework adapter declares ``requires_pydantic`` as one of the
   three :class:`PydanticCompat` enum values.
2. Every framework adapter sets the value *explicitly* on its subclass
   (not relying on the :class:`BaseAdapter` default) so the determination
   is deliberate, not accidental.
3. :func:`requires_pydantic` raises :class:`RuntimeError` with a clear
   message when the runtime Pydantic does not match an adapter's
   declaration.
"""

from __future__ import annotations

from typing import Set, List, Type
from unittest import mock

import pytest

from layerlens.instrument.adapters._base import (
    BaseAdapter,
    PydanticCompat,
    requires_pydantic,
)

# Frameworks whose adapter classes are expected to declare
# requires_pydantic explicitly. Keep this list aligned with the registry's
# ``_ADAPTER_MODULES`` framework subset (excluding providers/protocols
# which are pydantic-agnostic and inherit the default).
#
# ``benchmark_import`` is intentionally absent because its
# ``BenchmarkImportAdapter`` does NOT subclass :class:`BaseAdapter` (it
# never registered an ``ADAPTER_CLASS``).  ``langfuse_importer`` and
# ``browser_use`` registry entries point at module paths that don't
# exist on disk; the registry handles that defensively.
_FRAMEWORK_ADAPTERS: List[str] = [
    "langgraph",
    "langchain",
    "crewai",
    "autogen",
    "semantic_kernel",
    "langfuse",
    "openai_agents",
    "google_adk",
    "bedrock_agents",
    "pydantic_ai",
    "llama_index",
    "smolagents",
    "agno",
    "strands",
    "ms_agent_framework",
    "salesforce_agentforce",
    "embedding",
]


def _import_adapter_class(framework: str) -> Type[BaseAdapter]:
    """Import the adapter class for a given framework.

    Skips adapters whose import-time pydantic compat check would fail
    under the active runtime — those are tested in their own dedicated
    test which mocks ``PYDANTIC_V2``.
    """
    from layerlens.instrument.adapters._base.registry import _ADAPTER_MODULES

    module_path = _ADAPTER_MODULES[framework]
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        # The registry references a module that doesn't exist on disk
        # (pre-existing for ``langfuse_importer`` and ``browser_use``).
        # Fall back to the package path matching the framework name.
        fallback = f"layerlens.instrument.adapters.frameworks.{framework}"
        module = importlib.import_module(fallback)
    cls = getattr(module, "ADAPTER_CLASS", None)
    if cls is None:
        pytest.skip(f"{framework} has no ADAPTER_CLASS — not a registered adapter")
    if not isinstance(cls, type) or not issubclass(cls, BaseAdapter):
        pytest.skip(f"{framework}.ADAPTER_CLASS is not a BaseAdapter subclass")
    return cls


# ---------------------------------------------------------------------------
# Test 1: every framework adapter declares one of the three values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("framework", _FRAMEWORK_ADAPTERS)
def test_adapter_declares_valid_compat(framework: str) -> None:
    """Every registered framework adapter declares ``requires_pydantic``."""
    cls = _import_adapter_class(framework)
    declared = cls.requires_pydantic
    assert isinstance(declared, PydanticCompat), (
        f"{framework}.requires_pydantic must be a PydanticCompat enum, got {type(declared).__name__}: {declared!r}"
    )
    assert declared in {
        PydanticCompat.V1_ONLY,
        PydanticCompat.V2_ONLY,
        PydanticCompat.V1_OR_V2,
    }


# ---------------------------------------------------------------------------
# Test 2: lint — every framework adapter sets the attribute explicitly,
# not relying on the BaseAdapter default by accident.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("framework", _FRAMEWORK_ADAPTERS)
def test_adapter_sets_compat_explicitly(framework: str) -> None:
    """Every framework adapter must override ``requires_pydantic`` itself.

    Walks the MRO and checks that ``requires_pydantic`` appears in the
    adapter subclass's own ``__dict__`` (or that of an intermediate
    framework-specific base class), not only on
    :class:`BaseAdapter`. Guards against a future framework adapter
    silently inheriting V1_OR_V2 when it should declare V2_ONLY.
    """
    cls = _import_adapter_class(framework)

    declaring_classes: Set[str] = set()
    for klass in cls.__mro__:
        if klass is BaseAdapter:
            break
        if "requires_pydantic" in klass.__dict__:
            declaring_classes.add(klass.__name__)

    assert declaring_classes, (
        f"{framework} adapter ({cls.__name__}) does not set "
        "``requires_pydantic`` on its own class. Add an explicit declaration "
        "(V1_ONLY, V2_ONLY, or V1_OR_V2) — relying on the BaseAdapter "
        "default is forbidden by the Round-2 item 20 lint."
    )


# ---------------------------------------------------------------------------
# Test 3: requires_pydantic() raises RuntimeError with a clear message
# when the runtime Pydantic doesn't match.
# ---------------------------------------------------------------------------


def test_requires_pydantic_v1_or_v2_never_raises() -> None:
    """``V1_OR_V2`` declarations are always accepted."""
    requires_pydantic(PydanticCompat.V1_OR_V2)  # must not raise


def test_requires_pydantic_v2_only_raises_under_v1() -> None:
    """A V2_ONLY declaration raises ``RuntimeError`` under v1 runtime."""
    with mock.patch(
        "layerlens.instrument.adapters._base.pydantic_compat.PYDANTIC_V2",
        False,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            requires_pydantic(PydanticCompat.V2_ONLY)
    msg = str(exc_info.value)
    assert "Pydantic v2" in msg
    assert "v2_only" in msg
    # Message must include actionable guidance.
    assert "pip install" in msg


def test_requires_pydantic_v1_only_raises_under_v2() -> None:
    """A V1_ONLY declaration raises ``RuntimeError`` under v2 runtime."""
    with mock.patch(
        "layerlens.instrument.adapters._base.pydantic_compat.PYDANTIC_V2",
        True,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            requires_pydantic(PydanticCompat.V1_ONLY)
    msg = str(exc_info.value)
    assert "Pydantic v1" in msg
    assert "v1_only" in msg
    assert "pip install" in msg


def test_requires_pydantic_v2_only_passes_under_v2() -> None:
    """A V2_ONLY declaration is accepted under v2 runtime."""
    with mock.patch(
        "layerlens.instrument.adapters._base.pydantic_compat.PYDANTIC_V2",
        True,
    ):
        requires_pydantic(PydanticCompat.V2_ONLY)  # must not raise


def test_requires_pydantic_v1_only_passes_under_v1() -> None:
    """A V1_ONLY declaration is accepted under v1 runtime."""
    with mock.patch(
        "layerlens.instrument.adapters._base.pydantic_compat.PYDANTIC_V2",
        False,
    ):
        requires_pydantic(PydanticCompat.V1_ONLY)  # must not raise


def test_requires_pydantic_message_includes_caller_module() -> None:
    """Error message names the calling adapter module for actionability."""

    # Wrap the call in a function so the caller module is detectable.
    def _shim() -> None:
        requires_pydantic(PydanticCompat.V2_ONLY)

    # Force the helper down its raise path.
    with mock.patch(
        "layerlens.instrument.adapters._base.pydantic_compat.PYDANTIC_V2",
        False,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            _shim()
    # The shim lives in this test module; the helper walks one frame
    # back to identify the caller of requires_pydantic itself.
    assert __name__ in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 4: AdapterInfo surfaces the class-level requires_pydantic via .info()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("framework", _FRAMEWORK_ADAPTERS)
def test_adapter_info_exposes_compat(framework: str) -> None:
    """``BaseAdapter.info()`` reflects the class-level declaration."""
    cls = _import_adapter_class(framework)
    try:
        instance = cls()  # type: ignore[call-arg]
    except TypeError:
        pytest.skip(f"{framework} adapter cannot be instantiated with no args")

    info_obj = instance.info()
    assert info_obj.requires_pydantic == cls.requires_pydantic


# ---------------------------------------------------------------------------
# Test 5: documented expectations for the V2_ONLY frameworks
# ---------------------------------------------------------------------------


_EXPECTED_V2_ONLY: Set[str] = {
    "langchain",
    "langgraph",
    "crewai",
    "pydantic_ai",
    "langfuse",
}


@pytest.mark.parametrize("framework", sorted(_EXPECTED_V2_ONLY))
def test_known_v2_only_frameworks(framework: str) -> None:
    """Document expected V2_ONLY status for the well-known cases.

    A regression in this matrix (e.g., loosening langchain to V1_OR_V2)
    fails this test loudly — the determinations were made deliberately
    based on framework version pins and source imports, not by accident.
    """
    cls = _import_adapter_class(framework)
    assert cls.requires_pydantic is PydanticCompat.V2_ONLY, (
        f"{framework} is expected V2_ONLY (see adapter docstring for the "
        f"specific Pydantic v2 imports / framework pin that justifies it)"
    )
