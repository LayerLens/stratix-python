"""Smoke tests for the 6 ported protocol adapters + certification suite.

Each protocol adapter imports cleanly and exposes a class that extends
:class:`BaseProtocolAdapter` (the protocol-specific ABC ported from
``ateam/stratix/sdk/python/adapters/protocols/base.py``). Deeper tests
covering wire conformance, certification check execution, and per-adapter
event emission are PR-scoped and follow the established LLM/framework
test pattern.
"""

from __future__ import annotations

from typing import Type

import pytest

from layerlens.instrument.adapters.protocols.a2a import A2AAdapter
from layerlens.instrument.adapters.protocols.ap2 import AP2Adapter
from layerlens.instrument.adapters.protocols.mcp import MCPExtensionsAdapter
from layerlens.instrument.adapters.protocols.ucp import UCPAdapter
from layerlens.instrument.adapters.protocols.a2ui import A2UIAdapter
from layerlens.instrument.adapters.protocols.agui import AGUIAdapter

_FLAT_ADAPTERS: list[tuple[str, Type[object]]] = [
    ("ap2", AP2Adapter),
    ("a2ui", A2UIAdapter),
    ("ucp", UCPAdapter),
]

_PACKAGE_ADAPTERS: list[tuple[str, Type[object]]] = [
    ("a2a", A2AAdapter),
    ("agui", AGUIAdapter),
    ("mcp", MCPExtensionsAdapter),
]


@pytest.mark.parametrize(
    "name,cls",
    _FLAT_ADAPTERS + _PACKAGE_ADAPTERS,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_protocol_adapter_imports(name: str, cls: Type[object]) -> None:
    """Each protocol adapter class is importable and is a class."""
    assert cls.__name__
    assert isinstance(cls, type)


@pytest.mark.parametrize(
    "name,cls",
    _PACKAGE_ADAPTERS,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_package_adapter_class_export(name: str, cls: Type[object]) -> None:
    """Subdirectory protocols export ``ADAPTER_CLASS`` for registry."""
    import importlib

    module = importlib.import_module(f"layerlens.instrument.adapters.protocols.{name}")
    assert getattr(module, "ADAPTER_CLASS", None) is cls


def test_certification_suite_imports() -> None:
    """The certification module exposes ``ProtocolCertificationSuite``."""
    from layerlens.instrument.adapters.protocols.certification import (
        CheckResult,
        CertificationResult,
        ProtocolCertificationSuite,
    )

    assert ProtocolCertificationSuite is not None
    assert CheckResult is not None
    assert CertificationResult is not None


def test_base_protocol_adapter_importable() -> None:
    from layerlens.instrument.adapters.protocols.base import BaseProtocolAdapter

    assert BaseProtocolAdapter is not None


def test_protocol_support_modules_importable() -> None:
    """Shared protocol support modules port cleanly."""
    from layerlens.instrument.adapters.protocols import (
        health,
        exceptions,
        connection_pool,
    )

    assert health is not None
    assert exceptions is not None
    assert connection_pool is not None
