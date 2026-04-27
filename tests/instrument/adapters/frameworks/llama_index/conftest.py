"""Shared test fixtures for LlamaIndex adapter tests.

Ported from ``ateam/tests/adapters/llama_index/conftest.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.llama_index.lifecycle`` ->
  ``layerlens.instrument.adapters.frameworks.llama_index.lifecycle``
* ``MockStratix`` carries ``org_id = "test-org"`` so adapters constructed
  with this stand-in pass any future multi-tenant fail-fast checks
  (mirrors the langchain test-restoration approach).
"""

from __future__ import annotations

from typing import Any

import pytest

from layerlens.instrument.adapters.frameworks.llama_index.lifecycle import (
    LlamaIndexAdapter,
)


class MockStratix:
    """Mock STRATIX instance for testing."""

    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


@pytest.fixture
def mock_stratix() -> MockStratix:
    return MockStratix()


@pytest.fixture
def adapter(mock_stratix: MockStratix) -> LlamaIndexAdapter:
    a = LlamaIndexAdapter(stratix=mock_stratix)
    a.connect()
    return a


@pytest.fixture
def adapter_no_stratix() -> LlamaIndexAdapter:
    a = LlamaIndexAdapter()
    a.connect()
    return a
