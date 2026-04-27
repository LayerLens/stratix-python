"""Shared test fixtures for the Agno adapter test suite.

Ported from ``ateam/tests/adapters/agno/conftest.py``.

Renames:
- ``stratix.sdk.python.adapters.agno.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.agno.lifecycle``
- The mock client class is renamed ``MockStratix`` → ``MockLayerLens``
  but the existing pytest fixture name ``mock_stratix`` is preserved
  because the constructor on ``AgnoAdapter`` still accepts a
  positional / keyword ``stratix=`` argument (kept for ateam
  call-site compatibility, see ``AgnoAdapter.__init__``).
"""

from __future__ import annotations

from typing import Any

import pytest

from layerlens.instrument.adapters.frameworks.agno.lifecycle import AgnoAdapter


class MockLayerLens:
    """Mock LayerLens / Stratix client for testing adapter event emission."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


@pytest.fixture
def mock_stratix() -> MockLayerLens:
    return MockLayerLens()


@pytest.fixture
def adapter(mock_stratix: MockLayerLens) -> AgnoAdapter:
    a = AgnoAdapter(stratix=mock_stratix)
    a.connect()
    return a


@pytest.fixture
def adapter_no_stratix() -> AgnoAdapter:
    a = AgnoAdapter()
    a.connect()
    return a
