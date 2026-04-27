"""Shared test fixtures for Google ADK adapter tests.

Ported as-is from ``ateam/tests/adapters/google_adk/conftest.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.google_adk.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.google_adk.lifecycle``
"""

from __future__ import annotations

from typing import Any

import pytest

from layerlens.instrument.adapters.frameworks.google_adk.lifecycle import (
    GoogleADKAdapter,
)


class MockStratix:
    """Mock STRATIX instance for testing."""

    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this stratix pass the
    # BaseAdapter fail-fast check.
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
def adapter(mock_stratix: MockStratix) -> GoogleADKAdapter:
    instance = GoogleADKAdapter(stratix=mock_stratix)
    instance.connect()
    return instance


@pytest.fixture
def adapter_no_stratix() -> GoogleADKAdapter:
    instance = GoogleADKAdapter(org_id="test-org")
    instance.connect()
    return instance
