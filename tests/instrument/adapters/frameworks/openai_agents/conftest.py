"""Shared test fixtures for OpenAI Agents adapter tests.

Ported from ``ateam/tests/adapters/openai_agents/conftest.py``.

Renames:
- ``stratix.sdk.python.adapters.openai_agents.lifecycle`` ->
  ``layerlens.instrument.adapters.frameworks.openai_agents.lifecycle``
- ``MockStratix`` -> ``MockStratix`` (unchanged class name preserves
  ateam test ergonomics; only the surrounding namespace moved).

The mock exposes ``org_id`` so :class:`BaseAdapter`'s multi-tenancy
guard (``_resolve_org_id``) is satisfied without forcing every test to
plumb the kwarg manually.
"""

from __future__ import annotations

from typing import Any

import pytest

from layerlens.instrument.adapters.frameworks.openai_agents.lifecycle import (
    OpenAIAgentsAdapter,
)

TEST_ORG_ID = "test-org"


class MockStratix:
    """Mock STRATIX/LayerLens client for testing.

    Carries an ``org_id`` so :class:`BaseAdapter` accepts the adapter at
    construction time (the master branch enforces multi-tenant org_id
    resolution).
    """

    def __init__(self, org_id: str = TEST_ORG_ID) -> None:
        self.org_id = org_id
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
def adapter(mock_stratix: MockStratix) -> OpenAIAgentsAdapter:
    instance = OpenAIAgentsAdapter(stratix=mock_stratix)
    instance.connect()
    return instance


@pytest.fixture
def adapter_no_stratix() -> OpenAIAgentsAdapter:
    instance = OpenAIAgentsAdapter(org_id=TEST_ORG_ID)
    instance.connect()
    return instance
