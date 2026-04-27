"""Shared test fixtures for Bedrock Agents adapter tests.

Ported from ``ateam/tests/adapters/bedrock_agents/conftest.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from layerlens.instrument.adapters.frameworks.bedrock_agents.lifecycle import (
    BedrockAgentsAdapter,
)


class MockStratix:
    """Mock LayerLens / STRATIX instance for testing."""

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
def adapter(mock_stratix: MockStratix) -> BedrockAgentsAdapter:
    a = BedrockAgentsAdapter(stratix=mock_stratix)
    a.connect()
    return a


@pytest.fixture
def adapter_no_stratix() -> BedrockAgentsAdapter:
    a = BedrockAgentsAdapter()
    a.connect()
    return a
