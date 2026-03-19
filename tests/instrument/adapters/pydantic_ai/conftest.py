"""Shared test fixtures for Pydantic AI adapter tests."""

import pytest
from layerlens.instrument.adapters.pydantic_ai.lifecycle import PydanticAIAdapter


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self):
        self.events = []

    def emit(self, event_type: str, payload: dict):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str = None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


@pytest.fixture
def mock_stratix():
    return MockStratix()


@pytest.fixture
def adapter(mock_stratix):
    adapter = PydanticAIAdapter(stratix=mock_stratix)
    adapter.connect()
    return adapter


@pytest.fixture
def adapter_no_stratix():
    adapter = PydanticAIAdapter()
    adapter.connect()
    return adapter
