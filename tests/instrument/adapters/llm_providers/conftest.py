"""Shared fixtures for LLM provider adapter tests."""

import pytest


class MockStratix:
    """Mock STRATIX instance for testing provider adapters."""

    def __init__(self):
        self.events = []

    def emit(self, event_type, payload):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type=None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events

    def clear_events(self):
        self.events.clear()


@pytest.fixture
def mock_stratix():
    """Provide a fresh MockStratix instance."""
    return MockStratix()
