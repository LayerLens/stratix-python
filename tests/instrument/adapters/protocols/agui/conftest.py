"""
AG-UI adapter test fixtures.
"""

import pytest
from typing import Any

from layerlens.instrument.adapters.protocols.agui.adapter import AGUIAdapter
from layerlens.instrument.adapters._capture import CaptureConfig


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self):
        self.events: list[Any] = []
        self.agent_id = "test-agent"
        self.framework = "agui"
        self.is_policy_violated = False

    def __bool__(self):
        return True

    def emit(self, *args, **kwargs):
        self.events.append(args)


class MockAGUIStream:
    """
    Mock AG-UI SSE event generator.

    Replays a configurable AG-UI event sequence for testing.
    """

    def __init__(self, events: list[dict] | None = None):
        self.events = events or self._default_events()
        self._index = 0

    def _default_events(self) -> list[dict]:
        return [
            {"type": "RUN_STARTED", "threadId": "thread-1", "runId": "run-1"},
            {"type": "TEXT_MESSAGE_START", "messageId": "msg-1"},
            {"type": "TEXT_MESSAGE_CONTENT", "content": "Hello"},
            {"type": "TEXT_MESSAGE_CONTENT", "content": " World"},
            {"type": "TEXT_MESSAGE_END", "messageId": "msg-1"},
            {"type": "RUN_FINISHED", "threadId": "thread-1", "runId": "run-1"},
        ]

    def __iter__(self):
        for event in self.events:
            yield event

    def __next__(self):
        if self._index >= len(self.events):
            raise StopIteration
        event = self.events[self._index]
        self._index += 1
        return event


@pytest.fixture
def mock_stratix():
    return MockStratix()


@pytest.fixture
def mock_stream():
    return MockAGUIStream()


@pytest.fixture
def agui_adapter(mock_stratix):
    adapter = AGUIAdapter(stratix=mock_stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    return adapter


@pytest.fixture
def agui_adapter_no_streams(mock_stratix):
    config = CaptureConfig.full()
    config.l6b_protocol_streams = False
    adapter = AGUIAdapter(stratix=mock_stratix, capture_config=config)
    adapter.connect()
    return adapter
