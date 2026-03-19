"""
A2A adapter test fixtures.

Provides MockA2AServer and adapter instances for testing.
"""

import pytest
from typing import Any

from layerlens.instrument.adapters.protocols.a2a.adapter import A2AAdapter
from layerlens.instrument.adapters._capture import CaptureConfig


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self):
        self.events: list[Any] = []
        self.agent_id = "test-agent"
        self.framework = "a2a"
        self.is_policy_violated = False

    def __bool__(self):
        return True

    def emit(self, *args, **kwargs):
        self.events.append(args)


class MockA2AServer:
    """
    In-process mock A2A server for testing.

    Supports configurable responses and failure modes.
    """

    def __init__(self):
        self.requests: list[dict] = []
        self.responses: dict[str, Any] = {}
        self.fail_next = False
        self.agent_card = {
            "name": "MockAgent",
            "description": "A mock A2A agent for testing",
            "url": "http://mock-agent.test",
            "protocolVersion": "0.2.1",
            "capabilities": {"streaming": True, "pushNotifications": False},
            "skills": [
                {
                    "id": "search",
                    "name": "Web Search",
                    "description": "Search the web",
                    "tags": ["search", "web"],
                    "examples": ["Search for Python tutorials"],
                }
            ],
            "authentication": {"scheme": "bearer"},
        }

    def handle_request(self, request: dict) -> dict:
        self.requests.append(request)
        method = request.get("method", "")

        if self.fail_next:
            self.fail_next = False
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32001, "message": "Task not found"},
                "id": request.get("id"),
            }

        if method in self.responses:
            return self.responses[method]

        return {
            "jsonrpc": "2.0",
            "result": {"status": "ok"},
            "id": request.get("id"),
        }

    def get_agent_card(self) -> dict:
        return self.agent_card


@pytest.fixture
def mock_stratix():
    return MockStratix()


@pytest.fixture
def mock_server():
    return MockA2AServer()


@pytest.fixture
def a2a_adapter(mock_stratix):
    adapter = A2AAdapter(stratix=mock_stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    return adapter
