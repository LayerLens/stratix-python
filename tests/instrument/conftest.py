"""Shared fixtures for instrument tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_stratix():
    """Create a mock STRATIX instance for tests that need one."""
    stratix = MagicMock()
    stratix.agent_id = "test-agent"
    stratix.trial_id = "test-trial"
    stratix.policy_ref = "test-policy@1.0.0"
    stratix._event_buffer = []
    return stratix


@pytest.fixture
def mock_client():
    """Create a mock LayerLens API client."""
    client = MagicMock()
    client.traces = MagicMock()
    client.traces.upload = MagicMock()
    return client
