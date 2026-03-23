from __future__ import annotations

import json
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_client():
    client = Mock()
    client.traces = Mock()
    client.traces.upload = Mock()
    return client


@pytest.fixture
def capture_trace(mock_client):
    uploaded = {}

    def _capture(path):
        with open(path) as f:
            uploaded["trace"] = json.load(f)

    mock_client.traces.upload.side_effect = _capture
    return uploaded
