import os
from unittest import mock

import pytest

from layerlens.instrument import _upload


@pytest.fixture(autouse=True)
def _upload_sync_mode():
    """Force synchronous uploads in all tests so assertions don't race the worker thread."""
    _upload._sync_mode = True
    yield
    _upload._sync_mode = False


@pytest.fixture
def env_vars():
    """Clean environment variables for testing."""
    env_keys = [
        "LAYERLENS_STRATIX_API_KEY",
    ]
    original_values = {key: os.environ.get(key) for key in env_keys}

    # Clear environment variables
    for key in env_keys:
        if key in os.environ:
            del os.environ[key]

    yield

    # Restore original values
    for key, value in original_values.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]


@pytest.fixture
def mock_env_vars():
    """Mock environment variables with test values."""
    with mock.patch.dict(
        os.environ,
        {
            "LAYERLENS_STRATIX_API_KEY": "test-api-key",
        },
    ):
        yield
