import os
from unittest import mock

import pytest

from layerlens.instrument import _upload

# Enable the ``pytester`` fixture so the schema-lock enforcement (LAY-3613) can be
# guarded by an in-process inner pytest run (see test_event_schema.py).
pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _upload_sync_mode():
    """Force synchronous uploads in all tests so assertions don't race the worker thread."""
    _upload._sync_mode = True
    yield
    _upload._sync_mode = False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "live: run against the real LayerLens API")
    config.addinivalue_line(
        "markers",
        "privacy_evidence: proves no PII/secrets/payment/delegation content leaves the "
        "SDK under capture_content=False (doubles as SOC2/GDPR evidence)",
    )
    config.addinivalue_line(
        "markers",
        "invariant: a structural/runtime contract guard (keys-must-match, layer "
        "suppression, no-content sweep, secret scrub, cost pricing). Run as a "
        "required CI gate via `-m invariant` so the build fails if an invariant breaks.",
    )


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
