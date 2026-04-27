import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Prepend the worktree's ``src`` directory so test imports resolve to
# the in-tree code first. The global Python 3.12 environment editable-
# installs ``layerlens`` from a sibling worktree which would otherwise
# shadow our changes (CLAUDE.md: tests must actually exercise the code
# under review, not a sibling clone of it).
_WORKTREE_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _WORKTREE_SRC not in sys.path:
    sys.path.insert(0, _WORKTREE_SRC)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "live: run against the real LayerLens API")


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
