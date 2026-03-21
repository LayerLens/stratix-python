from __future__ import annotations

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner(mix_stderr=False)


@pytest.fixture
def cli_env():
    """Environment variables for CLI tests."""
    return {"LAYERLENS_STRATIX_API_KEY": "test-key-123"}
