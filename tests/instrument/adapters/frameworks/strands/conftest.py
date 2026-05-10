"""Shared test fixtures for AWS Strands adapter tests.

Ported as-is from ``ateam/tests/adapters/strands/conftest.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.strands.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.strands.lifecycle``
* ``stratix.sdk.python.adapters.base`` →
  ``layerlens.instrument.adapters._base``
* ``stratix.sdk.python.adapters.capture.CaptureConfig`` →
  ``layerlens.instrument.adapters._base.CaptureConfig``
* ``stratix.sdk.python.adapters.replay_models.ReplayableTrace`` →
  ``layerlens.instrument.adapters._base.ReplayableTrace``
* ``stratix.sdk.python.adapters.registry._ADAPTER_MODULES`` →
  ``layerlens.instrument.adapters._base.registry._ADAPTER_MODULES``
* The wrapper marker attribute renamed by the source from
  ``_stratix_original`` to ``_layerlens_original``.

Multi-tenancy: per the transitional "stratix attribute" pattern (see
migration doc §2.3 step 2 — keystone PR #118 still DRAFT), the
``MockStratix`` / ``EventCollector`` test stub gets an ``org_id``
attribute. The post-merge sweep PR will rebase to canonical kwarg once
#118 lands.
"""

import pytest

from layerlens.instrument.adapters.frameworks.strands.lifecycle import StrandsAdapter


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self):
        self.events = []
        self.org_id = "test-org"

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
    adapter = StrandsAdapter(stratix=mock_stratix)
    adapter.connect()
    return adapter


@pytest.fixture
def adapter_no_stratix():
    adapter = StrandsAdapter()
    adapter.connect()
    return adapter
