"""Unit tests for the live linkage verifier's assertion semantics (LAY-3638,
F-L2-001/002/003). Imported from the live package but run unconditionally — the
live collection gate only applies to tests under tests/e2e/live/."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from tests.e2e.live._linkage import (
    verify_linkage,
    linkage_configured,
    linkage_skip_reason,
)


def _client(integration_id):
    client = Mock()
    trace = Mock()
    trace.integration_id = integration_id
    client.traces.get.return_value = trace
    return client


@pytest.mark.invariant
class TestVerifyLinkage:
    def test_require_unstamped_raises(self, monkeypatch):
        monkeypatch.delenv("LAYERLENS_LIVE_INTEGRATION_ID", raising=False)
        with pytest.raises(AssertionError):
            verify_linkage(_client(None), "t1", require=True, poll_status=False)

    def test_require_stamped_links(self, monkeypatch):
        monkeypatch.delenv("LAYERLENS_LIVE_INTEGRATION_ID", raising=False)
        result = verify_linkage(_client("i1"), "t1", require=True, poll_status=False)
        assert result["linked"] is True
        assert result["integration_id"] == "i1"

    def test_default_is_record_only_no_raise(self, monkeypatch):
        # Default (no require, no env): records what the API returned, asserts nothing.
        monkeypatch.delenv("LAYERLENS_LIVE_INTEGRATION_ID", raising=False)
        result = verify_linkage(_client(None), "t1", poll_status=False)
        assert result["linked"] is False  # no exception

    def test_linkage_configured_and_skip_reason(self, monkeypatch):
        monkeypatch.delenv("LAYERLENS_LIVE_INTEGRATION_ID", raising=False)
        assert linkage_configured() is False
        assert linkage_skip_reason() is not None  # a visible skip reason, not a silent green
        monkeypatch.setenv("LAYERLENS_LIVE_INTEGRATION_ID", "i9")
        assert linkage_configured() is True
        assert linkage_skip_reason() is None
