"""Recorded-real-response replay for the Agentforce adapter (LAY-3614).

Agentforce was the other adapter that shipped on a *fictional* schema (it once
queried made-up DMOs; rewritten to the real ``ssot__`` STDM in LAY-3599). This
replays a REAL captured Salesforce session — the OAuth token exchange plus the
seven STDM SOQL responses, captured live from the dev org and committed scrubbed
— through the real ``_SalesforceConnection`` over ``httpx.MockTransport`` and
asserts the imported events. The fixture is the actual Salesforce wire we do not
control, so a future STDM shape change would surface here.
"""

from __future__ import annotations

from typing import Any

import pytest

import layerlens.instrument.adapters.frameworks.agentforce as af
from layerlens.instrument.adapters.frameworks.agentforce import AgentforceAdapter

from .conftest import find_events, capture_framework_trace
from ..._recorded import load_recorded, mock_transport


@pytest.fixture
def _recorded_httpx(monkeypatch):
    """Route the adapter's ``httpx.Client(...)`` through a MockTransport serving
    the recorded Salesforce interactions in order (the doubles' injection seam)."""
    fixture = load_recorded("agentforce", "default")
    transport, _ = mock_transport(fixture)
    real_httpx = af.httpx

    class _Shim:
        def Client(self, **kwargs: Any) -> Any:
            kwargs.pop("transport", None)
            return real_httpx.Client(transport=transport, timeout=kwargs.get("timeout", 30.0))

        def __getattr__(self, name: str) -> Any:
            return getattr(real_httpx, name)

    monkeypatch.setattr(af, "httpx", _Shim())
    return fixture


class TestAgentforceRecorded:
    def test_import_real_stdm_session(self, mock_client, _recorded_httpx):
        uploaded = capture_framework_trace(mock_client)
        adapter = AgentforceAdapter(mock_client)
        adapter.connect(
            credentials={
                "client_id": "x",
                "client_secret": "y",
                "instance_url": "https://unit-test.my.salesforce.com",
            }
        )
        summary = adapter.import_sessions(limit=2)
        adapter.disconnect()

        # The real STDM rows parse: two sessions imported, no errors.
        assert summary["sessions_imported"] == 2
        assert summary["errors"] == 0
        assert summary["next_cursor"]  # max start timestamp seen

        events = uploaded["events"]
        # The real session/interaction/step rows drive the full event family —
        # the fictional-DMO version would have produced none of these.
        assert len(find_events(events, "agent.lifecycle")) == 2
        assert len(find_events(events, "agent.input")) == 2
        assert len(find_events(events, "agent.output")) == 2
        assert find_events(events, "agent.interaction")
        assert find_events(events, "model.invoke")

    def test_provenance(self):
        prov = load_recorded("agentforce", "default")["provenance"]
        assert prov["provider"] == "agentforce"
        assert prov["captured_at"] and prov["captured_at"] != "pending-creds"
