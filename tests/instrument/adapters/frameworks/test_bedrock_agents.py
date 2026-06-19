"""Adapter-mechanics tests for the Bedrock Agents adapter (LAY-3600).

Lifecycle (hook registration/unregistration via the real boto3 event system),
dependency gating, and error isolation. The real-stream emission contract
(``completion`` EventStream proxy: transparency + per-trace emission) lives in
``test_bedrock_agents_doubles.py``.
"""

from __future__ import annotations

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.stub import Stubber  # noqa: E402

import layerlens.instrument.adapters.frameworks.bedrock_agents as _mod  # noqa: E402
from layerlens.instrument._context import _current_run, _current_collector  # noqa: E402
from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter  # noqa: E402

from .conftest import find_event, capture_framework_trace  # noqa: E402


def _stub_response() -> dict:
    return {"completion": {}, "contentType": "text/plain", "sessionId": "sess-1", "memoryId": "mem-1"}


def _make_boto_client():
    return boto3.client("bedrock-agent-runtime", region_name="us-east-1")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_connect_registers_hooks(self, mock_client):
        boto = _make_boto_client()
        adapter = BedrockAgentsAdapter(mock_client)
        adapter.connect(target=boto)

        stubber = Stubber(boto)
        stubber.activate()
        stubber.add_response("invoke_agent", _stub_response())

        fired = {"before": False, "after": False}
        boto.meta.events.register(_mod._BEFORE_HOOK, lambda **kw: fired.update(before=True))
        boto.meta.events.register(_mod._AFTER_HOOK, lambda **kw: fired.update(after=True))

        boto.invoke_agent(agentId="a1", agentAliasId="al1", sessionId="sess-1", inputText="hi")

        assert fired["before"]
        assert fired["after"]
        adapter.disconnect()

    def test_disconnect_unregisters_hooks(self, mock_client):
        boto = _make_boto_client()
        adapter = BedrockAgentsAdapter(mock_client)
        adapter.connect(target=boto)
        adapter.disconnect()

        stubber = Stubber(boto)
        stubber.activate()
        stubber.add_response("invoke_agent", _stub_response())
        # No collector active, no events emitted, no crash.
        boto.invoke_agent(agentId="a1", agentAliasId="al1", sessionId="sess-1", inputText="hi")

    def test_connect_returns_target(self, mock_client):
        boto = _make_boto_client()
        adapter = BedrockAgentsAdapter(mock_client)
        assert adapter.connect(target=boto) is boto
        adapter.disconnect()

    def test_connect_without_target_raises(self, mock_client):
        with pytest.raises(ValueError, match="requires a bedrock-agent-runtime"):
            BedrockAgentsAdapter(mock_client).connect(target=None)

    def test_adapter_info(self, mock_client):
        info = BedrockAgentsAdapter(mock_client).adapter_info()
        assert info.name == "bedrock_agents"
        assert info.adapter_type == "framework"
        assert not info.connected

    def test_connected_flag(self, mock_client):
        boto = _make_boto_client()
        adapter = BedrockAgentsAdapter(mock_client)
        assert not adapter.adapter_info().connected
        adapter.connect(target=boto)
        assert adapter.adapter_info().connected
        adapter.disconnect()
        assert not adapter.adapter_info().connected

    def test_raises_when_boto3_missing(self, mock_client, monkeypatch):
        monkeypatch.setattr(_mod, "_HAS_BOTO3", False)
        with pytest.raises(ImportError, match="bedrock"):
            BedrockAgentsAdapter(mock_client).connect(target=_make_boto_client())

    def test_disconnect_tolerates_unregister_failure(self, mock_client):
        boto = _make_boto_client()
        adapter = BedrockAgentsAdapter(mock_client)
        adapter.connect(target=boto)

        real_unregister = boto.meta.events.unregister
        boto.meta.events.unregister = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        adapter.disconnect()  # must not raise
        assert not adapter.is_connected
        boto.meta.events.unregister = real_unregister


# ---------------------------------------------------------------------------
# Error isolation — instrumentation must never crash the host call
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    def test_noop_when_disconnected(self, mock_client):
        adapter = BedrockAgentsAdapter(mock_client)
        adapter._before_invoke(params={"agentId": "a1", "inputText": "hi"})
        adapter._after_invoke(parsed={"completion": {}})
        assert not mock_client.traces.upload.called

    def test_hooks_survive_bad_data(self, mock_client):
        """Malformed params/parsed must not raise — and must not leak a run."""
        boto = _make_boto_client()
        adapter = BedrockAgentsAdapter(mock_client)
        adapter.connect(target=boto)
        # Each _before_invoke is paired with an _after_invoke that ends the run
        # (no top-level completion -> empty output + _end_run).
        adapter._before_invoke(params={"agentId": "a1", "inputText": "hi"})
        adapter._after_invoke()
        adapter._before_invoke(params={"agentId": "a1", "inputText": "hi"})
        adapter._after_invoke(parsed=None)
        adapter._before_invoke(params={"agentId": "a1", "inputText": "hi"})
        adapter._after_invoke(parsed={"completion": None})
        # params=None raises inside the hook before a run is begun (swallowed).
        adapter._before_invoke(params=None)
        adapter.disconnect()
        # No run leaked into the ambient ContextVars.
        assert _current_run.get() is None
        assert _current_collector.get() is None

    def test_invoke_error_ends_run_and_records_error(self, mock_client):
        """A transport error (after-call-error) must end the run + emit agent.error."""
        uploaded = capture_framework_trace(mock_client)
        boto = _make_boto_client()
        adapter = BedrockAgentsAdapter(mock_client)
        adapter.connect(target=boto)
        adapter._before_invoke(params={"agentId": "a1", "inputText": "hi"})
        assert _current_run.get() is not None  # run is open
        adapter._on_invoke_error(exception=RuntimeError("connection reset"))
        # Run ended (ContextVars cleared) and the trace flushed with agent.error.
        assert _current_run.get() is None
        adapter.disconnect()
        assert find_event(uploaded["events"], "agent.error")["payload"]["error"] == "connection reset"
