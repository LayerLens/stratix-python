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
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.bedrock_agents import BedrockAgentsAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402


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


# ---------------------------------------------------------------------------
# W5 / G10 — capture_content=False redaction (privacy gate)
#
# Proves the per-adapter ``_set_if_capturing`` gate actually SCRUBS the
# framework's content (agent.input, agent.output, tool.call args/output) when
# ``CaptureConfig(capture_content=False)`` — and that the same drive WITH
# ``capture_content=True`` DOES carry it (so the assertion is not vacuous).
# A recognizable SENTINEL is embedded in every content field; under the gate
# it must appear NOWHERE in the emitted payloads, while structural metadata
# (tool_name, agent_id, ...) survives.
# ---------------------------------------------------------------------------

_REDACTION_SENTINEL = "REDACT_ME_b3dr0ck_9WERT"  # appears only in content fields


class _OneShotStream:
    """Single-read ``completion`` EventStream stand-in (mirrors the doubles)."""

    def __init__(self, events):
        self._events = list(events)
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self._events):
            raise StopIteration
        event = self._events[self._idx]
        self._idx += 1
        return event

    def close(self):  # pragma: no cover - parity with the real EventStream API
        pass


def _sentinel_stream() -> "_OneShotStream":
    """A real-shaped completion turn whose every content field carries the SENTINEL:
    an action-group tool call (params + output) followed by the final answer chunk."""
    action_group = {
        "trace": {
            "agentId": "a-sentinel",
            "agentAliasId": "al-sentinel",
            "sessionId": "sess-sentinel",
            "trace": {
                "orchestrationTrace": {
                    "invocationInput": {
                        "invocationType": "ACTION_GROUP",
                        "traceId": "trace-ag-sentinel",
                        "actionGroupInvocationInput": {
                            "actionGroupName": "BookingActions",
                            "function": "rebook",
                            "parameters": [{"name": "passenger", "type": "string", "value": _REDACTION_SENTINEL}],
                        },
                    }
                }
            },
        }
    }
    observation = {
        "trace": {
            "agentId": "a-sentinel",
            "agentAliasId": "al-sentinel",
            "sessionId": "sess-sentinel",
            "trace": {
                "orchestrationTrace": {
                    "observation": {
                        "type": "ACTION_GROUP",
                        "traceId": "trace-ag-sentinel",
                        "actionGroupInvocationOutput": {"text": f"result for {_REDACTION_SENTINEL}"},
                    }
                }
            },
        }
    }
    chunk = {"chunk": {"bytes": f"Answer mentioning {_REDACTION_SENTINEL}".encode("utf-8")}}
    return _OneShotStream([action_group, observation, chunk])


def _drive_with_capture(mock_client, *, capture_content: bool) -> dict:
    """Drive one InvokeAgent turn (SENTINEL input + SENTINEL-bearing stream) under
    the given capture_content setting and return the accumulated uploaded trace."""
    uploaded = capture_framework_trace(mock_client)
    boto = _make_boto_client()

    def _inject(**kwargs):
        parsed = kwargs.get("parsed", {})
        if isinstance(parsed, dict):
            parsed["completion"] = _sentinel_stream()

    # Injector registered BEFORE connect so it populates parsed["completion"]
    # before the adapter's after-call hook wraps it in the proxy.
    boto.meta.events.register(_mod._AFTER_HOOK, _inject)
    adapter = BedrockAgentsAdapter(mock_client, capture_config=CaptureConfig(capture_content=capture_content))
    adapter.connect(target=boto)

    stubber = Stubber(boto)
    stubber.activate()
    stubber.add_response("invoke_agent", _stub_response())

    resp = boto.invoke_agent(
        agentId="a-sentinel",
        agentAliasId="al-sentinel",
        sessionId="sess-sentinel",
        inputText=f"Please rebook {_REDACTION_SENTINEL}",
        enableTrace=True,
    )
    list(resp["completion"])  # draining flushes the proxy's trace
    adapter.disconnect()
    return uploaded


class TestCaptureContentRedaction:
    def test_content_present_when_capturing(self, mock_client):
        """Control: capture_content=True carries the SENTINEL through every
        content field, so the gated assertions below are not vacuous."""
        uploaded = _drive_with_capture(mock_client, capture_content=True)
        events = uploaded["events"]

        import json as _json

        assert find_event(events, "agent.input")["payload"]["input"] == f"Please rebook {_REDACTION_SENTINEL}"
        assert _REDACTION_SENTINEL in find_event(events, "agent.output")["payload"]["output"]

        ag = next(tc for tc in find_events(events, "tool.call") if tc["payload"].get("tool_type") == "action_group")
        # input is the (structured) action-group parameters; output is the result text.
        assert _REDACTION_SENTINEL in _json.dumps(ag["payload"]["input"])  # tool-call args
        assert _REDACTION_SENTINEL in ag["payload"]["output"]  # tool-call result

    def test_content_scrubbed_when_capture_content_false(self, mock_client):
        """The gate: capture_content=False strips agent.input/agent.output and
        tool.call args/output — the SENTINEL appears NOWHERE in any payload —
        while structural metadata survives."""
        uploaded = _drive_with_capture(mock_client, capture_content=False)
        events = uploaded["events"]

        inp = find_event(events, "agent.input")
        assert "input" not in inp["payload"]  # content key dropped entirely
        assert inp["payload"]["agent_id"] == "a-sentinel"  # structure survives

        out = find_event(events, "agent.output")
        assert "output" not in out["payload"]

        ag = next(tc for tc in find_events(events, "tool.call") if tc["payload"].get("tool_type") == "action_group")
        assert "input" not in ag["payload"]  # tool-call args gated
        assert "output" not in ag["payload"]  # tool-call result gated
        assert ag["payload"]["tool_name"] == "BookingActions"  # structure survives
        assert ag["payload"]["function"] == "rebook"

        # Belt-and-suspenders: the SENTINEL must not survive anywhere in the trace.
        import json as _json

        blob = _json.dumps(events)
        assert _REDACTION_SENTINEL not in blob, "capture_content=False leaked content into the trace"
