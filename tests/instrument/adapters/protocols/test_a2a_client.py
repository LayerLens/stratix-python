from __future__ import annotations

from unittest.mock import MagicMock

from layerlens.instrument._events import (
    A2A_DELEGATION,
    A2A_TASK_CREATED,
    A2A_TASK_UPDATED,
)
from layerlens.instrument.adapters.protocols.a2a.client import A2AClientWrapper
from layerlens.instrument.adapters.protocols.a2a.task_lifecycle import TaskState


def _emitted_event_names(adapter):
    return [call.args[0] for call in adapter.emit.call_args_list]


def _last_payload_for(adapter, event_name):
    for call in reversed(adapter.emit.call_args_list):
        if call.args[0] == event_name:
            return call.args[1]
    raise AssertionError(f"{event_name} was never emitted")


class TestSendTask:
    def test_emits_task_created(self):
        adapter = MagicMock()
        wrapper = A2AClientWrapper(adapter, target_url="https://peer")
        wrapper.send_task("t1", [{"role": "user", "content": "hi"}], task_type="plan", agent_id="agent-42")
        names = _emitted_event_names(adapter)
        assert A2A_TASK_CREATED in names
        created = _last_payload_for(adapter, A2A_TASK_CREATED)
        assert created["task_id"] == "t1"
        assert created["receiver_url"] == "https://peer"
        assert created["task_type"] == "plan"
        assert created["submitter_agent_id"] == "agent-42"
        assert created["message_count"] == 1

    def test_emits_delegation_when_agent_id_set(self):
        adapter = MagicMock()
        A2AClientWrapper(adapter, "https://peer").send_task("t1", [], agent_id="agent-42")
        assert A2A_DELEGATION in _emitted_event_names(adapter)

    def test_no_delegation_without_agent_id(self):
        adapter = MagicMock()
        A2AClientWrapper(adapter, "https://peer").send_task("t1", [])
        assert A2A_DELEGATION not in _emitted_event_names(adapter)

    def test_returns_parent_span_id_for_correlation(self):
        adapter = MagicMock()
        parent = A2AClientWrapper(adapter, "https://peer").send_task("t1", [])
        assert isinstance(parent, str) and len(parent) == 16


class TestCompleteTask:
    def test_completed_emits_update_with_latency(self):
        adapter = MagicMock()
        wrapper = A2AClientWrapper(adapter, "https://peer")
        wrapper.send_task("t1", [])
        adapter.emit.reset_mock()
        wrapper.complete_task("t1", "completed", artifacts=[{"content": "x"}])
        payload = _last_payload_for(adapter, A2A_TASK_UPDATED)
        assert payload["task_id"] == "t1"
        assert payload["status"] == "completed"
        assert payload["artifact_count"] == 1
        assert "latency_ms" in payload

    def test_failure_carries_error_code_and_message(self):
        adapter = MagicMock()
        wrapper = A2AClientWrapper(adapter, "https://peer")
        wrapper.send_task("t1", [])
        adapter.emit.reset_mock()
        wrapper.complete_task("t1", "failed", error_code="E_TIMEOUT", error_message="timed out")
        payload = _last_payload_for(adapter, A2A_TASK_UPDATED)
        assert payload["error_code"] == "E_TIMEOUT"
        assert payload["error"] == "timed out"

    def test_complete_without_send_has_no_latency(self):
        adapter = MagicMock()
        A2AClientWrapper(adapter, "https://peer").complete_task("t-never-sent", "completed")
        payload = _last_payload_for(adapter, A2A_TASK_UPDATED)
        assert "latency_ms" not in payload


class TestDelegation:
    def test_delegate_task_emits_delegation(self):
        adapter = MagicMock()
        A2AClientWrapper(adapter, "https://peer").delegate_task(
            "sender", "receiver", task_id="t9", context={"priority": "high"}
        )
        payload = _last_payload_for(adapter, A2A_DELEGATION)
        assert payload["from_agent"] == "sender"
        assert payload["target_agent"] == "receiver"
        assert payload["task_id"] == "t9"
        assert payload["context_keys"] == ["priority"]


class TestCancelTask:
    def test_cancel_emits_cancelled_status(self):
        adapter = MagicMock()
        wrapper = A2AClientWrapper(adapter, "https://peer")
        wrapper.send_task("t1", [])
        adapter.emit.reset_mock()
        wrapper.cancel_task("t1")
        payload = _last_payload_for(adapter, A2A_TASK_UPDATED)
        assert payload["status"] == TaskState.CANCELLED.value
