from __future__ import annotations

from unittest.mock import MagicMock

from layerlens.instrument._events import A2A_TASK_CREATED, A2A_TASK_UPDATED
from layerlens.instrument.adapters.protocols.a2a.server import A2AServerWrapper


def _event_names(adapter):
    return [call.args[0] for call in adapter.emit.call_args_list]


def _last(adapter, event_name):
    for call in reversed(adapter.emit.call_args_list):
        if call.args[0] == event_name:
            return call.args[1]
    raise AssertionError(f"{event_name} was never emitted")


def _adapter():
    # The server now calls adapter._fingerprint for card-served events; the
    # MagicMock provides it automatically. A real task response carries the
    # server-assigned task id under result.id (spec §9.4.1).
    return MagicMock()


def _task_resp(task_id: str, state: str):
    return {"jsonrpc": "2.0", "id": "req-1", "result": {"kind": "task", "id": task_id, "status": {"state": state}}}


class TestMessageSend:
    def test_emits_created_and_completed_when_handler_succeeds(self):
        adapter = _adapter()
        handler = MagicMock(return_value=_task_resp("t1", "completed"))
        wrapper = A2AServerWrapper(adapter, original_handler=handler)

        response = wrapper.handle_request(
            {"method": "message/send", "id": "req-1", "params": {"message": {"messageId": "m1"}}},
            headers={"authorization": "Bearer x"},
        )
        assert response == _task_resp("t1", "completed")
        names = _event_names(adapter)
        assert A2A_TASK_CREATED in names
        assert A2A_TASK_UPDATED in names
        created = _last(adapter, A2A_TASK_CREATED)
        # The task id is the SERVER-ASSIGNED id from the response Task.
        assert created["task_id"] == "t1"
        assert created["source"] == "server"
        assert created["method"] == "message/send"
        assert "authorization" in created["headers_present"]
        updated = _last(adapter, A2A_TASK_UPDATED)
        assert updated["status"] == "completed"

    def test_handler_exception_emits_failed_update_then_reraises(self):
        adapter = _adapter()

        def handler(_body):
            raise RuntimeError("500 internal")

        wrapper = A2AServerWrapper(adapter, original_handler=handler)
        try:
            wrapper.handle_request(
                {"method": "message/send", "id": "req-1", "params": {"message": {"messageId": "m1"}}}
            )
        except RuntimeError as exc:
            assert "500" in str(exc)
        else:  # pragma: no cover - should have raised
            raise AssertionError("handler exception should have propagated")
        payload = _last(adapter, A2A_TASK_UPDATED)
        assert payload["status"] == "failed"
        assert "500" in payload["error"]

    def test_generates_task_id_when_response_lacks_one(self):
        adapter = _adapter()
        # Handler returns a non-Task (no result.id) -> fall back to a generated id.
        wrapper = A2AServerWrapper(adapter, original_handler=lambda _b: {"jsonrpc": "2.0", "id": "abc"})
        wrapper.handle_request({"method": "message/send", "id": "abc", "params": {"message": {"messageId": "m"}}})
        created = _last(adapter, A2A_TASK_CREATED)
        assert created["task_id"]


class TestMessageStream:
    def test_message_stream_is_recognized(self):
        adapter = _adapter()
        wrapper = A2AServerWrapper(adapter, original_handler=lambda _b: _task_resp("ts", "working"))
        wrapper.handle_request({"method": "message/stream", "id": "r", "params": {"message": {"messageId": "m"}}})
        created = _last(adapter, A2A_TASK_CREATED)
        assert created["method"] == "message/stream"
        assert created["task_id"] == "ts"


class TestTaskCancel:
    def test_emits_update_with_canceled_status(self):
        adapter = _adapter()
        handler = MagicMock(return_value=None)
        wrapper = A2AServerWrapper(adapter, original_handler=handler)
        wrapper.handle_request({"method": "tasks/cancel", "id": "req-1", "params": {"id": "t1"}})
        payload = _last(adapter, A2A_TASK_UPDATED)
        assert payload["task_id"] == "t1"
        # The real spec spelling is single-L 'canceled' (D5).
        assert payload["status"] == "canceled"


class TestObsoleteVocabulary:
    def test_tasks_send_is_ignored(self):
        # The v0.1 'tasks/send' is absent from a2a-sdk 1.1.0 — it must NOT
        # trigger task tracking (D1). A spec-grounded server speaks message/send.
        adapter = _adapter()
        wrapper = A2AServerWrapper(adapter, original_handler=lambda _b: _task_resp("t1", "completed"))
        wrapper.handle_request({"method": "tasks/send", "id": "req-1", "params": {"task": {"id": "t1"}}})
        assert A2A_TASK_CREATED not in _event_names(adapter)


class TestHandlerDelegation:
    def test_response_returned_verbatim_from_original_handler(self):
        adapter = _adapter()
        handler = MagicMock(return_value=_task_resp("t2", "working"))
        wrapper = A2AServerWrapper(adapter, original_handler=handler)
        result = wrapper.handle_request(
            {"method": "message/send", "id": "req-1", "params": {"message": {"messageId": "m"}}}
        )
        assert result == _task_resp("t2", "working")

    def test_returns_none_when_no_handler_registered(self):
        adapter = _adapter()
        wrapper = A2AServerWrapper(adapter)
        assert (
            wrapper.handle_request({"method": "message/send", "id": "req-1", "params": {"message": {"messageId": "m"}}})
            is None
        )


class TestAgentCard:
    def test_emits_card_served_event(self):
        adapter = _adapter()
        A2AServerWrapper(adapter).handle_agent_card_request()
        assert adapter.emit.call_args.args[0] == "a2a.agent.card.served"
