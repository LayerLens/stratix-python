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


class TestTaskSend:
    def test_emits_created_and_completed_when_handler_succeeds(self):
        adapter = MagicMock()
        handler = MagicMock(return_value={"result": {"status": "completed"}})
        wrapper = A2AServerWrapper(adapter, original_handler=handler)

        response = wrapper.handle_request(
            {"method": "tasks/send", "id": "req-1", "params": {"task": {"id": "t1"}}},
            headers={"authorization": "Bearer x"},
        )
        assert response == {"result": {"status": "completed"}}
        names = _event_names(adapter)
        assert A2A_TASK_CREATED in names
        assert A2A_TASK_UPDATED in names
        created = _last(adapter, A2A_TASK_CREATED)
        assert created["task_id"] == "t1"
        assert created["source"] == "server"
        assert "authorization" in created["headers_present"]
        updated = _last(adapter, A2A_TASK_UPDATED)
        assert updated["status"] == "completed"

    def test_handler_exception_emits_failed_update_then_reraises(self):
        adapter = MagicMock()

        def handler(_body):
            raise RuntimeError("500 internal")

        wrapper = A2AServerWrapper(adapter, original_handler=handler)
        try:
            wrapper.handle_request({"method": "tasks/send", "id": "req-1", "params": {"task": {"id": "t1"}}})
        except RuntimeError as exc:
            assert "500" in str(exc)
        else:  # pragma: no cover - should have raised
            raise AssertionError("handler exception should have propagated")
        payload = _last(adapter, A2A_TASK_UPDATED)
        assert payload["status"] == "failed"
        assert "500" in payload["error"]

    def test_generates_task_id_when_body_lacks_one(self):
        adapter = MagicMock()
        wrapper = A2AServerWrapper(adapter)
        wrapper.handle_request({"method": "tasks/send", "id": "abc"})
        created = _last(adapter, A2A_TASK_CREATED)
        assert created["task_id"]


class TestTaskCancel:
    def test_emits_update_with_cancelled_status(self):
        adapter = MagicMock()
        handler = MagicMock(return_value=None)
        wrapper = A2AServerWrapper(adapter, original_handler=handler)
        wrapper.handle_request({"method": "tasks/cancel", "id": "req-1", "params": {"id": "t1"}})
        payload = _last(adapter, A2A_TASK_UPDATED)
        assert payload["task_id"] == "t1"
        assert payload["status"] == "cancelled"


class TestHandlerDelegation:
    def test_response_returned_verbatim_from_original_handler(self):
        adapter = MagicMock()
        handler = MagicMock(return_value={"result": {"status": "working"}})
        wrapper = A2AServerWrapper(adapter, original_handler=handler)
        result = wrapper.handle_request({"method": "tasks/send", "id": "req-1", "params": {"task": {"id": "t2"}}})
        assert result == {"result": {"status": "working"}}

    def test_returns_none_when_no_handler_registered(self):
        adapter = MagicMock()
        wrapper = A2AServerWrapper(adapter)
        assert wrapper.handle_request({"method": "tasks/send", "id": "req-1", "params": {"task": {"id": "t1"}}}) is None


class TestAgentCard:
    def test_emits_card_served_event(self):
        adapter = MagicMock()
        A2AServerWrapper(adapter).handle_agent_card_request()
        assert adapter.emit.call_args.args[0] == "a2a.agent.card.served"
