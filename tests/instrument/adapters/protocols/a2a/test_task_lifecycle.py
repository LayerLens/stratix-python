"""
Tests for A2A task lifecycle: submitted → working → completed event chain.
"""

import pytest

from layerlens.instrument.adapters.protocols.a2a.task_lifecycle import (
    TaskStateMachine,
    TaskState,
    TERMINAL_STATES,
)


class TestTaskStateMachine:
    def test_initial_state(self):
        sm = TaskStateMachine("task-1")
        assert sm.state == TaskState.SUBMITTED
        assert not sm.is_terminal

    def test_submitted_to_working(self):
        sm = TaskStateMachine("task-1")
        assert sm.transition("working")
        assert sm.state == TaskState.WORKING

    def test_working_to_completed(self):
        sm = TaskStateMachine("task-1")
        sm.transition("working")
        assert sm.transition("completed")
        assert sm.state == TaskState.COMPLETED
        assert sm.is_terminal

    def test_working_to_failed(self):
        sm = TaskStateMachine("task-1")
        sm.transition("working")
        assert sm.transition("failed")
        assert sm.state == TaskState.FAILED
        assert sm.is_terminal

    def test_working_to_cancelled(self):
        sm = TaskStateMachine("task-1")
        sm.transition("working")
        assert sm.transition("cancelled")
        assert sm.state == TaskState.CANCELLED

    def test_working_to_input_required(self):
        sm = TaskStateMachine("task-1")
        sm.transition("working")
        assert sm.transition("input_required")
        assert sm.state == TaskState.INPUT_REQUIRED

    def test_input_required_to_working(self):
        sm = TaskStateMachine("task-1")
        sm.transition("working")
        sm.transition("input_required")
        assert sm.transition("working")
        assert sm.state == TaskState.WORKING

    def test_invalid_transition(self):
        sm = TaskStateMachine("task-1")
        assert not sm.transition("completed")  # submitted → completed invalid

    def test_terminal_state_no_transition(self):
        sm = TaskStateMachine("task-1")
        sm.transition("working")
        sm.transition("completed")
        assert not sm.transition("working")  # completed → working invalid

    def test_history(self):
        sm = TaskStateMachine("task-1")
        sm.transition("working")
        sm.transition("completed")
        assert len(sm.history) == 2
        assert sm.history[0] == (TaskState.SUBMITTED, TaskState.WORKING)
        assert sm.history[1] == (TaskState.WORKING, TaskState.COMPLETED)

    def test_to_dict(self):
        sm = TaskStateMachine("task-1")
        sm.transition("working")
        d = sm.to_dict()
        assert d["task_id"] == "task-1"
        assert d["state"] == "working"

    def test_unknown_state_string(self):
        sm = TaskStateMachine("task-1")
        assert not sm.transition("invalid_state")


class TestA2ATaskEvents:
    """Test task event emission through the adapter."""

    def test_task_submitted_event(self, a2a_adapter, mock_stratix):
        a2a_adapter.on_task_submitted(
            task_id="task-001",
            receiver_url="http://agent.test",
            task_type="search",
        )
        assert len(mock_stratix.events) == 1

    def test_task_completed_event(self, a2a_adapter, mock_stratix):
        a2a_adapter.on_task_submitted(
            task_id="task-001",
            receiver_url="http://agent.test",
        )
        a2a_adapter.on_task_completed(
            task_id="task-001",
            final_status="completed",
            artifacts=[{"type": "text", "content": "result"}],
        )
        assert len(mock_stratix.events) == 2

    def test_task_completed_with_duration(self, a2a_adapter, mock_stratix):
        a2a_adapter.on_task_submitted(
            task_id="task-002",
            receiver_url="http://agent.test",
        )
        a2a_adapter.on_task_completed(
            task_id="task-002",
            final_status="completed",
        )
        # Duration should be non-None since submitted was recorded
        completed_event = mock_stratix.events[1][0]
        assert completed_event.duration_ms is not None

    def test_task_delegation_emits_handoff(self, a2a_adapter, mock_stratix):
        a2a_adapter.on_task_delegation(
            from_agent="agent-a",
            to_agent="agent-b",
            context={"task": "search"},
        )
        assert len(mock_stratix.events) == 1
        handoff_event = mock_stratix.events[0][0]
        assert handoff_event.event_type == "agent.handoff"

    def test_stream_event(self, a2a_adapter, mock_stratix):
        a2a_adapter.on_stream_event(
            sequence=0,
            payload={"status": "working"},
        )
        assert len(mock_stratix.events) == 1
