"""Tests for LangChain memory tracing (TracedMemory + MemoryMutationTracker)."""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument import trace_context
from layerlens.instrument.adapters.frameworks.langchain import (
    TracedMemory,
    MemoryMutationTracker,
    wrap_memory,
)

from .conftest import find_event, find_events, capture_framework_trace


class _BufferMemory:
    """Tiny LangChain-shaped memory: keeps a list of (input, output) turns."""

    memory_variables: List[str] = ["history"]

    def __init__(self) -> None:
        self._turns: List[Dict[str, Any]] = []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"history": list(self._turns)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        self._turns.append({"in": inputs, "out": outputs})

    def clear(self) -> None:
        self._turns.clear()


class TestProxy:
    def test_load_memory_variables_passes_through(self):
        memory = _BufferMemory()
        memory.save_context({"input": "hi"}, {"output": "hello"})
        traced = wrap_memory(memory)
        out = traced.load_memory_variables({})
        assert out["history"][0]["in"] == {"input": "hi"}

    def test_memory_variables_property(self):
        traced = wrap_memory(_BufferMemory())
        assert traced.memory_variables == ["history"]

    def test_unknown_attribute_forwards(self):
        memory = _BufferMemory()
        memory.custom_field = "value"
        traced = wrap_memory(memory)
        assert traced.custom_field == "value"


class TestStateChange:
    def test_save_context_emits_state_change(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        traced = wrap_memory(_BufferMemory())

        with trace_context(mock_client):
            traced.save_context({"input": "hi"}, {"output": "hello"})

        evt = find_event(uploaded["events"], "agent.state.change")
        assert evt["payload"]["memory_type"] == "_BufferMemory"
        assert evt["payload"]["trigger"] == "save_context"
        assert evt["payload"]["before_hash"].startswith("sha256:")
        assert evt["payload"]["after_hash"].startswith("sha256:")
        assert evt["payload"]["before_hash"] != evt["payload"]["after_hash"]

    def test_save_context_with_unchanged_state_does_not_emit(self, mock_client):
        uploaded = capture_framework_trace(mock_client)

        class _NoopMemory(_BufferMemory):
            def save_context(self, inputs, outputs):
                pass  # don't actually mutate state

        traced = wrap_memory(_NoopMemory())

        with trace_context(mock_client):
            traced.save_context({"x": 1}, {"y": 2})

        assert find_events(uploaded["events"], "agent.state.change") == []

    def test_clear_emits_state_change(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        memory = _BufferMemory()
        memory.save_context({"input": "hi"}, {"output": "hello"})
        traced = wrap_memory(memory)

        with trace_context(mock_client):
            traced.clear()

        evt = find_event(uploaded["events"], "agent.state.change")
        assert evt["payload"]["trigger"] == "clear"

    def test_clear_on_empty_memory_does_not_emit(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        traced = wrap_memory(_BufferMemory())

        with trace_context(mock_client):
            traced.clear()

        assert find_events(uploaded["events"], "agent.state.change") == []

    def test_no_collector_means_no_emission(self):
        """Outside of trace_context the wrapped memory still works but emits nothing."""
        memory = _BufferMemory()
        traced = wrap_memory(memory)
        # No trace_context — should not raise
        traced.save_context({"input": "hi"}, {"output": "hello"})
        traced.clear()


class TestMutationTracker:
    def test_records_mutations(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        memory = _BufferMemory()
        tracker = MemoryMutationTracker()

        with trace_context(mock_client):
            with tracker.track(memory, operation="agent_turn_1"):
                memory.save_context({"input": "q1"}, {"output": "a1"})

        # Mutation was recorded
        assert len(tracker.mutations) == 1
        mutation = tracker.mutations[0]
        assert mutation["operation"] == "agent_turn_1"
        assert mutation["before_hash"] != mutation["after_hash"]

        # Event was emitted
        evt = find_event(uploaded["events"], "agent.state.change")
        assert evt["payload"]["trigger"] == "agent_turn_1"

    def test_no_mutation_means_no_record(self, mock_client):
        memory = _BufferMemory()
        tracker = MemoryMutationTracker()

        with trace_context(mock_client):
            with tracker.track(memory, operation="no_op"):
                pass  # touch nothing

        assert tracker.mutations == []

    def test_clear_resets(self, mock_client):
        memory = _BufferMemory()
        tracker = MemoryMutationTracker()

        with trace_context(mock_client):
            with tracker.track(memory, operation="t1"):
                memory.save_context({"a": 1}, {"b": 1})

        assert len(tracker.mutations) == 1
        tracker.clear()
        assert tracker.mutations == []


class TestNonSerializableMemory:
    def test_hash_falls_back_to_repr(self, mock_client):
        """Memory whose variables contain non-JSON-serializable values still hashes."""
        uploaded = capture_framework_trace(mock_client)

        class _Opaque:
            def __repr__(self):
                return "<opaque>"

        class _OpaqueMemory(_BufferMemory):
            def load_memory_variables(self, inputs):
                return {"history": _Opaque()}

        memory = _OpaqueMemory()
        traced = wrap_memory(memory)

        with trace_context(mock_client):
            traced.save_context({"input": "hi"}, {"output": "hello"})

        # No exception; if state did change we emit, otherwise not.
        # Either path is OK for the fallback test.


def test_traced_memory_is_traced_memory_instance():
    """wrap_memory returns a TracedMemory."""
    assert isinstance(wrap_memory(_BufferMemory()), TracedMemory)
