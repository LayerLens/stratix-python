"""Tests for STRATIX EventParser utility."""

import pytest
from datetime import datetime

from layerlens.instrument.utils.event_parser import (
    EventParser,
    ModelInvocation,
    ToolCall,
    StateChange,
)


@pytest.fixture
def parser():
    """Create EventParser instance."""
    return EventParser()


@pytest.fixture
def sample_events():
    """Sample events for testing."""
    return [
        {
            "identity": {
                "event_type": "model_invoke",
                "span_id": "span-1",
                "agent_id": "agent-1",
                "timestamps": {"created_at": "2024-01-15T10:00:00Z"},
            },
            "payload": {
                "layer": "L3",
                "model": "gpt-4",
                "prompt": "Hello",
                "response": "Hi there!",
                "tokens_in": 10,
                "tokens_out": 5,
                "latency_ms": 150.0,
            },
        },
        {
            "identity": {
                "event_type": "tool_call",
                "span_id": "span-2",
                "agent_id": "agent-1",
                "timestamps": {"created_at": "2024-01-15T10:00:01Z"},
            },
            "payload": {
                "layer": "L5a",
                "tool_name": "search",
                "inputs": {"query": "test"},
                "output": {"results": ["a", "b"]},
                "latency_ms": 200.0,
                "success": True,
            },
        },
        {
            "identity": {
                "event_type": "state_change",
                "span_id": "span-3",
                "agent_id": "agent-2",
                "timestamps": {"created_at": "2024-01-15T10:00:02Z"},
            },
            "payload": {
                "layer": "L1",
                "node": "processor",
                "field": "status",
                "old_value": "pending",
                "new_value": "complete",
            },
        },
    ]


class TestEventParserLayers:
    """Tests for layer extraction."""

    def test_extract_by_layer_l3(self, parser, sample_events):
        """Test extracting L3 events."""
        result = parser.extract_by_layer(sample_events, "L3")
        assert len(result) == 1
        assert result[0]["payload"]["model"] == "gpt-4"

    def test_extract_by_layer_l5a(self, parser, sample_events):
        """Test extracting L5a events."""
        result = parser.extract_by_layer(sample_events, "L5a")
        assert len(result) == 1
        assert result[0]["payload"]["tool_name"] == "search"

    def test_extract_by_layer_l1(self, parser, sample_events):
        """Test extracting L1 events."""
        result = parser.extract_by_layer(sample_events, "L1")
        assert len(result) == 1
        assert result[0]["payload"]["node"] == "processor"

    def test_extract_by_layer_empty(self, parser, sample_events):
        """Test extracting from non-existent layer."""
        result = parser.extract_by_layer(sample_events, "L2")
        assert len(result) == 0

    def test_extract_by_layer_invalid(self, parser, sample_events):
        """Test invalid layer raises error."""
        with pytest.raises(ValueError, match="Invalid layer"):
            parser.extract_by_layer(sample_events, "L99")


class TestEventParserModelInvocations:
    """Tests for model invocation extraction."""

    def test_extract_model_invocations(self, parser, sample_events):
        """Test extracting model invocations."""
        result = parser.extract_model_invocations(sample_events)
        assert len(result) == 1
        assert isinstance(result[0], ModelInvocation)

    def test_model_invocation_fields(self, parser, sample_events):
        """Test model invocation has correct fields."""
        result = parser.extract_model_invocations(sample_events)[0]
        assert result.model == "gpt-4"
        assert result.prompt == "Hello"
        assert result.response == "Hi there!"
        assert result.tokens_in == 10
        assert result.tokens_out == 5
        assert result.latency_ms == 150.0
        assert result.event_id == "span-1"

    def test_model_invocation_empty(self, parser):
        """Test empty list returns empty result."""
        result = parser.extract_model_invocations([])
        assert len(result) == 0

    def test_model_invocation_alternate_fields(self, parser):
        """Test extraction with alternate field names."""
        events = [{
            "event_type": "llm_call",
            "payload": {
                "model": "claude-3",
                "input": "Hi",
                "output": "Hello",
                "input_tokens": 5,
                "output_tokens": 10,
                "duration_ms": 100.0,
            },
        }]
        result = parser.extract_model_invocations(events)
        assert len(result) == 1
        assert result[0].model == "claude-3"
        assert result[0].prompt == "Hi"
        assert result[0].response == "Hello"
        assert result[0].tokens_in == 5
        assert result[0].tokens_out == 10


class TestEventParserToolCalls:
    """Tests for tool call extraction."""

    def test_extract_tool_calls(self, parser, sample_events):
        """Test extracting tool calls."""
        result = parser.extract_tool_calls(sample_events)
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)

    def test_tool_call_fields(self, parser, sample_events):
        """Test tool call has correct fields."""
        result = parser.extract_tool_calls(sample_events)[0]
        assert result.tool_name == "search"
        assert result.inputs == {"query": "test"}
        assert result.output == {"results": ["a", "b"]}
        assert result.latency_ms == 200.0
        assert result.success is True
        assert result.event_id == "span-2"

    def test_tool_call_alternate_fields(self, parser):
        """Test extraction with alternate field names."""
        events = [{
            "event_type": "function_call",
            "payload": {
                "name": "calculator",
                "arguments": {"x": 1, "y": 2},
                "result": 3,
                "duration_ms": 50.0,
            },
        }]
        result = parser.extract_tool_calls(events)
        assert len(result) == 1
        assert result[0].tool_name == "calculator"
        assert result[0].inputs == {"x": 1, "y": 2}
        assert result[0].output == 3


class TestEventParserStateChanges:
    """Tests for state change extraction."""

    def test_extract_state_changes(self, parser, sample_events):
        """Test extracting state changes."""
        result = parser.extract_state_changes(sample_events)
        assert len(result) == 1
        assert isinstance(result[0], StateChange)

    def test_state_change_fields(self, parser, sample_events):
        """Test state change has correct fields."""
        result = parser.extract_state_changes(sample_events)[0]
        assert result.node == "processor"
        assert result.field == "status"
        assert result.event_id == "span-3"
        # Hashes should be computed from old_value/new_value
        assert result.old_hash != ""
        assert result.new_hash != ""
        assert result.old_hash != result.new_hash

    def test_state_change_with_hashes(self, parser):
        """Test state change with pre-computed hashes."""
        events = [{
            "event_type": "state_mutation",
            "payload": {
                "agent": "node-1",
                "key": "data",
                "old_hash": "abc123",
                "new_hash": "def456",
            },
        }]
        result = parser.extract_state_changes(events)
        assert len(result) == 1
        assert result[0].node == "node-1"
        assert result[0].field == "data"
        assert result[0].old_hash == "abc123"
        assert result[0].new_hash == "def456"


class TestEventParserByType:
    """Tests for event type filtering."""

    def test_extract_by_type(self, parser, sample_events):
        """Test filtering by event type."""
        result = parser.extract_by_type(sample_events, "model_invoke")
        assert len(result) == 1
        assert result[0]["payload"]["model"] == "gpt-4"

    def test_extract_by_type_no_match(self, parser, sample_events):
        """Test filtering with no matches."""
        result = parser.extract_by_type(sample_events, "nonexistent")
        assert len(result) == 0


class TestEventParserByAgent:
    """Tests for agent ID filtering."""

    def test_extract_by_agent(self, parser, sample_events):
        """Test filtering by agent ID."""
        result = parser.extract_by_agent(sample_events, "agent-1")
        assert len(result) == 2

    def test_extract_by_agent_specific(self, parser, sample_events):
        """Test filtering specific agent."""
        result = parser.extract_by_agent(sample_events, "agent-2")
        assert len(result) == 1
        assert result[0]["payload"]["node"] == "processor"


class TestDataclasses:
    """Tests for dataclass instantiation."""

    def test_model_invocation_dataclass(self):
        """Test ModelInvocation can be instantiated."""
        mi = ModelInvocation(
            model="gpt-4",
            prompt="test",
            response="ok",
            tokens_in=10,
            tokens_out=5,
            latency_ms=100.0,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert mi.model == "gpt-4"
        assert mi.event_id == ""  # default

    def test_tool_call_dataclass(self):
        """Test ToolCall can be instantiated."""
        tc = ToolCall(
            tool_name="search",
            inputs={"q": "test"},
            output="result",
            latency_ms=50.0,
            timestamp="2024-01-15T10:00:00Z",
        )
        assert tc.tool_name == "search"
        assert tc.success is True  # default
        assert tc.error is None  # default

    def test_state_change_dataclass(self):
        """Test StateChange can be instantiated."""
        sc = StateChange(
            node="processor",
            field="status",
            old_hash="abc",
            new_hash="def",
            timestamp="2024-01-15T10:00:00Z",
        )
        assert sc.node == "processor"
        assert sc.event_id == ""  # default
