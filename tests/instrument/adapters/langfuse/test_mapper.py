"""Tests for Langfuse <-> STRATIX bidirectional mapper."""

import pytest
from datetime import datetime, timezone

from layerlens.instrument.adapters.langfuse.mapper import (
    STRATIXToLangfuseMapper,
    LangfuseToSTRATIXMapper,
)


# --- Sample Langfuse data ---

def _sample_trace(
    trace_id="lf-trace-1",
    input_val="What is AI?",
    output_val="AI is artificial intelligence.",
    observations=None,
    metadata=None,
    tags=None,
):
    """Create a sample Langfuse trace dict."""
    return {
        "id": trace_id,
        "name": "test-agent",
        "input": input_val,
        "output": output_val,
        "timestamp": "2024-06-01T10:00:00+00:00",
        "endTime": "2024-06-01T10:00:05+00:00",
        "observations": observations or [],
        "metadata": metadata or {"env": "test"},
        "tags": tags or ["test"],
        "sessionId": "session-1",
        "userId": "user-1",
        "scores": [{"name": "quality", "value": 0.9}],
    }


def _sample_generation(
    model="gpt-4",
    prompt_tokens=100,
    completion_tokens=50,
    total_cost=0.005,
    level="",
):
    return {
        "id": "gen-1",
        "type": "GENERATION",
        "name": "gpt-4-call",
        "model": model,
        "startTime": "2024-06-01T10:00:01+00:00",
        "endTime": "2024-06-01T10:00:02+00:00",
        "usage": {
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": prompt_tokens + completion_tokens,
        },
        "totalCost": total_cost,
        "level": level,
        "statusMessage": "Error in generation" if level == "ERROR" else "",
        "modelParameters": {"temperature": 0.7},
    }


def _sample_span(name="retrieval", is_tool=False, level=""):
    metadata = {}
    if is_tool:
        metadata["type"] = "TOOL"
    return {
        "id": "span-1",
        "type": "SPAN",
        "name": name,
        "startTime": "2024-06-01T10:00:00.500+00:00",
        "endTime": "2024-06-01T10:00:01+00:00",
        "input": {"query": "test"},
        "output": {"result": "found"},
        "metadata": metadata,
        "level": level,
        "statusMessage": "Span error" if level == "ERROR" else "",
    }


# --- Forward mapping tests (Langfuse -> STRATIX) ---

class TestLangfuseToSTRATIXMapper:

    def setup_method(self):
        self.mapper = LangfuseToSTRATIXMapper()

    def test_map_trace_basic(self):
        """Map a basic trace with input/output."""
        trace = _sample_trace()
        events = self.mapper.map_trace(trace)
        types = [e["event_type"] for e in events]
        assert "agent.input" in types
        assert "agent.output" in types
        assert "environment.config" in types

    def test_map_trace_preserves_trace_id(self):
        trace = _sample_trace(trace_id="lf-123")
        events = self.mapper.map_trace(trace)
        for e in events:
            assert e["trace_id"] == "lf-123"

    def test_map_trace_input_payload(self):
        trace = _sample_trace(input_val="Hello world")
        events = self.mapper.map_trace(trace)
        input_events = [e for e in events if e["event_type"] == "agent.input"]
        assert len(input_events) == 1
        assert input_events[0]["payload"]["input_text"] == "Hello world"
        assert input_events[0]["payload"]["framework"] == "langfuse"

    def test_map_trace_output_payload(self):
        trace = _sample_trace(output_val="Goodbye")
        events = self.mapper.map_trace(trace)
        output_events = [e for e in events if e["event_type"] == "agent.output"]
        assert len(output_events) == 1
        assert output_events[0]["payload"]["output_text"] == "Goodbye"

    def test_map_trace_metadata(self):
        trace = _sample_trace(metadata={"env": "production"})
        events = self.mapper.map_trace(trace)
        config_events = [e for e in events if e["event_type"] == "environment.config"]
        assert len(config_events) == 1
        assert config_events[0]["payload"]["config"]["env"] == "production"

    def test_map_trace_langfuse_metadata_preserved(self):
        trace = _sample_trace()
        events = self.mapper.map_trace(trace)
        input_events = [e for e in events if e["event_type"] == "agent.input"]
        meta = input_events[0].get("metadata", {})
        assert meta.get("langfuse_trace_id") == "lf-trace-1"
        assert meta.get("langfuse_session_id") == "session-1"
        assert meta.get("langfuse_user_id") == "user-1"
        assert meta.get("langfuse_tags") == ["test"]

    def test_map_trace_no_input(self):
        trace = _sample_trace(input_val=None)
        events = self.mapper.map_trace(trace)
        input_events = [e for e in events if e["event_type"] == "agent.input"]
        assert len(input_events) == 0

    def test_map_trace_no_output(self):
        trace = _sample_trace(output_val=None)
        events = self.mapper.map_trace(trace)
        output_events = [e for e in events if e["event_type"] == "agent.output"]
        assert len(output_events) == 0

    def test_map_trace_no_metadata(self):
        trace = _sample_trace()
        trace["metadata"] = None  # Explicitly set to None after creation
        events = self.mapper.map_trace(trace)
        config_events = [e for e in events if e["event_type"] == "environment.config"]
        assert len(config_events) == 0

    def test_map_generation(self):
        """Map a generation observation to model.invoke."""
        gen = _sample_generation()
        trace = _sample_trace(observations=[gen])
        events = self.mapper.map_trace(trace)
        model_events = [e for e in events if e["event_type"] == "model.invoke"]
        assert len(model_events) == 1
        payload = model_events[0]["payload"]
        assert payload["model"] == "gpt-4"
        assert payload["tokens_prompt"] == 100
        assert payload["tokens_completion"] == 50
        assert payload["tokens_total"] == 150

    def test_map_generation_latency(self):
        gen = _sample_generation()
        trace = _sample_trace(observations=[gen])
        events = self.mapper.map_trace(trace)
        model_events = [e for e in events if e["event_type"] == "model.invoke"]
        payload = model_events[0]["payload"]
        assert payload.get("latency_ms") is not None
        assert payload["latency_ms"] == pytest.approx(1000.0, abs=100)

    def test_map_generation_cost(self):
        gen = _sample_generation(total_cost=0.005)
        trace = _sample_trace(observations=[gen])
        events = self.mapper.map_trace(trace)
        cost_events = [e for e in events if e["event_type"] == "cost.record"]
        assert len(cost_events) == 1
        assert cost_events[0]["payload"]["cost_usd"] == 0.005

    def test_map_generation_no_cost(self):
        gen = _sample_generation(total_cost=None)
        gen.pop("totalCost", None)
        trace = _sample_trace(observations=[gen])
        events = self.mapper.map_trace(trace)
        cost_events = [e for e in events if e["event_type"] == "cost.record"]
        assert len(cost_events) == 0

    def test_map_generation_parameters(self):
        gen = _sample_generation()
        trace = _sample_trace(observations=[gen])
        events = self.mapper.map_trace(trace)
        model_events = [e for e in events if e["event_type"] == "model.invoke"]
        assert model_events[0]["payload"]["parameters"] == {"temperature": 0.7}

    def test_map_generation_error(self):
        gen = _sample_generation(level="ERROR")
        trace = _sample_trace(observations=[gen])
        events = self.mapper.map_trace(trace)
        model_events = [e for e in events if e["event_type"] == "model.invoke"]
        assert model_events[0]["payload"]["error"] == "Error in generation"
        violation_events = [e for e in events if e["event_type"] == "policy.violation"]
        assert len(violation_events) >= 1

    def test_map_span_as_agent_code(self):
        """Map a regular span to agent.code."""
        span = _sample_span(name="retrieval")
        trace = _sample_trace(observations=[span])
        events = self.mapper.map_trace(trace)
        code_events = [e for e in events if e["event_type"] == "agent.code"]
        assert len(code_events) == 1
        assert code_events[0]["payload"]["step_name"] == "retrieval"

    def test_map_span_with_io(self):
        span = _sample_span()
        trace = _sample_trace(observations=[span])
        events = self.mapper.map_trace(trace)
        code_events = [e for e in events if e["event_type"] == "agent.code"]
        payload = code_events[0]["payload"]
        assert payload["input"] == {"query": "test"}
        assert payload["output"] == {"result": "found"}

    def test_map_tool_span(self):
        """Map a TOOL-type span to tool.call."""
        span = _sample_span(name="web_search", is_tool=True)
        trace = _sample_trace(observations=[span])
        events = self.mapper.map_trace(trace)
        tool_events = [e for e in events if e["event_type"] == "tool.call"]
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "web_search"

    def test_map_tool_span_by_name_prefix(self):
        """Spans starting with 'tool_' are mapped as tool.call."""
        span = _sample_span(name="tool_search")
        trace = _sample_trace(observations=[span])
        events = self.mapper.map_trace(trace)
        tool_events = [e for e in events if e["event_type"] == "tool.call"]
        assert len(tool_events) == 1

    def test_map_span_error(self):
        span = _sample_span(level="ERROR")
        trace = _sample_trace(observations=[span])
        events = self.mapper.map_trace(trace)
        violation_events = [e for e in events if e["event_type"] == "policy.violation"]
        assert len(violation_events) >= 1
        assert violation_events[0]["payload"]["violation_type"] == "error"

    def test_map_span_warning(self):
        span = _sample_span(level="WARNING")
        trace = _sample_trace(observations=[span])
        events = self.mapper.map_trace(trace)
        violation_events = [e for e in events if e["event_type"] == "policy.violation"]
        assert len(violation_events) >= 1
        assert violation_events[0]["payload"]["violation_type"] == "warning"

    def test_observations_sorted_by_time(self):
        """Observations are sorted by startTime before mapping."""
        gen = _sample_generation()
        gen["startTime"] = "2024-06-01T10:00:02+00:00"
        span = _sample_span()
        span["startTime"] = "2024-06-01T10:00:01+00:00"
        trace = _sample_trace(observations=[gen, span])
        events = self.mapper.map_trace(trace)
        # Span should come before generation in non-input/output events
        non_io = [e for e in events if e["event_type"] not in ("agent.input", "agent.output", "environment.config")]
        if len(non_io) >= 2:
            assert non_io[0]["sequence_id"] < non_io[-1]["sequence_id"]

    def test_map_full_trace(self):
        """Map a complete trace with generation + span + tool."""
        gen = _sample_generation()
        span = _sample_span(name="processing")
        tool_span = _sample_span(name="tool_search", is_tool=True)
        trace = _sample_trace(observations=[gen, span, tool_span])
        events = self.mapper.map_trace(trace)
        types = {e["event_type"] for e in events}
        assert "agent.input" in types
        assert "agent.output" in types
        assert "model.invoke" in types
        assert "agent.code" in types
        assert "tool.call" in types
        assert "cost.record" in types
        assert "environment.config" in types

    def test_sequence_ids_monotonic(self):
        gen = _sample_generation()
        span = _sample_span()
        trace = _sample_trace(observations=[gen, span])
        events = self.mapper.map_trace(trace)
        seq_ids = [e["sequence_id"] for e in events]
        assert seq_ids == sorted(seq_ids)

    def test_map_dict_input(self):
        """Dict inputs are JSON-serialized to input_text."""
        trace = _sample_trace(input_val={"messages": [{"role": "user", "content": "hi"}]})
        events = self.mapper.map_trace(trace)
        input_events = [e for e in events if e["event_type"] == "agent.input"]
        assert isinstance(input_events[0]["payload"]["input_text"], str)

    def test_map_empty_observations(self):
        trace = _sample_trace(observations=[])
        events = self.mapper.map_trace(trace)
        types = [e["event_type"] for e in events]
        assert "agent.input" in types
        assert "agent.output" in types


# --- Reverse mapping tests (STRATIX -> Langfuse) ---

class TestSTRATIXToLangfuseMapper:

    def setup_method(self):
        self.mapper = STRATIXToLangfuseMapper()

    def _make_event(self, event_type, payload, trace_id="t1"):
        return {
            "event_type": event_type,
            "payload": payload,
            "trace_id": trace_id,
            "timestamp": "2024-06-01T10:00:00+00:00",
        }

    def test_map_agent_input(self):
        events = [
            self._make_event("agent.input", {"input_text": "Hello", "agent_id": "my-agent"}),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        assert result["trace"]["input"] == "Hello"
        assert result["trace"]["name"] == "my-agent"

    def test_map_agent_output(self):
        events = [
            self._make_event("agent.output", {"output_text": "Goodbye"}),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        assert result["trace"]["output"] == "Goodbye"

    def test_map_model_invoke_to_generation(self):
        events = [
            self._make_event("model.invoke", {
                "model": "gpt-4",
                "tokens_prompt": 100,
                "tokens_completion": 50,
                "tokens_total": 150,
                "parameters": {"temperature": 0.5},
            }),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        obs = result["observations"]
        assert len(obs) == 1
        assert obs[0]["type"] == "GENERATION"
        assert obs[0]["model"] == "gpt-4"
        assert obs[0]["usage"]["promptTokens"] == 100
        assert obs[0]["modelParameters"] == {"temperature": 0.5}

    def test_map_model_invoke_error(self):
        events = [
            self._make_event("model.invoke", {"model": "gpt-4", "error": "timeout"}),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        obs = result["observations"]
        assert obs[0]["level"] == "ERROR"
        assert obs[0]["statusMessage"] == "timeout"

    def test_map_tool_call_to_span(self):
        events = [
            self._make_event("tool.call", {
                "tool_name": "web_search",
                "input": {"query": "test"},
                "output": {"result": "found"},
            }),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        obs = result["observations"]
        assert len(obs) == 1
        assert obs[0]["type"] == "SPAN"
        assert obs[0]["name"] == "web_search"
        assert obs[0]["metadata"]["type"] == "TOOL"

    def test_map_tool_call_error(self):
        events = [
            self._make_event("tool.call", {"tool_name": "search", "error": "not found"}),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        obs = result["observations"]
        assert obs[0]["level"] == "ERROR"

    def test_map_agent_code_to_span(self):
        events = [
            self._make_event("agent.code", {
                "step_name": "preprocessing",
                "input": "raw data",
                "output": "processed data",
            }),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        obs = result["observations"]
        assert len(obs) == 1
        assert obs[0]["type"] == "SPAN"
        assert obs[0]["name"] == "preprocessing"

    def test_map_cost_attached_to_generation(self):
        events = [
            self._make_event("model.invoke", {"model": "gpt-4"}),
            self._make_event("cost.record", {"model": "gpt-4", "cost_usd": 0.01}),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        obs = result["observations"]
        assert obs[0].get("totalCost") == 0.01

    def test_map_cost_no_matching_generation(self):
        """Cost record without a generation is silently ignored."""
        events = [
            self._make_event("cost.record", {"model": "gpt-4", "cost_usd": 0.01}),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        assert result["observations"] == []

    def test_map_environment_config(self):
        events = [
            self._make_event("environment.config", {"config": {"env": "prod"}}),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        assert result["trace"]["metadata"]["environment_config"] == {"env": "prod"}

    def test_map_handoff(self):
        events = [
            self._make_event("agent.handoff", {
                "from_agent": "manager", "to_agent": "worker", "context": "task-1",
            }),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        obs = result["observations"]
        assert len(obs) == 1
        assert "handoff" in obs[0]["name"]
        assert obs[0]["metadata"]["from_agent"] == "manager"

    def test_map_state_change(self):
        events = [
            self._make_event("agent.state.change", {
                "state_type": "task", "before": "pending", "after": "running",
            }),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        obs = result["observations"]
        assert len(obs) == 1
        assert "state-change" in obs[0]["name"]

    def test_exported_trace_tagged(self):
        events = [
            self._make_event("agent.input", {"input_text": "test"}),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="t1")
        assert "stratix-exported" in result["trace"]["tags"]

    def test_stratix_trace_id_in_metadata(self):
        events = [
            self._make_event("agent.input", {"input_text": "test"}),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="my-trace")
        assert result["trace"]["metadata"]["stratix_trace_id"] == "my-trace"

    def test_full_roundtrip_event_types(self):
        """Map a full set of STRATIX events and verify all observations created."""
        events = [
            self._make_event("agent.input", {"input_text": "Hello", "agent_id": "bot"}),
            self._make_event("agent.code", {"step_name": "preprocess"}),
            self._make_event("model.invoke", {"model": "gpt-4", "tokens_prompt": 10, "tokens_completion": 5}),
            self._make_event("tool.call", {"tool_name": "search"}),
            self._make_event("cost.record", {"model": "gpt-4", "cost_usd": 0.001}),
            self._make_event("environment.config", {"config": {"key": "val"}}),
            self._make_event("agent.handoff", {"from_agent": "a", "to_agent": "b"}),
            self._make_event("agent.state.change", {"state_type": "x"}),
            self._make_event("agent.output", {"output_text": "Bye"}),
        ]
        result = self.mapper.map_events_to_trace(events, trace_id="full-test")
        trace = result["trace"]
        obs = result["observations"]

        assert trace["input"] == "Hello"
        assert trace["output"] == "Bye"
        assert trace["name"] == "bot"
        # 5 observations: code, generation, tool, handoff, state
        assert len(obs) == 5
        obs_types = [o["type"] for o in obs]
        assert obs_types.count("GENERATION") == 1
        assert obs_types.count("SPAN") == 4
