"""Tests for 3 output formatters."""

import json

import pytest

from layerlens.instrument.simulators.outputs import get_output_formatter, list_outputs
from layerlens.instrument.simulators.outputs.base import BaseOutputFormatter
from layerlens.instrument.simulators.span_model import (
    SimulatedSpan,
    SimulatedTrace,
    SpanKind,
    SpanType,
    TokenUsage,
)


def _make_trace() -> SimulatedTrace:
    agent_span = SimulatedSpan(
        span_id="agent001",
        span_type=SpanType.AGENT,
        name="agent Test_Agent",
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_005_000_000_000,
        kind=SpanKind.SERVER,
        agent_name="Test_Agent",
        attributes={
            "gen_ai.agent.name": "Test_Agent",
        },
    )
    llm_span = SimulatedSpan(
        span_id="llm001",
        parent_span_id="agent001",
        span_type=SpanType.LLM,
        name="chat gpt-4o",
        start_time_unix_nano=1_700_000_000_100_000_000,
        end_time_unix_nano=1_700_000_001_500_000_000,
        kind=SpanKind.CLIENT,
        provider="openai",
        model="gpt-4o",
        token_usage=TokenUsage(prompt_tokens=250, completion_tokens=180),
        finish_reasons=["stop"],
        response_id="chatcmpl-abc123",
        temperature=0.7,
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.input_tokens": 250,
            "gen_ai.usage.output_tokens": 180,
            "gen_ai.response.finish_reasons": ["stop"],
        },
    )
    tool_span = SimulatedSpan(
        span_id="tool001",
        parent_span_id="agent001",
        span_type=SpanType.TOOL,
        name="tool Get_Order",
        start_time_unix_nano=1_700_000_001_600_000_000,
        end_time_unix_nano=1_700_000_002_000_000_000,
        kind=SpanKind.INTERNAL,
        tool_name="Get_Order",
        tool_call_id="call_xyz",
        attributes={"gen_ai.tool.name": "Get_Order"},
    )
    eval_span = SimulatedSpan(
        span_id="eval001",
        parent_span_id="agent001",
        span_type=SpanType.EVALUATION,
        name="evaluation accuracy",
        start_time_unix_nano=1_700_000_002_100_000_000,
        end_time_unix_nano=1_700_000_002_500_000_000,
        kind=SpanKind.INTERNAL,
        eval_dimension="factual_accuracy",
        eval_score=0.92,
        eval_label="pass",
        attributes={
            "gen_ai.evaluation.score.value": 0.92,
            "gen_ai.evaluation.name": "factual_accuracy",
        },
    )
    return SimulatedTrace(
        trace_id="aabbccdd" * 4,
        spans=[agent_span, llm_span, tool_span, eval_span],
        source_format="openai",
        scenario="customer_service",
        topic="Shipping_Delay",
        resource_attributes={
            "service.name": "test-service",
            "telemetry.sdk.name": "opentelemetry",
        },
        scope_name="stratix.openai",
        scope_version="0.1.0",
    )


class TestOutputRegistry:
    def test_list_outputs_has_3(self):
        outputs = list_outputs()
        assert len(outputs) == 3

    def test_all_outputs_retrievable(self):
        for name in list_outputs():
            formatter = get_output_formatter(name)
            assert isinstance(formatter, BaseOutputFormatter)

    def test_unknown_output_raises(self):
        with pytest.raises(ValueError, match="Unknown output"):
            get_output_formatter("nonexistent")


class TestOTLPJSONOutput:
    def test_structure(self):
        formatter = get_output_formatter("otlp_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        assert "resourceSpans" in result
        rs = result["resourceSpans"]
        assert len(rs) == 1
        assert "resource" in rs[0]
        assert "scopeSpans" in rs[0]

    def test_resource_attributes(self):
        formatter = get_output_formatter("otlp_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        resource = result["resourceSpans"][0]["resource"]
        assert "attributes" in resource
        attr_keys = {a["key"] for a in resource["attributes"]}
        assert "service.name" in attr_keys

    def test_scope(self):
        formatter = get_output_formatter("otlp_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        scope_spans = result["resourceSpans"][0]["scopeSpans"][0]
        assert scope_spans["scope"]["name"] == "stratix.openai"
        assert scope_spans["scope"]["version"] == "0.1.0"

    def test_spans(self):
        formatter = get_output_formatter("otlp_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 4

    def test_span_fields(self):
        formatter = get_output_formatter("otlp_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
        llm_span = spans[1]
        assert llm_span["traceId"] == trace.trace_id
        assert llm_span["spanId"] == "llm001"
        assert llm_span["parentSpanId"] == "agent001"
        assert llm_span["name"] == "chat gpt-4o"
        assert llm_span["kind"] == 3  # CLIENT
        assert "startTimeUnixNano" in llm_span
        assert "endTimeUnixNano" in llm_span

    def test_status(self):
        formatter = get_output_formatter("otlp_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert spans[0]["status"]["code"] == 1  # OK

    def test_attribute_encoding(self):
        formatter = get_output_formatter("otlp_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
        llm_attrs = spans[1]["attributes"]
        # Find gen_ai.system attribute
        system_attr = next(
            (a for a in llm_attrs if a["key"] == "gen_ai.system"), None
        )
        assert system_attr is not None
        assert system_attr["value"]["stringValue"] == "openai"

    def test_int_attribute_encoding(self):
        formatter = get_output_formatter("otlp_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        spans = result["resourceSpans"][0]["scopeSpans"][0]["spans"]
        llm_attrs = spans[1]["attributes"]
        tokens_attr = next(
            (a for a in llm_attrs if a["key"] == "gen_ai.usage.input_tokens"), None
        )
        assert tokens_attr is not None
        assert tokens_attr["value"]["intValue"] == "250"

    def test_batch_format(self):
        formatter = get_output_formatter("otlp_json")
        traces = [_make_trace(), _make_trace()]
        result = formatter.format_batch(traces)
        assert len(result) == 2

    def test_json_serializable(self):
        formatter = get_output_formatter("otlp_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        serialized = json.dumps(result)
        assert len(serialized) > 0
        restored = json.loads(serialized)
        assert "resourceSpans" in restored


class TestLangfuseJSONOutput:
    def test_structure(self):
        formatter = get_output_formatter("langfuse_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        assert "id" in result
        assert "observations" in result
        assert "metadata" in result

    def test_trace_metadata(self):
        formatter = get_output_formatter("langfuse_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        assert result["metadata"]["scenario"] == "customer_service"
        assert result["metadata"]["topic"] == "Shipping_Delay"

    def test_observations(self):
        formatter = get_output_formatter("langfuse_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        obs = result["observations"]
        assert len(obs) == 4
        # LLM observation should be GENERATION type
        llm_obs = obs[1]
        assert llm_obs["type"] == "GENERATION"
        assert llm_obs["model"] == "gpt-4o"
        assert "usage" in llm_obs

    def test_observation_parent(self):
        formatter = get_output_formatter("langfuse_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        obs = result["observations"]
        assert "parentObservationId" not in obs[0]  # agent (root)
        assert obs[1]["parentObservationId"] == "agent001"  # llm

    def test_tags(self):
        formatter = get_output_formatter("langfuse_json")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        assert "customer_service" in result["tags"]

    def test_json_serializable(self):
        formatter = get_output_formatter("langfuse_json")
        result = formatter.format_trace(_make_trace())
        serialized = json.dumps(result)
        assert len(serialized) > 0


class TestSTRATIXNativeOutput:
    def test_structure(self):
        formatter = get_output_formatter("stratix_native")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        assert "trace_id" in result
        assert "events" in result
        assert "scenario" in result

    def test_event_types(self):
        formatter = get_output_formatter("stratix_native")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        event_types = [e["event_type"] for e in result["events"]]
        assert "agent.input" in event_types
        assert "model.invoke" in event_types
        assert "tool.call" in event_types
        assert "evaluation.result" in event_types

    def test_identity(self):
        formatter = get_output_formatter("stratix_native")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        llm_event = result["events"][1]
        assert llm_event["identity"]["trace_id"] == trace.trace_id
        assert llm_event["identity"]["span_id"] == "llm001"
        assert llm_event["identity"]["parent_span_id"] == "agent001"

    def test_model_invoke_payload(self):
        formatter = get_output_formatter("stratix_native")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        llm_event = result["events"][1]
        payload = llm_event["payload"]
        assert payload["provider"] == "openai"
        assert payload["model"] == "gpt-4o"
        assert payload["prompt_tokens"] == 250
        assert payload["completion_tokens"] == 180

    def test_tool_call_payload(self):
        formatter = get_output_formatter("stratix_native")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        tool_event = result["events"][2]
        assert tool_event["payload"]["tool_name"] == "Get_Order"

    def test_evaluation_payload(self):
        formatter = get_output_formatter("stratix_native")
        trace = _make_trace()
        result = formatter.format_trace(trace)
        eval_event = result["events"][3]
        assert eval_event["payload"]["dimension"] == "factual_accuracy"
        assert eval_event["payload"]["score"] == 0.92

    def test_json_serializable(self):
        formatter = get_output_formatter("stratix_native")
        result = formatter.format_trace(_make_trace())
        serialized = json.dumps(result)
        assert len(serialized) > 0


class TestAllOutputsCommon:
    @pytest.mark.parametrize("output_name", list_outputs())
    def test_format_trace(self, output_name):
        formatter = get_output_formatter(output_name)
        trace = _make_trace()
        result = formatter.format_trace(trace)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("output_name", list_outputs())
    def test_format_batch(self, output_name):
        formatter = get_output_formatter(output_name)
        traces = [_make_trace(), _make_trace()]
        result = formatter.format_batch(traces)
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.parametrize("output_name", list_outputs())
    def test_json_serializable(self, output_name):
        formatter = get_output_formatter(output_name)
        result = formatter.format_trace(_make_trace())
        json.dumps(result)  # Should not raise
