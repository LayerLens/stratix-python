"""Tests for TraceBuilder fluent API."""

from layerlens.instrument.simulators.span_model import SpanStatus, SpanType
from layerlens.instrument.simulators.trace_builder import TraceBuilder


class TestTraceBuilder:
    def test_basic_build(self):
        trace = (
            TraceBuilder(seed=42)
            .with_scenario("customer_service")
            .with_source("openai")
            .add_agent_span("Test_Agent")
            .add_llm_span(provider="openai", model="gpt-4o")
            .build()
        )
        assert trace.trace_id is not None
        assert trace.scenario == "customer_service"
        assert trace.source_format == "openai"
        assert trace.span_count == 2

    def test_span_types(self):
        trace = (
            TraceBuilder(seed=42)
            .add_agent_span("Agent")
            .add_llm_span()
            .add_tool_span(name="Tool")
            .add_evaluation_span(dimension="accuracy", score=0.9)
            .build()
        )
        types = [s.span_type for s in trace.spans]
        assert types == [SpanType.AGENT, SpanType.LLM, SpanType.TOOL, SpanType.EVALUATION]

    def test_parent_child_relationship(self):
        trace = (
            TraceBuilder(seed=42)
            .add_agent_span("Agent")
            .add_llm_span()
            .add_tool_span(name="Tool")
            .build()
        )
        agent = trace.spans[0]
        llm = trace.spans[1]
        tool = trace.spans[2]
        assert agent.parent_span_id is None
        assert llm.parent_span_id == agent.span_id
        assert tool.parent_span_id == agent.span_id

    def test_llm_span_defaults(self):
        trace = TraceBuilder(seed=42).add_agent_span("A").add_llm_span().build()
        llm = trace.spans[1]
        assert llm.provider == "openai"
        assert llm.model == "gpt-4o"
        assert llm.token_usage is not None
        assert llm.token_usage.prompt_tokens == 200
        assert llm.token_usage.completion_tokens == 150
        assert llm.finish_reasons == ["stop"]
        assert llm.response_id.startswith("chatcmpl-")

    def test_llm_span_custom(self):
        trace = (
            TraceBuilder(seed=42)
            .add_agent_span("A")
            .add_llm_span(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                prompt_tokens=500,
                completion_tokens=300,
                temperature=0.5,
                max_tokens=1000,
                finish_reasons=["end_turn"],
            )
            .build()
        )
        llm = trace.spans[1]
        assert llm.provider == "anthropic"
        assert llm.model == "claude-sonnet-4-20250514"
        assert llm.token_usage.prompt_tokens == 500
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1000

    def test_tool_span(self):
        trace = (
            TraceBuilder(seed=42)
            .add_agent_span("A")
            .add_tool_span(
                name="Get_Order",
                description="Retrieve order details",
                tool_input={"order_id": "123"},
                tool_output={"status": "shipped"},
            )
            .build()
        )
        tool = trace.spans[1]
        assert tool.tool_name == "Get_Order"
        assert tool.tool_description == "Retrieve order details"
        assert tool.tool_call_id.startswith("call_")
        assert tool.tool_input == {"order_id": "123"}
        assert tool.tool_output == {"status": "shipped"}

    def test_evaluation_span(self):
        trace = (
            TraceBuilder(seed=42)
            .add_agent_span("A")
            .add_evaluation_span(dimension="factual_accuracy", score=0.92)
            .build()
        )
        eval_span = trace.spans[1]
        assert eval_span.eval_dimension == "factual_accuracy"
        assert eval_span.eval_score == 0.92
        assert eval_span.eval_label == "pass"

    def test_evaluation_fail_label(self):
        trace = (
            TraceBuilder(seed=42)
            .add_agent_span("A")
            .add_evaluation_span(dimension="safety", score=0.3)
            .build()
        )
        assert trace.spans[1].eval_label == "fail"

    def test_with_error(self):
        trace = (
            TraceBuilder(seed=42)
            .add_agent_span("A")
            .add_llm_span()
            .with_error(error_type="rate_limit", span_index=-1)
            .build()
        )
        llm = trace.spans[1]
        assert llm.error_type == "rate_limit"
        assert llm.status == SpanStatus.ERROR
        assert llm.http_status_code == 429

    def test_with_streaming(self):
        trace = (
            TraceBuilder(seed=42)
            .add_agent_span("A")
            .add_llm_span(completion_tokens=100)
            .with_streaming(ttft_ms=120.0, tpot_ms=35.0)
            .build()
        )
        llm = trace.spans[1]
        assert llm.is_streaming is True
        assert llm.ttft_ms == 120.0
        assert llm.tpot_ms == 35.0
        assert llm.chunk_count == 20  # 100 / 5

    def test_with_session(self):
        trace = (
            TraceBuilder(seed=42)
            .with_session(turn=2)
            .add_agent_span("A")
            .build()
        )
        assert trace.session_id is not None
        assert trace.turn_number == 2

    def test_deterministic_build(self):
        def build():
            return (
                TraceBuilder(seed=42)
                .add_agent_span("A")
                .add_llm_span()
                .add_tool_span(name="T")
                .build()
            )

        t1 = build()
        t2 = build()
        assert t1.trace_id == t2.trace_id
        assert t1.spans[0].span_id == t2.spans[0].span_id
        assert t1.spans[1].start_time_unix_nano == t2.spans[1].start_time_unix_nano

    def test_agent_span_encompasses_children(self):
        trace = (
            TraceBuilder(seed=42)
            .add_agent_span("A")
            .add_llm_span(duration_ms=1000.0)
            .add_tool_span(name="T", latency_ms=500.0)
            .add_llm_span(duration_ms=800.0)
            .build()
        )
        agent = trace.spans[0]
        max_child_end = max(s.end_time_unix_nano for s in trace.spans[1:])
        assert agent.end_time_unix_nano > max_child_end

    def test_complex_trace(self):
        trace = (
            TraceBuilder(seed=42)
            .with_scenario("customer_service", topic="Shipping_Delay")
            .with_source("agentforce_otlp")
            .add_agent_span("Case_Resolution_Agent")
            .add_llm_span(
                provider="openai",
                model="gpt-4o",
                prompt_tokens=250,
                completion_tokens=180,
            )
            .add_tool_span(name="Get_Order_Details", latency_ms=350.0)
            .add_llm_span(
                provider="openai",
                model="gpt-4o",
                prompt_tokens=400,
                completion_tokens=220,
            )
            .add_evaluation_span(dimension="factual_accuracy", score=0.92)
            .with_error(error_type="rate_limit", span_index=-2)
            .with_streaming(ttft_ms=120.0, tpot_ms=35.0)
            .build()
        )
        assert trace.span_count == 5
        assert trace.scenario == "customer_service"
        assert trace.topic == "Shipping_Delay"
        assert trace.source_format == "agentforce_otlp"
        assert trace.total_tokens == 250 + 180 + 400 + 220

    def test_with_content(self):
        trace = (
            TraceBuilder(seed=42)
            .add_agent_span("A")
            .add_llm_span()
            .with_content(
                span_index=1,
                input_messages=[{"role": "user", "content": "Hello"}],
                output_message={"role": "assistant", "content": "Hi!"},
            )
            .build()
        )
        llm = trace.spans[1]
        assert len(llm.input_messages) == 1
        assert llm.output_message["content"] == "Hi!"

    def test_empty_trace(self):
        trace = TraceBuilder(seed=42).build()
        assert trace.span_count == 0
        assert trace.trace_id is not None
