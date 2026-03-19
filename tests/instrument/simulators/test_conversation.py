"""Tests for multi-turn conversation builder."""

import pytest

from layerlens.instrument.simulators.config import ConversationConfig
from layerlens.instrument.simulators.content.template_provider import TemplateContentProvider
from layerlens.instrument.simulators.conversation import ConversationBuilder
from layerlens.instrument.simulators.span_model import SpanType


class TestConversationBuilder:
    def test_disabled_returns_empty(self):
        config = ConversationConfig(enabled=False)
        builder = ConversationBuilder(config, seed=42)
        provider = TemplateContentProvider(seed=42)
        traces = builder.build_conversation(
            scenario="customer_service",
            topic="Shipping_Delay",
            provider="openai",
            model="gpt-4o",
            content_provider=provider,
        )
        assert traces == []

    def test_generates_traces(self):
        config = ConversationConfig(enabled=True, turns_min=3, turns_max=3)
        builder = ConversationBuilder(config, seed=42)
        provider = TemplateContentProvider(seed=42)
        traces = builder.build_conversation(
            scenario="customer_service",
            topic="Shipping_Delay",
            provider="openai",
            model="gpt-4o",
            content_provider=provider,
        )
        assert len(traces) == 3

    def test_shared_session_id(self):
        config = ConversationConfig(enabled=True, turns_min=3, turns_max=3)
        builder = ConversationBuilder(config, seed=42)
        provider = TemplateContentProvider(seed=42)
        traces = builder.build_conversation(
            scenario="customer_service",
            topic="Shipping_Delay",
            provider="openai",
            model="gpt-4o",
            content_provider=provider,
        )
        session_ids = {t.session_id for t in traces}
        assert len(session_ids) == 1
        assert None not in session_ids

    def test_turns_in_range(self):
        config = ConversationConfig(enabled=True, turns_min=2, turns_max=5)
        builder = ConversationBuilder(config, seed=42)
        provider = TemplateContentProvider(seed=42)
        traces = builder.build_conversation(
            scenario="customer_service",
            topic="Shipping_Delay",
            provider="openai",
            model="gpt-4o",
            content_provider=provider,
        )
        assert 2 <= len(traces) <= 5

    def test_unique_trace_ids(self):
        config = ConversationConfig(enabled=True, turns_min=4, turns_max=4)
        builder = ConversationBuilder(config, seed=42)
        provider = TemplateContentProvider(seed=42)
        traces = builder.build_conversation(
            scenario="customer_service",
            topic="Shipping_Delay",
            provider="openai",
            model="gpt-4o",
            content_provider=provider,
        )
        ids = [t.trace_id for t in traces]
        assert len(ids) == len(set(ids))

    def test_each_trace_has_agent_span(self):
        config = ConversationConfig(enabled=True, turns_min=3, turns_max=3)
        builder = ConversationBuilder(config, seed=42)
        provider = TemplateContentProvider(seed=42)
        traces = builder.build_conversation(
            scenario="customer_service",
            topic="Shipping_Delay",
            provider="openai",
            model="gpt-4o",
            content_provider=provider,
        )
        for trace in traces:
            agent_spans = [s for s in trace.spans if s.span_type == SpanType.AGENT]
            assert len(agent_spans) >= 1

    def test_each_trace_has_llm_spans(self):
        config = ConversationConfig(enabled=True, turns_min=3, turns_max=3)
        builder = ConversationBuilder(config, seed=42)
        provider = TemplateContentProvider(seed=42)
        traces = builder.build_conversation(
            scenario="customer_service",
            topic="Shipping_Delay",
            provider="openai",
            model="gpt-4o",
            content_provider=provider,
        )
        for trace in traces:
            llm_spans = [s for s in trace.spans if s.span_type == SpanType.LLM]
            assert len(llm_spans) >= 2  # At least 2 LLM calls per turn

    def test_tool_spans_on_first_last_turn(self):
        config = ConversationConfig(enabled=True, turns_min=3, turns_max=3)
        builder = ConversationBuilder(config, seed=42)
        provider = TemplateContentProvider(seed=42)
        traces = builder.build_conversation(
            scenario="customer_service",
            topic="Shipping_Delay",
            provider="openai",
            model="gpt-4o",
            content_provider=provider,
        )
        # First turn should have tool spans
        first_tools = [s for s in traces[0].spans if s.span_type == SpanType.TOOL]
        assert len(first_tools) >= 1
        # Last turn should have tool spans
        last_tools = [s for s in traces[-1].spans if s.span_type == SpanType.TOOL]
        assert len(last_tools) >= 1

    def test_evaluation_on_last_turn_only(self):
        config = ConversationConfig(enabled=True, turns_min=3, turns_max=3)
        builder = ConversationBuilder(config, seed=42)
        provider = TemplateContentProvider(seed=42)
        traces = builder.build_conversation(
            scenario="customer_service",
            topic="Shipping_Delay",
            provider="openai",
            model="gpt-4o",
            content_provider=provider,
        )
        # Non-last turns: no eval
        for trace in traces[:-1]:
            eval_spans = [s for s in trace.spans if s.span_type == SpanType.EVALUATION]
            assert len(eval_spans) == 0
        # Last turn: has eval
        eval_spans = [s for s in traces[-1].spans if s.span_type == SpanType.EVALUATION]
        assert len(eval_spans) == 1

    def test_include_content(self):
        config = ConversationConfig(enabled=True, turns_min=2, turns_max=2)
        builder = ConversationBuilder(config, seed=42)
        provider = TemplateContentProvider(seed=42)
        traces = builder.build_conversation(
            scenario="customer_service",
            topic="Shipping_Delay",
            provider="openai",
            model="gpt-4o",
            content_provider=provider,
            include_content=True,
        )
        # LLM spans should have input messages
        for trace in traces:
            llm_spans = [s for s in trace.spans if s.span_type == SpanType.LLM]
            first_llm = llm_spans[0]
            assert first_llm.input_messages is not None
            assert len(first_llm.input_messages) > 0

    def test_no_content_by_default(self):
        config = ConversationConfig(enabled=True, turns_min=2, turns_max=2)
        builder = ConversationBuilder(config, seed=42)
        provider = TemplateContentProvider(seed=42)
        traces = builder.build_conversation(
            scenario="customer_service",
            topic="Shipping_Delay",
            provider="openai",
            model="gpt-4o",
            content_provider=provider,
            include_content=False,
        )
        for trace in traces:
            llm_spans = [s for s in trace.spans if s.span_type == SpanType.LLM]
            first_llm = llm_spans[0]
            assert not first_llm.input_messages

    def test_deterministic(self):
        config = ConversationConfig(enabled=True, turns_min=3, turns_max=5)

        def run():
            builder = ConversationBuilder(config, seed=42)
            provider = TemplateContentProvider(seed=42)
            return builder.build_conversation(
                scenario="sales",
                topic="Pricing_Inquiry",
                provider="openai",
                model="gpt-4o",
                content_provider=provider,
            )

        r1 = run()
        r2 = run()
        assert len(r1) == len(r2)
        for t1, t2 in zip(r1, r2):
            assert len(t1.spans) == len(t2.spans)
            for s1, s2 in zip(t1.spans, t2.spans):
                assert s1.span_type == s2.span_type

    def test_different_scenarios(self):
        """Conversation builder works with all 5 scenarios."""
        config = ConversationConfig(enabled=True, turns_min=2, turns_max=2)
        scenarios = [
            ("customer_service", "Shipping_Delay"),
            ("sales", "Pricing_Inquiry"),
            ("order_management", "Order_Tracking"),
            ("knowledge_faq", "Policy_Question"),
            ("it_helpdesk", "Password_Reset"),
        ]
        for scenario, topic in scenarios:
            builder = ConversationBuilder(config, seed=42)
            provider = TemplateContentProvider(seed=42)
            traces = builder.build_conversation(
                scenario=scenario,
                topic=topic,
                provider="openai",
                model="gpt-4o",
                content_provider=provider,
            )
            assert len(traces) == 2, f"Failed for {scenario}"

    def test_growing_prompt_tokens(self):
        """Later turns should have more prompt tokens (growing context)."""
        config = ConversationConfig(enabled=True, turns_min=4, turns_max=4)
        builder = ConversationBuilder(config, seed=42)
        provider = TemplateContentProvider(seed=42)
        traces = builder.build_conversation(
            scenario="customer_service",
            topic="Shipping_Delay",
            provider="openai",
            model="gpt-4o",
            content_provider=provider,
        )
        # Get first LLM span's prompt tokens from each turn
        prompt_tokens = []
        for trace in traces:
            llm_spans = [s for s in trace.spans if s.span_type == SpanType.LLM]
            if llm_spans and llm_spans[0].token_usage:
                prompt_tokens.append(llm_spans[0].token_usage.prompt_tokens)
        assert len(prompt_tokens) == 4
        # The minimum range increases by 50 per turn (turn * 50 added),
        # so last turn tokens should exceed first turn tokens
        assert prompt_tokens[-1] > prompt_tokens[0], (
            f"Last turn tokens ({prompt_tokens[-1]}) should exceed "
            f"first turn tokens ({prompt_tokens[0]})"
        )
