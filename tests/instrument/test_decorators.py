"""Tests for STRATIX Python SDK Decorators."""

import pytest

from layerlens.instrument import STRATIX, trace_tool, trace_model


class TestTraceToolDecorator:
    """Tests for the @trace_tool decorator."""

    def test_basic_tool_tracing(self):
        """Test that trace_tool captures function execution."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        @stratix.trace_tool(name="lookup_order", version="1.0.0")
        def lookup_order(order_id: str) -> dict:
            return {"order_id": order_id, "status": "shipped"}

        ctx = stratix.start_trial()

        # Call the decorated function
        result = lookup_order("ORD-123")

        assert result == {"order_id": "ORD-123", "status": "shipped"}

        # Check event was emitted
        events = stratix.get_events()
        assert len(events) == 1

        event = events[0]
        assert event.payload.event_type == "tool.call"
        assert event.payload.tool.name == "lookup_order"
        assert event.payload.tool.version == "1.0.0"

    def test_tool_captures_input(self):
        """Test that trace_tool captures function input."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        @stratix.trace_tool(name="process_data")
        def process_data(x: int, y: int, multiplier: float = 1.0) -> float:
            return (x + y) * multiplier

        ctx = stratix.start_trial()

        result = process_data(10, 20, multiplier=2.0)

        assert result == 60.0

        events = stratix.get_events()
        event = events[0]

        # Check input was captured (ToolCallEvent has .input field)
        assert event.payload.input is not None
        assert "args" in event.payload.input or "kwargs" in event.payload.input

    def test_tool_captures_output(self):
        """Test that trace_tool captures function output."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        @stratix.trace_tool(name="get_user")
        def get_user(user_id: str) -> dict:
            return {"id": user_id, "name": "John Doe"}

        ctx = stratix.start_trial()

        result = get_user("user-123")

        events = stratix.get_events()
        event = events[0]

        # Check output was captured (ToolCallEvent has .output field)
        assert event.payload.output is not None
        assert "result" in event.payload.output

    def test_tool_captures_exception(self):
        """Test that trace_tool captures exceptions."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        @stratix.trace_tool(name="failing_tool")
        def failing_tool() -> None:
            raise ValueError("Something went wrong")

        ctx = stratix.start_trial()

        # Call should raise
        with pytest.raises(ValueError, match="Something went wrong"):
            failing_tool()

        # Event should still be emitted with error
        events = stratix.get_events()
        assert len(events) == 1

        event = events[0]
        assert event.payload.error == "Something went wrong"

    def test_tool_captures_latency(self):
        """Test that trace_tool captures execution latency."""
        import time

        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        @stratix.trace_tool(name="slow_tool")
        def slow_tool() -> str:
            time.sleep(0.1)
            return "done"

        ctx = stratix.start_trial()

        result = slow_tool()

        events = stratix.get_events()
        event = events[0]

        # Latency should be at least 100ms
        assert event.payload.latency_ms >= 100

    def test_tool_without_context(self):
        """Test that trace_tool works without context (just runs function)."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        @stratix.trace_tool(name="my_tool")
        def my_tool(x: int) -> int:
            return x * 2

        # Call without starting a trial
        result = my_tool(5)

        assert result == 10  # Function still works

    def test_integration_type(self):
        """Test setting integration type for tool."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        @stratix.trace_tool(name="external_api", integration="service")
        def call_api() -> dict:
            return {"response": "ok"}

        ctx = stratix.start_trial()

        call_api()

        events = stratix.get_events()
        event = events[0]

        assert event.payload.tool.integration == "service"


class TestTraceModelDecorator:
    """Tests for the @trace_model decorator."""

    def test_basic_model_tracing(self):
        """Test that trace_model captures model invocation."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        @stratix.trace_model(provider="openai", name="gpt-4", version="1.0.0")
        def call_model(prompt: str, temperature: float = 0.7) -> dict:
            return {
                "content": "Hello!",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }
            }

        ctx = stratix.start_trial()

        result = call_model("Say hello", temperature=0.5)

        events = stratix.get_events()
        assert len(events) == 1

        event = events[0]
        assert event.payload.event_type == "model.invoke"
        assert event.payload.model.provider == "openai"
        assert event.payload.model.name == "gpt-4"
        assert event.payload.model.version == "1.0.0"

    def test_model_extracts_token_counts(self):
        """Test that trace_model extracts token counts from response."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        @stratix.trace_model(provider="anthropic", name="claude-3")
        def call_claude(prompt: str) -> dict:
            return {
                "content": "Response",
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 25,
                    "total_tokens": 75,
                }
            }

        ctx = stratix.start_trial()

        result = call_claude("Hello")

        events = stratix.get_events()
        event = events[0]

        # ModelInvokeEvent has token counts directly on payload
        assert event.payload.prompt_tokens == 50
        assert event.payload.completion_tokens == 25
        assert event.payload.total_tokens == 75

    def test_model_captures_parameters(self):
        """Test that trace_model captures model parameters."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        @stratix.trace_model(provider="openai", name="gpt-4")
        def call_model(prompt: str, temperature: float = 0.7, max_tokens: int = 100) -> dict:
            return {"content": "Response"}

        ctx = stratix.start_trial()

        result = call_model("Hello", temperature=0.9, max_tokens=200)

        events = stratix.get_events()
        event = events[0]

        params = event.payload.model.parameters
        assert params.get("temperature") == 0.9
        assert params.get("max_tokens") == 200


@pytest.mark.asyncio
class TestAsyncDecorators:
    """Tests for async decorated functions."""

    async def test_async_tool_tracing(self):
        """Test that trace_tool works with async functions."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        @stratix.trace_tool(name="async_lookup")
        async def async_lookup(item_id: str) -> dict:
            return {"id": item_id, "found": True}

        ctx = stratix.start_trial()

        result = await async_lookup("item-123")

        assert result == {"id": "item-123", "found": True}

        events = stratix.get_events()
        assert len(events) == 1
        assert events[0].payload.event_type == "tool.call"

    async def test_async_model_tracing(self):
        """Test that trace_model works with async functions."""
        stratix = STRATIX(
            policy_ref="test-policy@1.0.0",
            agent_id="test_agent",
            auto_emit_code=False,
            auto_emit_config=False,
        )

        @stratix.trace_model(provider="openai", name="gpt-4")
        async def async_call_model(prompt: str) -> dict:
            return {"content": "Async response"}

        ctx = stratix.start_trial()

        result = await async_call_model("Hello async")

        assert result == {"content": "Async response"}

        events = stratix.get_events()
        assert len(events) == 1
        assert events[0].payload.event_type == "model.invoke"
