"""Anthropic-specific request and response metadata capture tests.

Targets LAY-3328 ADP-072 (umbrella story) and its task tickets:

* LAY-3333: extended thinking — content + budget_tokens captured
* LAY-3334: full request param + response metadata extraction

Privacy rules from the ACs:

* System prompt content is NEVER captured; only ``has_system`` + ``system_length``.
* ``messages`` content is NEVER captured; only count and per-role distribution.
* ``tools`` payload is NEVER captured fully; only count + names.
* ``metadata`` is NEVER captured wholesale; only the ``user_id`` field (the
  Anthropic-recommended cost-attribution key).
* ``thinking`` request payload is NEVER captured wholesale; only ``budget_tokens``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from anthropic.types import Usage, Message, TextBlock, ToolUseBlock

from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider

from .conftest import make_anthropic_response  # noqa: F401  (kept for parity with other tests)
from ...conftest import find_event


def _make_message(*, content: list[Any], stop_reason: str = "end_turn") -> Message:
    return Message(
        id="msg-test",
        type="message",
        role="assistant",
        model="claude-3-7-sonnet-20250219",
        content=content,
        usage=Usage(input_tokens=10, output_tokens=5),
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# derive_params: privacy-safe request summarization (LAY-3334)
# ---------------------------------------------------------------------------


class TestDeriveParamsPrivacy:
    def test_system_content_not_captured(self, mock_client, capture_trace):
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(return_value=_make_message(content=[TextBlock(type="text", text="ok")]))

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        secret = "You are an assistant with knowledge of internal API key sk-xxxxxxxxxxxxxxxx"

        @trace(mock_client)
        def my_agent():
            anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=100,
                system=secret,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        # Privacy AC: system content must not leak anywhere in params.
        assert params.get("has_system") is True
        assert params.get("system_length") == len(secret)
        assert "system" not in params
        for v in params.values():
            assert "sk-xxxxxxxxxxxxxxxx" not in str(v)

    def test_messages_count_and_role_distribution(self, mock_client, capture_trace):
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(return_value=_make_message(content=[TextBlock(type="text", text="ok")]))

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=100,
                messages=[
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                ],
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert params["messages_count"] == 3
        assert params["message_roles"] == {"user": 2, "assistant": 1}
        # Raw content not present.
        assert "messages" not in params
        for v in params.values():
            assert "u1" != v and "a1" != v

    def test_tools_count_and_names_no_schema(self, mock_client, capture_trace):
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(return_value=_make_message(content=[TextBlock(type="text", text="ok")]))

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=100,
                messages=[],
                tools=[
                    {
                        "name": "get_weather",
                        "description": "internal description",
                        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                    {
                        "name": "send_email",
                        "description": "another secret",
                        "input_schema": {},
                    },
                ],
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert params["tools_count"] == 2
        assert params["tool_names"] == ["get_weather", "send_email"]
        assert "tools" not in params
        # Tool schemas/descriptions not in params.
        for v in params.values():
            assert "internal description" != v
            assert "another secret" != v

    def test_tool_choice_type_and_name(self, mock_client, capture_trace):
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(return_value=_make_message(content=[TextBlock(type="text", text="ok")]))

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=100,
                messages=[],
                tool_choice={"type": "tool", "name": "get_weather"},
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert params["tool_choice_type"] == "tool"
        assert params["tool_choice_name"] == "get_weather"
        assert "tool_choice" not in params

    def test_metadata_user_id_only_captured(self, mock_client, capture_trace):
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(return_value=_make_message(content=[TextBlock(type="text", text="ok")]))

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=100,
                messages=[],
                metadata={
                    "user_id": "user_abc_123",
                    # Other metadata fields must NOT be captured.
                    "session_id": "should_not_leak",
                    "internal_pii_field": "definitely_should_not_leak",
                },
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert params["metadata_user_id"] == "user_abc_123"
        # No raw metadata key, no other metadata fields.
        assert "metadata" not in params
        for v in params.values():
            assert "should_not_leak" != v
            assert "definitely_should_not_leak" != v


# ---------------------------------------------------------------------------
# Extended thinking budget capture (LAY-3333)
# ---------------------------------------------------------------------------


class TestStandardResponseUnaffectedByThinkingFeature:
    """LAY-3333 DoD: "Test response without thinking (standard response) works unchanged".

    A request that doesn't enable thinking must produce a normal model.invoke
    event with no ``thinking_*`` fields, no ``has_thinking=True``, and no
    spurious extra metadata. Same shape as pre-LAY-3333.
    """

    def test_baseline_response_omits_thinking_fields(self, mock_client, capture_trace):
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(
            return_value=_make_message(content=[TextBlock(type="text", text="plain answer")])
        )

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hi"}],
                # NO thinking param, NO metadata, NO tools.
            )
            return "done"

        my_agent()
        payload = find_event(capture_trace["events"], "model.invoke")["payload"]
        params = payload["parameters"]

        # Thinking-related fields must be absent when thinking isn't requested.
        assert "thinking_budget_tokens" not in params
        assert "thinking_type" not in params

        # Response side: no thinking blocks → has_thinking is False, no
        # reasoning/thinking tokens in usage.
        assert payload.get("has_thinking") is False
        usage = payload.get("usage", {})
        assert "thinking_tokens" not in usage
        assert "reasoning_tokens" not in usage

        # And content block counts reflect just the text block.
        assert payload["content_block_counts"] == {"text": 1}

    def test_thinking_path_does_not_alter_standard_usage_keys(self, mock_client, capture_trace):
        # Baseline usage shape — what callers downstream rely on. The thinking
        # plumbing must not change it.
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(
            return_value=_make_message(content=[TextBlock(type="text", text="plain answer")])
        )

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            anthropic_client.messages.create(model="claude-3-7-sonnet-20250219", max_tokens=100, messages=[])
            return "done"

        my_agent()
        usage = find_event(capture_trace["events"], "model.invoke")["payload"]["usage"]
        # The standard usage keys are present and stable.
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5
        assert usage["prompt_tokens"] == 10  # equals input_tokens when no cache
        assert usage["completion_tokens"] == 5


class TestThinkingBudgetTokens:
    def test_budget_tokens_captured_from_request(self, mock_client, capture_trace):
        anthropic_client = Mock()
        anthropic_client.messages.create = Mock(
            return_value=_make_message(content=[TextBlock(type="text", text="answer")])
        )

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        @trace(mock_client)
        def my_agent():
            anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=200,
                messages=[],
                thinking={"type": "enabled", "budget_tokens": 2048},
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert params["thinking_budget_tokens"] == 2048
        assert params["thinking_type"] == "enabled"
        # Raw thinking dict not in params.
        assert "thinking" not in params


# ---------------------------------------------------------------------------
# extract_meta: content block counts + tool-use names (LAY-3334)
# ---------------------------------------------------------------------------


class TestExtractMetaContentBlocks:
    def test_text_only_response_block_counts(self):
        msg = _make_message(content=[TextBlock(type="text", text="hi")])
        meta = AnthropicProvider.extract_meta(msg)
        assert meta["content_block_counts"] == {"text": 1}
        assert "tool_use_names" not in meta
        assert meta["has_thinking"] is False

    def test_tool_use_blocks_counted_with_names(self):
        msg = _make_message(
            content=[
                TextBlock(type="text", text="I'll call some tools."),
                ToolUseBlock(type="tool_use", id="t1", name="get_weather", input={"city": "sf"}),
                ToolUseBlock(type="tool_use", id="t2", name="send_email", input={"to": "x@y"}),
            ],
            stop_reason="tool_use",
        )
        meta = AnthropicProvider.extract_meta(msg)
        assert meta["content_block_counts"] == {"text": 1, "tool_use": 2}
        assert meta["tool_use_names"] == ["get_weather", "send_email"]
        assert meta["has_thinking"] is False

    def test_thinking_block_detected(self):
        # ``extract_meta`` only does duck-typed attribute reads on each block,
        # so we bypass anthropic's Pydantic Message validation (which rejects
        # arbitrary thinking-block shapes) by constructing a duck-typed
        # response shim directly.
        from types import SimpleNamespace

        msg = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="Let me reason..."),
                SimpleNamespace(type="text", text="answer"),
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            id="msg-test",
            model="claude-3-7-sonnet-20250219",
            role="assistant",
            stop_reason="end_turn",
            stop_sequence=None,
        )
        meta = AnthropicProvider.extract_meta(msg)
        assert meta["content_block_counts"] == {"thinking": 1, "text": 1}
        assert meta["has_thinking"] is True
