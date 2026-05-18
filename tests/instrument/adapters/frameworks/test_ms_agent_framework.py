"""Tests for the Microsoft Agent Framework adapter.

These exercise the message-processing path against synthetic
ChatMessageContent-shaped objects so the tests don't need a working
semantic-kernel install. The wrapper itself is exercised by feeding a
mock async-iterable into ``instrument_chat``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from layerlens.instrument.adapters.frameworks.ms_agent_framework import (
    MSAgentFrameworkAdapter,
    _detect_provider,
)

from .conftest import find_event, find_events, capture_framework_trace

# ---------------------------------------------------------------------------
# Synthetic message helpers
# ---------------------------------------------------------------------------


def _msg(agent_name=None, items=(), metadata=None):
    return SimpleNamespace(
        agent_name=agent_name,
        items=list(items),
        metadata=metadata,
    )


def _make_item(cls_name, **fields):
    """Build an item whose ``type(item).__name__`` matches a specific class.

    SimpleNamespace's type name is fixed, so we create a fresh class per call.
    """
    cls = type(cls_name, (), {})
    obj = cls()
    for key, value in fields.items():
        setattr(obj, key, value)
    return obj


def _func_call(name, arguments):
    return _make_item("FunctionCallContent", name=name, arguments=arguments)


def _func_result(name, result):
    return _make_item("FunctionResultContent", name=name, result=result)


def _make_invoke(messages):
    """Build a fake `chat.invoke` that yields the given messages."""

    async def invoke(*_args, **_kwargs):
        for m in messages:
            yield m

    return invoke


def _run_chat(adapter, chat, messages):
    chat.invoke = _make_invoke(messages)
    adapter.instrument_chat(chat)

    async def consume():
        collected = []
        async for m in chat.invoke():
            collected.append(m)
        return collected

    return asyncio.run(consume())


# ---------------------------------------------------------------------------
# Adapter info / detection
# ---------------------------------------------------------------------------


class TestAdapterInfo:
    def test_name_and_type(self, mock_client):
        adapter = MSAgentFrameworkAdapter(mock_client)
        info = adapter.adapter_info()
        assert info.name == "ms_agent_framework"
        assert info.adapter_type == "framework"


class TestProviderDetection:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-4o", "openai"),
            ("o3-mini", "openai"),
            ("claude-3-5-sonnet", "anthropic"),
            ("gemini-1.5-pro", "google"),
            ("mistral-large", "mistral"),
            ("phi-3", "microsoft"),
            ("llama-3", "meta"),
            ("some-random-deployment", "azure_openai"),
        ],
    )
    def test_classification(self, model, expected):
        assert _detect_provider(model) == expected

    def test_none_returns_none(self):
        assert _detect_provider(None) is None


# ---------------------------------------------------------------------------
# Lifecycle wrapping
# ---------------------------------------------------------------------------


class TestInvokeWrapping:
    def test_invoke_emits_agent_input_and_output(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = MSAgentFrameworkAdapter(mock_client)

        chat = SimpleNamespace(name="ChatGroup", agent=SimpleNamespace(name="primary"))
        _run_chat(adapter, chat, [_msg(agent_name="primary")])

        events = uploaded["events"]
        agent_in = find_event(events, "agent.input")
        agent_out = find_event(events, "agent.output")
        assert agent_in["payload"]["agent_name"] == "primary"
        assert agent_out["payload"]["agent_name"] == "primary"
        # Sanity: framework label is set
        assert agent_in["payload"]["framework"] == "ms_agent_framework"

    def test_invoke_emits_environment_config_once(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = MSAgentFrameworkAdapter(mock_client)

        chat = SimpleNamespace(
            name="GroupChat",
            agents=[SimpleNamespace(name="a1"), SimpleNamespace(name="a2")],
            selection_strategy=_make_item("RoundRobinSelectionStrategy"),
            termination_strategy=_make_item("DefaultTermination"),
        )

        _run_chat(adapter, chat, [_msg(agent_name="a1")])

        configs = find_events(uploaded["events"], "environment.config")
        assert len(configs) == 1
        assert configs[0]["payload"]["agents"] == ["a1", "a2"]
        assert configs[0]["payload"]["selection_strategy"] == "RoundRobinSelectionStrategy"

    def test_disconnect_restores_originals(self, mock_client):
        # Skip connect() — it checks the optional semantic-kernel dependency
        # which isn't installed in the default test env. instrument_chat
        # itself doesn't check the dep.
        adapter = MSAgentFrameworkAdapter(mock_client)

        chat = SimpleNamespace(name="c", invoke=_make_invoke([]))
        original_invoke = chat.invoke
        adapter.instrument_chat(chat)
        assert chat.invoke is not original_invoke
        adapter.disconnect()
        assert chat.invoke is original_invoke

    def test_error_in_invoke_emits_agent_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = MSAgentFrameworkAdapter(mock_client)

        async def failing_invoke(*_a, **_kw):
            yield _msg(agent_name="primary")
            raise RuntimeError("kaboom")

        chat = SimpleNamespace(name="c", invoke=failing_invoke)
        adapter.instrument_chat(chat)

        async def consume():
            async for _ in chat.invoke():
                pass

        with pytest.raises(RuntimeError):
            asyncio.run(consume())

        agent_err = find_event(uploaded["events"], "agent.error")
        assert "kaboom" in agent_err["payload"]["error"]


# ---------------------------------------------------------------------------
# Per-message processing
# ---------------------------------------------------------------------------


class TestMessageProcessing:
    def test_tool_call_and_result_emitted(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = MSAgentFrameworkAdapter(mock_client)

        chat = SimpleNamespace(name="c", agent=SimpleNamespace(name="primary"))
        _run_chat(
            adapter,
            chat,
            [
                _msg(agent_name="primary", items=[_func_call("search", {"q": "AI"})]),
                _msg(agent_name="primary", items=[_func_result("search", ["r1", "r2"])]),
            ],
        )

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        tool_result = find_event(events, "tool.result")
        assert tool_call["payload"]["tool_name"] == "search"
        assert tool_result["payload"]["tool_name"] == "search"
        assert tool_call["payload"]["input"] == {"q": "AI"}

    def test_model_invoke_and_cost_from_metadata(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = MSAgentFrameworkAdapter(mock_client)

        chat = SimpleNamespace(name="c", agent=SimpleNamespace(name="primary"))
        _run_chat(
            adapter,
            chat,
            [
                _msg(
                    agent_name="primary",
                    metadata={
                        "model": "gpt-4o",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                    },
                )
            ],
        )

        model_invoke = find_event(uploaded["events"], "model.invoke")
        assert model_invoke["payload"]["model"] == "gpt-4o"
        assert model_invoke["payload"]["provider"] == "openai"

        cost = find_event(uploaded["events"], "cost.record")
        assert cost["payload"]["tokens_prompt"] == 10
        assert cost["payload"]["tokens_completion"] == 20

    def test_handoff_emitted_on_agent_turn_change(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = MSAgentFrameworkAdapter(mock_client)

        chat = SimpleNamespace(name="c", agent=SimpleNamespace(name="primary"))
        _run_chat(
            adapter,
            chat,
            [
                _msg(agent_name="primary"),
                _msg(agent_name="researcher"),  # turn transition
                _msg(agent_name="researcher"),  # no transition
                _msg(agent_name="writer"),  # another transition
            ],
        )

        handoffs = find_events(uploaded["events"], "agent.handoff")
        # Two transitions -> two handoffs
        assert len(handoffs) == 2
        assert handoffs[0]["payload"]["from_agent"] == "primary"
        assert handoffs[0]["payload"]["to_agent"] == "researcher"
        assert handoffs[1]["payload"]["from_agent"] == "researcher"
        assert handoffs[1]["payload"]["to_agent"] == "writer"

    def test_unknown_item_types_are_ignored(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = MSAgentFrameworkAdapter(mock_client)

        opaque = _make_item("TextContent", name="ignore_me")

        chat = SimpleNamespace(name="c", agent=SimpleNamespace(name="primary"))
        _run_chat(adapter, chat, [_msg(agent_name="primary", items=[opaque])])

        assert find_events(uploaded["events"], "tool.call") == []
        assert find_events(uploaded["events"], "tool.result") == []
