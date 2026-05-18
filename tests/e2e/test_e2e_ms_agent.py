"""End-to-end: MSAgentFrameworkAdapter against real semantic-kernel types.

We don't spin up a real LLM-backed AgentChat (that would need credentials
and is fragile), but we DO use real ``ChatMessageContent``,
``FunctionCallContent``, and ``FunctionResultContent`` instances from
semantic-kernel — so the adapter's message-processing path runs against
the actual SK types it'd see in production.

The chat itself is a thin object with an ``invoke`` that yields the real
SK content objects. instrument_chat wraps it, we await the wrapped
async generator, and verify the layerlens events that come out.
"""

from __future__ import annotations

import sys
import asyncio
from types import SimpleNamespace

import pytest

if sys.version_info < (3, 10):
    pytest.skip("semantic-kernel requires Python >= 3.10", allow_module_level=True)

sk_contents = pytest.importorskip("semantic_kernel.contents")
ChatMessageContent = sk_contents.ChatMessageContent
FunctionCallContent = sk_contents.FunctionCallContent
FunctionResultContent = sk_contents.FunctionResultContent
AuthorRole = sk_contents.AuthorRole

from layerlens.instrument.adapters.frameworks.ms_agent_framework import (
    MSAgentFrameworkAdapter,
)

from .conftest import events_of, first_event


def _msg(role: AuthorRole, content: str, *, agent_name: str | None = None, items=(), metadata=None):
    """Build a real SK ChatMessageContent.

    ChatMessageContent.items isn't a kwarg in newer SK builds, so we set
    it after construction.
    """
    m = ChatMessageContent(role=role, content=content, name=agent_name)
    if items:
        # Append in place — m.items is a real list
        for it in items:
            m.items.append(it)
    if metadata is not None:
        m.metadata.update(metadata)
    return m


def _fake_chat(yielded_messages, agent_name: str = "primary"):
    """Build a minimal chat-shaped object whose ``invoke`` yields real SK
    ChatMessageContent objects."""

    async def invoke(*_args, **_kwargs):
        for m in yielded_messages:
            yield m

    return SimpleNamespace(
        name="GroupChat",
        agent=SimpleNamespace(name=agent_name),
        invoke=invoke,
    )


def _drain(chat) -> list:
    """Iterate the wrapped chat's invoke to completion and return collected."""

    async def run():
        out = []
        async for m in chat.invoke():
            out.append(m)
        return out

    return asyncio.run(run())


class TestRealSKMessagesProduceLayerLensEvents:
    def test_simple_assistant_message(self, client_and_uploads):
        client, uploads = client_and_uploads
        adapter = MSAgentFrameworkAdapter(client)

        chat = _fake_chat([_msg(AuthorRole.ASSISTANT, "hello world", agent_name="primary")])
        adapter.instrument_chat(chat)
        result = _drain(chat)

        assert len(result) == 1
        # We emit agent.input and agent.output framing the invocation
        agent_in = first_event(uploads, "agent.input")
        agent_out = first_event(uploads, "agent.output")
        assert agent_in["payload"]["framework"] == "ms_agent_framework"
        assert agent_in["payload"]["agent_name"] == "primary"
        assert agent_out["payload"]["agent_name"] == "primary"

    def test_function_call_and_result_extracted(self, client_and_uploads):
        client, uploads = client_and_uploads
        adapter = MSAgentFrameworkAdapter(client)

        call = FunctionCallContent(
            id="call-1",
            name="search",
            arguments='{"q": "AI safety"}',
        )
        result = FunctionResultContent(
            id="call-1",
            name="search",
            result="found 3 papers",
        )

        chat = _fake_chat(
            [
                _msg(AuthorRole.ASSISTANT, "calling search", agent_name="primary", items=[call]),
                _msg(AuthorRole.TOOL, "search returned", agent_name="primary", items=[result]),
            ]
        )
        adapter.instrument_chat(chat)
        _drain(chat)

        tool_call = first_event(uploads, "tool.call")
        tool_result = first_event(uploads, "tool.result")
        assert tool_call["payload"]["tool_name"] == "search"
        assert tool_result["payload"]["tool_name"] == "search"

    def test_group_chat_turn_transition_emits_handoff(self, client_and_uploads):
        client, uploads = client_and_uploads
        adapter = MSAgentFrameworkAdapter(client)

        chat = _fake_chat(
            [
                _msg(AuthorRole.ASSISTANT, "researching...", agent_name="researcher"),
                _msg(AuthorRole.ASSISTANT, "writing draft", agent_name="writer"),
                _msg(AuthorRole.ASSISTANT, "reviewing", agent_name="reviewer"),
            ],
            agent_name="researcher",  # chat starts in researcher
        )
        adapter.instrument_chat(chat)
        _drain(chat)

        handoffs = events_of(uploads, "agent.handoff")
        # researcher -> writer, writer -> reviewer
        assert len(handoffs) == 2
        pairs = [(h["payload"]["from_agent"], h["payload"]["to_agent"]) for h in handoffs]
        assert ("researcher", "writer") in pairs
        assert ("writer", "reviewer") in pairs

    def test_model_metadata_produces_model_invoke_and_cost(self, client_and_uploads):
        client, uploads = client_and_uploads
        adapter = MSAgentFrameworkAdapter(client)

        chat = _fake_chat(
            [
                _msg(
                    AuthorRole.ASSISTANT,
                    "an answer",
                    agent_name="primary",
                    metadata={
                        "model": "gpt-4o",
                        "usage": {"prompt_tokens": 15, "completion_tokens": 8},
                    },
                ),
            ]
        )
        adapter.instrument_chat(chat)
        _drain(chat)

        model_invoke = first_event(uploads, "model.invoke")
        assert model_invoke["payload"]["model"] == "gpt-4o"
        assert model_invoke["payload"]["provider"] == "openai"

        cost = first_event(uploads, "cost.record")
        assert cost["payload"]["tokens_prompt"] == 15
        assert cost["payload"]["tokens_completion"] == 8

    def test_environment_config_fires_once_per_chat(self, client_and_uploads):
        client, uploads = client_and_uploads
        adapter = MSAgentFrameworkAdapter(client)

        chat = _fake_chat([_msg(AuthorRole.ASSISTANT, "hi", agent_name="primary")])
        adapter.instrument_chat(chat)

        _drain(chat)
        _drain(chat)  # second invocation — should not re-emit environment.config

        envs = events_of(uploads, "environment.config")
        assert len(envs) == 1, f"expected 1 environment.config event, got {len(envs)}"
