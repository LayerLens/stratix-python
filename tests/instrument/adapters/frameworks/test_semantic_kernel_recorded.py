"""Recorded-real-response replay for the Semantic Kernel framework (LAY-3614).

Drives a REAL ``semantic_kernel.Kernel`` whose ``OpenAIChatCompletion`` service
is backed by an ``AsyncOpenAI`` client over ``httpx.MockTransport`` serving the
captured OpenAI response, with the real ``SemanticKernelAdapter`` attached. The
natural run unit is a registered prompt function invoked via ``kernel.invoke`` —
SK's function-invocation filter opens/closes the adapter's run boundary (so the
collector flushes), while the prompt rendering routes through the patched
``_inner_get_chat_message_contents`` on the chat service. This exercises the
full path — real provider response shape -> real SK ``ChatMessageContent`` /
``CompletionUsage`` objects -> real adapter -> emitted events — which the unit
suite (hand-built doubles) and the matrix (fake services) never combine. Reuses
the openai corpus fixture (SK's OpenAI chat service consumes the provider's
chat.completion response).

The strong tell that the real provider shape flowed through is the token triple
12/1/13: SK parses the recorded response's ``usage`` into a ``CompletionUsage``
object on ``ChatMessageContent.metadata['usage']``, and the adapter normalizes
that real object into ``tokens_prompt`` / ``tokens_completion`` / ``tokens_total``.
(Unlike LangChain/PydanticAI, SK reports the *configured* ``ai_model_id`` for the
model field — it does not echo the response's resolved model id — so the model
assertion documents config, and the tokens carry the real-response proof.)
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from openai import AsyncOpenAI

pytest.importorskip("semantic_kernel")  # skips in the base venv (not installed there)

from semantic_kernel import Kernel  # noqa: E402
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)

from layerlens.instrument.adapters.frameworks.semantic_kernel import SemanticKernelAdapter

from .conftest import find_event, capture_framework_trace
from ..._recorded import load_recorded, mock_transport


def _kernel(fixture):
    transport, _ = mock_transport(fixture)
    # SK's OpenAI chat service is async-only; inject the MockTransport through the
    # documented ``async_client=`` seam (an AsyncOpenAI built on a custom
    # http_client). ai_model_id is the model we *request*.
    async_client = AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(transport=transport))
    service = OpenAIChatCompletion(ai_model_id="gpt-4o-mini", async_client=async_client)
    kernel = Kernel()
    kernel.add_service(service)
    return kernel


class TestSemanticKernelRecorded:
    def test_prompt_function_over_recorded_openai(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        kernel = _kernel(fixture)
        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        function = kernel.add_function(
            plugin_name="replay",
            function_name="say",
            prompt="{{$question}}",
            prompt_execution_settings=OpenAIChatPromptExecutionSettings(max_tokens=10),
        )
        result = asyncio.run(kernel.invoke(function, question="Reply with exactly: pong"))
        adapter.disconnect()

        assert str(result) == "pong"

        events = uploaded["events"]

        # The patched chat service emits model.invoke with usage normalized off the
        # real CompletionUsage parsed from the recorded chat.completion body.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "gpt-4o-mini"  # configured ai_model_id, not the response echo
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 1
        assert mi["payload"]["tokens_total"] == 13

        # cost.record echoes the same real per-call token accounting.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "semantic_kernel"
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 13
