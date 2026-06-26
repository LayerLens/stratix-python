"""Recorded-real-response replay for the AutoGen framework (LAY-3614).

Drives a REAL ``autogen_ext.models.openai.OpenAIChatCompletionClient`` over
``httpx.MockTransport`` serving the captured OpenAI ``chat.completions`` response,
with the real ``AutoGenAdapter`` attached. This exercises the full path — real
provider response shape -> the real OpenAI SDK deserialization done inside the
AutoGen model client -> AutoGen's real ``LLMCallEvent`` (whose ``response`` is the
model client's ``result.model_dump()``) -> real adapter logging handler ->
emitted events — which the matrix layer (fake models) and the unit doubles
(hand-built ``LLMCallEvent`` stand-ins) never combine. Reuses the openai corpus
fixture (AutoGen's OpenAI client consumes the provider's response).

Injection seam (clean, public, version-stable for autogen-ext 0.7.x): the OpenAI
client config allow-lists ``http_client``, which passes straight through to the
underlying ``openai.AsyncOpenAI`` constructor — the same documented seam LangChain
and pydantic_ai use. No framework internals are monkeypatched.

Determinism: the run is fully offline (the MockTransport answers the single
``POST /v1/chat/completions``); ``model_client.create`` is awaited via
``asyncio.run`` (no pytest-asyncio dependency, matching the openai-agents replay
test). The adapter manages its own collector and flushes on ``disconnect()``
(``_end_trace`` -> ``collector.flush``); the base venv autouse forces
``_upload._sync_mode = True``, so the upload is captured by the time
``disconnect()`` returns.

Strong tell that the real provider shape flowed through: ``model.invoke`` reports
``gpt-4o-mini-2024-07-18`` (the dated id echoed in the recorded *response* body,
read by the adapter off ``response["model"]``), not the ``gpt-4o-mini`` we
*requested*; and the token counts (12 / 1 / 13) are lifted from the real ``usage``
block the OpenAI SDK parsed.
"""

from __future__ import annotations

import sys
import asyncio

import httpx
import pytest

if sys.version_info < (3, 10):  # pragma: no cover - env guard
    pytest.skip("autogen requires Python >= 3.10", allow_module_level=True)
try:
    from autogen_core.models import UserMessage  # noqa: E402
    from autogen_ext.models.openai import OpenAIChatCompletionClient  # noqa: E402
except (ImportError, Exception):  # pragma: no cover - env guard
    pytest.skip("autogen not installed or incompatible", allow_module_level=True)

from layerlens.instrument.adapters.frameworks.autogen import AutoGenAdapter  # noqa: E402

from .conftest import find_event, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402


def _model_client(fixture):
    transport, _ = mock_transport(fixture)
    # autogen-ext's OpenAI client config allow-lists ``http_client`` and forwards
    # it to ``openai.AsyncOpenAI`` — the documented public custom-client seam.
    return OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=transport),
    )


class TestAutoGenRecorded:
    def test_model_client_over_recorded_openai(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        client = _model_client(fixture)
        adapter = AutoGenAdapter(mock_client)
        adapter.connect()
        try:
            result = asyncio.run(client.create([UserMessage(content="Reply with exactly: pong", source="user")]))
        finally:
            adapter.disconnect()

        # The real OpenAI SDK deserialized the recorded chat.completion.
        assert result.content == "pong"
        assert result.usage.prompt_tokens == 12
        assert result.usage.completion_tokens == 1

        events = uploaded["events"]

        # AutoGen's real LLMCallEvent carries ``response = result.model_dump()``;
        # the adapter reads the model id off ``response["model"]`` (the dated id in
        # the recorded *response* body) and the flat prompt/completion token
        # counts off the real ``usage`` block.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["framework"] == "autogen"
        assert mi["payload"]["model"] == "gpt-4o-mini-2024-07-18"
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 1
        assert mi["payload"]["tokens_total"] == 13

        # The adapter mirrors the same real token accounting onto cost.record.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "autogen"
        assert cost["payload"]["model"] == "gpt-4o-mini-2024-07-18"
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 13
