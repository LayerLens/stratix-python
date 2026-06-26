"""Recorded-real-response replay for the OpenAI Agents SDK adapter (LAY-3614).

Drives a REAL ``agents.Runner.run`` over ``httpx.MockTransport`` serving the
captured OpenAI ``chat.completions`` response, with the real
``OpenAIAgentsAdapter`` registered as the SDK's trace processor. This exercises
the full path — real provider response shape -> the real Agents SDK's own
``OpenAIChatCompletionsModel`` deserialization -> real ``generation``/``agent``
spans -> real adapter span handlers -> emitted events — which the matrix layer
(fake models) and the hand-built unit doubles (synthetic ``GenerationSpanData``)
never combine. Reuses the openai corpus fixture (the framework consumes the
provider's response).

Injection seam (clean, public, version-stable for agents 0.x):

* ``set_default_openai_client(AsyncOpenAI(http_client=<MockTransport>))`` routes
  the SDK's model client through the recorded transport — the documented way to
  supply a custom client, identical in spirit to LangChain's ``http_client=``.
* ``set_default_openai_api("chat_completions")`` selects the
  ``OpenAIChatCompletionsModel`` path so the recorded ``chat.completion`` body
  (not the Responses API) is what the real SDK parses.

Determinism: the run is fully offline (the MockTransport answers the single
``POST /v1/chat/completions``); the SDK's own backend trace exporter no-ops with
no ``OPENAI_API_KEY`` (which this test clears), so nothing touches the network.
The trace lifecycle (``on_trace_end`` -> ``collector.flush``) fires synchronously
inside ``Runner.run``, and the base venv forces ``_upload._sync_mode = True``, so
the upload is captured by the time ``Runner.run`` returns.

Note on asserted values: the ``generation`` span records the *requested* model
name (``gpt-4o-mini``), not the response's dated id — that is the real SDK's span
behaviour, not ours. The strong values pinned to the real response body are the
token counts (12 / 1 / 13), which the SDK lifts straight from the recorded
``usage`` block.
"""

from __future__ import annotations

import sys
import asyncio

import httpx
import pytest

if sys.version_info < (3, 10):
    pytest.skip("openai-agents requires Python >= 3.10", allow_module_level=True)
try:
    import agents  # noqa: F401
except (ImportError, Exception):  # pragma: no cover - env guard
    pytest.skip("openai-agents not installed or incompatible", allow_module_level=True)

from agents import (  # noqa: E402
    Agent,
    Runner,
    set_trace_processors,
    set_default_openai_api,
    set_default_openai_client,
)

from openai import AsyncOpenAI  # noqa: E402
from layerlens.instrument.adapters.frameworks.openai_agents import (  # noqa: E402
    OpenAIAgentsAdapter,
)

from .conftest import find_event  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402


@pytest.fixture
def agents_sdk_offline(monkeypatch):
    """Route the Agents SDK through a recorded MockTransport, offline.

    Yields the fixture body; restores all touched *global* SDK state on teardown
    (the SDK keeps the default client / api / processors as module globals).
    """
    # Hermetic: the SDK's default backend trace exporter no-ops without a key,
    # so this guarantees no trace HTTP even if CI exports one.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    fixture = load_recorded("openai", "default")
    transport, _ = mock_transport(fixture)
    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        http_client=httpx.AsyncClient(transport=transport),
    )
    # use_for_tracing=False: don't hand this client's (fake) key to the SDK's
    # trace exporter — we only want it for model calls.
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    try:
        yield
    finally:
        # The SDK has no "reset default client" API; the next test that needs a
        # real client sets its own, and clearing processors is the load-bearing
        # cleanup (matches test_openai_agents.py's clean_processors autouse).
        set_trace_processors([])


class TestOpenAIAgentsRecorded:
    def test_runner_over_recorded_openai(self, mock_client, agents_sdk_offline):
        from .conftest import capture_framework_trace

        uploaded = capture_framework_trace(mock_client)

        adapter = OpenAIAgentsAdapter(mock_client)
        adapter.connect()
        try:
            agent = Agent(
                name="pong_agent",
                instructions="Reply with exactly: pong",
                model="gpt-4o-mini",
            )
            result = asyncio.run(Runner.run(agent, "Reply with exactly: pong"))
        finally:
            adapter.disconnect()

        # The real SDK deserialized the recorded chat.completion.
        assert result.final_output == "pong"

        events = uploaded["events"]
        # Real generation span -> adapter -> model.invoke, carrying the real
        # OpenAI usage block lifted from the recorded response body.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["framework"] == "openai-agents"
        assert mi["payload"]["model"] == "gpt-4o-mini"
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 1
        assert mi["payload"]["tokens_total"] == 13

        # The adapter mirrors the same real token accounting onto cost.record.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "openai-agents"
        assert cost["payload"]["model"] == "gpt-4o-mini"
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 13

        # The run also wraps the generation in a real agent span.
        agent_in = find_event(events, "agent.input")
        assert agent_in["payload"]["agent_name"] == "pong_agent"
        agent_out = find_event(events, "agent.output")
        assert agent_out["payload"]["status"] == "ok"
