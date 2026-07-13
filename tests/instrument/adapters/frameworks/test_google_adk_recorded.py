"""Recorded-real-response replay for the Google ADK adapter (LAY-3614, G5).

Drives a REAL ADK ``Runner`` (``LlmAgent`` + ``InMemorySessionService`` + the
adapter's ``BasePlugin``) whose ``Gemini`` model's ``api_client`` is a real
``google.genai`` client backed by ``httpx.MockTransport`` serving a captured
``generateContent`` response. This exercises the full path — real provider JSON
body -> the real genai client's own ``GenerateContentResponse`` deserialization
-> the real ADK ``LlmResponse`` -> the real plugin ``after_model_callback`` ->
real adapter parser -> emitted events — which the unit doubles (hand-built
``Mock`` ``LlmResponse`` objects) never combine with a real provider body.

The strong tells that the real provider shape flowed through:
``model.invoke.provider == "google"`` and ``finish_reason == "STOP"`` come off
the real response, and the token counts (``tokens_prompt=6`` /
``tokens_completion=1`` / ``tokens_total=7``) are lifted from the recorded
``usageMetadata`` block. Note ``tokens_total`` is the adapter's prompt+completion
sum (7), NOT the API's ``totalTokenCount`` (24, which folds in Gemini's
``thoughtsTokenCount``) — a real distinction only a real body surfaces.
"""

from __future__ import annotations

import sys
import asyncio
from typing import Any

import httpx
import pytest

if sys.version_info < (3, 10):
    pytest.skip("google-adk requires Python >= 3.10", allow_module_level=True)

pytest.importorskip("google.adk")
pytest.importorskip("google.genai")

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.adk.models.google_llm import Gemini  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter  # noqa: E402

from .conftest import find_event, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402


def _mock_gemini(transport: httpx.MockTransport) -> Gemini:
    """A real ADK ``Gemini`` whose ``api_client`` is routed through the recorded
    MockTransport (the documented subclass-and-override injection seam). ADK
    calls the async client, so both sync/async client args carry the transport."""

    class _RecordedGemini(Gemini):
        @property
        def api_client(self) -> Any:
            return genai.Client(
                api_key="test-key",
                http_options=types.HttpOptions(
                    client_args={"transport": transport},
                    async_client_args={"transport": transport},
                ),
            )

    return _RecordedGemini(model="gemini-2.5-flash")


async def _run(adapter: GoogleADKAdapter, transport: httpx.MockTransport) -> None:
    agent = LlmAgent(
        name="pong_agent",
        model=_mock_gemini(transport),
        instruction="Reply with exactly: pong",
    )
    session_service = InMemorySessionService()
    await session_service.create_session(app_name="app", user_id="u", session_id="s")
    runner = Runner(app_name="app", agent=agent, session_service=session_service, plugins=[adapter.plugin])
    async for _ in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(role="user", parts=[types.Part(text="Reply with exactly: pong")]),
    ):
        pass


class TestGoogleADKRecorded:
    def test_runner_over_recorded_genai(self, mock_client):
        fixture = load_recorded("google_genai", "default")
        transport, _ = mock_transport(fixture)
        uploaded = capture_framework_trace(mock_client)

        adapter = GoogleADKAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            asyncio.run(_run(adapter, transport))
        finally:
            adapter.disconnect()

        events = uploaded["events"]

        # Real genai response -> real ADK LlmResponse -> adapter model.invoke.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "gemini-2.5-flash"
        assert mi["payload"]["provider"] == "google"
        assert mi["payload"]["finish_reason"] == "STOP"
        assert mi["payload"]["tokens_prompt"] == 6
        assert mi["payload"]["tokens_completion"] == 1
        # prompt+completion (7), not the API's totalTokenCount=24 (adds thoughts).
        assert mi["payload"]["tokens_total"] == 7

        # cost.record mirrors the same real token accounting.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["model"] == "gemini-2.5-flash"
        assert cost["payload"]["tokens_prompt"] == 6
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 7
