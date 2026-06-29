"""Recorded-real-response replay for the CrewAI framework (LAY-3614).

Drives a REAL ``crewai.Crew`` (Agent + Task, sequential process) whose
``crewai.LLM`` resolves to the native ``OpenAICompletion`` provider and is backed
by an ``httpx.MockTransport`` serving the captured OpenAI chat.completion
response, with the real ``CrewAIAdapter`` attached to the real ``crewai_event_bus``.
This exercises the full path — real provider response shape -> real CrewAI native
OpenAI provider + typed event bus -> real adapter -> emitted events — which the
matrix layer (fake models) and the hand-built unit doubles never combine.

Why this layer matters here specifically
-----------------------------------------
The unit doubles in ``test_crewai.py`` hand-build ``LLMCallCompletedEvent`` with
``response={"content": ..., "usage": {...}}`` — a dict carrying ``usage`` nested
*inside* ``response``. The REAL crewai 1.x native OpenAI provider emits a very
different shape: ``response`` is the bare assistant string (``"pong"``) and the
token usage lives in a SEPARATE ``event.usage`` field. So the strong real-response
value that reaches the trace deterministically is the crew-aggregated
``total_tokens`` (= the recorded ``usage.total_tokens``), surfaced on the
crew-level ``cost.record`` — together with the real assistant content on
``agent.output`` and the configured model on ``model.invoke``. (The per-call
prompt/completion split and the response-echoed model id ``gpt-4o-mini-2024-07-18``
do not currently reach the trace through the real event shape; this replay test
pins the behavior that the recorded usage genuinely flows end-to-end.)

Injection seam (PUBLIC, deterministic)
---------------------------------------
``crewai.LLM(model="openai/...")`` routes to the native ``OpenAICompletion``
pydantic model. Its ``client_params`` constructor field is merged into the kwargs
used to build the underlying ``openai.OpenAI`` client, so we pass
``client_params={"http_client": httpx.Client(transport=...)}``. The provider
*eagerly* builds both a sync and an async client at construction and would reject
a sync ``httpx.Client`` for the async one — so we construct without a key (the
eager build is then skipped) and set the public ``api_key`` field afterward; the
sync chat-completions path lazily builds the OpenAI client from ``client_params``
at call time. No framework private attribute is touched.

CrewAI dispatches bus handlers on a thread-pool executor, so after ``kickoff()``
we call the public ``crewai_event_bus.flush()`` barrier (waits for every pending
handler future) before ``disconnect()`` — this is what makes the captured trace
deterministic rather than racing the executor.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import httpx
import pytest

crewai = pytest.importorskip("crewai")  # py>=3.10 + crewai installed (skips in the base venv)

# The replay drives a real crewai native OpenAI provider via the keyless
# client_params seam and the modern typed event bus, both of which the 1.14 line
# ships. The combined base lock resolves whatever the cross-adapter solve allows
# (0.193.2 or 1.6.1 by platform), which lack the seam/event surface; the pinned
# matrix (crewai==1.14.6) row exercises this replay. Skip below 1.14.
from packaging.version import Version  # noqa: E402

if Version(crewai.__version__) < Version("1.14"):
    pytest.skip(
        f"crewai recorded replay requires >= 1.14; got {crewai.__version__}",
        allow_module_level=True,
    )

from crewai import LLM, Crew, Task, Agent, Process  # noqa: E402
from crewai.events import crewai_event_bus

from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter

from .conftest import find_event, find_events, capture_framework_trace
from ..._recorded import load_recorded, mock_transport


def _llm(fixture: Dict[str, Any]) -> LLM:
    transport, _ = mock_transport(fixture)
    # Construct with no key so the provider skips its eager async-client build
    # (which would reject our sync httpx.Client); the sync chat-completions path
    # then lazily builds the OpenAI client from ``client_params`` at call time.
    prior_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        llm = LLM(
            model="openai/gpt-4o-mini",
            client_params={"http_client": httpx.Client(transport=transport)},
        )
    except ImportError as exc:
        # The matrix pins crewai==1.14.6, whose native provider builds lazily and
        # accepts this keyless client_params seam. The base ``test`` job resolves
        # crewai unpinned on py>=3.10 and can pull an old release (e.g. 1.6.1)
        # whose provider demands OPENAI_API_KEY at construction and lacks the
        # seam — skip there rather than fail on an unsupported version (the matrix
        # row still exercises this replay on the pinned version).
        pytest.skip(f"crewai native-provider client_params seam unavailable: {exc}")
    finally:
        if prior_key is not None:
            os.environ["OPENAI_API_KEY"] = prior_key
    # Public field — supplies the key the lazy sync build needs.
    llm.api_key = "test-key"
    return llm


def _crew(fixture: Dict[str, Any]) -> Crew:
    llm = _llm(fixture)
    agent = Agent(
        role="responder",
        goal="Reply to the user",
        backstory="A terse assistant.",
        llm=llm,
        verbose=False,
        max_iter=1,
    )
    task = Task(
        description="Reply with exactly: pong",
        expected_output="pong",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)


class TestCrewAIRecorded:
    def test_crew_over_recorded_openai(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        adapter = CrewAIAdapter(mock_client, capture_config=CaptureConfig.full())
        # ``scoped_handlers`` isolates our bus subscription to this test (the
        # proven pattern in ``test_crewai.TestEventBusIntegration``).
        with crewai_event_bus.scoped_handlers():
            adapter.connect()
            result = _crew(fixture).kickoff()
            # Deterministic barrier: drain the bus executor so every adapter
            # handler has fired (and its collector flushed) before we assert.
            crewai_event_bus.flush(timeout=10.0)
            adapter.disconnect()

        assert str(result) == "pong"

        events = uploaded["events"]

        # The real native provider parses the recorded ChatCompletion: the
        # configured model id flows onto model.invoke.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "gpt-4o-mini"

        # The crew aggregates the REAL recorded usage (usage.total_tokens == 13)
        # and surfaces it on cost.record — the strong tell that the recorded
        # response shape flowed through crewai's own token accounting.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "crewai"
        assert cost["payload"]["tokens_total"] == 13

        # The agent emits the real assistant content (choices[0].message.content).
        agent_outs = find_events(events, "agent.output")
        agent_level = [e for e in agent_outs if e["payload"].get("agent_role") == "responder"]
        assert agent_level, "expected an agent-level agent.output"
        assert agent_level[0]["payload"]["status"] == "ok"
        assert agent_level[0]["payload"]["output"] == "pong"

        # The crew-level agent.output carries the same real total token count.
        crew_level = [e for e in agent_outs if e["payload"].get("crew_name")]
        assert crew_level, "expected a crew-level agent.output"
        assert crew_level[0]["payload"]["tokens_total"] == 13
