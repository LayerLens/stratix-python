"""Offline error + attestation + redaction floor for the CrewAI framework adapter.

Closes the W1 census cells that the existing ``test_crewai.py`` proves only via
direct ``adapter._on_*`` handler calls or a synthetic error string, by driving
the *real* ``crewai_event_bus`` (and a real ``crewai.Crew.kickoff()``) so a
regression fails in plain CI with no credentials and no network:

* Error-paths — a REAL ``openai`` SDK exception (the shape a crewai LLM call
                actually raises) is carried on a real ``LLMCallFailedEvent`` that
                is *emitted on the real crewai event bus* (not a direct handler
                call, not the synthetic ``"rate limit exceeded"`` string the
                existing suite uses). It surfaces as ``agent.error`` with the
                adapter's honest ``error_type == "llm_error"`` classification and
                the real exception message flowing through verbatim.
* Attestation — a small REAL ``crewai.Crew`` kickoff (real Agent+Task, real
                native OpenAI provider over a mocked transport) flushes a trace
                whose attestation chain reconstructs and ``verify_chain(...)``
                returns valid; a tamper control proves the check is not vacuous.
* Redaction   — a real bus-driven crew lifecycle with ``capture_content=False``
                keeps task/tool/crew content — and a SENTINEL sweep over
                ``json.dumps(events)`` — out of the stored trace, with a
                ``capture_content=True`` vacuity control proving the same path
                DOES carry the content otherwise.

The only mock is the network boundary (``httpx.MockTransport`` for the real
crewai native OpenAI provider); every crewai object, event, dispatch and the
adapter's own parser are real.

BUG-5 (real hierarchical-crew / ``delegate_work_to_coworker`` handoff) is now
fixed and covered by ``tests/e2e/test_e2e_crewai_delegation.py`` (the adapter
normalizes the sanitized tool name so real crewai delegation emits handoffs).
"""

from __future__ import annotations

import os
import sys
import json
import datetime

import pytest

# Mirror test_crewai.py's guards: crewai uses ``type | None`` (TypeError on
# py<3.10) and the modern typed event bus only lands on the 1.14 line, which the
# pinned matrix (crewai==1.14.6) resolves. importorskip only catches ImportError,
# so guard explicitly.
if sys.version_info < (3, 10):
    pytest.skip("crewai requires Python >= 3.10", allow_module_level=True)
try:
    import crewai  # noqa: F401
except (ImportError, TypeError):
    pytest.skip("crewai not installed or incompatible", allow_module_level=True)

from packaging.version import Version  # noqa: E402

if Version(crewai.__version__) < Version("1.14"):
    pytest.skip(
        f"crewai floor requires >= 1.14; got {crewai.__version__}",
        allow_module_level=True,
    )

import httpx  # noqa: E402
from crewai import LLM, Crew, Task, Agent, Process  # noqa: E402
from crewai.events import (  # noqa: E402
    TaskStartedEvent,
    LLMCallFailedEvent,
    TaskCompletedEvent,
    LLMCallStartedEvent,
    ToolUsageStartedEvent,
    CrewKickoffFailedEvent,
    ToolUsageFinishedEvent,
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    crewai_event_bus,
)
from crewai.tasks.task_output import TaskOutput  # noqa: E402

import openai  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.crewai import CrewAIAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ---------------------------------------------------------------------------
# Real-event-bus helpers (the proven pattern from test_crewai.py)
# ---------------------------------------------------------------------------
def _emit_wait(event) -> None:
    """Emit an event on the REAL crewai bus and block until every handler ran.

    crewai dispatches handlers on a thread-pool executor and ``emit`` returns a
    Future; waiting on it per-event both drains the executor and preserves the
    lifecycle ordering the adapter depends on (crew-start before task-start ...).
    """
    fut = crewai_event_bus.emit(None, event=event)
    if fut is not None:
        fut.result(timeout=5.0)


def _drive_bus_lifecycle(sentinel: str) -> None:
    """Drive a full crew lifecycle THROUGH THE REAL BUS with content-bearing fields.

    crew start (inputs) -> task start (context) -> tool start/finish (args/output)
    -> task complete (output) -> crew complete (output) + flush. Every content
    slot carries ``sentinel``.
    """
    now = datetime.datetime.now()
    _emit_wait(CrewKickoffStartedEvent(crew_name="floor-crew", inputs={"topic": sentinel}))
    _emit_wait(TaskStartedEvent(context=f"task context {sentinel}", task_name="floor-task", agent_role="researcher"))
    _emit_wait(ToolUsageStartedEvent(tool_name="web_search", tool_args=f"search for {sentinel}", agent_key="r1"))
    _emit_wait(
        ToolUsageFinishedEvent(
            tool_name="web_search",
            tool_args=f"search for {sentinel}",
            started_at=now,
            finished_at=now,
            output=f"found {sentinel}",
        )
    )
    _emit_wait(
        TaskCompletedEvent(
            output=TaskOutput(description="floor-task", raw=f"task result {sentinel}", agent="researcher"),
            task_name="floor-task",
        )
    )
    _emit_wait(
        CrewKickoffCompletedEvent(
            crew_name="floor-crew",
            output=TaskOutput(description="final", raw=f"final result {sentinel}", agent="researcher"),
            total_tokens=100,
        )
    )


def _build_recorded_llm(fixture) -> LLM:
    """A real ``crewai.LLM`` whose native OpenAI provider is backed by a mocked
    transport serving the recorded ChatCompletion — the proven seam from
    test_crewai_recorded.py (no key => skips the eager async-client build)."""
    transport, _ = mock_transport(fixture)
    prior_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        llm = LLM(
            model="openai/gpt-4o-mini",
            client_params={"http_client": httpx.Client(transport=transport)},
        )
    except ImportError as exc:  # pragma: no cover - only on unsupported crewai
        pytest.skip(f"crewai native-provider client_params seam unavailable: {exc}")
    finally:
        if prior_key is not None:
            os.environ["OPENAI_API_KEY"] = prior_key
    llm.api_key = "test-key"
    return llm


# ---------------------------------------------------------------------------
# Real error-shape floor (real openai exception, fired on the real bus)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_openai_error_surfaces_as_agent_error(self, mock_client):
        # A genuine openai SDK exception — the shape a real crewai LLM call raises
        # (crewai populates LLMCallFailedEvent.error with str(exception)). NOT the
        # synthetic "rate limit exceeded" string the existing suite feeds.
        response = httpx.Response(404, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
        err = openai.NotFoundError(
            "Error code: 404 - {'error': {'message': 'The model `gpt-4o-mini-ghost` does not exist', "
            "'type': 'invalid_request_error', 'code': 'model_not_found'}}",
            response=response,
            body=None,
        )
        # Prove it is the real class, not a hand-rolled stand-in.
        assert type(err).__name__ == "NotFoundError"
        assert isinstance(err, openai.OpenAIError)
        real_message = str(err)

        uploaded = capture_framework_trace(mock_client)
        adapter = CrewAIAdapter(mock_client, capture_config=CaptureConfig.full())
        with crewai_event_bus.scoped_handlers():
            adapter.connect()
            # Realistic failure sequence, all fired on the REAL bus.
            _emit_wait(CrewKickoffStartedEvent(crew_name="C", inputs={}))
            _emit_wait(
                LLMCallStartedEvent(model="openai/gpt-4o-mini", messages=[], call_type="llm_call", call_id="c_fail")
            )
            _emit_wait(LLMCallFailedEvent(error=real_message, model="openai/gpt-4o-mini", call_id="c_fail"))
            # The auth/404 failure crashes the crew — its terminal event flushes.
            _emit_wait(CrewKickoffFailedEvent(crew_name="C", error=real_message))
            adapter.disconnect()

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        # The LLM-level error is the one carrying the model.
        llm_errors = [e for e in errors if e["payload"].get("error_type") == "llm_error"]
        assert len(llm_errors) == 1, f"expected exactly one llm_error agent.error, saw {[e['payload'] for e in errors]}"
        payload = llm_errors[0]["payload"]

        # Honest adapter classification for an LLM failure (bite: lost if the
        # adapter stops classifying or stops emitting on llm-failure).
        assert payload["error_type"] == "llm_error"
        assert payload["status"] == "error"
        assert payload["model"] == "openai/gpt-4o-mini"
        # The REAL exception message flows through verbatim (bite: dropped/mangled
        # error text fails here). Tied to the real HTTP status of the class.
        assert payload["error"] == real_message
        assert "404" in payload["error"]
        assert payload["framework"] == "crewai"


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real crew kickoff
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_kickoff(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        adapter = CrewAIAdapter(mock_client, capture_config=CaptureConfig.full())
        with crewai_event_bus.scoped_handlers():
            adapter.connect()
            llm = _build_recorded_llm(fixture)
            agent = Agent(
                role="responder",
                goal="Reply to the user",
                backstory="A terse assistant.",
                llm=llm,
                verbose=False,
                max_iter=1,
            )
            task = Task(description="Reply with exactly: pong", expected_output="pong", agent=agent)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            result = crew.kickoff()
            crewai_event_bus.flush(timeout=10.0)
            adapter.disconnect()

        assert str(result) == "pong"

        events = uploaded["events"]
        assert events, "real crew kickoff must flush a non-empty trace"
        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real crew trace"
        assert len(envelopes) == len(events), (
            f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
        )
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result_ok = verify_chain(envelopes)
        assert result_ok.valid, f"attestation chain invalid: {result_ok.error}"

        # Vacuity control: verify_chain must REJECT a broken link, proving the
        # pass above is not trivially true. (Requires a chain long enough to
        # break at an interior link.)
        assert len(envelopes) >= 2
        tampered = list(envelopes)
        tampered[1] = AttestationEnvelope(
            hash=tampered[1].hash,
            scope=tampered[1].scope,
            previous_hash="sha256:deadbeef-not-the-prior-hash",
        )
        broken = verify_chain(tampered)
        assert not broken.valid and broken.break_index == 1, "verify_chain failed to detect a broken link"


# ---------------------------------------------------------------------------
# Redaction content-absence over a real bus-driven crew lifecycle
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: with capture_content=True the SAME real bus lifecycle
        DOES carry the SENTINEL and the content keys it rides on."""
        uploaded = capture_framework_trace(mock_client)
        adapter = CrewAIAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        with crewai_event_bus.scoped_handlers():
            adapter.connect()
            _drive_bus_lifecycle(SENTINEL)
            adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        assert any("input" in e["payload"] for e in find_events(events, "agent.input"))
        assert any("output" in e["payload"] for e in find_events(events, "agent.output"))
        assert "input" in find_event(events, "tool.call")["payload"]
        assert "output" in find_event(events, "tool.result")["payload"]

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps task/tool/crew content — and the SENTINEL —
        out of every stored event."""
        uploaded = capture_framework_trace(mock_client)
        adapter = CrewAIAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        with crewai_event_bus.scoped_handlers():
            adapter.connect()
            _drive_bus_lifecycle(SENTINEL)
            adapter.disconnect()

        events = uploaded["events"]
        assert events, "the lifecycle must still emit structural events without content"

        # 1) SENTINEL sweep over the serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) The content keys must be absent from every payload that would carry them.
        for e in find_events(events, "agent.input"):
            assert "input" not in e["payload"], "agent.input leaked 'input' under capture_content=False"
            assert "context" not in e["payload"], "agent.input leaked task 'context' under capture_content=False"
        for e in find_events(events, "agent.output"):
            assert "output" not in e["payload"], "agent.output leaked 'output' under capture_content=False"
        assert "input" not in find_event(events, "tool.call")["payload"], "tool.call leaked 'input'"
        assert "output" not in find_event(events, "tool.result")["payload"], "tool.result leaked 'output'"
