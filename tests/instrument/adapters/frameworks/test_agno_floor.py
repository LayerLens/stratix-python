"""Offline redaction + error + attestation + cost + config-allowlist floor for the Agno adapter.

Closes the W2 census cells that ``test_agno.py`` proves only via synthetic
inputs (hand-built ``RuntimeError``/``ValueError``, no SENTINEL sweep, no offline
``verify_chain`` attestation, no cost-with-``cost_usd`` assertion, no config
allowlist sweep) so a regression fails in plain CI with no credentials and no
network. Every object is a REAL agno object (``Agent``/``Team``/``RunMetrics``/
``ToolExecution``/``OpenAIChat``); the only mock is the network boundary
(``httpx.MockTransport`` for the recorded OpenAI body).

Classes:

* ``TestRedactionFloor`` — a real agno run lifecycle (agent I/O + a real
  ``ToolExecution``) with ``capture_content=False`` keeps every structural event
  but strips input/output/tool content — verified by a SENTINEL sweep over
  ``json.dumps(events)`` — with a ``capture_content=True`` vacuity control proving
  the same path DOES carry the content otherwise.
* ``TestRealErrorShape`` — a REAL ``openai.NotFoundError`` object (the SDK class,
  not the synthetic ``RuntimeError`` the existing suite feeds) propagated through
  the adapter's run wrapper surfaces on ``agent.output`` with the honest
  ``error_type == "NotFoundError"`` and the real message verbatim.

  NOTE ON SCOPE (documented, see the held finding in this task's report): a *real*
  agno model/provider error does NOT raise out of ``Agent.run()`` — agno swallows
  it into ``RunOutput.status == RunStatus.error`` and returns normally. The
  adapter does not inspect ``result.status``, so that real-provider-error path is
  a separate source bug held for the source owner; this floor covers the run
  wrapper's exception-classification path (the defensive net) with a real SDK
  exception object.
* ``TestAttestationOffline`` — a real agno run flushes a trace whose attestation
  chain reconstructs and ``verify_chain(...)`` returns valid; a tamper control
  breaking link 1 proves the check is not vacuous.
* ``TestCostFloor`` — a real agno ``Agent`` + ``OpenAIChat`` over the recorded
  OpenAI body (real ``RunMetrics`` 12/1/13) emits a ``cost.record`` carrying a
  computed ``cost_usd`` (Group-B adjudication: GREEN — the framework base pricing
  path prices any model in ``PRICING`` from the real token counts).
* ``TestConfigAllowlist`` — ``environment.config`` surfaces ONLY the declared,
  non-generic roster (a generic ``agno_agent`` member and an unnamed member are
  dropped) and never carries run I/O even under ``capture_content=True``.
* ``TestConcurrencyIsolation`` — two ``arun()`` coroutines interleaved on ONE
  adapter instance produce two isolated traces (distinct trace_ids, no
  cross-contamination) via the adapter's per-run ContextVar collector.
"""

from __future__ import annotations

import json
import asyncio
from typing import Any, Iterator, Optional

import httpx
import pytest

agno = pytest.importorskip("agno")

from agno.metrics import RunMetrics  # noqa: E402
from agno.run.agent import RunContentEvent, RunCompletedEvent  # noqa: E402
from agno.team.team import Team  # noqa: E402
from agno.agent.agent import Agent  # noqa: E402
from agno.models.base import Model  # noqa: E402
from agno.models.openai import OpenAIChat  # noqa: E402
from agno.models.response import ModelResponse, ToolExecution  # noqa: E402

import openai  # noqa: E402
from layerlens.models import CreateTracesResponse  # noqa: E402
from layerlens.instrument._upload import shutdown_uploads  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter  # noqa: E402
from layerlens.instrument.adapters.providers.pricing import PRICING, calculate_cost  # noqa: E402
from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage  # noqa: E402

from .conftest import find_event, record_for_schema_lock, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ---------------------------------------------------------------------------
# A real agno Model subclass — deterministic, no network. Rolls its usage onto
# the run's RunMetrics exactly like a real model so the adapter's post-hoc
# RunMetrics extraction runs for real. (Mirrors test_agno.py::_TestModel; a copy,
# because private test helpers are not imported across test modules.)
# ---------------------------------------------------------------------------
class _FloorModel(Model):
    def __init__(
        self,
        *,
        content: str = "ok",
        input_tokens: int = 10,
        output_tokens: int = 5,
        model_id: str = "gpt-4o-mini",
        delay: float = 0.0,
    ) -> None:
        super().__init__(id=model_id, name="FloorModel", provider="test")
        self._content = content
        self._in = input_tokens
        self._out = output_tokens
        self._delay = delay

    def _resp(self) -> ModelResponse:
        return ModelResponse(
            content=self._content,
            input_tokens=self._in,
            output_tokens=self._out,
            total_tokens=self._in + self._out,
        )

    def invoke(self, *a: Any, **k: Any) -> ModelResponse:
        return self._resp()

    async def ainvoke(self, *a: Any, **k: Any) -> ModelResponse:
        return self._resp()

    def invoke_stream(self, *a: Any, **k: Any) -> Iterator[ModelResponse]:
        yield self._resp()

    async def ainvoke_stream(self, *a: Any, **k: Any):  # type: ignore[override]
        yield self._resp()

    def _parse_provider_response(self, response: Any, **k: Any) -> ModelResponse:
        return self._resp()

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return self._resp()

    def response(self, messages: Any, **k: Any) -> ModelResponse:
        resp = self._resp()
        rr = k.get("run_response")
        if rr and rr.metrics:
            rr.metrics.input_tokens += resp.input_tokens or 0
            rr.metrics.output_tokens += resp.output_tokens or 0
            rr.metrics.total_tokens += resp.total_tokens or 0
        return resp

    async def aresponse(self, messages: Any, **k: Any) -> ModelResponse:
        if self._delay:
            await asyncio.sleep(self._delay)
        return self.response(messages, **k)


def _agent(name: Optional[str] = "floor_agent", model: Optional[_FloorModel] = None) -> Agent:
    a = Agent(model=model or _FloorModel(), name=name or "floor_agent")
    if name is None:
        a.name = None
    return a


class _InjectedResult:
    """A stand-in RunOutput carrying real agno RunMetrics/ToolExecution objects.

    The injected-result pattern (used throughout test_agno.py) drives the adapter
    against REAL agno objects without a live model producing the run — the
    adapter's content-gating / token / tool parsing runs identically to a real
    RunOutput."""

    def __init__(self, content: Any, metrics: Any, tools: Any) -> None:
        self.content = content
        self.metrics = metrics
        self.tools = tools


def _drive_injected_run(mock_client, config, *, content, tool_args, tool_result, message):
    """Connect to a real Agent, swap its run for one returning a real
    metrics/tool-bearing result, drive run(message), return uploaded events."""
    agent = _agent()
    uploaded = capture_framework_trace(mock_client)
    adapter = AgnoAdapter(mock_client, capture_config=config)
    adapter.connect(target=agent)
    adapter._unwrap_agent(agent)
    adapter._originals.pop(id(agent), None)
    result = _InjectedResult(
        content=content,
        metrics=RunMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        tools=[ToolExecution(tool_name="web_search", tool_args=tool_args, result=tool_result)],
    )
    agent.run = lambda *a, **kw: result
    adapter._instrument_agent(agent)
    agent.run(message)
    adapter.disconnect()
    return uploaded


# ---------------------------------------------------------------------------
# Redaction content-absence over a real agno run lifecycle + SENTINEL sweep
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: with capture_content=True the SAME lifecycle DOES
        carry the SENTINEL and the content keys it rides on."""
        uploaded = _drive_injected_run(
            mock_client,
            CaptureConfig(capture_content=True),
            content=f"answer {SENTINEL}",
            tool_args={"q": f"find {SENTINEL}"},
            tool_result=f"found {SENTINEL}",
            message=f"remember {SENTINEL}",
        )
        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        assert find_event(events, "agent.input")["payload"]["input"] == f"remember {SENTINEL}"
        assert find_event(events, "agent.output")["payload"]["output"] == f"answer {SENTINEL}"
        assert "input" in find_event(events, "tool.call")["payload"]
        assert "output" in find_event(events, "tool.result")["payload"]

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps every structural event but strips agent
        I/O + tool content — and the SENTINEL — out of the stored trace."""
        uploaded = _drive_injected_run(
            mock_client,
            CaptureConfig(capture_content=False),
            content=f"answer {SENTINEL}",
            tool_args={"q": f"find {SENTINEL}"},
            tool_result=f"found {SENTINEL}",
            message=f"remember {SENTINEL}",
        )
        events = uploaded["events"]

        # Structural events all still emitted (redaction removes CONTENT, not shape).
        assert find_event(events, "agent.input") is not None
        assert find_event(events, "agent.output") is not None
        assert find_event(events, "model.invoke") is not None
        assert find_event(events, "tool.call") is not None
        assert find_event(events, "tool.result") is not None

        # 1) SENTINEL sweep over the serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) The content keys must be absent from every payload that would carry them.
        assert "input" not in find_event(events, "agent.input")["payload"], "agent.input leaked 'input'"
        assert "output" not in find_event(events, "agent.output")["payload"], "agent.output leaked 'output'"
        assert "input" not in find_event(events, "tool.call")["payload"], "tool.call leaked 'input'"
        assert "output" not in find_event(events, "tool.result")["payload"], "tool.result leaked 'output'"


# ---------------------------------------------------------------------------
# Real error-shape floor (a real openai SDK exception through the run wrapper)
# ---------------------------------------------------------------------------
def _real_notfound() -> openai.NotFoundError:
    resp = httpx.Response(404, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    return openai.NotFoundError(
        "Error code: 404 - {'error': {'message': 'The model `gpt-4o-mini-ghost` does not exist', "
        "'type': 'invalid_request_error', 'code': 'model_not_found'}}",
        response=resp,
        body=None,
    )


class TestRealErrorShape:
    def test_real_openai_exception_stamps_honest_error_type(self, mock_client):
        # A genuine openai SDK exception object — NOT the synthetic RuntimeError the
        # existing suite feeds. (Agno swallows real provider errors into
        # RunStatus.error rather than raising — see the module docstring / held
        # finding; this exercises the adapter's exception-classification net.)
        err = _real_notfound()
        assert type(err).__name__ == "NotFoundError"
        assert isinstance(err, openai.OpenAIError)
        real_message = str(err)

        agent = _agent(name="err_agent")

        def _raising(*a: Any, **kw: Any) -> Any:
            raise err

        # The underlying run raises the REAL SDK exception; the adapter wraps it.
        agent.run = _raising
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        with pytest.raises(openai.NotFoundError):
            agent.run("hi")
        adapter.disconnect()

        out = find_event(uploaded["events"], "agent.output")
        # Honest classification from the REAL exception class (bite: lost if the
        # adapter stops stamping type(error).__name__ or stops emitting on error).
        assert out["payload"]["error_type"] == "NotFoundError"
        # The real exception message flows through verbatim (bite: dropped/mangled).
        assert out["payload"]["error"] == real_message
        assert "404" in out["payload"]["error"]
        assert out["payload"]["framework"] == "agno"


# ---------------------------------------------------------------------------
# Swallowed-provider-error floor — agno does NOT raise a real model error out of
# Agent.run(); it returns RunOutput.status == RunStatus.error with the error text
# in .content. The adapter must record that as a FAILED run, not a healthy output.
# ---------------------------------------------------------------------------
class TestSwallowedProviderError:
    def test_real_provider_error_surfaces_on_output(self, mock_client):
        # A REAL 404 from the OpenAI SDK (over MockTransport). agno swallows the
        # exception into RunStatus.error and returns normally — the run wrapper
        # never sees a raised exception, so only result.status marks the failure.
        model = OpenAIChat(
            id="gpt-4o-mini-ghost",
            api_key="test-key",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(
                        404,
                        json={
                            "error": {
                                "message": "The model `gpt-4o-mini-ghost` does not exist",
                                "type": "invalid_request_error",
                                "code": "model_not_found",
                            }
                        },
                    )
                )
            ),
        )
        agent = Agent(model=model, name="err_agent", telemetry=False)
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        result = agent.run("hi")  # agno swallows -> RunStatus.error, returns normally
        adapter.disconnect()

        # Sanity: agno DID mark the run failed (and did NOT raise).
        assert str(result.status).endswith("error")

        out = find_event(uploaded["events"], "agent.output")
        # BITE: a failed run must carry status=error + an honest error_type, not
        # be recorded as an ordinary successful output (both keys absent today).
        assert out["payload"].get("status") == "error"
        assert out["payload"].get("error_type") is not None
        # The provider error text is surfaced on `error` — not silently stashed in
        # `output` as though it were the agent's answer.
        assert "does not exist" in out["payload"].get("error", "")
        assert "output" not in out["payload"]


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real agno run
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_run(self, mock_client):
        agent = _agent(name="attest_agent", model=_FloorModel(content="pong", input_tokens=12, output_tokens=1))
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        agent.run("ping")
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "real agno run must flush a non-empty trace"
        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real agno trace"
        assert len(envelopes) == len(events), (
            f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
        )
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result_ok = verify_chain(envelopes)
        assert result_ok.valid, f"attestation chain invalid: {result_ok.error}"

        # Vacuity control: verify_chain must REJECT a broken interior link.
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
# Cost floor — cost_usd on a real token shape (Group-B adjudication: GREEN)
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_usd_present_on_real_token_shape(self, mock_client):
        # Real agno Agent + OpenAIChat over the recorded OpenAI body: the real
        # RunMetrics reports 12/1/13 parsed off the recorded usage{} block.
        fixture = load_recorded("openai", "default")
        transport, _ = mock_transport(fixture)
        model = OpenAIChat(id="gpt-4o-mini", api_key="test-key", http_client=httpx.Client(transport=transport))
        agent = Agent(model=model, name="cost_agent", telemetry=False)

        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client, capture_config=CaptureConfig.full())
        agent = adapter.connect(target=agent)
        result = agent.run("Reply with exactly: pong")
        adapter.disconnect()
        assert result.content == "pong"

        cost = find_event(uploaded["events"], "cost.record")
        assert cost["payload"]["tokens_total"] == 13
        assert cost["payload"]["model"] == "gpt-4o-mini"
        # cost_usd is computed by the framework base pricing path from model+tokens.
        assert "cost_usd" in cost["payload"], "cost.record carries no cost_usd for a priced model"
        expected = calculate_cost(
            "gpt-4o-mini",
            NormalizedTokenUsage(prompt_tokens=12, completion_tokens=1, total_tokens=13),
            PRICING,
        )
        assert expected is not None and expected > 0
        assert cost["payload"]["cost_usd"] == expected


# ---------------------------------------------------------------------------
# Config-roster allowlist — only declared, non-generic members are surfaced
# ---------------------------------------------------------------------------
class TestConfigAllowlist:
    def _mixed_team(self) -> Team:
        # A declared name, a GENERIC placeholder (agno_agent, on the denylist),
        # and an unnamed member — only the declared non-generic one is permitted.
        generic = _agent(name="agno_agent")
        unnamed = _agent(name=None)
        return Team(
            members=[_agent(name="researcher"), generic, unnamed],
            name="research_team",
            model=_FloorModel(),
        )

    def test_config_roster_allowlist_drops_generic_and_unnamed(self, mock_client):
        team = self._mixed_team()
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect(target=team)
        adapter._unwrap_agent(team)
        adapter._originals.pop(id(team), None)
        team.run = lambda *a, **kw: _InjectedResult(
            content="final",
            metrics=RunMetrics(input_tokens=1, output_tokens=1, total_tokens=2),
            tools=None,
        )
        adapter._instrument_agent(team)
        team.run("do research")
        adapter.disconnect()

        cfg = find_event(uploaded["events"], "environment.config")
        # ALLOWLIST BITE: generic + unnamed members are filtered out of the roster.
        assert cfg["payload"]["config"]["team_members"] == ["researcher"]
        assert cfg["payload"]["config"]["model"] == "gpt-4o-mini"
        assert cfg["payload"]["agent_name"] == "research_team"
        # Config is content-free: never carries run I/O even under capture_content=True.
        assert "input" not in cfg["payload"]
        assert "output" not in cfg["payload"]


# ---------------------------------------------------------------------------
# Interleaved-run isolation — per-run ContextVar collector (conc_async cell)
# ---------------------------------------------------------------------------
def _collect_separate_traces(mock_client):
    """Accumulate SEPARATE trace payloads per upload (one per run)."""
    traces = []

    def _capture(path: str) -> CreateTracesResponse:
        with open(path) as f:
            data = json.load(f)
        traces.append(data[0])
        record_for_schema_lock(data[0].get("events", []))
        # Non-empty trace_ids: an empty/None return is treated as a REJECT and
        # would drop the trace from the isolation check.
        return CreateTracesResponse(trace_ids=[data[0].get("trace_id") or "mock-trace-id"])

    mock_client.traces.upload.side_effect = _capture
    return traces


# ---------------------------------------------------------------------------
# Streaming-run floor — agent.run(stream=True) / arun(stream=True) return a lazy
# RunOutputEvent generator. The run's telemetry must be emitted once the CALLER
# drains the stream (accumulated content + a priced cost.record), never read off
# the generator itself before consumption — which records a BLANK run (empty
# agent.output, no model.invoke, no cost.record) because .content/.metrics are
# read off the generator object, not the aggregated RunOutput.
# ---------------------------------------------------------------------------
_STREAM_SSE = (
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
    '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
    '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"pong"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
    '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
    '"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":1,"total_tokens":13}}\n\n'
    "data: [DONE]\n\n"
)


def _streaming_openai_agent() -> Agent:
    """A real agno Agent + OpenAIChat whose streaming call is served the recorded
    SSE above (real chunk deltas + a real ``usage`` chunk) over MockTransport — so
    the run's real ``RunCompletedEvent`` carries a real ``RunMetrics`` 12/1/13, no
    network."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_STREAM_SSE.encode()
        )

    model = OpenAIChat(
        id="gpt-4o-mini",
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return Agent(model=model, name="stream_agent", telemetry=False)


class TestStreamingRun:
    def test_streamed_run_emits_output_and_priced_cost(self, mock_client):
        # A REAL agno streaming run: agent.run(stream=True) returns a lazy
        # RunOutputEvent generator the caller drains. stream_events=True surfaces
        # the terminal RunCompletedEvent carrying the aggregated content + token
        # metrics (agno gates that event behind stream_events — it is how an
        # observability-instrumented streaming customer sees the run totals).
        agent = _streaming_openai_agent()
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client, capture_config=CaptureConfig.full())
        agent = adapter.connect(target=agent)

        stream = agent.run("ping", stream=True, stream_events=True)
        events_seen = list(stream)  # drain -> the proxy accumulates + emits on exhaustion
        adapter.disconnect()
        shutdown_uploads(10.0)

        # Transparency: the proxy yielded every source event unchanged.
        assert any(getattr(e, "event", None) == "RunCompleted" for e in events_seen), (
            "the streamed generator must reach the caller unchanged"
        )
        assert any(getattr(e, "content", None) == "pong" for e in events_seen)

        events = uploaded["events"]
        # BITE: today _wrap_sync passes the generator to _on_run_end BEFORE the
        # caller consumes it, so agent.output carries no output and there is no
        # model.invoke / cost.record (the streamed run records blank).
        out = find_event(events, "agent.output")
        assert out is not None and out["payload"].get("output") == "pong", (
            "streamed run recorded a BLANK agent.output"
        )
        assert find_event(events, "model.invoke") is not None, "streamed run emitted no model.invoke"
        cost = find_event(events, "cost.record")
        assert cost is not None, "streamed run emitted no cost.record"
        assert cost["payload"]["tokens_total"] == 13
        assert cost["payload"]["model"] == "gpt-4o-mini"
        # cost_usd is priced from the streamed run's real token counts.
        assert "cost_usd" in cost["payload"], "streamed cost.record carries no cost_usd"
        expected = calculate_cost(
            "gpt-4o-mini",
            NormalizedTokenUsage(prompt_tokens=12, completion_tokens=1, total_tokens=13),
            PRICING,
        )
        assert expected is not None and expected > 0
        assert cost["payload"]["cost_usd"] == expected

    def test_streamed_arun_emits_output_and_priced_cost(self, mock_client):
        # The async streaming path (arun(stream=True) -> async generator) shares
        # the same root cause and today additionally CRASHES (the wrapper awaits an
        # async_generator). Inject a REAL agno RunOutputEvent async-stream (real
        # RunContent deltas + a terminal RunCompletedEvent carrying a real
        # RunMetrics) — the exact event shape a live arun(stream=True) yields — and
        # prove the async proxy accumulates content + emits a priced cost.record
        # once the caller drains it.
        agent = _agent(name="astream_agent")  # model id gpt-4o-mini

        async def _fake_astream(*a: Any, **kw: Any):
            yield RunContentEvent(content="po")
            yield RunContentEvent(content="ng")
            yield RunCompletedEvent(
                content="pong",
                metrics=RunMetrics(input_tokens=12, output_tokens=1, total_tokens=13),
            )

        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        adapter._unwrap_agent(agent)
        adapter._originals.pop(id(agent), None)
        agent.arun = _fake_astream
        adapter._instrument_agent(agent)

        async def _drive():
            seen = []
            async for ev in agent.arun("ping", stream=True):
                seen.append(ev)
            return seen

        seen = asyncio.run(_drive())
        adapter.disconnect()
        shutdown_uploads(10.0)

        assert any(getattr(e, "event", None) == "RunCompleted" for e in seen), (
            "the streamed async generator must reach the caller unchanged"
        )
        events = uploaded["events"]
        out = find_event(events, "agent.output")
        assert out is not None and out["payload"].get("output") == "pong", (
            "streamed arun recorded a BLANK agent.output"
        )
        cost = find_event(events, "cost.record")
        assert cost is not None, "streamed arun emitted no cost.record"
        assert cost["payload"]["tokens_total"] == 13
        assert cost["payload"]["model"] == "gpt-4o-mini"
        assert "cost_usd" in cost["payload"], "streamed arun cost.record carries no cost_usd"


class TestConcurrencyIsolation:
    def test_interleaved_arun_produces_isolated_traces(self, mock_client):
        """Two arun() coroutines whose lifecycles interleave (both begin before
        either ends, forced by the model's await) must produce two isolated
        traces — distinct trace_ids, each carrying only its own run's marker."""
        traces = _collect_separate_traces(mock_client)
        adapter = AgnoAdapter(mock_client, capture_config=CaptureConfig.full())
        agent_a = Agent(model=_FloorModel(content="alpha-out", delay=0.03), name="agent_a")
        agent_b = Agent(model=_FloorModel(content="bravo-out", delay=0.03), name="agent_b")
        adapter.connect(target=agent_a)
        adapter.connect(target=agent_b)  # ONE adapter instance wraps both agents

        async def _go():
            return await asyncio.gather(agent_a.arun("alpha-in"), agent_b.arun("bravo-in"))

        asyncio.run(_go())
        adapter.disconnect()

        shutdown_uploads(10.0)
        assert len(traces) == 2, f"interleaved runs merged or lost a trace: got {len(traces)}"
        assert len({t["trace_id"] for t in traces}) == 2, "traces must have distinct trace_ids"
        for marker in ("alpha", "bravo"):
            own = [t for t in traces if marker in json.dumps(t["events"])]
            assert len(own) == 1, f"run marker {marker!r} must appear in exactly 1 trace, found {len(own)}"
            other = "bravo" if marker == "alpha" else "alpha"
            assert other not in json.dumps(own[0]["events"]), (
                f"trace for run {marker!r} is contaminated with run {other!r} events"
            )
