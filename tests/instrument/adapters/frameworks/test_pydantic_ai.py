"""Tests for the PydanticAI adapter.

Tests use PydanticAI's TestModel to exercise the real agent loop —
no monkey-patching or mocking of PydanticAI internals (LAY-3567 B2).
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402
from pydantic_ai.models.wrapper import WrapperModel  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.pydantic_ai import (
    PydanticAIAdapter,
)  # noqa: E402

from .conftest import find_event, find_events, record_for_schema_lock, capture_framework_trace  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    name: Optional[str] = None,
    output_text: str = "Hello!",
    model_name: str = "test",  # noqa: ARG001 — accepted for API stability; TestModel no longer exposes this kwarg
    tools: Optional[list] = None,
) -> Agent:
    """Create a PydanticAI Agent with TestModel for deterministic testing."""
    agent = Agent(
        model=TestModel(custom_output_text=output_text),
        name=name,
    )
    if tools:
        for tool_fn in tools:
            agent.tool_plain(tool_fn)
    return agent


def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"72F in {city}"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestPydanticAIAdapterLifecycle:
    def test_connect_wraps_model_and_run_methods(self, mock_client):
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent()
        original_model = agent.model

        adapter.connect(target=agent)

        assert adapter.is_connected
        assert isinstance(agent.model, WrapperModel)
        assert agent.model.wrapped is original_model
        # run methods are shadowed on the instance
        assert "run_sync" in vars(agent)
        info = adapter.adapter_info()
        assert info.name == "pydantic-ai"
        assert info.adapter_type == "framework"
        assert info.connected is True

        adapter.disconnect()

    def test_disconnect_restores_agent(self, mock_client):
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent()
        original_model = agent.model

        adapter.connect(target=agent)
        adapter.disconnect()

        assert not adapter.is_connected
        assert agent.model is original_model
        for method in ("run", "run_sync", "run_stream"):
            assert method not in vars(agent), f"{method} still shadowed after disconnect"

    def test_connect_without_target_raises(self, mock_client):
        adapter = PydanticAIAdapter(mock_client)
        with pytest.raises(ValueError, match="requires a target agent"):
            adapter.connect()

    def test_connect_without_pydantic_ai_raises(self, mock_client, monkeypatch):
        import layerlens.instrument.adapters.frameworks.pydantic_ai as mod

        monkeypatch.setattr(mod, "_HAS_PYDANTIC_AI", False)
        adapter = PydanticAIAdapter(mock_client)
        with pytest.raises(ImportError, match="pydantic-ai"):
            adapter.connect(target=_make_agent())


# ---------------------------------------------------------------------------
# run_sync
# ---------------------------------------------------------------------------


class TestRunSync:
    def test_basic_run(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        agent = _make_agent(output_text="The weather is sunny")

        adapter.connect(target=agent)
        result = agent.run_sync("What is the weather?")
        adapter.disconnect()

        assert result.output == "The weather is sunny"
        events = uploaded["events"]

        inp = find_event(events, "agent.input")
        assert inp["payload"]["framework"] == "pydantic-ai"
        assert inp["payload"]["input"] == "What is the weather?"

        out = find_event(events, "agent.output")
        assert out["payload"]["status"] == "ok"
        assert out["payload"]["output"] == "The weather is sunny"
        assert out["payload"]["latency_ms"] >= 0
        assert out["payload"]["tokens_prompt"] > 0
        assert out["payload"]["tokens_completion"] > 0

        cost = find_event(events, "cost.record")
        assert cost["payload"]["tokens_total"] > 0

    def test_named_agent(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(name="my_agent", output_text="ok")

        adapter.connect(target=agent)
        agent.run_sync("test")
        adapter.disconnect()

        inp = find_event(uploaded["events"], "agent.input")
        assert inp["payload"]["agent_name"] == "my_agent"

    def test_unnamed_agent_does_not_emit_model_as_agent(self, mock_client):
        """An unnamed pydantic Agent must NOT surface the MODEL as its
        agent_name (model-as-agent is the fabrication the Agent column forbids).
        With no declared name, agent_name is omitted — honest "—", not the model."""
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(name=None, output_text="ok")  # no declared name

        adapter.connect(target=agent)
        agent.run_sync("test")
        adapter.disconnect()

        inp = find_event(uploaded["events"], "agent.input")
        model = inp["payload"].get("model")
        an = inp["payload"].get("agent_name")
        # agent_name must not be the model; ideally absent when there is no name.
        assert an != model, f"agent_name {an!r} is the model — model-as-agent"
        assert not an, f"unnamed agent should omit agent_name, got {an!r}"
        # And no honest agent identity is synthesized from a model.
        idents = find_events(uploaded["events"], "agent.identity")
        assert idents == [], "an unnamed, model-only agent must not get a fabricated identity"


# ---------------------------------------------------------------------------
# async run
# ---------------------------------------------------------------------------


class TestRunAsync:
    def test_async_run(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        agent = _make_agent(name="async_agent", output_text="Async result")

        adapter.connect(target=agent)
        result = asyncio.get_event_loop().run_until_complete(agent.run("async test"))
        adapter.disconnect()

        assert result.output == "Async result"

        inp = find_event(uploaded["events"], "agent.input")
        assert inp["payload"]["agent_name"] == "async_agent"
        assert inp["payload"]["input"] == "async test"

        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["status"] == "ok"


# ---------------------------------------------------------------------------
# Streaming (G8 / W4)
# ---------------------------------------------------------------------------


class TestStreaming:
    """Streaming coverage for the ``run_stream`` path.

    pydantic-ai streams in production via ``Agent.run_stream`` (an async
    context manager yielding a ``StreamedRunResult``); under the hood the
    instrumented model's ``_InstrumentedModel.request_stream`` drives the SSE.
    This exercises the real agent loop end-to-end with ``TestModel`` (fake, no
    network) so we pin the *emitted* contract for a streamed run.

    NO-TTFT LIMITATION (documented per task RULE 3): unlike the OpenAI/Anthropic
    provider adapters — which wrap the raw SDK chunk iterator and so can anchor
    ``ttft_ms`` on the first chunk and ``streaming_duration_ms`` on exhaustion —
    pydantic-ai exposes no stable per-chunk hook. The adapter therefore emits a
    SINGLE ``model.invoke`` with ``streaming=True`` from ``request_stream``'s
    ``__aexit__`` (using the ``StreamedRunResult``'s final usage/model), plus an
    ``agent.output`` with ``streaming=True`` from the run wrapper. There is no
    ``ttft_ms`` / ``streaming_duration_ms`` here, and the adapter does NOT
    aggregate per-chunk content (aggregation_added=false).

    STREAMED OUTPUT CONTENT: pydantic-ai's ``StreamedRunResult`` exposes its
    result only via the ``await get_output()`` coroutine — it has no ``.output``
    attribute the way the non-streaming ``AgentRunResult`` does. The streaming
    run wrapper (already async) awaits ``get_output()`` after the consumer's
    ``async with`` body has run and hands the resolved value into
    ``_finish_run_ok``, so the streamed ``agent.output`` carries the run's real
    ``output`` under capture_content=True — same contract as the non-streaming
    path. (Resolution is guarded: a consumer that abandons the stream falls back
    to the honest "no output" rather than crashing the run.)

    We assert: a ``streaming=True`` ``model.invoke`` is emitted and
    ``agent.output`` carries ``streaming=True`` with the resolved ``output``.
    """

    @staticmethod
    def _run_stream(agent: Agent, prompt: str) -> str:
        """Drive ``agent.run_stream`` to completion, returning the final output.

        Consumes the text stream (delta mode) the way a production caller would,
        which is what makes ``_InstrumentedModel.request_stream`` open *and*
        close its async context — the close is where ``model.invoke`` fires.
        """

        async def _go() -> str:
            async with agent.run_stream(prompt) as stream:
                async for _ in stream.stream_text(delta=True):
                    pass
                return await stream.get_output()

        return asyncio.get_event_loop().run_until_complete(_go())

    def test_run_stream_emits_streaming_model_invoke_and_output(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        agent = _make_agent(name="stream_agent", output_text="Streamed hello")

        adapter.connect(target=agent)
        output = self._run_stream(agent, "stream me")
        adapter.disconnect()

        assert output == "Streamed hello"
        events = uploaded["events"]

        # agent.input recorded the prompt for the streamed run.
        inp = find_event(events, "agent.input")
        assert inp["payload"]["agent_name"] == "stream_agent"
        assert inp["payload"]["input"] == "stream me"

        # The model-level streaming invoke is the heart of this test: the
        # instrumented WrapperModel must mark the streamed request as such.
        model_invokes = find_events(events, "model.invoke")
        assert len(model_invokes) >= 1
        streaming_invokes = [m for m in model_invokes if m["payload"].get("streaming") is True]
        assert len(streaming_invokes) >= 1, (
            f"expected a streaming=True model.invoke from request_stream; "
            f"got payloads {[m['payload'] for m in model_invokes]}"
        )
        mi = streaming_invokes[0]["payload"]
        # request_stream still reports the model name + a latency reading.
        assert isinstance(mi["model"], str) and mi["model"]
        assert mi["latency_ms"] >= 0
        # No per-chunk timing is surfaced on the pydantic-ai streaming path
        # (no stable per-chunk hook) — pin that limitation so a future change
        # that adds TTFT updates this test deliberately.
        assert "ttft_ms" not in mi
        assert "streaming_duration_ms" not in mi

        # The run wrapper marks the aggregate agent.output as streamed too.
        out = find_event(events, "agent.output")
        assert out["payload"]["status"] == "ok"
        assert out["payload"]["streaming"] is True
        # STREAMED OUTPUT CONTENT (see class docstring): the run wrapper resolves
        # the StreamedRunResult via ``await get_output()`` (it has no ``.output``
        # attribute), so the streamed agent.output carries the run's real output
        # content under capture_content=True — same as the non-streaming path.
        assert out["payload"]["output"] == "Streamed hello"

    def test_run_stream_records_usage_and_cost(self, mock_client):
        # A streamed run still resolves usage from the StreamedRunResult once
        # the stream completes, so the same cost.record contract as the
        # non-streaming path must hold.
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(name="stream_cost_agent", output_text="tokens please")

        adapter.connect(target=agent)
        self._run_stream(agent, "count my tokens")
        adapter.disconnect()

        events = uploaded["events"]
        out = find_event(events, "agent.output")
        # TestModel always reports some usage; the streamed agent.output carries
        # the normalized token fields.
        assert out["payload"]["tokens_prompt"] > 0
        assert out["payload"]["tokens_completion"] > 0

        cost = find_event(events, "cost.record")
        assert cost["payload"]["tokens_total"] > 0


# ---------------------------------------------------------------------------
# Model invocation events
# ---------------------------------------------------------------------------


class TestModelInvocation:
    def test_model_invoke_emitted(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="hello")

        adapter.connect(target=agent)
        agent.run_sync("hi")
        adapter.disconnect()

        model_invokes = find_events(uploaded["events"], "model.invoke")
        assert len(model_invokes) >= 1
        # TestModel reports its own model name ("test"); we just assert the
        # adapter captured whatever it was, non-empty.
        assert isinstance(model_invokes[0]["payload"]["model"], str)
        assert model_invokes[0]["payload"]["model"]
        assert model_invokes[0]["payload"]["tokens_prompt"] > 0

    def test_model_invoke_with_tools_has_two_calls(self, mock_client):
        """When a tool is called, TestModel makes 2 model requests:
        first to call the tool, then to produce the final text."""
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="Done", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("weather NYC")
        adapter.disconnect()

        model_invokes = find_events(uploaded["events"], "model.invoke")
        assert len(model_invokes) == 2

    def test_model_invoke_emits_response_id_when_provider_supplies_one(self, mock_client):
        """S18/F11: surface the provider's own request id, never fabricated."""

        class _ResponseIdModel(TestModel):
            async def request(self, messages, model_settings, model_request_parameters):
                response = await super().request(messages, model_settings, model_request_parameters)
                response.provider_response_id = "resp_test_123"
                return response

        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = Agent(model=_ResponseIdModel(custom_output_text="hello"))

        adapter.connect(target=agent)
        agent.run_sync("hi")
        adapter.disconnect()

        invoke = find_event(uploaded["events"], "model.invoke")
        assert invoke["payload"]["response_id"] == "resp_test_123"

    def test_model_invoke_has_no_response_id_when_provider_omits_one(self, mock_client):
        """TestModel does not populate provider_response_id — must stay absent, not fabricated."""
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="hello")

        adapter.connect(target=agent)
        agent.run_sync("hi")
        adapter.disconnect()

        invoke = find_event(uploaded["events"], "model.invoke")
        assert "response_id" not in invoke["payload"]


# ---------------------------------------------------------------------------
# Tool events
# ---------------------------------------------------------------------------


class TestToolEvents:
    def test_tool_call_and_result(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="Done", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("weather NYC")
        adapter.disconnect()

        events = uploaded["events"]

        tool_calls = find_events(events, "tool.call")
        assert len(tool_calls) == 1
        assert tool_calls[0]["payload"]["tool_name"] == "get_weather"

        tool_results = find_events(events, "tool.result")
        assert len(tool_results) == 1
        assert tool_results[0]["payload"]["tool_name"] == "get_weather"

    def test_tool_result_has_output(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        agent = _make_agent(output_text="Done", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("weather NYC")
        adapter.disconnect()

        tool_results = find_events(uploaded["events"], "tool.result")
        assert len(tool_results) == 1
        # The output should contain the tool's return value
        assert "72F" in str(tool_results[0]["payload"]["output"])

    def test_tool_result_has_latency(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="Done", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("weather")
        adapter.disconnect()

        tool_results = find_events(uploaded["events"], "tool.result")
        assert len(tool_results) == 1
        assert tool_results[0]["payload"]["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# Span hierarchy
# ---------------------------------------------------------------------------


class TestSpanHierarchy:
    def test_per_step_events_parented_to_root(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="Done", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("weather")
        adapter.disconnect()

        events = uploaded["events"]
        root = find_event(events, "agent.input")
        root_span = root["span_id"]

        for evt in find_events(events, "model.invoke"):
            assert evt["parent_span_id"] == root_span
        for evt in find_events(events, "tool.call"):
            assert evt["parent_span_id"] == root_span
        for evt in find_events(events, "tool.result"):
            assert evt["parent_span_id"] == root_span


# ---------------------------------------------------------------------------
# CaptureConfig gating
# ---------------------------------------------------------------------------


class TestCaptureConfigGating:
    def test_no_content_capture_omits_io(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        config = CaptureConfig(capture_content=False)
        adapter = PydanticAIAdapter(mock_client, capture_config=config)
        agent = _make_agent(output_text="done", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("secret prompt")
        adapter.disconnect()

        events = uploaded["events"]

        inp = find_event(events, "agent.input")
        assert "input" not in inp["payload"]

        tool_calls = find_events(events, "tool.call")
        assert len(tool_calls) >= 1
        assert "input" not in tool_calls[0]["payload"]

        tool_results = find_events(events, "tool.result")
        assert len(tool_results) >= 1
        assert "output" not in tool_results[0]["payload"]

        # cost.record should still exist
        assert len(find_events(events, "cost.record")) == 1

    def test_full_config_includes_content(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        config = CaptureConfig.full()
        adapter = PydanticAIAdapter(mock_client, capture_config=config)
        agent = _make_agent(output_text="Hi Alice", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("greet Alice")
        adapter.disconnect()

        events = uploaded["events"]

        inp = find_event(events, "agent.input")
        assert inp["payload"]["input"] == "greet Alice"

        out = find_event(events, "agent.output")
        assert out["payload"]["output"] == "Hi Alice"

        tool_calls = find_events(events, "tool.call")
        assert "input" in tool_calls[0]["payload"]


# ---------------------------------------------------------------------------
# Multiple runs
# ---------------------------------------------------------------------------


class TestMultipleRuns:
    def test_sequential_runs_separate_traces(self, mock_client):
        import json

        all_uploads: list = []

        def _capture(path: str) -> None:
            with open(path) as f:
                data = json.load(f)
            all_uploads.append(data[0])
            record_for_schema_lock(data[0].get("events", []))

        mock_client.traces.upload.side_effect = _capture

        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="ok")

        adapter.connect(target=agent)
        agent.run_sync("first")
        agent.run_sync("second")
        adapter.disconnect()

        assert len(all_uploads) == 2
        trace_ids = {u["trace_id"] for u in all_uploads}
        assert len(trace_ids) == 2


# ---------------------------------------------------------------------------
# Event structure
# ---------------------------------------------------------------------------


class TestEventStructure:
    def test_event_fields(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(name="test_agent", output_text="ok")

        adapter.connect(target=agent)
        agent.run_sync("hello")
        adapter.disconnect()

        events = uploaded["events"]
        for event in events:
            assert "event_type" in event
            assert "trace_id" in event
            assert "span_id" in event
            assert "sequence_id" in event
            assert "timestamp_ns" in event
            assert "payload" in event

        seq_ids = [e["sequence_id"] for e in events]
        assert seq_ids == sorted(seq_ids)
        assert len(set(seq_ids)) == len(seq_ids)

        trace_ids = {e["trace_id"] for e in events}
        assert len(trace_ids) == 1

    def test_attestation_present(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="ok")

        adapter.connect(target=agent)
        agent.run_sync("test")
        adapter.disconnect()

        assert uploaded.get("trace_id") is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestHonestGraphContract:
    """Lever A honest-graph contract for pydantic-ai.

    The Stratix app renders a trace as an agent GRAPH: the server builds nodes
    from the first HONEST producer-declared agent identity found across the
    trace's events (``payload.agent_name`` among them) and edges from
    ``agent.handoff``. For pydantic-ai the honest node identity is the DECLARED
    ``Agent(name=...)`` and NEVER the model id (model-as-agent is the fabrication
    the Agent column forbids).

    NO-HANDOFF LIMITATION (framework hook does not exist): pydantic-ai has no
    handoff/transition callback. Its "multi-agent" pattern is tool-based agent
    DELEGATION — a sub-agent is invoked as a plain function inside a tool and
    produces its OWN separate trace via its own adapter/collector — so this
    single-target adapter observes no from/to transition. Emitting an
    ``agent.handoff`` would be fabrication, so none is emitted. The graph
    contribution this adapter can honestly make is: attribute the ONE declared
    node identity onto EVERY node-bearing event (``agent.input`` / ``agent.output``
    AND ``model.invoke`` — including the streaming path), and stay BLANK
    everywhere when the agent has no declared name.
    """

    def test_declared_agent_name_attributed_to_model_invoke(self, mock_client):
        """A declared Agent name must ride on model.invoke (not just
        agent.input/output) so the graph engine attributes the model call to the
        honest node — and it must be the DECLARED name, never the model id."""
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(name="finance_agent", output_text="ok", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("weather NYC")
        adapter.disconnect()

        events = uploaded["events"]
        assert find_event(events, "agent.input")["payload"]["agent_name"] == "finance_agent"
        assert find_event(events, "agent.output")["payload"]["agent_name"] == "finance_agent"

        model_invokes = find_events(events, "model.invoke")
        assert model_invokes, "expected at least one model.invoke"
        for mi in model_invokes:
            an = mi["payload"].get("agent_name")
            assert an == "finance_agent", (
                "model.invoke must carry the declared agent_name so the graph "
                f"attributes the model call to the honest node; got {mi['payload']!r}"
            )
            # Never the model-as-agent anti-pattern.
            assert an != mi["payload"].get("model")

    def test_declared_agent_name_attributed_to_streaming_model_invoke(self, mock_client):
        """The streaming model.invoke emitted from request_stream must carry the
        declared agent_name too — the graph must not lose the node on a
        streamed run."""
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        agent = _make_agent(name="stream_named_agent", output_text="Streamed")

        async def _go() -> str:
            async with agent.run_stream("stream me") as stream:
                async for _ in stream.stream_text(delta=True):
                    pass
                return await stream.get_output()

        adapter.connect(target=agent)
        asyncio.get_event_loop().run_until_complete(_go())
        adapter.disconnect()

        events = uploaded["events"]
        streaming_invokes = [m for m in find_events(events, "model.invoke") if m["payload"].get("streaming") is True]
        assert streaming_invokes, "expected a streaming=True model.invoke"
        for mi in streaming_invokes:
            an = mi["payload"].get("agent_name")
            assert an == "stream_named_agent", (
                f"streaming model.invoke must carry the declared agent_name; got {mi['payload']!r}"
            )
            assert an != mi["payload"].get("model")

    def test_unnamed_agent_stays_blank_on_every_node_event(self, mock_client):
        """An unnamed Agent has no honest identity: NO event (agent.input,
        agent.output, model.invoke) may carry an agent_name, none may equal the
        model, no agent.identity is synthesized, and no handoff is fabricated."""
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(name=None, output_text="ok", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("weather NYC")
        adapter.disconnect()

        events = uploaded["events"]
        for event_type in ("agent.input", "agent.output", "model.invoke"):
            for e in find_events(events, event_type):
                an = e["payload"].get("agent_name")
                assert not an, f"{event_type} fabricated agent_name {an!r} on an unnamed agent"
                assert an != e["payload"].get("model")

        assert find_events(events, "agent.identity") == [], (
            "an unnamed, model-only agent must not get a fabricated identity"
        )
        # pydantic-ai has no handoff hook — never fabricate an edge.
        assert find_events(events, "agent.handoff") == []


class TestEdgeCases:
    def test_empty_prompt(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="ok")

        adapter.connect(target=agent)
        agent.run_sync("")
        adapter.disconnect()

        inp = find_event(uploaded["events"], "agent.input")
        assert inp["payload"]["framework"] == "pydantic-ai"

    def test_pydantic_model_output(self, mock_client):
        from pydantic import BaseModel

        class CityInfo(BaseModel):
            city: str
            temp: int

        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        agent = Agent(
            model=TestModel(custom_output_args={"city": "NYC", "temp": 72}),
            output_type=CityInfo,
        )

        adapter.connect(target=agent)
        result = agent.run_sync("weather")
        adapter.disconnect()

        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["output"] == {"city": "NYC", "temp": 72}

    def test_zero_token_usage_still_has_tokens(self, mock_client):
        """TestModel always produces some tokens, so we verify they're present."""
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="ok")

        adapter.connect(target=agent)
        agent.run_sync("test")
        adapter.disconnect()

        out = find_event(uploaded["events"], "agent.output")
        # TestModel always has some token usage
        assert "tokens_prompt" in out["payload"]
        assert len(find_events(uploaded["events"], "cost.record")) == 1

    def test_disconnect_idempotent(self, mock_client):
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent()
        adapter.connect(target=agent)
        adapter.disconnect()
        adapter.disconnect()  # should not raise

    def test_string_model_agent(self, mock_client):
        """Agents built with a model *string* (the documented usage, e.g.
        ``Agent("openai:gpt-4o-mini")``) must connect and instrument too."""
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = Agent("test")
        original_model = agent.model

        adapter.connect(target=agent)
        agent.run_sync("hello")
        adapter.disconnect()

        assert agent.model is original_model
        events = uploaded["events"]
        assert find_event(events, "agent.input")["payload"]["framework"] == "pydantic-ai"
        model_invokes = find_events(events, "model.invoke")
        assert len(model_invokes) >= 1
        assert model_invokes[0]["payload"]["model"]

    def test_tool_error_emits_agent_error(self, mock_client):
        def explode(text: str) -> str:
            """Always fails."""
            raise ValueError("boom from tool")

        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        agent = _make_agent(output_text="never", tools=[explode])

        adapter.connect(target=agent)
        with pytest.raises(Exception, match="boom from tool"):
            agent.run_sync("trigger the tool")
        adapter.disconnect()

        errors = find_events(uploaded["events"], "agent.error")
        assert len(errors) >= 1
        assert "boom from tool" in errors[0]["payload"]["error"]
