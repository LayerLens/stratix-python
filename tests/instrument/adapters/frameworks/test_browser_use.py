"""Tests for the BrowserUse framework adapter.

Drives the REAL browser-use objects — a real ``browser_use.Agent`` (constructed
offline with a real ``ChatOpenAI``), the real ``AgentHistoryList`` /
``AgentHistory`` / ``AgentOutput`` / ``ActionModel`` / ``ActionResult`` /
``BrowserStateHistory`` / ``UsageSummary`` shapes the adapter walks, and the
real bound-method wrap on ``agent.run``. The ONLY thing stubbed is the network:
``Agent.run`` never drives a browser, so the coroutine is replaced with one that
returns a real history the framework itself would have produced.

This is the lane the census called highest-value: it pins the REAL shapes, so a
browser-use version rename (``go_to_url`` -> ``navigate``, ``agent.browser``
-> ``browser_profile``, the dropped ``__version__``) fails here rather than in
production, and it needs no browser binary and no API key.
"""

from __future__ import annotations

import os
import sys
import json
import asyncio

import pytest

if sys.version_info < (3, 11):  # browser-use requires >= 3.11 (dist metadata)
    pytest.skip("browser-use requires Python >= 3.11", allow_module_level=True)

pytest.importorskip("browser_use")

# The real Agent/ChatOpenAI construct offline but ChatOpenAI wants a key present.
os.environ.setdefault("OPENAI_API_KEY", "sk-browser-use-unit-test-not-real")

from browser_use.agent.views import (  # noqa: E402
    AgentOutput,
    ActionResult,
    AgentHistory,
    StepMetadata,
    AgentHistoryList,
)
from browser_use.tokens.views import UsageSummary  # noqa: E402
from browser_use.agent.service import Agent  # noqa: E402
from browser_use.browser.views import BrowserStateHistory  # noqa: E402
from browser_use.tools.service import Tools  # noqa: E402
from browser_use.browser.profile import BrowserProfile  # noqa: E402
from browser_use.llm.openai.chat import ChatOpenAI  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.browser_use import (  # noqa: E402
    BrowserUseAdapter,
    instrument_browser_use,
    uninstrument_browser_use,
)

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

# ---------------------------------------------------------------------------
# Real browser-use object builders
# ---------------------------------------------------------------------------

_ACTION_MODEL = Tools().registry.create_action_model()
_AGENT_OUTPUT = AgentOutput.type_with_custom_actions(_ACTION_MODEL)


def _action(**spec):
    """A REAL ActionModel — 0.13's RootModel dumps to ``{action_name: params}``."""
    return _ACTION_MODEL(**spec)


def _step(actions, results, url="https://example.com/docs"):
    """A REAL AgentHistory step with real state/metadata."""
    return AgentHistory(
        model_output=_AGENT_OUTPUT(
            action=actions,
            evaluation_previous_goal="looking good",
            memory="m",
            next_goal="n",
        ),
        result=results,
        state=BrowserStateHistory(url=url, title="Docs", tabs=[], interacted_element=[], screenshot_path=None),
        metadata=StepMetadata(step_start_time=1.0, step_end_time=2.5, step_number=1),
    )


def _usage(prompt=120, completion=40, total=160, cost=0.0021, cached=0, by_model=None):
    """A REAL UsageSummary — the shape ``AgentHistoryList.usage`` carries."""
    return UsageSummary(
        total_prompt_tokens=prompt,
        total_prompt_cost=0.001,
        total_prompt_cached_tokens=cached,
        total_prompt_cached_cost=0.0,
        total_prompt_cache_creation_tokens=0,
        total_prompt_cache_creation_cost=0.0,
        total_completion_tokens=completion,
        total_completion_cost=0.0011,
        total_tokens=total,
        total_cost=cost,
        entry_count=3,
        by_model=by_model or {},
    )


def _zero_usage():
    """The REAL all-zero UsageSummary a run with no recorded LLM entry carries.

    Verbatim the shape ``TokenCost().get_usage_summary()`` returns on 0.13 when
    ``entry_count == 0``, and ``Agent.run`` assigns it to ``history.usage``
    UNCONDITIONALLY — including on the KeyboardInterrupt path — so any run that
    crashes (or is Ctrl-C'd) before its first LLM call carries exactly this.
    """
    return UsageSummary(
        total_prompt_tokens=0,
        total_prompt_cost=0.0,
        total_prompt_cached_tokens=0,
        total_prompt_cached_cost=0.0,
        total_prompt_cache_creation_tokens=0,
        total_prompt_cache_creation_cost=0.0,
        total_completion_tokens=0,
        total_completion_cost=0.0,
        total_tokens=0,
        total_cost=0.0,
        entry_count=0,
        by_model={},
    )


def _history(steps=None, usage=None):
    return AgentHistoryList(history=list(steps or []), usage=usage)


def _drifted_history(steps=None, usage=None):
    """A REAL AgentHistoryList carrying an off-shape usage/step.

    ``model_construct`` skips validation so a genuinely different browser-use
    VERSION (whose UsageSummary names differ, or whose step is corrupt) can be
    simulated on the real class — the adapter duck-types these deliberately, and
    the current pydantic model would reject them at construction.
    """
    return AgentHistoryList.model_construct(history=list(steps or []), usage=usage)


def _drifted_step(actions, results, url="https://example.com/docs"):
    return AgentHistory.model_construct(
        model_output=_AGENT_OUTPUT.model_construct(action=actions),
        result=results,
        state=BrowserStateHistory(url=url, title="Docs", tabs=[], interacted_element=[], screenshot_path=None),
        metadata=StepMetadata(step_start_time=1.0, step_end_time=2.5, step_number=1),
    )


def _default_history():
    """Two real steps: a navigate then an extract.

    NB the real ``ActionResult`` validates that ``success=True`` is only legal
    with ``is_done=True`` — a regular action that succeeds leaves ``success``
    as None, and only a failure sets ``success=False``. The adapter therefore
    omits ``success`` on most real actions, which is the honest outcome.
    """
    return _history(
        steps=[
            _step(
                [_action(navigate={"url": "https://example.com/docs", "new_tab": False})],
                [ActionResult(extracted_content=None)],
            ),
            _step(
                [_action(extract={"query": "the install command", "extract_links": False})],
                [ActionResult(extracted_content="pip install layerlens")],
            ),
        ],
        usage=_usage(),
    )


def _make_agent(task="find the install command", llm=None):
    """A REAL browser_use.Agent, constructed offline (no browser is started
    until run(), and we never let the real run() execute).

    ``headless`` is set EXPLICITLY: browser_use leaves it ``None`` by default and
    resolves it at profile build time to "headful if a display is available", so
    an unset value is False on a developer machine and True on a headless CI
    runner. Pinning it makes the config-read assertion deterministic AND stronger
    — it proves the adapter reads the profile's real value, not the ambient default.
    """
    return Agent(
        task=task,
        llm=llm or ChatOpenAI(model="gpt-4o"),
        browser_profile=BrowserProfile(headless=False),
    )


def _stub_run(agent, history=None, error=None):
    """Replace the REAL Agent.run coroutine with one that returns a real history
    (or raises). This is the network/browser boundary — the ONLY stub."""
    result = history if history is not None else _default_history()

    async def _run(max_steps: int = 500, on_step_start=None, on_step_end=None):
        if error is not None:
            # Real browser-use records its partial history on the agent even
            # when the run raises — mirror that so the error path is honest.
            agent.history = result
            raise error
        agent.history = result
        return result

    agent.run = _run
    return result


def _drive(mock_client, *, history=None, error=None, capture_config=None, agent=None, run_args=(), run_kwargs=None):
    """Instrument a REAL agent, drive the (stubbed) run, return uploaded events."""
    uploaded = capture_framework_trace(mock_client)
    agent = agent if agent is not None else _make_agent()
    _stub_run(agent, history=history, error=error)
    adapter = BrowserUseAdapter(mock_client, capture_config=capture_config or CaptureConfig.full())
    adapter.connect(target=agent)
    try:
        if error is not None:
            with pytest.raises(type(error)):
                asyncio.run(agent.run(*run_args, **(run_kwargs or {})))
        else:
            asyncio.run(agent.run(*run_args, **(run_kwargs or {})))
    finally:
        adapter.disconnect()
    return uploaded["events"]


# ---------------------------------------------------------------------------
# Lifecycle / hook mechanism
# ---------------------------------------------------------------------------
class TestLifecycle:
    def test_connect_requires_a_target(self, mock_client):
        adapter = BrowserUseAdapter(mock_client)
        with pytest.raises(ValueError, match="requires a target agent"):
            adapter.connect()

    def test_connect_rejects_an_object_without_run(self, mock_client):
        adapter = BrowserUseAdapter(mock_client)
        with pytest.raises(TypeError, match=r"\.run\(\)"):
            adapter.connect(target=object())

    def test_connect_rejects_a_sync_run(self, mock_client):
        """traced_run awaits the original — a sync .run() must fail fast at
        install time, not with a TypeError inside the user's first call."""

        class SyncAgent:
            task = "t"
            llm = None

            def run(self, max_steps=500):
                return None

        adapter = BrowserUseAdapter(mock_client)
        with pytest.raises(TypeError, match="async"):
            adapter.connect(target=SyncAgent())

    def test_connect_wraps_the_real_bound_run(self, mock_client):
        agent = _make_agent()
        original = agent.run
        adapter = BrowserUseAdapter(mock_client)
        adapter.connect(target=agent)
        assert agent.run is not original
        assert getattr(agent.run, "_layerlens_traced", False) is True
        adapter.disconnect()

    def test_disconnect_restores_the_real_class_method(self, mock_client):
        agent = _make_agent()
        adapter = BrowserUseAdapter(mock_client)
        adapter.connect(target=agent)
        adapter.disconnect()
        # Deleting the instance attribute re-exposes the real bound class method.
        assert "run" not in vars(agent)
        assert getattr(agent.run, "_layerlens_traced", False) is False
        assert agent.run.__func__ is Agent.run

    def test_disconnect_restores_a_pre_existing_instance_attribute(self, mock_client):
        """A stubbed instance-level run() must be put back verbatim."""
        agent = _make_agent()
        stub = _stub_run(agent) and agent.run
        adapter = BrowserUseAdapter(mock_client)
        adapter.connect(target=agent)
        adapter.disconnect()
        assert agent.run is stub

    def test_instrumenting_twice_is_a_noop(self, mock_client):
        agent = _make_agent()
        adapter = BrowserUseAdapter(mock_client)
        adapter.connect(target=agent)
        wrapped = agent.run
        adapter.instrument_agent(agent)
        assert agent.run is wrapped
        adapter.disconnect()

    def test_a_second_adapter_never_double_wraps(self, mock_client):
        """The _layerlens_traced marker (not just the per-instance registry)
        guards the install path — two adapters over one agent would otherwise
        duplicate every event."""
        agent = _make_agent()
        first = BrowserUseAdapter(mock_client)
        first.connect(target=agent)
        wrapped = agent.run
        second = BrowserUseAdapter(mock_client)
        second.connect(target=agent)
        assert agent.run is wrapped
        first.disconnect()
        second.disconnect()

    def test_framework_version_is_the_real_installed_version(self, mock_client):
        """browser-use 0.13 has no __version__ — the adapter must read the dist
        metadata, not silently report nothing."""
        import importlib.metadata

        agent = _make_agent()
        adapter = BrowserUseAdapter(mock_client)
        adapter.connect(target=agent)
        # Read BEFORE disconnect: the base class clears _metadata on disconnect.
        version = adapter.adapter_info().metadata["framework_version"]
        adapter.disconnect()
        assert version == importlib.metadata.version("browser-use")
        assert version != "unknown"

    def test_wrapper_preserves_the_run_signature(self, mock_client):
        agent = _make_agent()
        adapter = BrowserUseAdapter(mock_client)
        adapter.connect(target=agent)
        assert agent.run.__name__ == "run"
        adapter.disconnect()

    def test_instrument_browser_use_entrypoint(self, mock_client):
        agent = _make_agent()
        _stub_run(agent)
        adapter = instrument_browser_use(agent, client=mock_client)
        try:
            assert isinstance(adapter, BrowserUseAdapter)
            assert getattr(agent.run, "_layerlens_traced", False) is True
        finally:
            uninstrument_browser_use()
        assert getattr(agent.run, "_layerlens_traced", False) is False


# ---------------------------------------------------------------------------
# Run boundaries — agent.input / agent.output / agent.state.change
# ---------------------------------------------------------------------------
class TestRunBoundaries:
    def test_success_emits_the_full_contract(self, mock_client):
        events = _drive(mock_client)
        types = [e["event_type"] for e in events]
        for expected in (
            "agent.input",
            "environment.config",
            "tool.call",
            "model.invoke",
            "cost.record",
            "agent.output",
            "agent.state.change",
        ):
            assert expected in types, f"{expected} missing from {types}"

    def test_agent_input_carries_the_real_task(self, mock_client):
        events = _drive(mock_client)
        payload = find_event(events, "agent.input")["payload"]
        assert payload["framework"] == "browser_use"
        assert payload["input_text"] == "find the install command"

    def test_agent_input_is_on_the_root_span(self, mock_client):
        events = _drive(mock_client)
        payload = find_event(events, "agent.input")
        out = find_event(events, "agent.output")
        assert payload["span_id"] == out["span_id"]
        # A self-parented span IS the collector's real root marker (it declines
        # to synthesize a trace.root wrapper over it).
        assert payload["parent_span_id"] == payload["span_id"]

    def test_agent_output_reports_real_steps_and_latency(self, mock_client):
        events = _drive(mock_client)
        payload = find_event(events, "agent.output")["payload"]
        assert payload["total_steps"] == 2
        assert isinstance(payload["latency_ms"], float)
        assert payload["latency_ms"] >= 0.0
        assert payload["status"] == "ok"
        assert "error" not in payload

    def test_agent_output_uses_the_real_final_result(self, mock_client):
        """history.final_result() is browser-use's own answer accessor — a done
        action's extracted_content is the run's output."""
        history = _history(
            steps=[
                _step(
                    [_action(done={"text": "the answer is 42", "success": True})],
                    [ActionResult(is_done=True, success=True, extracted_content="the answer is 42")],
                )
            ],
            usage=_usage(),
        )
        events = _drive(mock_client, history=history)
        payload = find_event(events, "agent.output")["payload"]
        assert payload["output_text"] == "the answer is 42"

    def test_output_omitted_rather_than_invented(self, mock_client):
        """A run whose last action extracted nothing has NO final result (the
        real ``final_result()`` returns None). The AgentHistoryList must never be
        stringified as if it were the agent's answer — an honest absence beats a
        fabricated output."""
        history = _history(
            steps=[
                _step(
                    [_action(navigate={"url": "https://example.com/docs", "new_tab": False})],
                    [ActionResult(extracted_content=None)],
                )
            ],
            usage=_usage(),
        )
        assert history.final_result() is None  # the framework's own accessor agrees
        events = _drive(mock_client, history=history)
        payload = find_event(events, "agent.output")["payload"]
        assert "output_text" not in payload
        assert "AgentHistory" not in json.dumps(payload)

    def test_state_change_carries_the_ingest_required_shape(self, mock_client):
        events = _drive(mock_client)
        payload = find_event(events, "agent.state.change")["payload"]
        assert payload["state_key"] == "run_status"
        assert payload["state_type"] == "run_complete"
        assert payload["old_value"] == "running"
        assert payload["new_value"] == "complete"

    def test_state_change_is_the_last_event(self, mock_client):
        events = _drive(mock_client)
        emitted = [e["event_type"] for e in events if e["event_type"] != "trace.root"]
        assert emitted[-1] == "agent.state.change"

    def test_max_steps_reported_when_passed_as_a_keyword(self, mock_client):
        events = _drive(mock_client, run_kwargs={"max_steps": 7})
        assert find_event(events, "agent.input")["payload"]["max_steps"] == 7

    def test_max_steps_reported_when_passed_positionally(self, mock_client):
        """Agent.run(self, max_steps=500, ...) — ``await agent.run(20)`` is the
        idiomatic call and the caller DID declare it."""
        events = _drive(mock_client, run_args=(20,))
        assert find_event(events, "agent.input")["payload"]["max_steps"] == 20

    def test_max_steps_omitted_when_not_passed(self, mock_client):
        """The framework's own max_steps=500 default is NOT the caller's
        declaration and must never be reported as if it were."""
        events = _drive(mock_client)
        assert "max_steps" not in find_event(events, "agent.input")["payload"]


# ---------------------------------------------------------------------------
# Error path — the finally block still tells the truth
# ---------------------------------------------------------------------------
class TestErrorPath:
    def test_error_still_emits_the_partial_history(self, mock_client):
        events = _drive(mock_client, error=RuntimeError("browser crashed"))
        types = [e["event_type"] for e in events]
        # The real partial history (agent.history) is still walked.
        assert types.count("tool.call") == 2
        assert "agent.output" in types
        assert "agent.state.change" in types

    def test_error_state_change_reports_failure(self, mock_client):
        events = _drive(mock_client, error=RuntimeError("browser crashed"))
        payload = find_event(events, "agent.state.change")["payload"]
        assert payload["state_type"] == "run_failed"
        assert payload["new_value"] == "failed"
        assert payload["error"] == "browser crashed"
        assert payload["error_type"] == "RuntimeError"

    def test_error_surfaces_on_agent_output(self, mock_client):
        events = _drive(mock_client, error=RuntimeError("browser crashed"))
        payload = find_event(events, "agent.output")["payload"]
        assert payload["error"] == "browser crashed"
        assert payload["error_type"] == "RuntimeError"
        assert payload.get("status") != "ok"

    def test_instrumentation_never_breaks_the_users_run(self, mock_client):
        """A task whose repr explodes must not stop the browser automation:
        telemetry is best-effort, the customer's run is not."""

        class Unrenderable:
            def __str__(self):
                raise ValueError("cannot render")

            __repr__ = __str__

        agent = _make_agent()
        object.__setattr__(agent, "task", Unrenderable())
        expected = _stub_run(agent)
        uploaded = capture_framework_trace(mock_client)
        adapter = BrowserUseAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        result = asyncio.run(agent.run())
        adapter.disconnect()
        assert result is expected, "the user's run must still return its real result"
        # The run still reports its outcome even though the input could not render.
        assert find_events(uploaded["events"], "agent.state.change")

    def test_exception_propagates_unchanged(self, mock_client):
        agent = _make_agent()
        boom = RuntimeError("browser crashed")
        _stub_run(agent, error=boom)
        adapter = BrowserUseAdapter(mock_client)
        adapter.connect(target=agent)
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(agent.run())
        adapter.disconnect()
        assert caught.value is boom


# ---------------------------------------------------------------------------
# tool.call — one per REAL browser action
# ---------------------------------------------------------------------------
class TestToolCalls:
    def test_one_tool_call_per_real_action_with_real_names(self, mock_client):
        events = _drive(mock_client)
        calls = find_events(events, "tool.call")
        assert [c["payload"]["tool_name"] for c in calls] == ["navigate", "extract"]

    def test_actions_are_index_paired_with_their_step(self, mock_client):
        history = _history(
            steps=[
                _step(
                    [
                        _action(navigate={"url": "https://example.com", "new_tab": False}),
                        _action(click={"index": 3, "new_tab": False}),
                    ],
                    [ActionResult(extracted_content="ok"), ActionResult(success=False, error="element gone")],
                )
            ],
            usage=_usage(),
        )
        events = _drive(mock_client, history=history)
        calls = find_events(events, "tool.call")
        assert [(c["payload"]["step_index"], c["payload"]["action_index"]) for c in calls] == [(0, 0), (0, 1)]
        # The real ActionResult reports no bool for a regular success — the
        # adapter omits the key rather than defaulting it to True.
        assert "success" not in calls[0]["payload"]
        assert calls[1]["payload"]["success"] is False
        assert calls[1]["payload"]["error"] == "element gone"

    def test_url_and_output_come_from_the_real_state_and_result(self, mock_client):
        events = _drive(mock_client)
        calls = find_events(events, "tool.call")
        assert calls[0]["payload"]["url"] == "https://example.com/docs"
        assert calls[1]["payload"]["output"] == "pip install layerlens"

    def test_no_tool_call_carries_a_fabricated_latency(self, mock_client):
        """The history exposes no per-ACTION timing (N actions share one step);
        dividing the step duration across them would be fabrication."""
        events = _drive(mock_client)
        for call in find_events(events, "tool.call"):
            assert "latency_ms" not in call["payload"]

    def test_unnameable_action_is_skipped_not_fabricated(self, mock_client):
        """THE core honesty SKIP: no real action name -> no tool.call, ever."""

        class Nameless:
            def model_dump(self, **_kwargs):
                return {}

        history = _drifted_history(
            steps=[_drifted_step([Nameless()], [ActionResult(extracted_content=None)])],
            usage=_usage(),
        )
        events = _drive(mock_client, history=history)
        assert find_events(events, "tool.call") == []

    def test_success_omitted_when_the_result_reports_no_bool(self, mock_client):
        history = _history(
            steps=[_step([_action(wait={"seconds": 1})], [ActionResult(success=None)])],
            usage=_usage(),
        )
        events = _drive(mock_client, history=history)
        assert "success" not in find_event(events, "tool.call")["payload"]

    def test_a_real_done_action_reports_its_real_success_bool(self, mock_client):
        """``success=True`` is only legal on a done result — that one IS real."""
        history = _history(
            steps=[
                _step(
                    [_action(done={"text": "finished", "success": True})],
                    [ActionResult(is_done=True, success=True, extracted_content="finished")],
                )
            ],
            usage=_usage(),
        )
        events = _drive(mock_client, history=history)
        payload = find_event(events, "tool.call")["payload"]
        assert payload["tool_name"] == "done"
        assert payload["success"] is True

    def test_outcome_is_never_mispaired(self, mock_client):
        """Fewer results than actions -> the unmatched action reports NO outcome
        rather than borrowing its neighbour's."""
        history = _history(
            steps=[
                _step(
                    [
                        _action(navigate={"url": "https://example.com", "new_tab": False}),
                        _action(click={"index": 1, "new_tab": False}),
                    ],
                    [ActionResult(extracted_content="first")],
                )
            ],
            usage=_usage(),
        )
        events = _drive(mock_client, history=history)
        second = find_events(events, "tool.call")[1]["payload"]
        assert "success" not in second
        assert "output" not in second
        assert "error" not in second

    def test_a_malformed_step_never_breaks_the_run(self, mock_client):
        class Exploding:
            @property
            def model_output(self):
                raise ValueError("corrupt step")

        history = _drifted_history(steps=[Exploding(), _default_history().history[0]], usage=_usage())
        events = _drive(mock_client, history=history)
        assert [c["payload"]["tool_name"] for c in find_events(events, "tool.call")] == ["navigate"]
        assert "agent.output" in [e["event_type"] for e in events]

    def test_tool_calls_are_children_of_the_root_span(self, mock_client):
        events = _drive(mock_client)
        root = find_event(events, "agent.input")["span_id"]
        for call in find_events(events, "tool.call"):
            assert call["parent_span_id"] == root
            assert call["span_id"] != root


# ---------------------------------------------------------------------------
# model.invoke + cost.record — real token figures, honest skips
# ---------------------------------------------------------------------------
class TestModelUsage:
    def test_one_model_invoke_with_the_real_token_figures(self, mock_client):
        events = _drive(mock_client)
        invokes = find_events(events, "model.invoke")
        assert len(invokes) == 1
        payload = invokes[0]["payload"]
        assert payload["model"] == "gpt-4o"
        assert payload["provider"] == "openai"
        # The REAL UsageSummary names are total_prompt_tokens/total_completion_tokens.
        assert payload["tokens_prompt"] == 120
        assert payload["tokens_completion"] == 40
        assert payload["tokens_total"] == 160

    def test_no_model_invoke_when_the_model_is_unknowable(self, mock_client):
        """THE headline honesty SKIP: agent.llm reports no real model name ->
        nothing at all is emitted, tokens included."""

        class NamelessLLM:
            pass

        agent = _make_agent(llm=ChatOpenAI(model="gpt-4o"))
        agent.llm = NamelessLLM()
        events = _drive(mock_client, agent=agent)
        assert find_events(events, "model.invoke") == []
        assert find_events(events, "cost.record") == []
        # The rest of the trace is still emitted.
        assert find_events(events, "tool.call")

    def test_the_llm_class_name_is_never_a_model_name(self, mock_client):
        class ChatSomething:
            pass

        agent = _make_agent()
        agent.llm = ChatSomething()
        events = _drive(mock_client, agent=agent)
        assert find_events(events, "model.invoke") == []

    def test_model_invoke_emitted_without_tokens_when_usage_is_absent(self, mock_client):
        """Older browser-use reports no usage — the model is still real, so the
        invocation is recorded token-free rather than dropped."""
        events = _drive(mock_client, history=_history(steps=_default_history().history, usage=None))
        payload = find_event(events, "model.invoke")["payload"]
        assert payload["model"] == "gpt-4o"
        assert "tokens_prompt" not in payload
        # ... but an unpriceable, token-free cost is never invented.
        assert find_events(events, "cost.record") == []

    def test_no_model_invoke_carries_a_fabricated_latency(self, mock_client):
        events = _drive(mock_client)
        assert "latency_ms" not in find_event(events, "model.invoke")["payload"]

    def test_a_bool_is_never_a_token_count(self, mock_client):
        class BoolUsage:
            total_prompt_tokens = True
            total_completion_tokens = 40
            total_tokens = 40
            total_cost = 0.0
            total_prompt_cached_tokens = 0

        events = _drive(mock_client, history=_drifted_history(steps=_default_history().history, usage=BoolUsage()))
        payload = find_event(events, "model.invoke")["payload"]
        assert "tokens_prompt" not in payload
        assert payload["tokens_completion"] == 40


class TestCostRecord:
    def test_cost_record_uses_the_frameworks_own_real_cost(self, mock_client):
        """browser-use computes cost itself (UsageSummary.total_cost). Its real
        figure must win over the SDK's pricing table."""
        events = _drive(mock_client)
        payload = find_event(events, "cost.record")["payload"]
        assert payload["cost_usd"] == 0.0021
        assert payload["model"] == "gpt-4o"
        assert payload["tokens_prompt"] == 120
        assert payload["tokens_completion"] == 40

    def test_cached_tokens_come_from_the_real_usage_field(self, mock_client):
        events = _drive(mock_client, history=_history(steps=_default_history().history, usage=_usage(cached=64)))
        assert find_event(events, "cost.record")["payload"]["cached_tokens"] == 64

    def test_cost_priced_from_real_tokens_when_the_framework_reports_none(self, mock_client):
        """calculate_cost=False leaves total_cost at 0.0 — the shared pricer then
        fills a REAL cost from the real prompt/completion counts."""
        events = _drive(mock_client, history=_history(steps=_default_history().history, usage=_usage(cost=0.0)))
        payload = find_event(events, "cost.record")["payload"]
        assert payload["cost_usd"] > 0.0

    def test_no_cost_record_without_prompt_or_completion_tokens(self, mock_client):
        """A total-only usage cannot be priced: _price_cost_record computes from
        prompt/completion rates ONLY and would stamp a fabricated $0.00."""

        class TotalOnly:
            total_tokens = 160
            total_cost = 0.0

        events = _drive(mock_client, history=_drifted_history(steps=_default_history().history, usage=TotalOnly()))
        assert find_event(events, "model.invoke")["payload"]["tokens_total"] == 160
        assert find_events(events, "cost.record") == []

    def test_no_cost_record_for_a_real_all_zero_usage(self, mock_client):
        """A run that recorded no LLM entry (crash/Ctrl-C before the first call,
        or a 0-step run) carries a REAL all-zero UsageSummary. Zero tokens at a
        zero cost is not a $0.00 spend measurement — it is the ABSENCE of one, so
        no cost.record may be emitted."""
        events = _drive(mock_client, history=_history(steps=_default_history().history, usage=_zero_usage()))
        assert find_events(events, "cost.record") == []

    def test_all_zero_usage_leaves_no_fabricated_token_counts(self, mock_client):
        """The model is real, so model.invoke still stands — but a zero is not a
        measured count and must not be stamped as one."""
        events = _drive(mock_client, history=_history(steps=_default_history().history, usage=_zero_usage()))
        payload = find_event(events, "model.invoke")["payload"]
        assert payload["model"] == "gpt-4o"
        for key in ("tokens_prompt", "tokens_completion", "tokens_total"):
            assert key not in payload, f"{key} fabricated from an all-zero usage"

    def test_no_cost_record_when_prompt_and_completion_are_zero(self, mock_client):
        """The unpriceable guard must test the token VALUES, not merely whether
        the keys are present: prompt/completion of 0 price to a fabricated $0.00
        exactly as an absent prompt/completion would."""

        class ZeroValuedPromptCompletion:
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 160
            total_cost = 0.0

        events = _drive(
            mock_client,
            history=_drifted_history(steps=_default_history().history, usage=ZeroValuedPromptCompletion()),
        )
        assert find_events(events, "cost.record") == []

    def test_cost_usd_is_never_zero(self, mock_client):
        for usage in (
            _usage(),
            _usage(cost=0.0),
            _usage(cost=0.0, prompt=120, completion=40),
            _zero_usage(),
            _usage(prompt=0, completion=0, total=0, cost=0.0),
        ):
            events = _drive(mock_client, history=_history(steps=_default_history().history, usage=usage))
            for rec in find_events(events, "cost.record"):
                assert rec["payload"].get("cost_usd") != 0.0


class TestUsageTokens:
    def test_an_all_zero_usage_yields_no_token_figures(self):
        """The strip the shared _normalize_tokens performs (and that callers rely
        on via ``if tokens:``) must survive the port's attr-name deviation."""
        from layerlens.instrument.adapters.frameworks.browser_use import _usage_tokens

        assert _usage_tokens(_zero_usage()) == {}

    def test_real_counts_still_survive(self):
        from layerlens.instrument.adapters.frameworks.browser_use import _usage_tokens

        assert _usage_tokens(_usage()) == {
            "tokens_prompt": 120,
            "tokens_completion": 40,
            "tokens_total": 160,
        }

    def test_a_real_zero_alongside_a_real_count_is_kept(self):
        """Only an ALL-zero usage is absent data. A genuine 0 completion next to
        a real prompt count is a real measurement and must not be stripped."""
        from layerlens.instrument.adapters.frameworks.browser_use import _usage_tokens

        assert _usage_tokens(_usage(prompt=120, completion=0, total=120)) == {
            "tokens_prompt": 120,
            "tokens_completion": 0,
            "tokens_total": 120,
        }


# ---------------------------------------------------------------------------
# environment.config
# ---------------------------------------------------------------------------
class TestEnvironmentConfig:
    def test_config_reports_the_real_agent_setup(self, mock_client):
        events = _drive(mock_client)
        payload = find_event(events, "environment.config")["payload"]
        config = payload["config"]
        assert config["framework"] == "browser_use"
        assert config["model"] == "gpt-4o"
        assert config["provider"] == "openai"
        assert config["task"] == "find the install command"

    def test_config_reads_the_modern_browser_profile(self, mock_client):
        """0.13 replaced agent.browser.config with browser_profile — reading the
        old path captures nothing while pretending to."""
        events = _drive(mock_client)
        config = find_event(events, "environment.config")["payload"]["config"]
        assert config["browser.headless"] is False

    def test_config_never_reads_sensitive_data(self, mock_client):
        agent = _make_agent()
        agent.sensitive_data = {"https://example.com": {"pw": "hunter2"}}
        events = _drive(mock_client, agent=agent)
        config = find_event(events, "environment.config")["payload"]["config"]
        assert "sensitive_data" not in config
        assert "hunter2" not in str(events)

    def test_config_is_emitted_once_per_run_as_a_child_span(self, mock_client):
        events = _drive(mock_client)
        configs = find_events(events, "environment.config")
        assert len(configs) == 1
        assert configs[0]["parent_span_id"] == find_event(events, "agent.input")["span_id"]

    def test_config_survives_a_hostile_getattr(self, mock_client):
        class Hostile:
            task = "t"
            llm = None

            def __getattr__(self, item):
                raise RuntimeError(f"no {item}")

            async def run(self, max_steps=500):
                return _default_history()

        events = _drive(mock_client, agent=Hostile())
        assert find_events(events, "environment.config")

    def test_config_omits_every_unset_field(self, mock_client):
        events = _drive(mock_client)
        config = find_event(events, "environment.config")["payload"]["config"]
        # Agent has no max_steps attribute at all — never defaulted/fabricated.
        assert "max_steps" not in config
        assert None not in config.values()


# ---------------------------------------------------------------------------
# Identity — the Agent column must never be fabricated
# ---------------------------------------------------------------------------
class TestIdentity:
    def test_unnamed_agent_stamps_no_identity(self, mock_client):
        """A real browser_use.Agent has NO name attribute, so any stamped name
        would be a placeholder masquerading as a producer-declared identity."""
        events = _drive(mock_client)
        for e in events:
            assert "agent_name" not in e["payload"]
            assert "agent_id" not in e["payload"]

    def test_caller_supplied_name_is_stamped_on_every_event(self, mock_client):
        agent = _make_agent()
        _stub_run(agent)
        uploaded = capture_framework_trace(mock_client)
        adapter = BrowserUseAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent, agent_name="docs-scraper")
        asyncio.run(agent.run())
        adapter.disconnect()
        events = uploaded["events"]
        for event_type in ("agent.input", "agent.output", "agent.state.change", "tool.call", "model.invoke"):
            payload = find_event(events, event_type)["payload"]
            assert payload["agent_name"] == "docs-scraper"
            assert payload["agent_id"] == "docs-scraper"

    def test_a_generic_name_is_never_an_identity(self, mock_client):
        agent = _make_agent()
        _stub_run(agent)
        adapter = BrowserUseAdapter(mock_client, capture_config=CaptureConfig.full())
        uploaded = capture_framework_trace(mock_client)
        adapter.connect(target=agent, agent_name="agent")
        asyncio.run(agent.run())
        adapter.disconnect()
        for e in uploaded["events"]:
            assert "agent_name" not in e["payload"]

    def test_the_honest_identity_resolves_for_the_trace(self, mock_client):
        from layerlens.instrument._identity import honest_agent_identity

        agent = _make_agent()
        _stub_run(agent)
        uploaded = capture_framework_trace(mock_client)
        adapter = BrowserUseAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent, agent_name="docs-scraper")
        asyncio.run(agent.run())
        adapter.disconnect()
        # The collector synthesizes agent.identity at flush; strip it and re-derive.
        events = [e for e in uploaded["events"] if e["event_type"] != "agent.identity"]
        identity = honest_agent_identity(events)
        assert identity is not None
        assert identity["agent_name"] == "docs-scraper"

    def test_the_model_is_never_surfaced_as_the_agent(self, mock_client):
        from layerlens.instrument._identity import honest_agent_identity

        events = _drive(mock_client)
        assert honest_agent_identity([e for e in events if e["event_type"] != "agent.identity"]) is None


# ---------------------------------------------------------------------------
# Trace integrity
# ---------------------------------------------------------------------------
class TestTraceIntegrity:
    def test_history_resolves_from_the_run_result(self, mock_client):
        events = _drive(mock_client)
        assert len(find_events(events, "tool.call")) == 2

    def test_history_resolves_from_the_agent_on_the_error_path(self, mock_client):
        """The run raised, so there is no result — the agent's own partial
        history is the honest fallback."""
        events = _drive(mock_client, error=RuntimeError("boom"))
        assert len(find_events(events, "tool.call")) == 2

    def test_no_history_still_emits_the_run_boundaries(self, mock_client):
        agent = _make_agent()

        async def _run(max_steps: int = 500, on_step_start=None, on_step_end=None):
            return None

        agent.run = _run
        uploaded = capture_framework_trace(mock_client)
        adapter = BrowserUseAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        asyncio.run(agent.run())
        adapter.disconnect()
        events = uploaded["events"]
        assert find_event(events, "agent.output")["payload"]["total_steps"] == 0
        assert find_events(events, "tool.call") == []

    def test_all_events_share_one_trace(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        agent = _make_agent()
        _stub_run(agent)
        adapter = BrowserUseAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        asyncio.run(agent.run())
        adapter.disconnect()
        assert uploaded["trace_id"] is not None
        assert len(uploaded["events"]) > 5

    def test_concurrent_runs_do_not_clobber_each_other(self, mock_client):
        """Per-run state lives in ContextVars — two agents driven concurrently
        must each report their own step count."""
        from layerlens.instrument._collector import set_trace_observer

        traces = []
        set_trace_observer(lambda p: traces.append(p))
        try:
            one = _make_agent(task="one")
            two = _make_agent(task="two")
            _stub_run(one, history=_history(steps=_default_history().history[:1], usage=_usage()))
            _stub_run(two, history=_default_history())
            a1 = BrowserUseAdapter(mock_client, capture_config=CaptureConfig.full())
            a2 = BrowserUseAdapter(mock_client, capture_config=CaptureConfig.full())
            a1.connect(target=one)
            a2.connect(target=two)

            async def _both():
                await asyncio.gather(one.run(), two.run())

            asyncio.run(_both())
            a1.disconnect()
            a2.disconnect()
        finally:
            set_trace_observer(None)

        step_counts = sorted(
            find_event(t["events"], "agent.output")["payload"]["total_steps"] for t in traces if t["events"]
        )
        assert step_counts == [1, 2]
