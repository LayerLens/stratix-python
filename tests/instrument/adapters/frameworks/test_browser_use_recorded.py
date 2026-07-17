"""Recorded-real-response replay for the browser_use framework (LAY-3614).

Replays the REAL ``AgentHistoryList`` produced by a REAL headless-Chromium
browser-use run — a corporate travel desk's ``trip-research-agent`` reading a
Trip Options Board and picking the cheapest nonstop, freely-cancellable
Boston->Lisbon package under a $1,400 cap (recorded by
``samples/data/generators/browser_use.py``; the same run also produced the
shipped sample fixture ``samples/data/traces/industry/travel_browseruse_research.jsonl``).

WHY THIS SHAPE
--------------
The corpus rule is *record UPSTREAM of the parser, assert DOWNSTREAM of it*.
For this adapter the parser under test is the **history walk**, so the thing we
do not control — and therefore the thing worth recording — is browser-use's own
``AgentHistoryList``. The fixture is that object serialized by browser-use's OWN
``model_dump()``, and the replay re-materializes it through browser-use's OWN
``AgentHistoryList.load_from_dict`` against the REAL action registry
(``Tools().registry.create_action_model()``). So the objects the adapter walks
here are real framework objects: real ``ActionModel`` subclass instances (whose
``model_dump()`` really is ``{action_name: params}``), real ``ActionResult``
outcomes, real per-step ``state.url``, and the real ``UsageSummary``. The agent
is a REAL ``browser_use.Agent`` with a REAL ``ChatOpenAI`` llm, so the adapter's
model/provider resolution and its ``environment.config`` extraction run against
the real objects too.

The ONLY substitution is ``agent.run`` itself — it returns the recorded history
instead of launching a browser. That is this transport family's equivalent of
``httpx.MockTransport``: swap the producer, keep the real body and the real
parser. No browser, no network, no spend, deterministic in CI.

The strong tells that the real framework shape flowed through — none of these
are values a unit double would have to get right:

* ``model.invoke`` reports ``18061/565/18626`` tokens, which exist ONLY on the
  real ``UsageSummary`` under its ``total_prompt_tokens`` / ``total_completion_tokens``
  spellings. The shared ``_normalize_tokens`` probes ``prompt_tokens`` /
  ``input_tokens`` and would find NEITHER — so these numbers are proof the
  adapter's own ``_usage_tokens`` probe order is still correct against the real
  0.13 shape. Rename them upstream and this test goes red.
* ``cost.record`` carries a real ``cost_usd`` priced from those real tokens even
  though the real ``UsageSummary.total_cost`` is ``0.0`` (browser-use only
  prices when ``calculate_cost=True``, which the recording left off). This pins
  the honesty branch: the adapter must NOT stamp browser-use's ``0.0`` as a real
  cost, and must leave ``cost_usd`` for the shared price-on-emit chokepoint to
  fill from the genuine prompt/completion counts.
* the four ``tool.call`` events are the actions the browser REALLY executed
  (``navigate`` -> ``click`` -> ``click`` -> ``done``, the model's own path to
  reading the board), named by unpacking real ``ActionModel`` dumps rather than
  by any hand-set ``name`` field. Two of them share step 1 — the real run put
  two actions in one step — so the replay also exercises the adapter's
  action/result index-pairing against a genuine multi-action step.
"""

from __future__ import annotations

import os
import asyncio

import pytest

pytest.importorskip("browser_use")  # skips in the base venv (not installed there)

# browser-use pings its own telemetry/cloud-sync endpoints unless this is off.
# Set before importing the package: ``CONFIG`` reads the environment lazily, but
# a test must never make an outbound call.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("BROWSER_USE_CLOUD_SYNC", "false")

from browser_use import Agent  # noqa: E402
from browser_use.llm import ChatOpenAI  # noqa: E402
from browser_use.agent.views import AgentOutput, AgentHistoryList  # noqa: E402
from browser_use.tokens.views import UsageSummary  # noqa: E402
from browser_use.tools.service import Tools  # noqa: E402
from browser_use.browser.profile import BrowserProfile  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.browser_use import BrowserUseAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded  # noqa: E402

_SCENARIO = "travel_research"
_AGENT_NAME = "trip-research-agent"

#: The real token figures the recorded run's UsageSummary carries.
_PROMPT_TOKENS = 18061
_COMPLETION_TOKENS = 565
_TOTAL_TOKENS = 18626
_CACHED_TOKENS = 7040


def _real_history(fixture) -> AgentHistoryList:
    """Re-materialize the recorded body through browser-use's OWN loader.

    ``AgentHistoryList.model_dump()`` is a custom dump that emits only
    ``history`` (it drops ``usage``), so the recorder sealed the real
    ``UsageSummary`` alongside it and it is re-attached here.
    """
    body = fixture["response"]
    output_model = AgentOutput.type_with_custom_actions(Tools().registry.create_action_model())
    history = AgentHistoryList.load_from_dict({"history": body["history"]}, output_model)
    history.usage = UsageSummary(**body["usage"])
    return history


def _replay_agent(history: AgentHistoryList) -> Agent:
    """A REAL ``browser_use.Agent`` whose ``run`` serves the recorded history.

    Everything the adapter reads off the agent — ``llm`` (a real ``ChatOpenAI``
    that really declares ``provider='openai'``), ``task``, ``browser_profile`` —
    is the framework's own object. Only the browser drive is replaced.
    """
    agent = Agent(
        task="Pick the cheapest nonstop, freely-cancellable BOS->LIS option under $1,400.",
        llm=ChatOpenAI(model="gpt-4o-mini", api_key="test-key"),
        browser_profile=BrowserProfile(headless=True),
        use_vision=False,
    )

    async def _run(*_args, **_kwargs):
        return history

    agent.run = _run  # type: ignore[method-assign]
    return agent


@pytest.fixture
def uploaded_events(mock_client):
    """Replay the recorded real history through the real adapter, once."""
    history = _real_history(load_recorded("browser_use", _SCENARIO))
    uploaded = capture_framework_trace(mock_client)

    agent = _replay_agent(history)
    adapter = BrowserUseAdapter(mock_client, capture_config=CaptureConfig.full())
    adapter.connect(target=agent, agent_name=_AGENT_NAME)

    result = asyncio.run(agent.run(4))
    adapter.disconnect()

    # The adapter must return the framework's own object unchanged.
    assert result is history
    return uploaded["events"]


class TestBrowserUseRecorded:
    def test_model_invoke_carries_the_real_usage(self, uploaded_events):
        """The real UsageSummary's ``total_*`` spellings must still be found.

        The shared ``_normalize_tokens`` probes ``prompt_tokens``/``input_tokens``
        — neither of which the real 0.13 ``UsageSummary`` has — so these figures
        can only come from the adapter's own probe order reading the real object.
        """
        mi = find_event(uploaded_events, "model.invoke")
        assert mi["payload"]["framework"] == "browser_use"
        assert mi["payload"]["model"] == "gpt-4o-mini"
        # Off the REAL ChatOpenAI wrapper's own ``provider`` declaration.
        assert mi["payload"]["provider"] == "openai"
        assert mi["payload"]["tokens_prompt"] == _PROMPT_TOKENS
        assert mi["payload"]["tokens_completion"] == _COMPLETION_TOKENS
        assert mi["payload"]["tokens_total"] == _TOTAL_TOKENS

    def test_exactly_one_model_invoke_for_the_run(self, uploaded_events):
        """Usage lives on the history LIST, not per step — 3 steps, 1 invoke.

        Emitting one per step would multiply the real token figures into a
        fabricated total.
        """
        assert len(find_events(uploaded_events, "model.invoke")) == 1

    def test_cost_is_priced_from_the_real_tokens_not_browser_uses_zero(self, uploaded_events):
        """The real ``UsageSummary.total_cost`` is 0.0 (``calculate_cost`` off).

        The adapter must treat that as "not priced" rather than a measured zero,
        and let the shared price-on-emit chokepoint derive the real figure from
        the genuine prompt/completion counts at real gpt-4o-mini rates.
        """
        assert load_recorded("browser_use", _SCENARIO)["response"]["usage"]["total_cost"] == 0.0

        cost = find_event(uploaded_events, "cost.record")
        assert cost["payload"]["framework"] == "browser_use"
        assert cost["payload"]["model"] == "gpt-4o-mini"
        assert cost["payload"]["tokens_prompt"] == _PROMPT_TOKENS
        assert cost["payload"]["tokens_completion"] == _COMPLETION_TOKENS
        assert cost["payload"]["tokens_total"] == _TOTAL_TOKENS
        # The real cached-prompt figure off ``total_prompt_cached_tokens``.
        assert cost["payload"]["cached_tokens"] == _CACHED_TOKENS
        # 18061/1e6*0.15 + 565/1e6*0.60 — real gpt-4o-mini rates on real tokens.
        assert cost["payload"]["cost_usd"] == pytest.approx(0.00304815, rel=1e-6)

    def test_tool_calls_are_the_real_executed_actions(self, uploaded_events):
        """One tool.call per REAL action, named by unpacking real ActionModels.

        The recorded run's step 1 really executed TWO actions, so this also pins
        the index-pairing: N actions in one step are emitted as N distinct
        tool.calls that share the step's index while carrying their own
        ``action_index`` (a per-STEP emit would collapse them into one).
        """
        calls = find_events(uploaded_events, "tool.call")
        assert [c["payload"]["tool_name"] for c in calls] == ["navigate", "click", "click", "done"]
        assert [c["payload"]["step_index"] for c in calls] == [0, 1, 1, 2]
        assert [c["payload"]["action_index"] for c in calls] == [0, 0, 1, 0]
        assert all(c["payload"]["agent_name"] == _AGENT_NAME for c in calls)

        # The real per-step URL: step 0 navigates from a blank page (no URL yet),
        # so the board URL only appears from the step that ran ON the board.
        assert calls[0]["payload"].get("url") is None
        for on_board in calls[1:]:
            assert on_board["payload"]["url"] == "http://127.0.0.1:8731/index.html"

        # The real ActionResult of the terminal action.
        assert calls[3]["payload"]["success"] is True
        # The real navigate params, off the real ActionModel dump.
        assert "127.0.0.1:8731" in calls[0]["payload"]["input"]

    def test_run_output_is_the_agents_real_answer(self, uploaded_events):
        """``history.final_result()`` — never a stringified history object."""
        out = find_event(uploaded_events, "agent.output")
        assert out["payload"]["framework"] == "browser_use"
        assert out["payload"]["status"] == "ok"
        assert out["payload"]["total_steps"] == 3
        # The real answer the model produced off the real rendered board. NW-101
        # is the genuinely correct pick (NW-102 is cheaper but has a stop, NW-103
        # is nonstop but non-refundable), so this is not a "smallest number" grab.
        assert "NW-101" in out["payload"]["output_text"]

    def test_identity_is_producer_declared_not_invented(self, uploaded_events):
        """A real browser_use Agent declares no name of its own."""
        assert not hasattr(_replay_agent(_real_history(load_recorded("browser_use", _SCENARIO))), "name")
        for event_type in ("agent.input", "agent.output", "agent.state.change"):
            payload = find_event(uploaded_events, event_type)["payload"]
            assert payload["agent_name"] == _AGENT_NAME
            assert payload["agent_id"] == _AGENT_NAME

    def test_state_change_reports_the_real_run_completion(self, uploaded_events):
        change = find_event(uploaded_events, "agent.state.change")
        assert change["payload"]["state_key"] == "run_status"
        assert change["payload"]["state_type"] == "run_complete"
        assert change["payload"]["old_value"] == "running"
        assert change["payload"]["new_value"] == "complete"
        assert "error_type" not in change["payload"]

    def test_config_reports_the_real_agent_and_omits_what_it_cannot_read(self, uploaded_events):
        """environment.config off the REAL Agent + real BrowserProfile.

        browser-use 0.13 keeps ``max_steps``/``max_failures``/``use_vision`` in a
        settings object rather than on the Agent, so the adapter's allowlist finds
        nothing to read — and must therefore OMIT them rather than report the
        framework's defaults as if the developer had chosen them.
        """
        config = find_event(uploaded_events, "environment.config")["payload"]["config"]
        assert config["framework"] == "browser_use"
        assert config["model"] == "gpt-4o-mini"
        assert config["provider"] == "openai"
        assert config["browser.headless"] is True
        for absent in ("max_steps", "max_failures", "use_vision", "save_conversation_path"):
            assert absent not in config
        # sensitive_data holds the user's credentials and must never be captured.
        assert "sensitive_data" not in config

    def test_declared_max_steps_is_reported_from_the_call(self, uploaded_events):
        """``max_steps`` is a run() argument, not an agent attribute — the caller
        really passed 4, so it is honestly reported."""
        assert find_event(uploaded_events, "agent.input")["payload"]["max_steps"] == 4

    def test_provenance_is_a_real_capture(self):
        prov = load_recorded("browser_use", _SCENARIO)["provenance"]
        assert prov["provider"] == "openai"
        assert prov["model"] == "gpt-4o-mini"
        assert prov["sdk_version"].startswith("browser-use ")
        assert prov["captured_at"] and prov["captured_at"] != "pending-creds"
