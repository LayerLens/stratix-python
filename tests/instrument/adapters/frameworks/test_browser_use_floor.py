"""Offline redaction + error + attestation + cost floor for the BrowserUse adapter.

Every lane runs with no browser binary, no API key and no network — the real
``browser_use`` objects are constructed directly and the only stub is the
``Agent.run`` coroutine (the browser/network boundary):

* REDACTION   — a real run whose task, browsed URL, action params and scraped
                page text ALL carry a SENTINEL: under ``capture_content=False``
                the sentinel must not survive anywhere in the serialized trace,
                while the structure (tool names, step/action indices, token
                counts, the run-status transition) must. A ``capture_content=True``
                vacuity control proves the sweep can fail.
* ERROR       — a REAL browser-use SDK exception shape
                (``browser_use.agent.views.AgentError`` / the real
                ``LLMException`` raised by the real llm layer), not a synthetic
                RuntimeError, surfaces the honest failure + a surviving category.
* ATTESTATION — ``verify_chain`` over the real flushed trace, plus a tamper
                control that must fail.
* COST        — the framework's own real ``UsageSummary.total_cost`` is carried
                verbatim, and the honest-omission proof: no real model or no
                real prompt/completion tokens => NO cost.record (never $0.00).
"""

from __future__ import annotations

import os
import sys
import json
import asyncio

import pytest

if sys.version_info < (3, 11):
    pytest.skip("browser-use requires Python >= 3.11", allow_module_level=True)

pytest.importorskip("browser_use")

os.environ.setdefault("OPENAI_API_KEY", "sk-browser-use-floor-not-real")

from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.browser_use import BrowserUseAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from .test_browser_use import (  # noqa: E402
    _step,
    _usage,
    _action,
    _history,
    _stub_run,
    _make_agent,
    _zero_usage,
    _default_history,
    _drifted_history,
)

SENTINEL = "LL-SENTINEL-b7c41d9e"


def _sentinel_history():
    """A real history whose URL, action params and extracted content all carry
    the SENTINEL — every content slot the adapter can emit.

    The action is a real ``done``: the only shape whose ``ActionResult`` may
    legally report ``success=True``, so the structural-survival lane below has a
    real bool to assert on.
    """
    from browser_use.agent.views import ActionResult

    return _history(
        steps=[
            _step(
                [_action(done={"text": f"logged in as {SENTINEL}", "success": True})],
                [ActionResult(is_done=True, success=True, extracted_content=f"scraped {SENTINEL}")],
                url=f"https://example.com/?token={SENTINEL}",
            )
        ],
        usage=_usage(),
    )


def _run_with(mock_client, capture_config, history=None, task=None):
    uploaded = capture_framework_trace(mock_client)
    agent = _make_agent(task=task or f"log in with {SENTINEL}")
    _stub_run(agent, history=history if history is not None else _sentinel_history())
    adapter = BrowserUseAdapter(mock_client, capture_config=capture_config)
    adapter.connect(target=agent)
    asyncio.run(agent.run())
    adapter.disconnect()
    return uploaded


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: the SAME real run DOES carry the SENTINEL and the
        content keys it rides on when capture_content=True."""
        uploaded = _run_with(mock_client, CaptureConfig.full())
        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        assert "input_text" in find_event(events, "agent.input")["payload"]
        assert SENTINEL in find_event(events, "environment.config")["payload"]["config"]["task"]
        call = find_event(events, "tool.call")["payload"]
        assert SENTINEL in call["url"]
        assert SENTINEL in call["input"]
        assert SENTINEL in call["output"]

    def test_content_absent_when_not_capturing(self, mock_client):
        uploaded = _run_with(mock_client, CaptureConfig(capture_content=False))
        events = uploaded["events"]
        assert events, "the run must still emit structural events without content"

        # 1) SENTINEL sweep over the whole serialized trace — a browsed URL
        #    carries query-string session tokens / PII, so it is CONTENT.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) The content keys are absent from every payload that would carry them.
        assert "input_text" not in find_event(events, "agent.input")["payload"]
        assert "output_text" not in find_event(events, "agent.output")["payload"]
        assert "task" not in find_event(events, "environment.config")["payload"]["config"]
        call = find_event(events, "tool.call")["payload"]
        for key in ("url", "input", "output"):
            assert key not in call, f"tool.call leaked {key!r} under capture_content=False"

    def test_structure_and_topology_survive_redaction(self, mock_client):
        """Redaction must strip content WITHOUT blinding observability."""
        uploaded = _run_with(mock_client, CaptureConfig(capture_content=False))
        events = uploaded["events"]
        call = find_event(events, "tool.call")["payload"]
        assert call["tool_name"] == "done"
        assert call["step_index"] == 0
        assert call["action_index"] == 0
        assert call["success"] is True
        state = find_event(events, "agent.state.change")["payload"]
        assert state["state_key"] == "run_status"
        assert state["new_value"] == "complete"
        assert find_event(events, "agent.output")["payload"]["total_steps"] == 1
        invoke = find_event(events, "model.invoke")["payload"]
        assert invoke["model"] == "gpt-4o"
        assert invoke["tokens_prompt"] == 120


# ---------------------------------------------------------------------------
# Error path — a REAL browser-use exception shape
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_llm_exception_surfaces_honestly(self, mock_client):
        """browser-use raises its own ModelProviderError out of the llm layer —
        the shape a real run actually fails with, not a synthetic RuntimeError."""
        from browser_use.llm.exceptions import ModelProviderError

        err = ModelProviderError("rate limit exceeded", status_code=429, model="gpt-4o")
        assert type(err).__name__ == "ModelProviderError"

        uploaded = capture_framework_trace(mock_client)
        agent = _make_agent()
        _stub_run(agent, history=_default_history(), error=err)
        adapter = BrowserUseAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        with pytest.raises(ModelProviderError):
            asyncio.run(agent.run())
        adapter.disconnect()

        events = uploaded["events"]
        out = find_event(events, "agent.output")["payload"]
        assert out["error_type"] == "ModelProviderError"
        assert "rate limit exceeded" in out["error"]
        state = find_event(events, "agent.state.change")["payload"]
        assert state["state_type"] == "run_failed"
        assert state["new_value"] == "failed"
        assert state["error_type"] == "ModelProviderError"
        # The real partial history is still reported — a crashed run tells the truth.
        assert len(find_events(events, "tool.call")) == 2

    def test_error_category_survives_redaction(self, mock_client):
        """The free-text error is content and is stripped, but the FAILURE must
        stay visible: error_type + the run_failed transition are categories."""
        from browser_use.llm.exceptions import ModelProviderError

        uploaded = capture_framework_trace(mock_client)
        agent = _make_agent()
        _stub_run(agent, error=ModelProviderError(f"boom {SENTINEL}", status_code=500, model="gpt-4o"))
        adapter = BrowserUseAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect(target=agent)
        with pytest.raises(ModelProviderError):
            asyncio.run(agent.run())
        adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: error text survived capture_content=False"
        state = find_event(events, "agent.state.change")["payload"]
        assert state["error_type"] == "ModelProviderError"
        assert state["new_value"] == "failed"
        assert "error" not in state


# ---------------------------------------------------------------------------
# Attestation
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_a_real_run(self, mock_client):
        uploaded = _run_with(mock_client, CaptureConfig.full())
        events = uploaded["events"]
        assert events, "a real run must flush a non-empty trace"

        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real browser_use trace"
        assert len(envelopes) == len(events), f"{len(envelopes)} envelopes for {len(events)} events"
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

        # Tamper control: verify_chain must REJECT a broken link.
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
# Cost — the framework's real figure, and the honest omission
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_framework_cost_is_carried_verbatim(self, mock_client):
        uploaded = _run_with(mock_client, CaptureConfig.full())
        payload = find_event(uploaded["events"], "cost.record")["payload"]
        # browser-use's OWN measured cost, not the SDK pricing table's estimate.
        assert payload["cost_usd"] == 0.0021
        assert payload["tokens_prompt"] == 120
        assert payload["tokens_completion"] == 40

    def test_no_cost_record_when_the_model_is_unknowable(self, mock_client):
        class NamelessLLM:
            pass

        uploaded = capture_framework_trace(mock_client)
        agent = _make_agent()
        agent.llm = NamelessLLM()
        _stub_run(agent)
        adapter = BrowserUseAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        asyncio.run(agent.run())
        adapter.disconnect()
        assert find_events(uploaded["events"], "cost.record") == []
        assert find_events(uploaded["events"], "model.invoke") == []

    def test_no_cost_record_without_real_prompt_completion_tokens(self, mock_client):
        """_price_cost_record derives cost from prompt/completion rates ONLY, so
        a total-only usage would be stamped with a fabricated $0.00."""

        class TotalOnly:
            total_tokens = 160
            total_cost = 0.0

        uploaded = _run_with(
            mock_client,
            CaptureConfig.full(),
            history=_drifted_history(steps=_default_history().history, usage=TotalOnly()),
        )
        assert find_events(uploaded["events"], "cost.record") == []

    def test_cost_record_never_carries_a_zero(self, mock_client):
        # Driven across the priced run AND the two unpriceable shapes a real run
        # produces: the all-zero summary of a run that recorded no LLM entry, and
        # a summary whose prompt/completion are a real zero. Asserting only the
        # priced run would state the invariant without ever testing it.
        histories = [
            None,
            _history(steps=_default_history().history, usage=_zero_usage()),
            _history(steps=_default_history().history, usage=_usage(prompt=0, completion=0, total=0, cost=0.0)),
        ]
        for history in histories:
            uploaded = _run_with(mock_client, CaptureConfig.full(), history=history)
            for rec in find_events(uploaded["events"], "cost.record"):
                cost = rec["payload"].get("cost_usd")
                assert cost is None or cost > 0.0, "a fabricated $0.00 cost reached the trace"

    def test_no_cost_record_for_a_run_that_recorded_no_llm_entry(self, mock_client):
        """browser-use assigns the all-zero UsageSummary UNCONDITIONALLY at the
        end of run() — including the KeyboardInterrupt path — so a crash before
        the first LLM call must yield NO cost.record, not a $0.00 spend."""
        uploaded = _run_with(
            mock_client,
            CaptureConfig.full(),
            history=_history(steps=_default_history().history, usage=_zero_usage()),
        )
        assert find_events(uploaded["events"], "cost.record") == []
        # The rest of the trace is unaffected: the run is still fully reported.
        assert find_events(uploaded["events"], "tool.call")
        assert find_event(uploaded["events"], "agent.state.change")["payload"]["new_value"] == "complete"

    def test_cost_survives_redaction(self, mock_client):
        """cost.record is _ALWAYS_ENABLED and content-free — a customer must
        still see spend under capture_content=False."""
        uploaded = _run_with(mock_client, CaptureConfig(capture_content=False))
        payload = find_event(uploaded["events"], "cost.record")["payload"]
        assert payload["cost_usd"] == 0.0021
        assert payload["tokens_total"] == 160

    def test_minimal_config_still_reports_cost_and_status(self, mock_client):
        """minimal() disables L3/L5a — cost.record + agent.state.change are
        _ALWAYS_ENABLED and must survive."""
        uploaded = _run_with(mock_client, CaptureConfig.minimal())
        events = uploaded["events"]
        assert find_events(events, "cost.record")
        assert find_events(events, "agent.state.change")
        assert find_events(events, "tool.call") == []
        assert find_events(events, "model.invoke") == []
