"""Offline redaction + error + attestation + cost floor for the DSPy adapter.

Runs in plain CI with no credentials and no network: DSPy's own ``DummyLM`` is
the model, every other object (programs, signatures, the ``dspy.settings``
callback bus, ``BootstrapFewShot``) is real.

* Redaction   — a real nested program call with ``capture_content=False`` keeps
                the question/answer/prompt/tool content — and a SENTINEL sweep
                over the whole serialized trace — out of the stored trace, with a
                ``capture_content=True`` vacuity control proving the same path
                DOES carry it otherwise.
* Error       — a REAL ``dspy.utils.exceptions.AdapterParseError`` raised by a
                genuine DSPy parse failure (the LM answering the wrong field),
                not a synthetic RuntimeError. Its message embeds the raw LM
                response verbatim, so it is also a redaction surface.
* Attestation — the offline chain over a real program call reconstructs and
                ``verify_chain`` returns valid; a tamper control proves the
                check is not vacuous.
* Cost        — a token-bearing call prices; a zero-token call emits NO
                cost.record (the honest omission), and the prefix split is what
                lets a litellm-style ``openai/gpt-4o-mini`` resolve at all.
"""

from __future__ import annotations

import sys
import json

import pytest

if sys.version_info < (3, 10):
    pytest.skip("dspy requires Python >= 3.10", allow_module_level=True)

dspy = pytest.importorskip("dspy", reason="dspy not installed")

from importlib.metadata import version as _pkg_version  # noqa: E402

if int(_pkg_version("dspy").split(".")[0]) < 3:
    pytest.skip(
        "dspy adapter targets dspy >= 3 (the pinned matrix venv tests 3.x)",
        allow_module_level=True,
    )

from dspy.utils.dummies import DummyLM  # noqa: E402
from dspy.utils.exceptions import AdapterParseError  # noqa: E402

from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.dspy import DSPyAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

SENTINEL = "LL-SENTINEL-9d4b7e1a"


class FloorSignature(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class FloorProgram(dspy.Module):
    """A developer-declared program (an honest agent identity) wrapping a Predict."""

    def __init__(self) -> None:
        super().__init__()
        self.pred = dspy.Predict(FloorSignature)

    def forward(self, question: str):
        return self.pred(question=question)


class _UsageLM(DummyLM):
    """A DummyLM whose response carries usage, so DSPy's own
    ``_process_lm_response`` records it onto the history entry the adapter reads."""

    def __init__(self, answers, usage=None):
        super().__init__(answers)
        self._usage = usage

    def forward(self, prompt=None, messages=None, **kwargs):
        response = super().forward(prompt=prompt, messages=messages, **kwargs)
        if self._usage is not None:
            response.usage = self._usage
        return response


@pytest.fixture(autouse=True)
def _reset_dspy_settings():
    yield
    dspy.settings.configure(callbacks=[])


def _drive_program(mock_client, capture_config, *, sentinel=SENTINEL, lm=None):
    """Drive a real nested DSPy program with content-bearing fields."""
    uploaded = capture_framework_trace(mock_client)
    dspy.configure(lm=lm or DummyLM([{"answer": f"answer {sentinel}"} for _ in range(4)]), callbacks=[])
    adapter = DSPyAdapter(mock_client, capture_config=capture_config)
    adapter.connect()
    try:
        FloorProgram()(question=f"question {sentinel}")
    finally:
        adapter.disconnect()
    return uploaded


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_control_capture_content_true_does_carry_the_sentinel(self, mock_client):
        """Vacuity control: without it, the redaction test below could pass
        because the content never rode the trace at all."""
        uploaded = _drive_program(mock_client, CaptureConfig.full())
        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"

        # And it must ride the exact keys the redaction test asserts are gone.
        agent_in = find_events(events, "agent.input")[0]["payload"]
        assert SENTINEL in agent_in["input_text"]
        assert SENTINEL in json.dumps(agent_in["inputs"])
        agent_out = [e["payload"] for e in find_events(events, "agent.output")]
        assert any(SENTINEL in json.dumps(p.get("prediction", "")) for p in agent_out)
        assert any(SENTINEL in p.get("output_text", "") for p in agent_out)
        invoke = find_event(events, "model.invoke")["payload"]
        assert SENTINEL in invoke["prompt"]
        assert SENTINEL in invoke["output"]

    def test_capture_content_false_strips_content_but_keeps_structure(self, mock_client):
        uploaded = _drive_program(mock_client, CaptureConfig.standard())
        events = uploaded["events"]

        # 1) SENTINEL sweep over the whole serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) The content keys are gone by name.
        for event in find_events(events, "agent.input"):
            for key in ("input_text", "inputs"):
                assert key not in event["payload"], f"{key} leaked under capture_content=False"
        for event in find_events(events, "agent.output"):
            for key in ("output_text", "prediction"):
                assert key not in event["payload"], f"{key} leaked under capture_content=False"
        for event in find_events(events, "model.invoke"):
            for key in ("prompt", "output"):
                assert key not in event["payload"], f"model.invoke.{key} leaked under capture_content=False"

        # 3) Structure + topology + identity SURVIVE — redaction must not blind.
        inputs = find_events(events, "agent.input")
        assert {e["payload"]["module_type"] for e in inputs} == {"FloorProgram", "Predict"}
        assert inputs[0]["payload"]["input_keys"] == ["question"], "safe field NAMES must survive"
        outer = next(e for e in inputs if e["payload"]["module_type"] == "FloorProgram")
        assert outer["payload"]["agent_name"] == "FloorProgram"
        invoke = find_event(events, "model.invoke")["payload"]
        assert invoke["model"] == "dummy"
        configs = {e["payload"]["module_type"]: e["payload"] for e in find_events(events, "environment.config")}
        assert configs["Predict"]["signature"] == "FloorSignature"
        assert configs["Predict"]["input_fields"] == ["question"]
        # The outer program declares no signature — the key is OMITTED, never
        # defaulted to an empty string.
        assert "signature" not in configs["FloorProgram"]
        # The span tree still reconstructs.
        inner = next(e for e in inputs if e["payload"]["module_type"] == "Predict")
        assert inner["parent_span_id"] == outer["span_id"]

    def test_tool_content_is_stripped(self, mock_client):
        uploaded = capture_framework_trace(mock_client)

        def lookup(city: str) -> str:
            """Look up a city."""
            return f"result {SENTINEL}"

        dspy.configure(
            lm=DummyLM(
                [
                    {"next_thought": "look up", "next_tool_name": "lookup", "next_tool_args": {"city": SENTINEL}},
                    {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
                    {"reasoning": "found", "answer": "ok"},
                ]
            ),
            callbacks=[],
        )
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.standard())
        adapter.connect()
        try:
            dspy.ReAct(FloorSignature, tools=[lookup])(question=f"where is {SENTINEL}?")
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: tool content survived capture_content=False"
        tools = find_events(events, "tool.call")
        assert tools, "tool topology must survive redaction"
        lookup_call = next(e for e in tools if e["payload"]["tool_name"] == "lookup")
        # Name / success / latency are metadata and MUST survive.
        assert lookup_call["payload"]["success"] is True
        assert "latency_ms" in lookup_call["payload"]
        assert "input" not in lookup_call["payload"]
        assert "output" not in lookup_call["payload"]

    def test_failing_tool_error_string_is_stripped_but_the_failure_stays_visible(self, mock_client):
        """A raising tool puts str(exc) — which echoes the failing arguments — on
        tool.call. Unlike agent.output/model.invoke, tool.call has NO ``error``
        entry in _CONTENT_KEYS, so the emit-site gate is the only thing between
        that string and the stored trace."""
        uploaded = capture_framework_trace(mock_client)

        def lookup(city: str) -> str:
            """Look up a city."""
            raise ValueError(f"no such city: {SENTINEL}")

        dspy.configure(
            lm=DummyLM(
                [
                    {"next_thought": "look up", "next_tool_name": "lookup", "next_tool_args": {"city": "X"}},
                    {"next_thought": "give up", "next_tool_name": "finish", "next_tool_args": {}},
                    {"reasoning": "failed", "answer": "unknown"},
                ]
            ),
            callbacks=[],
        )
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.standard())
        adapter.connect()
        try:
            dspy.ReAct(FloorSignature, tools=[lookup])(question="where?")
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        failed = [e for e in find_events(events, "tool.call") if e["payload"]["success"] is False]
        assert failed, "a raising tool must still emit tool.call with success=False"
        payload = failed[0]["payload"]
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: the tool's error string survived redaction"
        assert "error" not in payload
        # The failure stays auditable via its surviving category + name.
        assert payload["error_type"] == "ValueError"
        assert payload["tool_name"] == "lookup"


# ---------------------------------------------------------------------------
# Real error shape
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    @staticmethod
    def _parse_failure_lm():
        # The LM answers a field the signature never declared -> DSPy's adapter
        # cannot parse the response and raises AdapterParseError for real.
        return DummyLM([{"wrong_field": f"bogus {SENTINEL}"} for _ in range(8)])

    def test_real_adapter_parse_error_surfaces_on_agent_output(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        dspy.configure(lm=self._parse_failure_lm(), callbacks=[])
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        raised: BaseException | None = None
        try:
            FloorProgram()(question="q?")
        except BaseException as exc:
            raised = exc
        finally:
            adapter.disconnect()

        # Prove the real dspy exception class reached us — not a stand-in.
        assert raised is not None, "the parse failure must propagate to the caller"
        assert isinstance(raised, AdapterParseError)
        assert type(raised).__module__.startswith("dspy")

        events = uploaded["events"]
        outputs = [e for e in find_events(events, "agent.output") if "error" in e["payload"]]
        assert outputs, "a failed module run must still emit an agent.output carrying the error"
        payload = outputs[0]["payload"]
        assert payload["error_type"] == "AdapterParseError"
        assert "failed to parse the LM response" in payload["error"]
        # The failed run is still a REAL run: identity + latency survive.
        assert payload["latency_ms"] >= 0

    def test_real_error_message_is_content_and_is_redacted(self, mock_client):
        """AdapterParseError embeds the raw LM response verbatim, so the error
        string is content — it must not survive capture_content=False."""
        uploaded = capture_framework_trace(mock_client)
        dspy.configure(lm=self._parse_failure_lm(), callbacks=[])
        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.standard())
        adapter.connect()
        try:
            FloorProgram()(question="q?")
        except AdapterParseError:
            pass
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: the LM response inside str(exc) survived"
        # ... but the failure is still VISIBLE via its surviving category.
        typed = [e for e in find_events(events, "agent.output") if e["payload"].get("error_type")]
        assert typed, "redaction must not hide that the run failed"
        assert typed[0]["payload"]["error_type"] == "AdapterParseError"
        assert "error" not in typed[0]["payload"]


# ---------------------------------------------------------------------------
# Attestation
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_a_real_program_call(self, mock_client):
        uploaded = _drive_program(mock_client, CaptureConfig.full())
        events = uploaded["events"]
        assert events, "a real DSPy program call must flush a non-empty trace"

        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real dspy trace"
        assert len(envelopes) == len(events), f"{len(envelopes)} envelopes for {len(events)} events"
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result_ok = verify_chain(envelopes)
        assert result_ok.valid, f"attestation chain invalid: {result_ok.error}"

        # Tamper control: the check must REJECT a broken link, proving the pass
        # above is not trivially true.
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
# Cost / tokens
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_priced_model_with_real_usage_carries_cost_usd(self, mock_client):
        lm = _UsageLM(
            [{"answer": "blue"} for _ in range(4)],
            usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        )
        lm.model = "openai/gpt-4o-mini"
        uploaded = _drive_program(mock_client, CaptureConfig.full(), lm=lm)
        events = uploaded["events"]

        cost = find_event(events, "cost.record")
        assert cost is not None, "a token-bearing call on a priced model must emit cost.record"
        payload = cost["payload"]
        assert payload["tokens_prompt"] == 1000
        assert payload["tokens_completion"] == 500
        assert payload["tokens_total"] == 1500
        # The litellm ``provider/model`` prefix must be split off, or the pricing
        # table never resolves and a priced call silently ships unpriced.
        assert payload["model"] == "gpt-4o-mini"
        assert payload["provider"] == "openai"
        assert payload["cost_usd"] > 0, "a priced model must carry a real cost_usd"

        # Framework token vocabulary only — mixing fails the schema lock.
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert key not in payload

    def test_unmeasurable_call_emits_no_cost_record(self, mock_client):
        """The honest omission: DummyLM reports zero tokens, so there is nothing
        to price and NO cost.record may be invented."""
        uploaded = _drive_program(mock_client, CaptureConfig.full())
        events = uploaded["events"]
        assert find_event(events, "model.invoke") is not None
        assert find_events(events, "cost.record") == [], "an unmeasurable call must not be priced"

    def test_unpriced_local_model_reports_tokens_without_cost(self, mock_client):
        """A local ollama model resolves no rate — tokens still land, cost_usd is
        honestly absent rather than fabricated as 0.0."""
        lm = _UsageLM(
            [{"answer": "blue"} for _ in range(4)],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        lm.model = "ollama/llama3"
        uploaded = _drive_program(mock_client, CaptureConfig.full(), lm=lm)

        cost = find_event(uploaded["events"], "cost.record")
        assert cost is not None
        assert cost["payload"]["tokens_total"] == 15
        assert cost["payload"]["provider"] == "ollama"
        assert cost["payload"].get("cost_usd") is None, "an unpriced model must not fabricate a cost"
