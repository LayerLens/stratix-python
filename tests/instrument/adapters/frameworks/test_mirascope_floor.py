"""Offline redaction + error + attestation + cost floor for the Mirascope adapter.

Runs in plain CI with no credentials and no network: every lane drives a REAL
``@llm.call`` through the real provider over an ``httpx.MockTransport`` (see
``_mirascope_support``), so a regression fails here rather than in a live lane.

* Redaction   — ``capture_content=False`` strips prompt/args/output while KEEPING
                the topology (tool_name/agent_name/model/tokens/success), proven
                by a SENTINEL sweep over the whole serialised trace plus a
                ``capture_content=True`` vacuity control showing the same path
                does carry the content otherwise.
* Error       — a REAL ``mirascope.llm.exceptions.NotFoundError`` (raised by the
                real provider's own error map off a real OpenAI 404 body), not a
                synthetic ``RuntimeError``.
* Attestation — offline ``verify_chain`` over the collected payload, with a
                tamper control proving the check is not vacuous.
* Cost        — real pricing off real token counts, plus the honest-omission
                proof when the model is unknown.
"""

from __future__ import annotations

import sys
import json

import pytest

if sys.version_info < (3, 10):  # pragma: no cover - matrix pins 3.11
    pytest.skip("mirascope 2.x requires Python >= 3.10", allow_module_level=True)

pytest.importorskip("mirascope.llm", reason="mirascope not installed")
pytest.importorskip("openai", reason="mirascope[openai] not installed")

import mirascope.llm as llm  # noqa: E402  # pyright: ignore[reportMissingImports]

from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.mirascope import (  # noqa: E402
    MirascopeAdapter,
    uninstrument_mirascope,
)

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ._mirascope_support import (  # noqa: E402
    MODEL_ID,
    BARE_MODEL,
    ok_handler,
    completion_body,
    mirascope_openai,
    not_found_handler,
    restore_call_classes,
    capture_raw_emissions,
)

SENTINEL = "LL-SENTINEL-7f3a9c2e"


@pytest.fixture(autouse=True)
def _no_leaked_adapter():
    yield
    uninstrument_mirascope()
    restore_call_classes()


def _drive(mock_client, capture_config, handler=None, expect_error=None):
    """Drive one REAL mirascope call carrying the SENTINEL in every content slot."""
    uploaded = capture_framework_trace(mock_client)
    with mirascope_openai(handler or ok_handler(completion_body(f"answer about {SENTINEL}"))):
        adapter = MirascopeAdapter(mock_client, capture_config=capture_config)
        adapter.connect()
        try:

            @llm.call(MODEL_ID)
            def recommend_book(genre: str, note: str):
                return f"Recommend a {genre} book. Note: {note}"

            if expect_error is not None:
                with pytest.raises(expect_error):
                    recommend_book(SENTINEL, note=SENTINEL)
            else:
                recommend_book(SENTINEL, note=SENTINEL)
        finally:
            # A raising call must not leak the class-level patch into later lanes.
            adapter.disconnect()
    return uploaded


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: the SAME real path DOES carry the SENTINEL when
        capture_content=True — so the absence test below can actually fail."""
        uploaded = _drive(mock_client, CaptureConfig.full())
        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL"
        assert "input" in find_event(events, "tool.call")["payload"]
        assert "output" in find_event(events, "tool.result")["payload"]

    def test_content_absent_when_not_capturing(self, mock_client):
        uploaded = _drive(mock_client, CaptureConfig(capture_content=False))
        events = uploaded["events"]
        assert events, "structure must survive capture_content=False"

        # 1) Whole-trace SENTINEL sweep.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) The content keys are absent from the payloads that would carry them.
        assert "input" not in find_event(events, "tool.call")["payload"]
        assert "output" not in find_event(events, "tool.result")["payload"]

        # 3) Topology/metadata SURVIVES — redaction must not blind the trace.
        call = find_event(events, "tool.call")["payload"]
        assert call["tool_name"] == "mirascope.recommend_book"
        assert call["agent_name"] == "recommend_book"
        assert call["success"] is True
        invoke = find_event(events, "model.invoke")["payload"]
        assert invoke["model"] == BARE_MODEL
        assert invoke["tokens_prompt"] == 17
        assert find_event(events, "cost.record")["payload"]["cost_usd"] > 0

    def test_adapter_itself_gates_content_at_emit(self, mock_client):
        """The adapter's OWN gate, isolated from the collector backstop.

        Bite-proven: the two assertions above pass even if every
        ``_set_if_capturing`` is replaced with a direct assignment, because
        ``TraceCollector.emit`` redacts before storing. This lane fails in that
        case — it is what actually holds the emit-time gate to account.
        """
        with capture_raw_emissions() as raw:
            _drive(mock_client, CaptureConfig(capture_content=False))

        assert raw, "no events reached the collector"
        by_type = {t: p for t, p in raw}
        assert "tool.call" in by_type and "tool.result" in by_type

        for event_type, payload in raw:
            for key in ("input", "output", "error"):
                assert key not in payload, (
                    f"adapter emitted un-gated '{key}' on {event_type} under capture_content=False "
                    "(the collector backstop would hide this leak downstream)"
                )
            assert SENTINEL not in json.dumps(payload, default=str), (
                f"adapter handed the SENTINEL to the collector on {event_type}"
            )

    def test_adapter_gates_error_text_at_emit(self, mock_client):
        """Same isolation for the error path — ``error`` rides tool.result, whose
        backstop would otherwise mask an un-gated emit."""
        with capture_raw_emissions() as raw:
            _drive(
                mock_client,
                CaptureConfig(capture_content=False),
                handler=not_found_handler(),
                expect_error=llm.exceptions.NotFoundError,
            )

        result = [p for t, p in raw if t == "tool.result"]
        assert result, "the error path must still emit tool.result"
        assert "error" not in result[0], "adapter emitted un-gated error text at capture_content=False"
        assert result[0]["error_type"] == "NotFoundError"

    def test_adapter_emits_content_at_emit_when_capturing(self, mock_client):
        """Vacuity control for the two lanes above: the same raw-capture seam DOES
        see the content when capture_content=True."""
        with capture_raw_emissions() as raw:
            _drive(mock_client, CaptureConfig.full())

        by_type = {t: p for t, p in raw}
        assert "input" in by_type["tool.call"]
        assert "output" in by_type["tool.result"]
        assert SENTINEL in json.dumps(by_type["tool.call"], default=str)

    def test_error_text_is_stripped_but_the_failure_stays_visible(self, mock_client):
        """An exception message can quote the prompt, so it is content — but the
        FACT of the failure must survive redaction."""
        uploaded = _drive(
            mock_client,
            CaptureConfig(capture_content=False),
            handler=not_found_handler(),
            expect_error=llm.exceptions.NotFoundError,
        )
        events = uploaded["events"]
        result = find_event(events, "tool.result")["payload"]
        assert "error" not in result, "raw error text leaked under capture_content=False"
        # The honest, content-free failure signal.
        assert result["success"] is False
        assert result["error_type"] == "NotFoundError"


# ---------------------------------------------------------------------------
# Error — a real SDK exception shape
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_mirascope_exception_surfaces_honestly(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        with mirascope_openai(not_found_handler()):
            adapter = MirascopeAdapter(mock_client, capture_config=CaptureConfig.full())
            adapter.connect()

            @llm.call(MODEL_ID)
            def ghost(x: str):
                return x

            with pytest.raises(llm.exceptions.NotFoundError) as excinfo:
                ghost("hi")
            adapter.disconnect()

        err = excinfo.value
        # Prove it is the real mirascope class produced by the real error map,
        # not a hand-rolled stand-in.
        assert type(err).__module__ == "mirascope.llm.exceptions"
        assert isinstance(err, llm.exceptions.APIError)
        real_message = str(err)

        events = uploaded["events"]
        result = find_event(events, "tool.result")["payload"]
        assert result["error_type"] == "NotFoundError"
        assert result["error"] == real_message
        assert "404" in result["error"]
        assert result["success"] is False
        assert result["framework"] == "mirascope"


# ---------------------------------------------------------------------------
# Attestation
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_a_real_call(self, mock_client):
        uploaded = _drive(mock_client, CaptureConfig.full())
        events = uploaded["events"]
        assert events, "a real call must flush a non-empty trace"

        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(
                hash=e["hash"],
                scope=HashScope(e["scope"]),
                previous_hash=e.get("previous_hash"),
            )
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured"
        assert len(envelopes) == len(events), f"{len(envelopes)} envelopes for {len(events)} events"
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        ok = verify_chain(envelopes)
        assert ok.valid, f"attestation chain invalid: {ok.error}"

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
# Cost
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_is_real_and_matches_the_reported_tokens(self, mock_client):
        uploaded = _drive(mock_client, CaptureConfig.full())
        cost = find_event(uploaded["events"], "cost.record")["payload"]
        assert cost["model"] == BARE_MODEL
        assert cost["tokens_prompt"] == 17
        assert cost["tokens_completion"] == 6

        # The value is the real table price for these exact counts — not a
        # placeholder and not a constant.
        from layerlens.instrument.adapters.providers.pricing import calculate_cost
        from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage

        expected = calculate_cost(
            BARE_MODEL,
            NormalizedTokenUsage(prompt_tokens=17, completion_tokens=6, total_tokens=23),
        )
        assert expected is not None and expected > 0
        assert cost["cost_usd"] == expected

    def test_unknown_model_is_never_priced(self, mock_client):
        """Honest omission — no model means no model.invoke and no cost."""
        uploaded = capture_framework_trace(mock_client)
        adapter = MirascopeAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()

        class _ModellessCall:
            __name__ = "modelless"

            def call(self, *args, **kwargs):
                return object()

        target = _ModellessCall()
        adapter.traced_call(target)
        target.call("x")
        adapter.disconnect()

        events = uploaded["events"]
        assert find_events(events, "tool.call"), "the call must stay visible"
        assert find_events(events, "cost.record") == []
        assert find_events(events, "model.invoke") == []
        assert "0.0" not in json.dumps([e["payload"] for e in events if "cost" in json.dumps(e)])
