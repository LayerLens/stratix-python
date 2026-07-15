"""Offline redaction + error + attestation + cost floor for the Marvin adapter.

Every lane drives the REAL ``marvin`` module through the REAL adapter patch with
no credentials and no network — a pydantic-ai ``TestModel`` stands in for the
provider transport, which is the seam ``marvin.Agent.get_model()`` resolves.

* Redaction   — a real ``marvin.classify``/``cast`` under ``capture_content=False``
                keeps the caller's data out of ``tool.call.input`` /
                ``model.invoke.args|kwargs|response``, proven by a SENTINEL sweep
                over ``json.dumps(events)``, with a ``capture_content=True``
                vacuity control proving the same path DOES carry it otherwise.
* Error       — a REAL marvin SDK exception (``marvin.extract`` rejects string
                targets without instructions — raised by marvin's own fns/extract.py,
                not a synthetic RuntimeError) surfaces as an honest failed
                ``tool.call`` whose ``error_type`` survives redaction while the
                free-text ``error`` does not.
* Attestation — a real classify flushes a trace whose chain reconstructs and
                ``verify_chain(...)`` accepts; a tamper control proves the check
                is not vacuous.
* Cost        — the honest OMISSION: marvin exposes no usage on its primitives,
                so no ``cost.record`` and no token/cost field is ever synthesized,
                even though the resolved model is one the pricing table knows.
"""

from __future__ import annotations

import os
import sys
import json
import tempfile

import pytest

if sys.version_info < (3, 10):
    pytest.skip("marvin requires Python >= 3.10", allow_module_level=True)

# ``import marvin`` calls ensure_db_tables_exist() at module scope — point it at a
# throwaway file BEFORE the import so a test run never touches the real database.
os.environ.setdefault(
    "MARVIN_DATABASE_URL",
    "sqlite+aiosqlite:///" + os.path.join(tempfile.mkdtemp(prefix="layerlens-marvin-"), "marvin.db"),
)

marvin = pytest.importorskip("marvin", reason="marvin not installed")

from pydantic_ai.models.test import TestModel  # noqa: E402

from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.marvin import MarvinAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

marvin.settings.enable_default_print_handler = False

#: Must never survive capture_content=False anywhere in the serialized trace.
SENTINEL = "ACCT-4429-PATIENT-Jane-Roe-DIAGNOSIS"


def _agent():
    return marvin.Agent(name="Sentiment Analyst", model=TestModel())


def _run(mock_client, capture_config, fn):
    adapter = MarvinAdapter(mock_client, capture_config=capture_config)
    adapter.connect()
    uploaded = capture_framework_trace(mock_client)
    try:
        fn()
    finally:
        adapter.disconnect()
    return uploaded


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_no_content_strips_the_callers_data_but_keeps_the_topology(self, mock_client):
        uploaded = _run(
            mock_client,
            CaptureConfig.standard(),
            lambda: marvin.classify(SENTINEL, labels=["positive", "negative"], agent=_agent()),
        )

        call = find_event(uploaded["events"], "tool.call")["payload"]
        assert "input" not in call, "the caller's data survived capture_content=False"
        assert "output" not in call
        # Structure/topology must SURVIVE redaction.
        assert call["framework"] == "marvin"
        assert call["tool_name"] == "marvin.classify"
        assert call["primitive"] == "classify"
        assert call["success"] is True
        assert call["latency_ms"] > 0
        assert call["agent_name"] == "Sentiment Analyst"

        invoke = find_event(uploaded["events"], "model.invoke")["payload"]
        for key in ("args", "kwargs", "response"):
            assert key not in invoke, f"model.invoke.{key} survived capture_content=False"
        assert invoke["model"] == "test", "the model must survive redaction"

    def test_sentinel_sweep_over_the_whole_serialized_trace(self, mock_client):
        uploaded = _run(
            mock_client,
            CaptureConfig.standard(),
            lambda: marvin.cast(SENTINEL, target=str, instructions="Echo it back", agent=_agent()),
        )
        assert uploaded["events"], "no trace was flushed"
        blob = json.dumps(uploaded["events"])
        assert SENTINEL not in blob, "the caller's data leaked somewhere in the serialized trace"

    def test_vacuity_control_capture_content_true_does_carry_it(self, mock_client):
        """Proves the sweep above can fail — the same path DOES carry the content."""
        uploaded = _run(
            mock_client,
            CaptureConfig.full(),
            lambda: marvin.cast(SENTINEL, target=str, instructions="Echo it back", agent=_agent()),
        )
        blob = json.dumps(uploaded["events"])
        assert SENTINEL in blob, "capture_content=True captured no content — the redaction test is vacuous"

        call = find_event(uploaded["events"], "tool.call")["payload"]
        assert any(SENTINEL in str(v) for v in call["input"])
        invoke = find_event(uploaded["events"], "model.invoke")["payload"]
        assert any(SENTINEL in str(v) for v in invoke["args"])

    def test_a_caller_label_does_not_survive_no_content(self, mock_client):
        """``classify(labels=[...])`` is the CALLER'S OWN taxonomy, not schema.

        A real label set is arbitrary customer text — ``labels=["billing dispute
        acct 4429", "patient consented"]`` — so it is content, unlike
        ``response_model`` (a symbol out of the customer's source). The sentinel
        sweep above cannot see this: it only ever puts the sentinel in the DATA
        argument, while the label set rides its own payload key on BOTH tool.call
        and model.invoke.
        """
        uploaded = _run(
            mock_client,
            CaptureConfig.standard(),
            lambda: marvin.classify("some text", labels=[SENTINEL, "benign"], agent=_agent()),
        )

        call = find_event(uploaded["events"], "tool.call")["payload"]
        assert "labels" not in call, "the caller's label taxonomy survived capture_content=False on tool.call"
        invoke = find_event(uploaded["events"], "model.invoke")["payload"]
        assert "labels" not in invoke, "the caller's label taxonomy survived capture_content=False on model.invoke"
        assert SENTINEL not in json.dumps(uploaded["events"]), "a caller-supplied label leaked into the trace"

        # Structure/topology must SURVIVE — only the taxonomy VALUES are stripped.
        assert call["primitive"] == "classify"
        assert call["success"] is True
        assert invoke["model"] == "test"

    def test_vacuity_control_labels_are_carried_with_content_on(self, mock_client):
        """Proves the lane above is about GATING, not about labels never being set."""
        uploaded = _run(
            mock_client,
            CaptureConfig.full(),
            lambda: marvin.classify("some text", labels=[SENTINEL, "benign"], agent=_agent()),
        )
        call = find_event(uploaded["events"], "tool.call")["payload"]
        assert SENTINEL in call["labels"], "labels were never captured at all — the gating lane is vacuous"
        invoke = find_event(uploaded["events"], "model.invoke")["payload"]
        assert SENTINEL in invoke["labels"]

    def test_api_key_is_never_echoed_even_with_content_on(self, mock_client):
        """No marvin 3.x primitive takes ``api_key``, but a @marvin.fn-decorated
        function's own parameters are echoed under ``model.invoke.kwargs`` — and one
        of those can be named ``api_key``. It stays excluded at every capture level.

        The value is deliberately NOT secret-SHAPED: a ``sk-...`` string is caught by
        the collector's secret-scrub chokepoint regardless, which would make this
        lane pass whether the exclusion works or not. This value survives the
        scrubber, so ``_EXCLUDED_KWARGS`` is the only thing keeping it out.
        """
        agent = _agent()
        key_value = "TENANT-KEY-SENTINEL-99"

        def call():
            @marvin.fn(agent=agent)
            def lookup(account: str, api_key: str) -> str:
                """Look the account up."""

            lookup(account=SENTINEL, api_key=key_value)

        uploaded = _run(mock_client, CaptureConfig.full(), call)

        blob = json.dumps(uploaded["events"])
        assert key_value not in blob, "api_key was echoed onto a payload"
        # Vacuity control: the SIBLING kwarg on the same call IS captured, so the
        # assertion above is about the exclusion — not about kwargs being empty.
        assert SENTINEL in blob, "no kwargs were captured at all — the api_key lane is vacuous"


# ---------------------------------------------------------------------------
# Error — a REAL marvin SDK exception shape
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_marvin_error_surfaces_honestly(self, mock_client):
        def call():
            with pytest.raises(ValueError, match="Instructions are required"):
                marvin.extract(SENTINEL, target=str, agent=_agent())

        uploaded = _run(mock_client, CaptureConfig.full(), call)

        call_payload = find_event(uploaded["events"], "tool.call")["payload"]
        assert call_payload["success"] is False
        assert call_payload["error_type"] == "ValueError"
        assert "Instructions are required" in call_payload["error"]
        assert call_payload["latency_ms"] > 0
        # The error path must not fabricate an output.
        assert "output" not in call_payload

    def test_error_text_is_gated_but_the_category_survives(self, mock_client):
        def call():
            with pytest.raises(ValueError):
                marvin.extract(SENTINEL, target=str, agent=_agent())

        uploaded = _run(mock_client, CaptureConfig.standard(), call)

        call_payload = find_event(uploaded["events"], "tool.call")["payload"]
        assert "error" not in call_payload, "the free-text error survived capture_content=False"
        assert call_payload["error_type"] == "ValueError", "the error CATEGORY must survive redaction"
        assert call_payload["success"] is False
        assert SENTINEL not in json.dumps(uploaded["events"])

    def test_the_exception_still_reaches_the_caller(self, mock_client):
        """Instrumentation must never swallow a customer's exception."""
        raised = {}

        def call():
            try:
                marvin.extract(SENTINEL, target=str, agent=_agent())
            except ValueError as exc:
                raised["exc"] = exc

        _run(mock_client, CaptureConfig.standard(), call)
        assert isinstance(raised.get("exc"), ValueError), "the adapter swallowed the caller's exception"


# ---------------------------------------------------------------------------
# Attestation
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_chain_verifies_over_a_real_classify(self, mock_client):
        uploaded = _run(
            mock_client,
            CaptureConfig.full(),
            lambda: marvin.classify("This is great", labels=["positive", "negative"], agent=_agent()),
        )

        events = uploaded["events"]
        assert events, "a real marvin call must flush a non-empty trace"
        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real marvin trace"
        assert len(envelopes) == len(events), f"{len(envelopes)} envelopes for {len(events)} events"
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

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
# Cost — the honest omission
# ---------------------------------------------------------------------------
class TestCostOmission:
    def test_a_priced_model_still_yields_no_cost_because_there_are_no_tokens(self, mock_client):
        """marvin.defaults.model is 'openai:gpt-4o' — a model the pricing table knows.

        The adapter still emits NO cost.record: marvin's primitives return only the
        parsed value and never expose usage, so any cost here would be invented.
        """
        assert marvin.defaults.model == "openai:gpt-4o", "marvin default changed; revisit this lane"

        uploaded = _run(
            mock_client,
            CaptureConfig.full(),
            # No agent= -> the model resolves from marvin.defaults (a PRICED model).
            lambda: marvin.classify(
                "great", labels=["positive", "negative"], agent=marvin.Agent(model=TestModel())
            ),
        )

        assert find_events(uploaded["events"], "cost.record") == [], (
            "a cost.record was emitted although marvin exposes no usage — that is fabricated cost"
        )
        for invoke in find_events(uploaded["events"], "model.invoke"):
            p = invoke["payload"]
            for key in ("cost_usd", "tokens_total", "tokens_prompt", "tokens_completion"):
                assert key not in p, f"model.invoke carried a synthesized {key}"

    def test_environment_config_reports_only_what_it_discovered(self, mock_client):
        uploaded = _run(
            mock_client,
            CaptureConfig.full(),
            lambda: marvin.classify("great", labels=["positive", "negative"], agent=_agent()),
        )
        cfg = find_event(uploaded["events"], "environment.config")["payload"]["config"]
        assert cfg["framework"] == "marvin"
        assert cfg["model"] == "openai:gpt-4o"
        # marvin exposes no provider setting — it must be omitted, not defaulted.
        assert "provider" not in cfg
