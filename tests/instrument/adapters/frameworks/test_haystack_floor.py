"""Offline error + attestation + redaction + cost floor for the Haystack adapter.

Closes the W2 census cells (``error``/``redaction``/``attest``/``cost``) that the
existing ``test_haystack.py`` proves only via hand-built tag dicts on the tracer
or a synthetic error string, by driving a *real* ``haystack.Pipeline`` (real
components, real ``OpenAIGenerator`` over a recorded OpenAI body) so a regression
fails in plain CI with no credentials and no network:

* Error-paths — a component failure surfaces on ``payload.error`` (Haystack does
                NOT emit ``agent.error``: a non-generator failure lands on the
                honest ``tool.result.error`` and the pipeline-level
                ``agent.output.error``, both verbatim). Two real shapes: a real
                custom ``@component`` whose ``run()`` raises (Haystack wraps it in
                a REAL ``PipelineRuntimeError``), and a real ``OpenAIGenerator``
                over a 429 transport whose REAL ``openai.RateLimitError`` is the
                wrapped ``__cause__`` — the shape a real Haystack generator call
                actually raises.
* Attestation — a real RAG ``Pipeline.run`` (custom retriever -> recorded
                generator) flushes a trace whose attestation chain reconstructs
                and ``verify_chain(...)`` returns valid; a tamper control breaks an
                interior link to prove the check is not vacuous.
* Redaction   — the same real RAG lifecycle with ``capture_content=False`` keeps
                every structural event but strips retriever/generator/pipeline
                content — and a SENTINEL sweep over ``json.dumps(events)`` — from
                the stored trace, with a ``capture_content=True`` vacuity control
                proving the same path DOES carry the content otherwise.
* Cost        — the real recorded token shape (12/1/13 on ``gpt-4o-mini``) prices
                to a real ``cost_usd`` on ``cost.record`` (the shared framework
                ``_price_cost_record`` fills it from the resolved model rate).

The only mock is the network boundary (``httpx.MockTransport`` for the real
``OpenAIGenerator``); every Haystack object, component, span, the pipeline run
and the adapter's own parser are real.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

# The base venv has no haystack-ai; this floor runs in the matrix ``haystack``
# venv (haystack-ai==2.30.1). importorskip only guards ImportError, which is the
# whole gate we need here.
pytest.importorskip("haystack")

import httpx  # noqa: E402
from haystack import Pipeline, component  # noqa: E402
from haystack.utils import Secret  # noqa: E402
from haystack.core.errors import PipelineRuntimeError  # noqa: E402
from haystack.components.generators.openai import OpenAIGenerator  # noqa: E402

import openai  # noqa: E402
import layerlens.instrument.adapters.frameworks.haystack as _mod  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.haystack import HaystackAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"
ERR_MARKER = "clause-store-unreachable-42"


@pytest.fixture(autouse=True)
def _arm_haystack_flag():
    # test_haystack.py mutates ``_mod._HAS_HAYSTACK`` in the same matrix pytest
    # process; re-arm the module flag so ``connect()`` sees the truthy value the
    # real installed import set, independent of cross-file run order.
    prev = _mod._HAS_HAYSTACK
    _mod._HAS_HAYSTACK = True
    yield
    _mod._HAS_HAYSTACK = prev


# ---------------------------------------------------------------------------
# Real Haystack components (real @component API; only the network is mocked)
# ---------------------------------------------------------------------------
@component
class _ClauseRetriever:
    """A real (non-generator) Haystack component standing in for a document
    retriever — the recalled clause text carries the SENTINEL, so it drives the
    adapter's tool.call/tool.result content path with real content."""

    @component.output_types(prompt=str)
    def run(self, query: str) -> Dict[str, str]:
        return {"prompt": f"Answer using clause [{SENTINEL}] for query: {query}. Reply with exactly: pong"}


@component
class _FailingRetriever:
    """A real Haystack component whose ``run()`` raises — the real way a pipeline
    component fails. Haystack wraps it in a real ``PipelineRuntimeError``."""

    @component.output_types(prompt=str)
    def run(self, query: str) -> Dict[str, str]:
        raise RuntimeError(f"{ERR_MARKER}: clause index backend unavailable")


def _recorded_generator(fixture: Dict[str, Any]) -> OpenAIGenerator:
    """A real ``OpenAIGenerator`` whose OpenAI client is backed by the recorded
    ChatCompletion body — Haystack's ``init_http_client`` forwards
    ``http_client_kwargs`` to ``httpx.Client(**kwargs)`` and ``httpx.Client``
    accepts ``transport=``, so the real SDK client deserializes the real body."""
    transport, _ = mock_transport(fixture)
    return OpenAIGenerator(
        api_key=Secret.from_token("test-key"),
        model="gpt-4o-mini",
        http_client_kwargs={"transport": transport},
    )


def _rag_pipeline(fixture: Dict[str, Any]) -> Pipeline:
    """retriever -> generator: the real Haystack RAG shape (one honest node per
    producer-declared component)."""
    pipe = Pipeline()
    pipe.add_component("retriever", _ClauseRetriever())
    pipe.add_component("llm", _recorded_generator(fixture))
    pipe.connect("retriever.prompt", "llm.prompt")
    return pipe


# ---------------------------------------------------------------------------
# Real error-shape floor (real component / real openai exception, real pipeline)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_component_failure_surfaces_verbatim(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = HaystackAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        pipe = Pipeline()
        pipe.add_component("retriever", _FailingRetriever())
        with pytest.raises(PipelineRuntimeError) as exc_info:
            pipe.run({"retriever": {"query": "find the indemnification clause"}})
        adapter.disconnect()

        # A REAL Haystack SDK exception class — not a hand-rolled stand-in — with
        # the component's own RuntimeError as the wrapped cause.
        assert type(exc_info.value).__name__ == "PipelineRuntimeError"
        assert type(exc_info.value).__module__.startswith("haystack")
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert ERR_MARKER in str(exc_info.value.__cause__)

        events = uploaded["events"]
        # Haystack surfaces a component failure on the honest ``tool.result.error``
        # (it does NOT emit ``agent.error``). The real error text flows verbatim —
        # bite: a dropped/mangled error, or a stopped emit-on-failure, fails here.
        result = find_event(events, "tool.result")
        assert "error" in result["payload"], "component failure dropped from tool.result"
        assert ERR_MARKER in result["payload"]["error"]
        assert result["payload"]["component_name"] == "retriever"

        # The pipeline-level ``agent.output`` also carries the wrapped failure.
        out = find_event(events, "agent.output")
        assert "error" in out["payload"], "pipeline failure dropped from agent.output"
        assert ERR_MARKER in out["payload"]["error"]

    def test_real_provider_error_surfaces_at_pipeline(self, mock_client):
        # A genuine 429 body -> the real OpenAI SDK raises RateLimitError, the shape
        # a real Haystack OpenAIGenerator call actually raises. max_retries=0 so the
        # SDK surfaces it instead of retrying the retryable status.
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Rate limit reached for gpt-4o-mini",
                        "type": "requests",
                        "code": "rate_limit_exceeded",
                    }
                },
            )

        uploaded = capture_framework_trace(mock_client)
        adapter = HaystackAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        gen = OpenAIGenerator(
            api_key=Secret.from_token("test-key"),
            model="gpt-4o-mini",
            http_client_kwargs={"transport": httpx.MockTransport(_handler)},
            max_retries=0,
        )
        pipe = Pipeline()
        pipe.add_component("llm", gen)
        with pytest.raises(PipelineRuntimeError) as exc_info:
            pipe.run({"llm": {"prompt": "hi"}})
        adapter.disconnect()

        # The REAL openai SDK exception is the wrapped cause (proves the real
        # provider shape flowed through, not a synthetic string).
        cause = exc_info.value.__cause__
        assert isinstance(cause, openai.RateLimitError)
        assert isinstance(cause, openai.OpenAIError)

        events = uploaded["events"]
        # A generator failure crashes the pipeline; the real openai error text —
        # HTTP status + the provider's own reason code — flows verbatim onto
        # ``agent.output.error``.
        out = find_event(events, "agent.output")
        assert "error" in out["payload"], "provider failure dropped from agent.output"
        assert "429" in out["payload"]["error"]
        assert "rate_limit_exceeded" in out["payload"]["error"]


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real pipeline run
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_pipeline(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        adapter = HaystackAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        pipe = _rag_pipeline(fixture)
        result = pipe.run({"retriever": {"query": "indemnification obligations"}})
        adapter.disconnect()

        assert result["llm"]["replies"][0] == "pong"

        events = uploaded["events"]
        assert events, "real pipeline run must flush a non-empty trace"
        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real pipeline trace"
        assert len(envelopes) == len(events), (
            f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
        )
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result_ok = verify_chain(envelopes)
        assert result_ok.valid, f"attestation chain invalid: {result_ok.error}"

        # Vacuity control: verify_chain must REJECT a broken interior link, proving
        # the pass above is not trivially true.
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
# Redaction content-absence over a real RAG pipeline lifecycle
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def _drive_rag(self, mock_client, *, capture_content: bool) -> Dict[str, Any]:
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)
        adapter = HaystackAdapter(mock_client, capture_config=CaptureConfig(capture_content=capture_content))
        adapter.connect()
        pipe = _rag_pipeline(fixture)
        result = pipe.run({"retriever": {"query": f"clause about {SENTINEL}"}})
        adapter.disconnect()
        assert result["llm"]["replies"][0] == "pong"
        return uploaded

    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: with capture_content=True the SAME real RAG lifecycle
        DOES carry the SENTINEL and the content keys it rides on."""
        uploaded = self._drive_rag(mock_client, capture_content=True)
        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        assert "input" in find_event(events, "agent.input")["payload"]
        assert "input" in find_event(events, "tool.call")["payload"]
        assert "output" in find_event(events, "tool.result")["payload"]
        assert "input" in find_event(events, "model.invoke")["payload"]
        assert "output" in find_event(events, "agent.output")["payload"]

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps every structural event but strips
        retriever/generator/pipeline content — and the SENTINEL — from the trace."""
        uploaded = self._drive_rag(mock_client, capture_content=False)
        events = uploaded["events"]
        assert events, "the lifecycle must still emit structural events without content"

        # 1) SENTINEL sweep over the serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) Every content key must be absent from the payload that would carry it.
        for e in find_events(events, "agent.input"):
            assert "input" not in e["payload"], "agent.input leaked 'input' under capture_content=False"
        for e in find_events(events, "tool.call"):
            assert "input" not in e["payload"], "tool.call leaked 'input' under capture_content=False"
        for e in find_events(events, "tool.result"):
            assert "output" not in e["payload"], "tool.result leaked 'output' under capture_content=False"
        for e in find_events(events, "model.invoke"):
            assert "input" not in e["payload"], "model.invoke leaked 'input' under capture_content=False"
            assert "output" not in e["payload"], "model.invoke leaked 'output' under capture_content=False"
        for e in find_events(events, "agent.output"):
            assert "output" not in e["payload"], "agent.output leaked 'output' under capture_content=False"

        # 3) The structural spine survives (graph/render still builds): the honest
        # component nodes are still present even with content stripped.
        assert find_events(events, "tool.call"), "structural tool.call dropped under capture_content=False"
        assert find_events(events, "model.invoke"), "structural model.invoke dropped under capture_content=False"


# ---------------------------------------------------------------------------
# Cost floor — a real recorded token shape must price to a real cost_usd
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_usd_present_on_real_token_shape(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        adapter = HaystackAdapter(mock_client)  # standard config
        adapter.connect()
        pipe = Pipeline()
        pipe.add_component("llm", _recorded_generator(fixture))
        result = pipe.run({"llm": {"prompt": "Reply with exactly: pong"}})
        adapter.disconnect()

        assert result["llm"]["replies"][0] == "pong"

        cost = find_event(uploaded["events"], "cost.record")
        # The real recorded token shape (12/1/13) parsed off the real generator
        # meta, on the response model id.
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 13
        assert cost["payload"]["model"] == "gpt-4o-mini-2024-07-18"

        # The shared framework cost augmentation prices the real model+tokens to a
        # real USD figure — bite: if pricing regresses (or the model unprices),
        # cost_usd is None and this fails.
        cost_usd = cost["payload"].get("cost_usd")
        assert cost_usd is not None, "cost_usd absent — the real recorded model failed to price"
        assert isinstance(cost_usd, float)
        assert cost_usd > 0
