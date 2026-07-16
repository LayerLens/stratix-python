"""Offline attestation + redaction + cost + capture-level floor for the Langfuse
bidirectional batch-sync adapter.

Langfuse is not a real-time instrumentation wrapper: it *imports* Langfuse
traces (trace/generation/span/event/score) into flat LayerLens events and
*exports* LayerLens events back to Langfuse's ingestion API. So the "real
framework object" here is the REAL ``LangfuseAdapter`` driving a REAL
``TraceCollector`` over REAL Langfuse JSON bodies — with the network
(``httpx.Client``) as the ONLY mock. Every parse, emit, flush, pricing hook and
attestation link is real, so a regression fails in plain CI with no credentials
and no network.

Closes the W2 census cells the existing ``test_langfuse.py`` proves only weakly
(or via a mock upload that never inspects the attestation chain):

* Attestation — a real one-trace import (generation + LLM-as-judge score) flushes
                a trace whose attestation chain reconstructs and
                ``verify_chain(...)`` returns valid; the envelope count matches the
                event count and a ``root_hash`` is present. A tamper control
                proves the check is not vacuous (breaking interior link 1 is
                rejected with ``break_index == 1``).
* Redaction   — a real import with ``capture_content=False`` keeps the structural
                events (model.invoke/cost/score/tool) but strips trace/generation/
                span/event/score CONTENT, proven by a SENTINEL sweep over the
                serialized trace; a ``capture_content=True`` vacuity control proves
                the SAME path carries the SENTINEL otherwise.
* Cost        — a real generation with an upstream ``calculatedTotalCost`` carries
                ``cost_usd`` verbatim, and a generation WITHOUT upstream cost is
                priced locally from model+tokens via the real PRICING table (the
                langfuse raw-emit fallback). Both bite: drop either path and
                ``cost_usd`` goes ``None``.
* Params      — a full capture-level sweep: ``CaptureConfig.minimal()`` suppresses
                ``model.invoke`` / ``agent.code`` / ``tool.call`` while
                ``cost.record`` + ``agent.state.change`` still flow; ``full()`` is
                the vacuity control that carries all three from the SAME
                observations.

HELD (NOT committed): the real-error-shape cell. Langfuse marks a failed
observation with ``level == "ERROR"`` (+ ``statusMessage``). The adapter reads
``statusMessage`` only for EVENT-type observations and NEVER inspects ``level``
nor emits ``agent.error`` — so an imported failed LLM call / span renders as a
plain success (source_bug_suspicion #2). Proven RED offline and reported as a
held finding; no ``agent.error`` assertion is committed here because it would be
red on current source.
"""

from __future__ import annotations

import json

import layerlens.instrument.adapters.frameworks.langfuse as _mod
from layerlens.attestation._verify import verify_chain
from layerlens.attestation._envelope import HashScope, AttestationEnvelope
from layerlens.instrument._capture_config import CaptureConfig

# httpx is genuinely importable in this env, but the suite never makes a real
# network call — mirror test_langfuse.py and force the dependency flag so the
# adapter's ``_check_dependency`` guard is satisfied while every HTTP round trip
# is served by a mock client.
_mod._HAS_HTTPX = True

from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ---------------------------------------------------------------------------
# Mock-network plumbing (the ONLY mock — the adapter + collector are real)
# ---------------------------------------------------------------------------
from unittest.mock import Mock  # noqa: E402


def _response(json_data=None, status_code=200):
    resp = Mock(spec=[])
    resp.status_code = status_code
    resp.json = Mock(return_value=json_data or {})
    resp.raise_for_status = Mock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


def _mock_http():
    http = Mock(spec=[])
    http.get = Mock()
    http.post = Mock()
    http.close = Mock()
    return http


def _connected_adapter(mock_client, config):
    """A REAL LangfuseAdapter wired to a mock httpx.Client (no network)."""
    adapter = LangfuseAdapter(mock_client, capture_config=config)
    http = _mock_http()
    adapter._http = http
    adapter._connected = True
    adapter._host = "https://test.langfuse.com"
    adapter._public_key = "pk-test"
    adapter._secret_key = "sk-test"
    adapter._metadata["host"] = "https://test.langfuse.com"
    return adapter, http


def _import_one(adapter, http, trace_body):
    """Serve the list-then-fetch pair and import exactly ONE Langfuse trace.

    Exactly one trace => exactly one collector flush, so the accumulated event
    list corresponds 1:1 with the single flush's attestation chain.
    """
    http.get.side_effect = [
        _response({"data": [{"id": trace_body["id"], "updatedAt": "2026-06-14T00:00:00Z"}]}),
        _response(trace_body),
    ]
    return adapter.import_traces(limit=1)


# ---------------------------------------------------------------------------
# Industry-realistic Langfuse trace fixtures (Media: content-moderation judge)
# ---------------------------------------------------------------------------
def _generation(s: str = "", *, model: str = "gpt-4", cost=0.005):
    """A Langfuse GENERATION observation (a moderation LLM call)."""
    gen = {
        "id": "gen-moderation",
        "type": "GENERATION",
        "name": "moderation-llm",
        "model": model,
        "input": f"Classify this comment for policy violations {s}",
        "output": f"verdict: allow {s}",
        "usage": {"promptTokens": 100, "completionTokens": 50, "totalTokens": 150},
    }
    if cost is not None:
        gen["calculatedTotalCost"] = cost
    return gen


def _judge_score(s: str = ""):
    """An LLM-as-judge score — the langfuse-distinctive evaluation.result path."""
    return {
        "name": "moderation_quality",
        "value": 0.92,
        "source": "API",
        "dataType": "NUMERIC",
        "comment": f"judge rationale {s}",
    }


def _trace(observations, scores=None, s: str = "", tid: str = "lf-media-001"):
    return {
        "id": tid,
        "name": "content-moderation-review",
        "input": f"moderate user comment {s}",
        "output": f"final decision: allow {s}",
        "metadata": {"queue": "media-moderation"},
        "observations": observations,
        "scores": scores or [],
    }


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real one-trace import
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_import(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter, http = _connected_adapter(mock_client, CaptureConfig.full())

        count = _import_one(adapter, http, _trace([_generation()], scores=[_judge_score()]))
        adapter.disconnect()

        assert count == 1
        events = uploaded["events"]
        assert events, "a real Langfuse import must flush a non-empty trace"
        # The langfuse-distinctive score path must be part of the attested chain.
        assert find_event(events, "evaluation.result")["payload"]["value"] == 0.92

        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the imported trace"
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
# Redaction content-absence over a real import
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: with capture_content=True the SAME real import carries
        the SENTINEL and the content keys it rides on."""
        uploaded = capture_framework_trace(mock_client)
        adapter, http = _connected_adapter(mock_client, CaptureConfig(capture_content=True))
        span = {
            "id": "sp",
            "type": "SPAN",
            "name": "retriever",
            "input": f"lookup {SENTINEL}",
            "output": f"docs {SENTINEL}",
        }
        _import_one(
            adapter,
            http,
            _trace([_generation(SENTINEL), span], scores=[_judge_score(SENTINEL)], s=SENTINEL),
        )
        adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control import must carry the SENTINEL when capturing content"
        assert find_event(events, "agent.input")["payload"]["content"].endswith(SENTINEL)
        assert find_event(events, "agent.output")["payload"]["content"].endswith(SENTINEL)
        assert SENTINEL in find_event(events, "model.invoke")["payload"]["messages"]
        assert SENTINEL in find_event(events, "model.invoke")["payload"]["output_message"]
        assert SENTINEL in find_event(events, "tool.call")["payload"]["input"]
        assert SENTINEL in find_event(events, "evaluation.result")["payload"]["comment"]

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps the structural events but strips every
        content slot — and the SENTINEL — out of the stored trace."""
        uploaded = capture_framework_trace(mock_client)
        adapter, http = _connected_adapter(mock_client, CaptureConfig(capture_content=False))
        span = {
            "id": "sp",
            "type": "SPAN",
            "name": "retriever",
            "input": f"lookup {SENTINEL}",
            "output": f"docs {SENTINEL}",
        }
        _import_one(
            adapter,
            http,
            _trace([_generation(SENTINEL), span], scores=[_judge_score(SENTINEL)], s=SENTINEL),
        )
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "the import must still emit structural events without content"

        # 1) SENTINEL sweep over the serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) Content keys must be absent from every payload that would carry them.
        for e in find_events(events, "agent.input"):
            assert "content" not in e["payload"], "agent.input leaked 'content' under capture_content=False"
        for e in find_events(events, "agent.output"):
            assert "content" not in e["payload"], "agent.output leaked 'content' under capture_content=False"
        mi = find_event(events, "model.invoke")
        assert "messages" not in mi["payload"], "model.invoke leaked 'messages'"
        assert "output_message" not in mi["payload"], "model.invoke leaked 'output_message'"
        assert "input" not in find_event(events, "tool.call")["payload"], "tool.call leaked 'input'"
        assert "output" not in find_event(events, "tool.call")["payload"], "tool.call leaked 'output'"
        assert "comment" not in find_event(events, "evaluation.result")["payload"], "score leaked 'comment'"

        # 3) Non-content signal SURVIVES redaction (structure is not thrown away).
        assert mi["payload"]["model"] == "gpt-4"
        assert find_event(events, "cost.record")["payload"]["tokens_total"] == 150
        assert find_event(events, "evaluation.result")["payload"]["value"] == 0.92


# ---------------------------------------------------------------------------
# Cost floor — real token shape, real PRICING fallback
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_usd_present_from_upstream(self, mock_client):
        """Langfuse's own calculatedTotalCost flows onto cost.record verbatim."""
        uploaded = capture_framework_trace(mock_client)
        adapter, http = _connected_adapter(mock_client, CaptureConfig.full())
        _import_one(adapter, http, _trace([_generation(cost=0.0123)]))
        adapter.disconnect()

        cost = find_event(uploaded["events"], "cost.record")
        assert cost["payload"]["cost_usd"] == 0.0123
        assert cost["payload"]["tokens_prompt"] == 100
        assert cost["payload"]["tokens_completion"] == 50
        assert cost["payload"]["tokens_total"] == 150

    def test_cost_usd_priced_locally_when_upstream_absent(self, mock_client):
        """When Langfuse omits calculatedTotalCost, the langfuse raw-emit fallback
        prices cost_usd from model+tokens via the real PRICING table. Bite: drop
        the _price_cost_record call and cost_usd goes None."""
        from layerlens.instrument.adapters.providers.pricing import PRICING, calculate_cost
        from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage

        uploaded = capture_framework_trace(mock_client)
        adapter, http = _connected_adapter(mock_client, CaptureConfig.full())
        _import_one(adapter, http, _trace([_generation(cost=None)]))
        adapter.disconnect()

        cost = find_event(uploaded["events"], "cost.record")
        expected = calculate_cost(
            "gpt-4",
            NormalizedTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            PRICING,
        )
        assert expected, "PRICING table must price gpt-4 for this floor to bite"
        assert cost["payload"].get("cost_usd") == expected, (
            "langfuse did not locally price a no-upstream-cost generation"
        )


# ---------------------------------------------------------------------------
# Capture-level params sweep (broadens the ◑ params cell)
# ---------------------------------------------------------------------------
class TestCaptureLevelParams:
    _OBS = [
        {
            "id": "g",
            "type": "GENERATION",
            "name": "moderation-llm",
            "model": "gpt-4",
            "input": "q",
            "output": "a",
            "usage": {"promptTokens": 100, "completionTokens": 50, "totalTokens": 150},
            "calculatedTotalCost": 0.005,
        },
        {"id": "sc", "type": "SPAN", "name": "code-executor", "input": "x", "output": "y"},
        {"id": "st", "type": "SPAN", "name": "retriever", "input": "x", "output": "y"},
        {"id": "e", "type": "EVENT", "name": "status", "statusMessage": "done", "input": "d"},
    ]

    def test_full_config_carries_all_levels(self, mock_client):
        """Vacuity control: at full() the SAME observations surface model.invoke,
        agent.code and tool.call."""
        uploaded = capture_framework_trace(mock_client)
        adapter, http = _connected_adapter(mock_client, CaptureConfig.full())
        _import_one(adapter, http, _trace(list(self._OBS)))
        adapter.disconnect()

        events = uploaded["events"]
        assert len(find_events(events, "model.invoke")) == 1
        assert len(find_events(events, "agent.code")) == 1
        assert len(find_events(events, "tool.call")) == 1
        assert len(find_events(events, "cost.record")) == 1
        assert len(find_events(events, "agent.state.change")) == 1

    def test_minimal_config_suppresses_metadata_levels(self, mock_client):
        """minimal() (l3_model_metadata / l2 off) must drop model.invoke,
        agent.code and tool.call while cost.record + agent.state.change still flow.
        Bite: if minimal stops gating, model.invoke/agent.code/tool.call reappear."""
        uploaded = capture_framework_trace(mock_client)
        adapter, http = _connected_adapter(mock_client, CaptureConfig.minimal())
        _import_one(adapter, http, _trace(list(self._OBS)))
        adapter.disconnect()

        events = uploaded["events"]
        assert len(find_events(events, "model.invoke")) == 0, "minimal must suppress model.invoke"
        assert len(find_events(events, "agent.code")) == 0, "minimal must suppress agent.code"
        assert len(find_events(events, "tool.call")) == 0, "minimal must suppress tool.call"
        # Cost + state always flow (spend accounting + lifecycle must survive gating).
        assert len(find_events(events, "cost.record")) == 1
        assert len(find_events(events, "agent.state.change")) == 1


# ---------------------------------------------------------------------------
# Error fidelity — an ERROR-level observation must surface as agent.error
# ---------------------------------------------------------------------------
class TestErrorFloor:
    def test_error_level_generation_emits_agent_error(self, mock_client):
        """A Langfuse generation flagged ``level == "ERROR"`` (with a statusMessage)
        is a FAILED LLM call. It must import as a distinct ``agent.error`` carrying
        the real error text + a real error_type — not silently as a healthy
        model.invoke. Bite: drop the level read and no agent.error is emitted."""
        uploaded = capture_framework_trace(mock_client)
        adapter, http = _connected_adapter(mock_client, CaptureConfig.full())
        err_gen = {
            "id": "gen-err",
            "type": "GENERATION",
            "name": "moderation-llm",
            "model": "gpt-4",
            "level": "ERROR",
            "statusMessage": "RateLimitError: 429 quota exceeded",
            "input": "q",
            "output": None,
            "usage": {"promptTokens": 100, "completionTokens": 0, "totalTokens": 100},
        }
        _import_one(adapter, http, _trace([err_gen]))
        adapter.disconnect()

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        assert len(errors) == 1, (
            f"ERROR-level Langfuse generation emitted no agent.error; types={[e['event_type'] for e in events]}"
        )
        assert "429 quota exceeded" in errors[0]["payload"].get("error", "")
        assert errors[0]["payload"].get("status") == "error"
        # A real error_type is recovered from the statusMessage exception prefix.
        assert errors[0]["payload"].get("error_type") == "RateLimitError"

    def test_error_content_redacted_but_status_survives(self, mock_client):
        """The free-text error message is content (stripped under
        capture_content=False), but the structural error signal — the agent.error
        event, its status, level and error_type — must SURVIVE redaction so a
        failure is never silently downgraded to a success."""
        uploaded = capture_framework_trace(mock_client)
        adapter, http = _connected_adapter(mock_client, CaptureConfig(capture_content=False))
        err_gen = {
            "id": "gen-err",
            "type": "GENERATION",
            "name": "moderation-llm",
            "model": "gpt-4",
            "level": "ERROR",
            "statusMessage": f"RateLimitError: {SENTINEL}",
            "usage": {"promptTokens": 10, "completionTokens": 0, "totalTokens": 10},
        }
        _import_one(adapter, http, _trace([err_gen]))
        adapter.disconnect()

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        assert len(errors) == 1, "the structural agent.error must survive redaction"
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: error message survived capture_content=False"
        assert "error" not in errors[0]["payload"], "agent.error leaked the free-text 'error' under redaction"
        assert errors[0]["payload"].get("status") == "error"
        assert errors[0]["payload"].get("level") == "ERROR"
        assert errors[0]["payload"].get("error_type") == "RateLimitError"


# ---------------------------------------------------------------------------
# Nesting fidelity — parentObservationId must resolve to the emitted span
# ---------------------------------------------------------------------------
class TestNestingFloor:
    def test_parent_observation_id_resolves_to_emitted_span(self, mock_client):
        """A child observation's ``parentObservationId`` must resolve to the SAME
        span_id the parent observation emits under — otherwise the child is
        orphaned onto a random phantom span and the retriever->LLM causal edge is
        lost. Bite: mint a fresh span for the parent ref and the child no longer
        nests under its real parent."""
        uploaded = capture_framework_trace(mock_client)
        adapter, http = _connected_adapter(mock_client, CaptureConfig.full())
        parent_span = {
            "id": "obs-parent",
            "type": "SPAN",
            "name": "retriever-chain",
            "input": "lookup",
            "output": "docs",
        }
        child_gen = {
            "id": "obs-child",
            "type": "GENERATION",
            "name": "moderation-llm",
            "model": "gpt-4",
            "parentObservationId": "obs-parent",
            "input": "q",
            "output": "a",
            "usage": {"promptTokens": 10, "completionTokens": 5, "totalTokens": 15},
        }
        _import_one(adapter, http, _trace([parent_span, child_gen]))
        adapter.disconnect()

        events = uploaded["events"]
        parent_evt = find_event(events, "tool.call")  # the parent SPAN import
        child_evt = find_event(events, "model.invoke")  # the child GENERATION import

        # The child must nest UNDER the parent's emitted span, not a phantom.
        assert child_evt["parent_span_id"] == parent_evt["span_id"], (
            f"orphaned nesting: child parent_span_id={child_evt['parent_span_id']} "
            f"!= parent span_id={parent_evt['span_id']}"
        )
        # The resolved parent must be a REAL emitted span in this trace.
        emitted_span_ids = {e["span_id"] for e in events}
        assert child_evt["parent_span_id"] in emitted_span_ids, "child references a span no event emits"
        # The generation's own cost.record must share the child's span (unchanged).
        assert find_event(events, "cost.record")["span_id"] == child_evt["span_id"]
