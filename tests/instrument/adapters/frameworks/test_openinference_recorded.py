"""Recorded-real-span replay for the OpenInference ingestion adapter (LAY-3614).

WHY THIS FIXTURE IS THE RIGHT "UPSTREAM"
----------------------------------------
The corpus rule is: record UPSTREAM of the parser, assert DOWNSTREAM of it. Every
other adapter's upstream is a provider's HTTP body — but ``openinference`` patches
nothing and calls no model: it *ingests OpenTelemetry spans*. So its upstream IS
the spans, and ``tests/fixtures/recorded/openinference/default.json`` is the real
thing we do not control:

* it holds REAL OpenInference/OTel spans exported from ONE real instrumented
  retail-support RAG run (see ``samples/data/generators/openinference.py``);
* the ``ChatCompletion`` LLM span was emitted by the REAL
  ``openinference-instrumentation-openai`` auto-instrumentor around a REAL
  ``openai`` gpt-4o-mini call — its instrumentation scope in the fixture is
  literally ``openinference.instrumentation.openai``, which is the provenance
  tell that no hand-authored span could carry;
* the AGENT / TOOL / RETRIEVER spans came from the real OpenInference
  ``OITracer`` wrapping the run's real steps;
* it is serialized as OTLP/JSON by the REAL OTel OTLP encoder, so the replay
  drives the adapter's genuine wire shapes — hex ids, ``{"intValue": "372"}``
  STRING-typed ints, ``STATUS_CODE_OK`` enum names, and the flattened
  ``retrieval.documents.{i}.document.*`` keys a real instrumentor emits. Those
  are exactly the coercions (``_otlp_value`` / ``_as_int`` / ``_normalize_status_name``
  / ``_flattened_index_count``) that a hand-built dict double would never test.

The strong tells that the REAL recorded body flowed through: ``model.invoke``
reports the resolved dated model id ``gpt-4o-mini-2024-07-18`` (the id OpenAI
*returned*, not the ``gpt-4o-mini`` that was requested) with the real usage
372/80/452, and ``cost.record.cost_usd`` is the real figure DERIVED from those
real counts through the real pricing table — none of which any double supplies.

HONEST NOTE ON ``status``: the OTLP/JSON path omits a status for the UNSET spans
(protobuf drops the 0-valued enum, so ``status`` is ``{}``), whereas the live
ReadableSpan path reports ``"UNSET"``. Both are honest — UNSET carries no failure
signal either way — so this lane asserts status only where the span really
declares one (the OK LLM span), rather than pinning a false equivalence.
"""

from __future__ import annotations

import json

import pytest

from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.openinference import OpenInferenceAdapter

from .conftest import find_event, find_events, capture_framework_trace
from ..._recorded import load_recorded


def _recorded_spans(fixture):
    """Yield each real span dict out of the fixture's OTLP/JSON document."""
    for rs in fixture["otlp"]["resourceSpans"]:
        for ss in rs["scopeSpans"]:
            for span in ss["spans"]:
                yield span


def _ingest(mock_client, fixture, *, capture_content: bool = True):
    config = CaptureConfig.full() if capture_content else CaptureConfig()
    adapter = OpenInferenceAdapter(mock_client, capture_config=config)
    adapter.connect()
    ingested = adapter.ingest_spans(_recorded_spans(fixture))
    adapter.flush()
    return ingested


class TestOpenInferenceRecorded:
    def test_fixture_is_a_real_auto_instrumented_capture(self):
        """Provenance guard: the LLM span must come from the REAL OpenInference
        auto-instrumentor. If this corpus were ever swapped for hand-authored
        spans, the instrumentation scope would not survive."""
        fixture = load_recorded("openinference", "default")
        scopes = {ss["scope"]["name"] for rs in fixture["otlp"]["resourceSpans"] for ss in rs["scopeSpans"]}
        assert "openinference.instrumentation.openai" in scopes
        assert fixture["provenance"]["provider"] == "openinference"

    def test_real_spans_map_to_the_event_contract(self, mock_client):
        fixture = load_recorded("openinference", "default")
        uploaded = capture_framework_trace(mock_client)

        _ingest(mock_client, fixture)
        events = uploaded["events"]

        # One real run's four spans -> the full documented mapping (the AGENT
        # span is a PAIR, and the priced LLM span adds a cost.record).
        assert sorted({e["event_type"] for e in events}) == [
            "agent.input",
            "agent.output",
            "cost.record",
            "model.invoke",
            "retrieval.query",
            "tool.call",
        ]

    def test_llm_span_carries_the_real_model_and_tokens(self, mock_client):
        fixture = load_recorded("openinference", "default")
        uploaded = capture_framework_trace(mock_client)

        _ingest(mock_client, fixture)

        mi = find_event(uploaded["events"], "model.invoke")["payload"]
        assert mi["framework"] == "openinference"
        # The RESOLVED dated id OpenAI really returned (the request asked for the
        # undated "gpt-4o-mini"), read off the real instrumentor's span.
        assert mi["model"] == "gpt-4o-mini-2024-07-18"
        assert mi["model_name"] == "gpt-4o-mini-2024-07-18"
        assert mi["provider"] == "openai"
        # Real usage off the real call, through the OTLP {"intValue": "372"}
        # STRING coercion — the dual-written flat token canon.
        assert mi["prompt_tokens"] == 372
        assert mi["input_tokens"] == 372
        assert mi["completion_tokens"] == 80
        assert mi["output_tokens"] == 80
        assert mi["total_tokens"] == 452
        # The real span declared OK (STATUS_CODE_OK), and it survived normalization.
        assert mi["status"] == "OK"
        assert mi["span_kind"] == "LLM"
        # A real measured duration off the real span bounds.
        assert mi["latency_ms"] > 0

    def test_cost_is_really_derived_from_the_real_tokens(self, mock_client):
        """gpt-4o-mini IS priced, so the real counts yield a real cost — this is
        the priced branch of ``_emit_cost_record``, not a fabricated 0.0."""
        fixture = load_recorded("openinference", "default")
        uploaded = capture_framework_trace(mock_client)

        _ingest(mock_client, fixture)

        cost = find_event(uploaded["events"], "cost.record")["payload"]
        assert cost["framework"] == "openinference"
        assert cost["model"] == "gpt-4o-mini-2024-07-18"
        assert cost["prompt_tokens"] == 372
        assert cost["completion_tokens"] == 80

        # Recomputed from the SAME real pricing table the adapter used, so the
        # assertion pins the real derivation rather than a copied constant.
        from layerlens.instrument.adapters.providers.pricing import PRICING, calculate_cost
        from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage

        expected = calculate_cost(
            "gpt-4o-mini-2024-07-18",
            NormalizedTokenUsage(prompt_tokens=372, completion_tokens=80, total_tokens=452),
            PRICING,
        )
        assert expected is not None and expected > 0
        assert cost["cost_usd"] == pytest.approx(expected)

    def test_retriever_span_counts_the_real_flattened_documents(self, mock_client):
        """The real instrumentor flattens documents into
        ``retrieval.documents.{i}.document.*`` — the indexed form, never a list."""
        fixture = load_recorded("openinference", "default")
        uploaded = capture_framework_trace(mock_client)

        _ingest(mock_client, fixture)

        rq = find_event(uploaded["events"], "retrieval.query")["payload"]
        assert rq["framework"] == "openinference"
        assert rq["span_kind"] == "RETRIEVER"
        # Three real policy documents were really retrieved for this question.
        assert rq["document_count"] == 3
        # The real customer question rode through as the query.
        assert "Summit Trail boots" in rq["query"]

    def test_tool_span_carries_the_real_tool_identity(self, mock_client):
        fixture = load_recorded("openinference", "default")
        uploaded = capture_framework_trace(mock_client)

        _ingest(mock_client, fixture)

        tc = find_event(uploaded["events"], "tool.call")["payload"]
        assert tc["tool_name"] == "order_lookup"
        assert tc["span_kind"] == "TOOL"
        # The real order record the real tool step returned.
        assert "SO-884213" in tc["output"]

    def test_agent_span_pair_never_fabricates_an_agent_name(self, mock_client):
        """The Agent column must stay an honest empty-state: an AGENT span's only
        identity is its span NAME, which ``_identity.py`` forbids as an
        Agent-column source. The name rides in ``agent_id`` (which no identity
        tier reads) and ``agent_name`` is NEVER written."""
        fixture = load_recorded("openinference", "default")
        uploaded = capture_framework_trace(mock_client)

        _ingest(mock_client, fixture)

        agent_in = find_event(uploaded["events"], "agent.input")["payload"]
        agent_out = find_event(uploaded["events"], "agent.output")["payload"]
        for payload in (agent_in, agent_out):
            assert payload["agent_id"] == "retail_support_agent"
            assert payload["operation"] == "agent"
            assert "agent_name" not in payload

        # The real grounded answer the real model produced.
        assert "POL-WAR-04" in agent_out["output_text"]
        # The output is stamped at span END, so it never precedes the input.
        assert agent_out["timestamp"] >= agent_in["timestamp"]

    def test_content_is_gated_but_the_real_structure_survives(self, mock_client):
        """Under the DEFAULT config the real prompt/answer text is withheld while
        the real model/token/cost structure still rides through.

        RE-JUDGED in LAY-3622 F2. This assertion used to read
        ``"output_text" not in out``, pinning a divergence: the adapter sets
        ``input_text``/``output_text`` to a present-but-EMPTY string so the turn stays
        valid for a strict consumer, and the collector-tier backstop then deleted the
        keys outright — the mitigation never reached the wire. The backstop now keeps
        a content key that is already empty (``_is_content_free``), so the adapter's
        documented invariant finally holds end-to-end and this lane asserts the
        present-but-EMPTY value instead of its absence.

        The privacy outcome is unchanged, and that is the point of the fix: an empty
        string carries no content. The real-answer leak check below is what proves it
        over the REAL recorded fixture rather than a constructed payload.
        """
        fixture = load_recorded("openinference", "default")
        uploaded = capture_framework_trace(mock_client)

        _ingest(mock_client, fixture, capture_content=False)

        mi = find_event(uploaded["events"], "model.invoke")["payload"]
        assert "prompt" not in mi
        assert "output" not in mi
        # Structure is not content: the real model + real tokens still report.
        assert mi["model"] == "gpt-4o-mini-2024-07-18"
        assert mi["prompt_tokens"] == 372

        outs = find_events(uploaded["events"], "agent.output")
        assert outs, "the agent turn must still be recorded when content is gated"
        out = outs[0]["payload"]
        # The ingest-visible surface survives as present-but-EMPTY (see docstring);
        # the un-required ``output`` duplicate is omitted at emit time and stays gone.
        assert out["output_text"] == ""
        assert "output" not in out
        # The real answer must not survive anywhere in the gated trace.
        assert "POL-WAR-04" not in json.dumps(uploaded["events"])
        # ...while the real structural identity of the turn does.
        assert out["agent_id"] == "retail_support_agent"
        assert out["span_kind"] == "AGENT"

    def test_real_source_trace_correlation_is_preserved(self, mock_client):
        """The events carry the SOURCE OTel trace/span ids, so the LayerLens trace
        still correlates back to the producer's own telemetry."""
        fixture = load_recorded("openinference", "default")
        uploaded = capture_framework_trace(mock_client)

        _ingest(mock_client, fixture)

        source_trace_ids = {span["traceId"] for span in _recorded_spans(fixture)}
        assert len(source_trace_ids) == 1
        source_trace_id = source_trace_ids.pop()

        # cost.record is built fresh by ``_emit_cost_record`` (not from
        # ``_common``), so it correlates on the ENVELOPE only — every other event
        # also carries the source trace id in its payload.
        for event in uploaded["events"]:
            if event["event_type"] != "cost.record":
                assert event["payload"]["trace_id"] == source_trace_id

        # The real source span ids ride the envelope, preserving the real
        # topology the run actually had: agent -> {tool, retriever, LLM}.
        by_type = {e["event_type"]: e for e in uploaded["events"]}
        agent_span_id = by_type["agent.input"]["span_id"]
        for event_type in ("tool.call", "retrieval.query", "model.invoke", "cost.record"):
            assert by_type[event_type]["parent_span_id"] == agent_span_id
        # The cost really hangs off the LLM span it was derived from.
        assert by_type["cost.record"]["span_id"] == by_type["model.invoke"]["span_id"]

        # The LLM span really was a child of the real agent span.
        assert by_type["model.invoke"]["payload"]["parent_span_id"] == agent_span_id


class TestOpenInferenceRecordedTeam:
    """Offline replay of the MULTI-AGENT recorded fixture (LAY-3622 E1).

    All 9 tests in the class above load ``('openinference', 'default')`` — the
    single-agent capture. ``team.json`` (42.8K, recorded from a real
    ``openinference-instrumentation-openai`` run against gpt-4o-mini) had no offline
    mapping assertion at all. It was not, as first thought, covered via the
    samples/render path: that path consumes a DIFFERENT artifact
    (``samples/data/traces/industry/retail_openinference_support_team.jsonl``), and a
    repo-wide search found NO reader for ``team.json`` beyond the two generic corpus
    hygiene lanes (secret-leak + provenance, which sweep every fixture by rglob).
    It was an orphan artifact — a real recorded multi-agent capture nothing asserted
    the mapping over.

    HONEST SCOPE NOTE: the fixture holds 3 AGENT, 3 LLM and 2 RETRIEVER spans and
    **zero CHAIN spans**, so this lane is an AGENT-pair lane, not the "AGENT/CHAIN
    pair" lane originally scoped. CHAIN dispatch is covered by the conformance corpus
    instead. Saying so rather than implying CHAIN coverage that does not exist.
    """

    def test_the_fixture_is_the_real_multi_agent_capture(self, mock_client) -> None:
        fixture = load_recorded("openinference", "team")
        spans = list(_recorded_spans(fixture))
        assert len(spans) == 8
        kinds: dict = {}
        for span in spans:
            for kv in span.get("attributes", []):
                if kv.get("key") == "openinference.span.kind":
                    value = kv["value"].get("stringValue")
                    kinds[value] = kinds.get(value, 0) + 1
        assert kinds == {"AGENT": 3, "LLM": 3, "RETRIEVER": 2}
        # Provenance: a REAL recorded run, not a hand-written fixture.
        provenance = fixture.get("provenance") or {}
        assert provenance.get("provider") == "openinference"
        assert "openinference-instrumentation-openai" in provenance.get("sdk_version", "")

    def test_the_multi_agent_mapping_is_complete(self, mock_client) -> None:
        uploaded = capture_framework_trace(mock_client)
        fixture = load_recorded("openinference", "team")
        _ingest(mock_client, fixture)

        counts: dict = {}
        for event in uploaded["events"]:
            counts[event["event_type"]] = counts.get(event["event_type"], 0) + 1
        # 3 AGENT spans -> 3 input/output PAIRS; 3 LLM -> 3 model.invoke + 3 priced
        # cost.record; 2 RETRIEVER -> 2 retrieval.query. 14 events total, which is
        # exactly the event_count the committed live render sweep read back from the
        # server for this same workload.
        assert counts == {
            "agent.input": 3,
            "agent.output": 3,
            "model.invoke": 3,
            "cost.record": 3,
            "retrieval.query": 2,
        }
        assert sum(counts.values()) == 14

    def test_the_three_real_agents_are_named_honestly(self, mock_client) -> None:
        uploaded = capture_framework_trace(mock_client)
        _ingest(mock_client, load_recorded("openinference", "team"))
        agent_ids = sorted({e["payload"]["agent_id"] for e in find_events(uploaded["events"], "agent.input")})
        # The real span names from the recorded run — no fabricated or generic node.
        assert agent_ids == ["returns-specialist", "support-triage-supervisor", "warranty-specialist"]

    def test_the_agent_topology_really_is_a_supervisor_over_two_specialists(self, mock_client) -> None:
        # This is what makes the trace render as a 3-node/2-edge DAG rather than
        # three unrelated nodes: exactly one AGENT span has no captured parent, and
        # the other two descend from it.
        uploaded = capture_framework_trace(mock_client)
        _ingest(mock_client, load_recorded("openinference", "team"))
        inputs = find_events(uploaded["events"], "agent.input")
        by_span = {e["span_id"]: e for e in inputs}
        roots = [e for e in inputs if e.get("parent_span_id") not in by_span]
        assert len(roots) == 1, "a multi-agent DAG needs exactly one root agent"
        assert roots[0]["payload"]["agent_id"] == "support-triage-supervisor"
        children = sorted(e["payload"]["agent_id"] for e in inputs if e.get("parent_span_id") == roots[0]["span_id"])
        assert children == ["returns-specialist", "warranty-specialist"]

    def test_every_priced_call_carries_a_real_cost(self, mock_client) -> None:
        # LAY-3622 Cluster A over a REAL recorded multi-agent workload: three real
        # billed gpt-4o-mini calls, three real costs, no fabricated zero.
        uploaded = capture_framework_trace(mock_client)
        _ingest(mock_client, load_recorded("openinference", "team"))
        costs = [e["payload"] for e in find_events(uploaded["events"], "cost.record")]
        assert len(costs) == 3
        for cost in costs:
            assert cost["cost_usd"] > 0, f"a real billed call priced at {cost['cost_usd']}"
            assert "cost_status" not in cost
            assert cost["prompt_tokens"] > 0 and cost["completion_tokens"] > 0

    #: Real content strings from the recorded run — a genuine model answer and a
    #: genuine retriever query. Neither is a topology name, so neither may survive
    #: ``capture_content=False`` for any legitimate reason.
    REAL_CONTENT = (
        "qualifies as a manufacturing defect covered under",
        "30 day return window refund return shipping cost final sale",
        "split seam manufacturing defect warranty coverage",
    )

    def test_redaction_keeps_the_topology_and_drops_the_content(self, mock_client) -> None:
        uploaded = capture_framework_trace(mock_client)
        _ingest(mock_client, load_recorded("openinference", "team"), capture_content=False)
        events = uploaded["events"]
        # Topology and counts survive — redaction must not blind observability...
        assert len(find_events(events, "agent.input")) == 3
        assert sorted({e["payload"]["agent_id"] for e in find_events(events, "agent.input")}) == [
            "returns-specialist",
            "support-triage-supervisor",
            "warranty-specialist",
        ]
        assert all(e["payload"]["total_tokens"] > 0 for e in find_events(events, "model.invoke"))
        # ...and no real prompt / answer / query text does, anywhere in the trace.
        blob = json.dumps(events)
        for content in self.REAL_CONTENT:
            assert content not in blob, f"real captured content survived redaction: {content!r}"

    def test_the_redaction_sweep_can_actually_fail(self, mock_client) -> None:
        # VACUITY CONTROL for the sweep above: with capture_content=True the SAME
        # strings MUST be present, otherwise the sweep would pass on a fixture whose
        # content the mapping never carried in the first place.
        uploaded = capture_framework_trace(mock_client)
        _ingest(mock_client, load_recorded("openinference", "team"), capture_content=True)
        blob = json.dumps(uploaded["events"])
        for content in self.REAL_CONTENT:
            assert content in blob, f"the sweep is vacuous — {content!r} is never captured at all"
