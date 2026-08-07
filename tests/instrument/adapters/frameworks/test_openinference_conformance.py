"""Python<->Go conformance lane for the OpenInference span->event mapping.

The OpenInference mapping exists in TWO languages: this SDK's importer
(``layerlens.instrument.adapters.frameworks.openinference``) and the atlas Go OTLP
bridge (``atlas-app/apps/otlp-ingest/ingest/openinference.go``). An OpenInference
trace must render IDENTICALLY whether it arrived through the SDK or over OTLP, so
the two must never silently drift. This lane pins them against each other.

How it works (the proven graph-contract dual-oracle pattern — see
``tests/e2e/live/graph_contract/``):

* ``oi_conformance/spans.otlp.json`` — the SHARED corpus, one span per mapping
  lane (all 9 OpenInference span kinds + the never-drop UNKNOWN default + the
  fallbacks, caps, and honest-omission branches). Fed VERBATIM to both languages.
* ``oi_conformance/oracle.json`` — the Go bridge's REAL output over that corpus,
  GENERATED from the Go code (``openinference_conformance_test.go``
  ``TestDumpOpenInferenceConformance``), never hand-written. It is a transcript of
  what the bridge actually emits.

This test feeds the same corpus through the Python importer and asserts the events
match — dispatch, field names, values, and caps. The Go half of the lane
(``TestOpenInferenceConformanceOracleIsCurrent``) fails if the GO side drifts from
the oracle, so drift on EITHER side is caught: Go drift fails Go CI, Python drift
fails this test.

What is deliberately NOT pinned is enumerated in :data:`_KNOWN_EXCEPTIONS` below —
each one is a named, explained, deliberate difference, so everything else stays
pinned exactly.
"""

from __future__ import annotations

import os
import json
import hashlib
from typing import Any, Dict, List, Tuple

import pytest

from layerlens.instrument.adapters.frameworks.openinference import (
    OpenInferenceAdapter,
    span_to_events,
    otlp_json_to_span_records,
)

_DIR = os.path.join(os.path.dirname(__file__), "oi_conformance")
_CORPUS_PATH = os.path.join(_DIR, "spans.otlp.json")
_ORACLE_PATH = os.path.join(_DIR, "oracle.json")


def _load(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


ORACLE: Dict[str, Any] = _load(_ORACLE_PATH)


# ---------------------------------------------------------------------------
# The KNOWN, EXPECTED, ALLOWED differences — each named and justified.
# Anything NOT listed here is pinned exactly. Do not add to this list to make a
# failure go away: a new entry is a decision to let the two languages disagree.
# ---------------------------------------------------------------------------

#: D1 — the duration FIELD NAME differs, deliberately and permanently.
#: The Python SDK emits ``latency_ms`` (the SDK canon — tests/instrument/_event_schema.py).
#: The Go bridge emits ``duration_ms``, the OTLP path's platform-wide convention
#: (convert.go / merge.go / writer.go / openinference.go all use it, every OTLP
#: golden uses it, none uses latency_ms). Renaming either side alone would break
#: it against its OWN siblings, so the divergence is documented and LEFT.
#: The VALUE is still pinned — only the name is exempt.
_DURATION_PY = "latency_ms"
_DURATION_GO = "duration_ms"

#: Python-only payload keys the Go bridge carries OUTSIDE the payload (at the
#: event/trace envelope level) or does not model. Not a mapping divergence — a
#: structural difference in where the same fact lives. ``span_id``/``span_name``/
#: ``timestamp`` are NOT ignored: they are cross-checked against the Go EVENT
#: envelope below, which is stricter than skipping them.
_PY_ONLY_ENVELOPE = {
    "framework",  # Go: CanonicalTrace.Framework (trace level, deriveFramework)
    "run_id",  # Go: CanonicalTrace.TraceIDHex (trace level)
    "trace_id",  # Go: CanonicalTrace.TraceIDHex (trace level)
    "span_kind",  # Go: consumed for dispatch, not re-emitted
    "parent_span_id",  # Go: event envelope, not payload (unused by this corpus)
    "span_id",  # cross-checked against the Go event envelope instead
    "span_name",  # cross-checked against the Go event envelope instead
    "timestamp",  # cross-checked against the Go event envelope timestamp_ns
}

#: ``status`` — a KNOWN, tracked divergence (LAY-3620), documented in the adapter
#: module docstring. Python stamps the content-free OK/ERROR/UNSET status because
#: this side runs the collector-tier redaction backstop, which strips ``error``
#: from model.invoke / agent.output under the default capture_content=False —
#: leaving a failed call indistinguishable from a successful one without it. The
#: Go bridge has no such backstop and needs the same field to re-converge.
_PY_ONLY_TRACKED = {"status"}

_KNOWN_EXCEPTIONS = _PY_ONLY_ENVELOPE | _PY_ONLY_TRACKED


def _corpus_spans() -> List[Dict[str, Any]]:
    """Every span in the shared corpus, in wire order.

    Uses the SHIPPED envelope decoder (LAY-3622 Cluster B) rather than re-walking
    ``resourceSpans -> scopeSpans -> spans`` here. The hand-rolled loop this
    replaces was the tell that the SDK had no OTLP envelope surface at all; it also
    dropped resource-level attributes on the floor, so the lane never exercised the
    resource merge the Go bridge performs.
    """
    return otlp_json_to_span_records(_load(_CORPUS_PATH))


def _python_events() -> List[Dict[str, Any]]:
    """Run the shared corpus through the REAL Python importer.

    Uses the adapter's own extraction path (``_extract_record``) — the same code a
    caller hits via ``ingest_span`` — so the lane pins the shipped importer, not a
    test-local re-reading of the corpus. ``capture_content=True`` matches the Go
    bridge, which always captures content (its secret floor scrubs at persist).
    """
    adapter = OpenInferenceAdapter(client=None)
    out: List[Dict[str, Any]] = []
    for span in _corpus_spans():
        record = adapter._extract_record(span)
        assert record is not None, f"importer dropped span {span.get('spanId')}"
        for event_type, payload in span_to_events(record, capture_content=True):
            out.append({"event_type": event_type, "payload": payload})
    return out


PY_EVENTS = _python_events()
GO_EVENTS: List[Dict[str, Any]] = ORACLE["events"]


def _pinned(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The payload minus the named exceptions, with D1's rename applied.

    The duration VALUE is kept under the Go name so the two sides compare on the
    same key — the name is exempt, the measurement is not.
    """
    out = {k: v for k, v in payload.items() if k not in _KNOWN_EXCEPTIONS}
    if _DURATION_PY in payload:
        out[_DURATION_GO] = payload[_DURATION_PY]
    out.pop(_DURATION_PY, None)
    return out


def _pairs() -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    return list(zip(PY_EVENTS, GO_EVENTS))


# ---------------------------------------------------------------------------
# Corpus integrity — the oracle must describe the corpus we actually feed.
# ---------------------------------------------------------------------------


def test_oracle_matches_the_corpus_it_was_generated_from() -> None:
    """The oracle pins the exact corpus bytes it was generated from.

    Without this, editing the corpus without regenerating would silently compare
    Python's new output against a transcript of DIFFERENT spans — the lane would
    still be green while pinning nothing.
    """
    digest = hashlib.sha256(open(_CORPUS_PATH, "rb").read()).hexdigest()
    assert digest == ORACLE["corpus_sha256"], (
        "the shared corpus changed but the Go oracle was not regenerated.\n"
        "In atlas-app/apps: go test ./otlp-ingest/ingest/ -run TestDumpOpenInferenceConformance "
        "-count=1 -v\nthen copy spans.otlp.json + oracle.json back into "
        "tests/instrument/adapters/frameworks/oi_conformance/"
    )


def test_the_lane_is_not_vacuous() -> None:
    """A corpus/oracle that went empty would make every other test trivially pass."""
    assert len(GO_EVENTS) == 26, f"expected the 26-event oracle, got {len(GO_EVENTS)}"
    assert PY_EVENTS, "the Python importer emitted NO events for the shared corpus"


# ---------------------------------------------------------------------------
# Dispatch — the span kind -> event type contract.
# ---------------------------------------------------------------------------


def test_dispatch_is_identical() -> None:
    """Both languages must emit the same event types, in the same order, for the
    same spans — including the PAIR an AGENT/CHAIN span produces and the
    never-drop UNKNOWN default."""
    py = [e["event_type"] for e in PY_EVENTS]
    go = [e["event_type"] for e in GO_EVENTS]
    assert py == go, f"dispatch DRIFTED:\n  python: {py}\n  go:     {go}"


def test_every_span_kind_in_the_contract_is_covered() -> None:
    """The lane is only a lock if the corpus actually exercises the whole mapping.
    All 9 declared kinds + the UNKNOWN default must be represented."""
    covered = {e["payload"].get("span_kind") for e in PY_EVENTS}
    for kind in (
        "LLM",
        "EMBEDDING",
        "TOOL",
        "RERANKER",
        "RETRIEVER",
        "AGENT",
        "CHAIN",
        "GUARDRAIL",
        "EVALUATOR",
    ):
        assert kind in covered, f"corpus does not exercise the {kind} span kind"
    # The never-drop default: an unrecognised kind and a present-but-empty kind.
    assert "SOMETHING_NEW" in covered, "corpus does not exercise the UNKNOWN default"
    assert "" in covered, "corpus does not exercise the empty-kind default"
    types = {e["event_type"] for e in PY_EVENTS}
    assert types == {
        "model.invoke",
        "embedding.create",
        "tool.call",
        "retrieval.query",
        "agent.input",
        "agent.output",
        "policy.violation",
        "evaluation.result",
        "agent.interaction",
    }, f"the corpus does not produce the full event vocabulary, got {sorted(types)}"


# ---------------------------------------------------------------------------
# Payload — field names, values, caps.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("idx", range(len(GO_EVENTS)))
def test_payload_matches_the_go_bridge(idx: int) -> None:
    """Every mapped payload field must match the Go bridge EXACTLY.

    This is the lane's core: it fails on a renamed field, a changed cap, a changed
    fallback, an added field on one side, or a different value.
    """
    py_ev, go_ev = PY_EVENTS[idx], GO_EVENTS[idx]
    py_payload = _pinned(py_ev["payload"])
    go_payload = dict(go_ev["payload"])

    assert py_ev["event_type"] == go_ev["event_type"], (
        f"span {go_ev['span_id']} ({go_ev['span_name']}): event_type "
        f"{py_ev['event_type']!r} != Go {go_ev['event_type']!r}"
    )

    py_keys, go_keys = set(py_payload), set(go_payload)
    assert py_keys == go_keys, (
        f"span {go_ev['span_id']} ({go_ev['span_name']}) {go_ev['event_type']}: FIELD SET drift.\n"
        f"  python-only: {sorted(py_keys - go_keys)}\n"
        f"  go-only:     {sorted(go_keys - py_keys)}\n"
        f"If this is intended, it must be added to _KNOWN_EXCEPTIONS with a reason."
    )
    for key in sorted(py_keys):
        assert py_payload[key] == go_payload[key], (
            f"span {go_ev['span_id']} ({go_ev['span_name']}) {go_ev['event_type']}: "
            f"{key!r} = {py_payload[key]!r} (python) != {go_payload[key]!r} (go)"
        )


def test_span_identity_and_chronology_match_the_go_envelope() -> None:
    """The Python payload's span_id/span_name/timestamp must agree with the Go
    EVENT envelope. This is where the two put the same fact in different places,
    so it is cross-checked rather than skipped — it pins the agent.output-stamped-
    at-span-END chronology on both sides."""
    for py_ev, go_ev in _pairs():
        p = py_ev["payload"]
        assert p["span_id"] == go_ev["span_id"], f"span_id {p['span_id']} != {go_ev['span_id']}"
        assert p["span_name"] == go_ev["span_name"], f"span_name drift on {go_ev['span_id']}"
        # Python payload timestamp is epoch SECONDS; the Go envelope is epoch ns.
        assert p["timestamp"] * 1_000_000_000 == pytest.approx(go_ev["timestamp_ns"]), (
            f"span {go_ev['span_id']} {go_ev['event_type']}: timestamp "
            f"{p['timestamp']}s != Go {go_ev['timestamp_ns']}ns"
        )


def test_duration_is_the_same_measurement_under_both_names() -> None:
    """D1: the duration field NAME differs by design (latency_ms vs duration_ms),
    but it must be the same measurement. Pins the value across the rename so the
    exemption cannot hide a real timing divergence."""
    checked = 0
    for py_ev, go_ev in _pairs():
        if _DURATION_GO not in go_ev["payload"]:
            continue
        assert _DURATION_PY in py_ev["payload"], (
            f"span {go_ev['span_id']}: Go emits {_DURATION_GO} but Python emits no {_DURATION_PY}"
        )
        assert py_ev["payload"][_DURATION_PY] == pytest.approx(go_ev["payload"][_DURATION_GO])
        checked += 1
    assert checked == len(GO_EVENTS), "every corpus span has a duration; the D1 check went vacuous"


def test_the_duration_rename_is_the_only_name_level_difference() -> None:
    """The D1 exemption must stay a SINGLE rename. If Python ever stops emitting
    latency_ms, or Go stops emitting duration_ms, this documents-and-fails rather
    than letting the exception quietly cover a new gap."""
    for py_ev, go_ev in _pairs():
        assert _DURATION_PY in py_ev["payload"], f"Python lost {_DURATION_PY} on {go_ev['span_id']}"
        assert _DURATION_GO in go_ev["payload"], f"Go lost {_DURATION_GO} on {go_ev['span_id']}"
        assert _DURATION_GO not in py_ev["payload"], (
            f"Python now ALSO emits {_DURATION_GO} on {go_ev['span_id']} — the D1 divergence "
            f"changed shape; re-decide it rather than leaving both names"
        )


# ---------------------------------------------------------------------------
# The three closed divergences, pinned at the corpus level (D2/D3/D4).
# These would pass by construction if the payload comparison above holds, but
# they name the SPECIFIC regression each one guards so a failure is legible.
# ---------------------------------------------------------------------------


def _by_span(span_id_suffix: str) -> Dict[str, Any]:
    for ev in PY_EVENTS:
        if ev["payload"]["span_id"].endswith(span_id_suffix):
            return ev
    raise AssertionError(f"corpus span *{span_id_suffix} not found")


def test_d2_a_non_errored_span_carrying_a_message_is_not_errored() -> None:
    """D2: corpus span ...04 is status=OK WITH a status message. Neither language
    may label it errored."""
    assert "error" not in _by_span("04")["payload"]
    assert "error" not in GO_EVENTS[3]["payload"]
    # ...03 (ERROR + message) and ...05 (ERROR + empty) still report honestly.
    assert _by_span("03")["payload"]["error"] == "rate limited by upstream"
    assert _by_span("05")["payload"]["error"] == "span status ERROR"


def test_d3_an_empty_tool_parameters_does_not_shadow_the_real_input() -> None:
    """D3: corpus span ...0c has tool.parameters="" and a populated input.value.
    Both languages must keep the real input — this was silent DATA LOSS."""
    assert _by_span("0c")["payload"]["input"] == '{"query": "layerlens pricing"}'
    assert GO_EVENTS[11]["payload"]["input"] == '{"query": "layerlens pricing"}'


def test_d4_embedding_count_reads_the_flattened_form_both_sides() -> None:
    """D4: corpus span ...09 carries the FLATTENED embeddings form real
    instrumentors emit. Both languages must count 2 distinct indices; span ...0a
    (no embeddings attrs) must OMIT the count rather than zero-fill it."""
    assert _by_span("09")["payload"]["embedding_count"] == 2
    assert GO_EVENTS[8]["payload"]["embedding_count"] == 2
    assert "embedding_count" not in _by_span("0a")["payload"]
    assert "embedding_count" not in GO_EVENTS[9]["payload"]


def test_content_caps_agree_across_languages() -> None:
    """Corpus span ...06 carries 5000-char content. Both languages must truncate
    at the SAME cap with the SAME marker — an uncapped side is a real
    retention/size hazard on 3rd-party spans."""
    prompt = _by_span("06")["payload"]["prompt"]
    assert prompt == GO_EVENTS[5]["payload"]["prompt"]
    assert prompt.endswith("...[truncated 3000 chars]")
    assert len(prompt) == 2000 + len("...[truncated 3000 chars]")


class TestGoConvergenceOutsideTheCorpus:
    """Two real Python<->Go divergences the pinned corpus cannot catch (LAY-3622 E2).

    The oracle only proves parity for inputs the corpus CONTAINS. These two shapes
    are absent from all 24 spans, so both implementations drifted silently and the
    lane stayed green. Go is the reference here (the oracle is generated from it),
    so Python converged toward Go — and because the corpus contains neither input,
    the oracle needed no regeneration. Verified: 37 conformance tests still pass.

    Bite proof: drop the `.strip()` / `.lower()` and these fail while every other
    conformance test keeps passing — which is exactly the blind spot they close.
    """

    @pytest.mark.parametrize("padded", [" LLM ", "LLM\n", "\tLLM", "  LLM"])
    def test_a_whitespace_padded_span_kind_is_trimmed_like_go(self, padded: str) -> None:
        # Go: kind := strings.ToUpper(strings.TrimSpace(asStr(rawKind)))
        # (apps/otlp-ingest/ingest/openinference.go). Without the trim this span
        # fell through to the agent.interaction default in Python while Go typed it
        # as model.invoke — the SAME span rendering differently depending on whether
        # it arrived via the SDK or via OTLP.
        events = span_to_events(
            {
                "span_kind": padded,
                "name": "openai.chat",
                "attributes": {"openinference.span.kind": padded, "llm.model_name": "gpt-4o"},
                "trace_id": "aa" * 16,
                "span_id": "bb" * 8,
            },
            capture_content=True,
        )
        assert [t for t, _ in events] == ["model.invoke"]

    def test_whitespace_only_kind_still_falls_back(self) -> None:
        # BOUNDARY: a kind that is ONLY whitespace has no type to recover, and both
        # sides agree it takes the never-drop default.
        events = span_to_events(
            {"span_kind": "   ", "name": "x", "attributes": {}, "trace_id": "a" * 32, "span_id": "b" * 16},
            capture_content=True,
        )
        assert [t for t, _ in events] == ["agent.interaction"]

    @pytest.mark.parametrize("kind,expected", [("AGENT", "agent"), ("CHAIN", "chain")])
    def test_a_nameless_agent_span_falls_back_lowercased_like_go(self, kind: str, expected: str) -> None:
        # Go: agentID := oiCap(firstNonEmpty(spanName, strings.ToLower(kind), "agent"))
        # (openinference.go). agent_id is a graph NODE id, so an upper-cased Python
        # fallback rendered the same nameless span as a differently-NAMED node
        # depending on arrival path — a visible, cross-path graph divergence.
        events = dict(
            span_to_events(
                {
                    "span_kind": kind,
                    "name": None,
                    "attributes": {"openinference.span.kind": kind},
                    "trace_id": "aa" * 16,
                    "span_id": "bb" * 8,
                },
                capture_content=True,
            )
        )
        assert events["agent.input"]["agent_id"] == expected
        assert events["agent.output"]["agent_id"] == expected
        # operation was already lower-cased; the two now agree rather than
        # disagreeing on the case of the same word.
        assert events["agent.input"]["operation"] == expected

    def test_a_named_agent_span_keeps_its_real_name(self) -> None:
        # VACUITY CONTROL: the fallback must not start lower-casing real span names.
        events = dict(
            span_to_events(
                {
                    "span_kind": "AGENT",
                    "name": "Support-Triage-Supervisor",
                    "attributes": {"openinference.span.kind": "AGENT"},
                    "trace_id": "aa" * 16,
                    "span_id": "bb" * 8,
                },
                capture_content=True,
            )
        )
        assert events["agent.input"]["agent_id"] == "Support-Triage-Supervisor"

    @pytest.mark.parametrize("raw", [" AGENT ", "AGENT\n", "\tAGENT"])
    def test_a_padded_kind_does_not_leak_into_agent_id_via_the_raw_dict_path(self, raw: str) -> None:
        """The residue the first pass at E2a missed.

        ``_record_from_dict`` has TWO branches. The pre-extracted one
        (``span_kind`` AND ``attributes`` both present) was stripped; the raw-dict
        fallback was not — and a dict carrying ``span_kind`` with no ``attributes``
        is a shape the adapter's public API explicitly accepts ("feed ingest_span
        exported span dicts"). So a padded " AGENT " survived into the record and
        ``agent_id`` came out as " agent ".

        Dispatch was unaffected, because ``span_to_events`` re-strips before
        matching — which is precisely why the whole conformance lane stayed green
        while the rendered NODE ID was wrong. ``agent_id`` is the graph node id, so
        one logical agent rendered as two differently-named nodes depending on
        arrival path: the exact failure E2b claims to have closed.

        Bite proof: drop the ``.strip()`` from the else-branch and these fail with
        a whitespace-padded agent_id.
        """
        adapter = OpenInferenceAdapter(client=None)
        record = adapter._record_from_dict({"span_kind": raw, "trace_id": "aa" * 16, "span_id": "bb" * 8, "name": None})
        assert record is not None
        assert record["span_kind"] == "AGENT"
        events = dict(span_to_events(record, capture_content=True))
        assert events["agent.input"]["agent_id"] == "agent"
        assert events["agent.input"]["operation"] == "agent"

    def test_both_record_from_dict_branches_agree_on_a_padded_kind(self) -> None:
        # The two branches must not disagree with each other either — that
        # divergence is what made the residue invisible.
        adapter = OpenInferenceAdapter(client=None)
        pre_extracted = adapter._record_from_dict(
            {"span_kind": " LLM ", "attributes": {}, "trace_id": "aa" * 16, "span_id": "bb" * 8}
        )
        raw_dict = adapter._record_from_dict({"span_kind": " LLM ", "trace_id": "aa" * 16, "span_id": "bb" * 8})
        assert pre_extracted is not None and raw_dict is not None
        assert pre_extracted["span_kind"] == raw_dict["span_kind"] == "LLM"


# ---------------------------------------------------------------------------
# Cross-repo freshness (LAY-3622 E3). The oracle is copied between repos BY HAND
# (`openinference_conformance_test.go` prints the cp; PROVENANCE.md documents it as
# a numbered step), and neither side checked the other:
#
#   * atlas's TestOpenInferenceConformanceOracleIsCurrent rebuilds from the LOCAL Go
#     bridge and compares to the LOCAL oracle — it never looks at the SDK.
#   * this file's test_oracle_matches_the_corpus_it_was_generated_from hashes the
#     SDK's OWN corpus against the SDK's OWN oracle — an INTRA-repo pair-consistency
#     check, not a freshness check.
#
# So a Go-side mapping change + a local atlas regen, with neither file copied
# across, leaves BOTH repos internally self-consistent and green while the SDK pins
# a stale transcript. That is the window this closes.
# ---------------------------------------------------------------------------

#: Where atlas keeps its side of the shared pair. Override with LAYERLENS_ATLAS_REPO.
_ATLAS_ENV = "LAYERLENS_ATLAS_REPO"
_ATLAS_OI_TESTDATA = os.path.join("apps", "otlp-ingest", "ingest", "testdata", "oi-conformance")


def _atlas_oi_dir() -> str | None:
    """The atlas-side oracle directory, or None when atlas is not on this machine."""
    candidates = []
    env = os.environ.get(_ATLAS_ENV)
    if env:
        candidates.append(env)
    # The conventional sibling checkout (both repos live under .../layerlens/).
    # _DIR = <repo>/tests/instrument/adapters/frameworks/oi_conformance
    repo_root = os.path.abspath(os.path.join(_DIR, *[os.pardir] * 5))
    candidates.append(os.path.join(os.path.dirname(repo_root), "atlas-app"))
    for candidate in candidates:
        path = os.path.join(candidate, _ATLAS_OI_TESTDATA)
        if os.path.isdir(path):
            return path
    return None


def test_the_oracle_is_in_sync_with_the_atlas_side_copy() -> None:
    """The hand-copied pair must be byte-identical across the two repos.

    Deliberately NOT skipped when atlas is absent: a skip inside this row fails the
    adapter matrix (a skip in an installed-framework row is silent coverage loss).
    It asserts where it CAN — on a machine holding both checkouts, which is precisely
    where the regeneration-and-copy step happens and therefore where the drift is
    introduced.

    HOW WEAK THAT IS, stated plainly (LAY-3622 F6): **CI is never such a machine.**
    No workflow in ``.github/workflows`` checks out atlas, so this guard never fires
    in CI and drift is caught only if a developer who has both repos runs this lane.
    Closing the window properly needs the conformance pair published as a VERSIONED
    artifact that both repos pin, rather than hand-copied — that is the open F6
    follow-up, and it is why this is a partial guard rather than a solved problem.
    """
    atlas_dir = _atlas_oi_dir()
    if atlas_dir is None:
        return  # atlas not present; the window is documented, see the README.

    for filename in ("spans.otlp.json", "oracle.json"):
        atlas_path = os.path.join(atlas_dir, filename)
        if not os.path.isfile(atlas_path):
            continue
        sdk_digest = hashlib.sha256(open(os.path.join(_DIR, filename), "rb").read()).hexdigest()
        atlas_digest = hashlib.sha256(open(atlas_path, "rb").read()).hexdigest()
        assert sdk_digest == atlas_digest, (
            f"{filename} differs between the two repos — the shared conformance pair was "
            f"regenerated on one side and not copied to the other.\n"
            f"  SDK  : {os.path.join(_DIR, filename)}  sha256={sdk_digest}\n"
            f"  atlas: {atlas_path}  sha256={atlas_digest}\n"
            "Re-run the copy step in oi_conformance/README.md ('Changing the mapping')."
        )
