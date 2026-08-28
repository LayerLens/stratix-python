"""Parse the recorded API responses with the real models — the SDK half of the gate.

``corpus/`` holds response bodies recorded from atlas-app's production Go structs
by that repo's ``TestSDKCorpus`` (see ``DOCS/api-contract/README.md`` there). This
module parses them with the SDK's own pydantic models, so a wire-shape change on
the API side fails here instead of at a customer.

Two passes, and the second is the one that matters:

1. **As recorded** — every body must parse. Catches renames, removals and type
   narrowing.
2. **With every null-capable key forced to null** — each key that the API could
   plausibly send as null is set to null, one at a time, and the body must still
   parse. This is the pass that would have caught LAY-3765 nine months before the
   customer did: the corpus recorded on 2026-07-16 would have parsed fine, because
   the scores were populated in it.

Why forcing nulls is the right shape for this test rather than just recording more
fixtures: a fixture only covers the states someone thought to record. The
producing side is Go, where "nullable" is a property of the *type* — a pointer
field without ``omitempty`` can emit null for any value the pipeline fails to
compute, on any deployment, at any time. Enumerating the keys and forcing each one
covers the whole space rather than the sampled part of it.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple
from pathlib import Path

import pytest

from layerlens.models import Evaluation, ResultsResponse

CORPUS = Path(__file__).parent / "corpus"

# Every key the API can send as null on a body the SDK reads, as established by
# atlas-app's TestEvaluationJSONNullableKeyAllowlist / the Go struct tags:
#
#   * `summary` is `interface{}` — nullable since forever.
#   * `readability_score` / `toxicity_score` are `*float64`. Deployed builds emit
#     null (LAY-3765); with the fix they omit the key. Both must parse.
#   * `ethics_score` is a non-pointer float64 today and CANNOT be null. Listed
#     anyway, as prophylaxis: it is the same shape on the adjacent line in the
#     same metric family, and one server-side change from the same fate.
#   * The token aggregates are `*int64`/`*float64` WITH omitempty, so they are
#     omitted rather than nulled — but a future tag edit would flip that.
_EVALUATION_NULLABLE_KEYS = (
    "summary",
    "readability_score",
    "toxicity_score",
    "ethics_score",
    "total_input_tokens",
    "total_output_tokens",
    "avg_input_tokens_per_prompt",
    "avg_output_tokens_per_prompt",
)

# `metrics` is `json.RawMessage` with no omitempty, so a nil value emits null.
# `results` is `[]LLMResult` with no omitempty; the SQL repositories leave the
# slice nil when nothing matches, so it emits null on any empty page.
_RESULT_ROW_NULLABLE_KEYS = ("metrics",)


def _evaluation_bodies() -> List[Tuple[str, Dict[str, Any]]]:
    """(name, evaluation-dict) for every recorded evaluation, list rows included."""
    bodies: List[Tuple[str, Dict[str, Any]]] = []

    for path in sorted(CORPUS.glob("evaluations_*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "evaluations" in payload:
            for index, row in enumerate(payload["evaluations"]):
                bodies.append((f"{path.name}[{index}]", row))
        else:
            bodies.append((path.name, payload))

    return bodies


def _results_bodies() -> List[Tuple[str, Dict[str, Any]]]:
    return [(path.name, json.loads(path.read_text())) for path in sorted(CORPUS.glob("results_*.json"))]


EVALUATION_BODIES = _evaluation_bodies()
RESULTS_BODIES = _results_bodies()


def _with_pagination(body: Dict[str, Any]) -> Dict[str, Any]:
    """Add the client-derived pagination the resource layer supplies."""
    metrics = body.get("metrics") or {}
    total_count = metrics.get("total_count", 0) if isinstance(metrics, dict) else 0

    return {
        **body,
        "results": body.get("results") or [],
        "pagination": {"page": 1, "page_size": 100, "total_pages": 1, "total_count": total_count},
    }


def test_corpus_is_non_empty() -> None:
    """Guards against a vacuous pass. Every assertion below is parametrized over
    the corpus, so an empty corpus would report success while testing nothing."""
    assert EVALUATION_BODIES, f"no evaluation bodies found under {CORPUS}"
    assert RESULTS_BODIES, f"no results bodies found under {CORPUS}"


def test_corpus_covers_the_not_computed_case() -> None:
    """The fixture design is the load-bearing part of this gate.

    A corpus of fully-populated happy-path evaluations parses cleanly and proves
    nothing. At least one recorded evaluation must have uncomputed scores — either
    absent (post-fix) or explicitly null (as deployed builds send them).
    """
    not_computed = [
        name
        for name, body in EVALUATION_BODIES
        if body.get("readability_score") is None and body.get("toxicity_score") is None
    ]

    assert not_computed, (
        "no recorded evaluation has uncomputed readability/toxicity scores; "
        "the corpus no longer covers LAY-3765 and this gate is disarmed"
    )


def test_corpus_covers_both_the_null_and_the_omitted_encoding() -> None:
    """Deployed builds send null; builds carrying the fix omit the key. A client
    has to read both, so both must be recorded."""
    has_explicit_null = any(
        "readability_score" in body and body["readability_score"] is None for _, body in EVALUATION_BODIES
    )
    has_omitted = any("readability_score" not in body for _, body in EVALUATION_BODIES)

    assert has_explicit_null, "corpus lost the `readability_score: null` recording (what deployed builds send)"
    assert has_omitted, "corpus lost the omitted-key recording (what the fixed build sends)"


@pytest.mark.parametrize("name,body", EVALUATION_BODIES, ids=[name for name, _ in EVALUATION_BODIES])
def test_recorded_evaluation_parses(name: str, body: Dict[str, Any]) -> None:
    """Pass 1: every recorded evaluation body parses as-is."""
    evaluation = Evaluation(**body)

    assert evaluation.id == body["id"]
    assert evaluation.status.value == body["status"]


@pytest.mark.parametrize("name,body", EVALUATION_BODIES, ids=[name for name, _ in EVALUATION_BODIES])
@pytest.mark.parametrize("key", _EVALUATION_NULLABLE_KEYS)
def test_recorded_evaluation_parses_with_key_forced_to_null(name: str, body: Dict[str, Any], key: str) -> None:
    """Pass 2: forcing any null-capable key to null must not break the body.

    This is the pass that would have caught LAY-3765.
    """
    forced = copy.deepcopy(body)
    forced[key] = None

    # Round-trip through JSON so the model sees wire-shaped input, not Python objects.
    Evaluation(**json.loads(json.dumps(forced)))


@pytest.mark.parametrize("name,body", EVALUATION_BODIES, ids=[name for name, _ in EVALUATION_BODIES])
@pytest.mark.parametrize("key", _EVALUATION_NULLABLE_KEYS)
def test_recorded_evaluation_parses_with_key_removed(name: str, body: Dict[str, Any], key: str) -> None:
    """A key the API omits must be as readable as one it nulls.

    `omitempty` on a nullable field turns null into absence, so the same field
    reaches clients both ways depending on which build they are talking to.
    """
    reduced = copy.deepcopy(body)
    reduced.pop(key, None)

    Evaluation(**json.loads(json.dumps(reduced)))


@pytest.mark.parametrize("name,body", RESULTS_BODIES, ids=[name for name, _ in RESULTS_BODIES])
def test_recorded_results_page_parses(name: str, body: Dict[str, Any]) -> None:
    """Pass 1 for `/results`, including the custom-scorer metrics shape."""
    response = ResultsResponse.model_validate(_with_pagination(body))

    assert response.evaluation_id == body["evaluation_id"]
    assert len(response.results) == len(body.get("results") or [])


@pytest.mark.parametrize("name,body", RESULTS_BODIES, ids=[name for name, _ in RESULTS_BODIES])
@pytest.mark.parametrize("key", _RESULT_ROW_NULLABLE_KEYS)
def test_recorded_results_page_parses_with_row_key_forced_to_null(name: str, body: Dict[str, Any], key: str) -> None:
    """Pass 2 for `/results`, applied to every row."""
    forced = copy.deepcopy(body)
    for row in forced.get("results") or []:
        row[key] = None

    ResultsResponse.model_validate(_with_pagination(json.loads(json.dumps(forced))))


@pytest.mark.parametrize("name,body", RESULTS_BODIES, ids=[name for name, _ in RESULTS_BODIES])
def test_recorded_results_page_parses_with_null_results_list(name: str, body: Dict[str, Any]) -> None:
    """`"results": null` is what an empty page looks like on deployed builds."""
    forced = copy.deepcopy(body)
    forced["results"] = None

    response = ResultsResponse.model_validate(_with_pagination(forced))

    assert response.results == []


def test_recorded_results_durations_are_read_as_nanoseconds() -> None:
    """The corpus carries `duration` as the int64 nanosecond count the API sends;
    pin the conversion so a regression to seconds is caught here too."""
    from datetime import timedelta

    body = json.loads((CORPUS / "results_get_flat_metrics.json").read_text())
    wire_nanoseconds = body["results"][0]["duration"]

    assert wire_nanoseconds == 2_500_000_000, "corpus no longer carries a nanosecond duration"

    response = ResultsResponse.model_validate(_with_pagination(body))

    assert response.results[0].duration == timedelta(seconds=2.5)
    assert response.results[0].duration != timedelta(seconds=wire_nanoseconds)


def test_recorded_scorer_metrics_expose_a_failed_scorer_as_none() -> None:
    """A custom scorer that failed reports `score: null`; it must not read as 0."""
    body = json.loads((CORPUS / "results_get_scorer_metrics.json").read_text())

    response = ResultsResponse.model_validate(_with_pagination(body))
    metrics = response.results[0].metrics

    assert metrics is not None
    failed = [outcome for outcome in metrics.values() if getattr(outcome, "status", None) == "failed"]

    assert failed, "corpus no longer covers a failed custom scorer"
    assert all(outcome.score is None for outcome in failed)
