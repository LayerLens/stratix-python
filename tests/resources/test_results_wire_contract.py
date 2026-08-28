"""The `/results` path against the bytes the API actually sends (LAY-3765 / LAY-2772).

Three defects, all found by comparing the SDK's models to the production Go
structs rather than to the SDK's own fixtures. Every body here is either read
from the recorded corpus generated from those structs, or built and serialized
through ``httpx.Response(..., json=...)`` so the SDK parses real wire bytes.

That distinction is the reason these bugs survived. ``test_results.py`` fed a live
``datetime.timedelta`` object straight into ``Result(**data)`` and then asserted
``duration.total_seconds() == 2.5`` — a property true of the fixture itself and
false of every production response. A fixture that never crosses the wire cannot
detect a unit mismatch on the wire.
"""

from __future__ import annotations

import json
import asyncio
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import timedelta

import httpx
import pytest

from layerlens import Stratix
from layerlens.models import Result
from layerlens._exceptions import StratixError, APIResponseValidationError

CORPUS = Path(__file__).resolve().parents[1] / "contract" / "corpus"

BASE_URL = "https://api.test.invalid/api/v1"

_ORGANIZATION_BODY = {
    "data": {
        "id": "68a1f0c2d4e5b6a7c8d9e100",
        "name": "Contract Fixture Org",
        "projects": [{"id": "68a1f0c2d4e5b6a7c8d9e200", "name": "default"}],
    }
}


def _corpus(name: str) -> Any:
    return json.loads((CORPUS / name).read_text())


def _client(results_pages: List[Any]) -> Stratix:
    """A real Stratix serving `results_pages` in order, one per `/results` request.

    Pages are handed out sequentially so pagination walks are exercised exactly as
    a customer's ``get_all()`` would drive them.
    """
    remaining = list(results_pages)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/organizations"):
            return httpx.Response(200, json=_ORGANIZATION_BODY)
        if request.url.path.endswith("/results"):
            body = (
                remaining.pop(0) if remaining else {"evaluation_id": "e", "results": [], "metrics": {"total_count": 0}}
            )
            if isinstance(body, httpx.Response):
                return body
            return httpx.Response(200, json=body)
        return httpx.Response(404, json={"status": "error", "error": "unrouted in test"})

    transport = httpx.MockTransport(handle)

    class _MockTransportStratix(Stratix):
        def _init_transport(self, *args: Any, **kwargs: Any) -> httpx.BaseTransport:
            return transport

    return _MockTransportStratix(api_key="test-key", base_url=BASE_URL)


def _page(results: List[Dict[str, Any]], *, total_count: Optional[int] = None) -> Dict[str, Any]:
    return {
        "evaluation_id": "68a1f0c2d4e5b6a7c8d9e001",
        "results": results,
        "metrics": {
            "total_count": len(results) if total_count is None else total_count,
            "min_toxicity_score": 0.0,
            "max_toxicity_score": 0.1,
            "min_readability_score": 0.8,
            "max_readability_score": 0.9,
        },
    }


def _row(**overrides: Any) -> Dict[str, Any]:
    """A per-prompt row shaped as models.LLMResult marshals it."""
    row: Dict[str, Any] = {
        "subset": "mathematics",
        "prompt": "What is the derivative of x^2?",
        "result": "2x",
        "truth": "2x",
        "duration": 2_500_000_000,  # int64 NANOSECONDS, as time.Duration marshals
        "score": 1.0,
        "metrics": {"toxicity": 0.02, "readability": 0.81},
        "input_tokens": 512,
        "output_tokens": 128,
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------------- #
# duration: int64 nanoseconds on the wire, read as seconds by pydantic
# --------------------------------------------------------------------------- #


def test_duration_is_read_as_nanoseconds() -> None:
    """`Duration time.Duration` has no MarshalJSON, so it goes out as an int64
    nanosecond count. pydantic reads a bare int into `timedelta` as SECONDS, so
    2.5 seconds became roughly 79 years — with no exception raised anywhere.

    The atlas frontend divides this same field by 1e9, so the wire unit is
    unambiguous and the SDK is the side that is wrong.
    """
    client = _client([_page([_row(duration=2_500_000_000)])])

    response = client.results.get_by_id(evaluation_id="68a1f0c2d4e5b6a7c8d9e001")

    assert response is not None
    assert response.results[0].duration == timedelta(seconds=2.5)


def test_duration_from_recorded_corpus() -> None:
    """Same assertion against the body recorded from the real Go struct."""
    body = _corpus("results_get_flat_metrics.json")

    assert body["results"][0]["duration"] == 2_500_000_000, "corpus no longer carries a nanosecond duration"

    client = _client([body])
    response = client.results.get_by_id(evaluation_id="68a1f0c2d4e5b6a7c8d9e001")

    assert response is not None
    assert response.results[0].duration == timedelta(seconds=2.5)


@pytest.mark.parametrize(
    "nanoseconds,expected",
    [
        (0, timedelta(0)),
        (1_000, timedelta(microseconds=1)),
        (18_750_000_000, timedelta(seconds=18.75)),
        (3_600_000_000_000, timedelta(hours=1)),
    ],
)
def test_duration_conversion_across_magnitudes(nanoseconds: int, expected: timedelta) -> None:
    assert Result(**_row(duration=nanoseconds)).duration == expected


def test_duration_still_accepts_a_timedelta() -> None:
    """Constructing a Result in Python keeps working; only wire ints are rescaled."""
    assert Result(**_row(duration=timedelta(seconds=2.5))).duration == timedelta(seconds=2.5)


# --------------------------------------------------------------------------- #
# metrics: the worker writes a nested scorer object, not a flat float map
# --------------------------------------------------------------------------- #


def test_metrics_accepts_custom_scorer_objects() -> None:
    """Every evaluation run with custom scorers produces a `/results` page the SDK
    could not parse: the worker writes `{scorerID: {score, status, error}}`, while
    the model declared `Dict[str, Optional[float]]`.

    A failed scorer reports `score: null`, so the score itself must be optional.
    """
    body = _corpus("results_get_scorer_metrics.json")
    client = _client([body])

    response = client.results.get_by_id(evaluation_id="68a1f0c2d4e5b6a7c8d9e002")

    assert response is not None
    metrics = response.results[0].metrics
    assert metrics is not None

    succeeded = metrics["68a1f0c2d4e5b6a7c8d9e0f1"]
    failed = metrics["68a1f0c2d4e5b6a7c8d9e0f2"]

    assert succeeded.score == pytest.approx(0.8)
    assert succeeded.status == "success"

    assert failed.score is None, "a scorer that failed has no score; it must not read as 0"
    assert failed.status == "failed"
    assert failed.error == "scorer model timed out"


def test_metrics_still_accepts_the_flat_builtin_shape() -> None:
    """Built-in metrics stay plain floats — the union must not break them."""
    result = Result(**_row(metrics={"toxicity": 0.02, "readability": 0.81}))

    assert result.metrics == {"toxicity": 0.02, "readability": 0.81}


def test_metrics_accepts_null() -> None:
    """`Metrics json.RawMessage` carries no omitempty, so a nil value serializes as
    null rather than being omitted."""
    assert Result(**_row(metrics=None)).metrics is None


def test_metrics_accepts_a_missing_key() -> None:
    row = _row()
    del row["metrics"]

    assert Result(**row).metrics is None


# --------------------------------------------------------------------------- #
# silent truncation: a short list returned as if it were complete
# --------------------------------------------------------------------------- #


def test_get_all_raises_instead_of_truncating_on_a_bad_page() -> None:
    """The most serious of the three.

    `get_by_id` swallowed every parse failure (`except Exception: return None`) and
    `get_all_by_id` treated that None as END OF PAGES. A customer calling
    `get_all()` therefore received a short list with no error, no warning, and no
    way to tell it apart from a complete one — strictly worse than the crash they
    reported, because a crash is visible.

    Page 2 here is unparseable. The call must fail loudly rather than hand back
    page 1 as the whole answer.
    """
    good_page = _page([_row()], total_count=250)
    broken_page = {
        "evaluation_id": "68a1f0c2d4e5b6a7c8d9e001",
        "results": [_row(score="not-a-number")],
        "metrics": {"total_count": 250},
    }

    client = _client([good_page, broken_page])

    with pytest.raises(APIResponseValidationError) as caught:
        client.results.get_all_by_id(evaluation_id="68a1f0c2d4e5b6a7c8d9e001")

    message = str(caught.value)
    assert "/results" in message
    assert "score" in message, "the error must name the offending field path"


def test_get_by_id_raises_a_layerlens_error_on_a_bad_body() -> None:
    """The raised error must be catchable as a `layerlens` type and must carry the
    offending payload, so a customer can report the drift instead of guessing."""
    client = _client([{"evaluation_id": "e", "results": [_row(score="not-a-number")], "metrics": {"total_count": 1}}])

    with pytest.raises(StratixError) as caught:
        client.results.get_by_id(evaluation_id="68a1f0c2d4e5b6a7c8d9e001")

    assert isinstance(caught.value, APIResponseValidationError)
    assert caught.value.body is not None, "the error must carry the body that failed to parse"


def test_get_all_walks_every_page_and_returns_them_all() -> None:
    """The happy path still paginates to completion."""
    pages = [
        _page([_row(prompt=f"prompt-{i}") for i in range(100)], total_count=250),
        _page([_row(prompt=f"prompt-{100 + i}") for i in range(100)], total_count=250),
        _page([_row(prompt=f"prompt-{200 + i}") for i in range(50)], total_count=250),
    ]
    client = _client(pages)

    results = client.results.get_all_by_id(evaluation_id="68a1f0c2d4e5b6a7c8d9e001")

    assert len(results) == 250
    assert results[0].prompt == "prompt-0"
    assert results[-1].prompt == "prompt-249"


def test_get_all_raises_when_a_page_is_empty_before_the_last() -> None:
    """An empty page mid-walk means the server's total_count and its rows disagree.

    Breaking out of the loop here is what silently truncated the list. There is no
    honest way to return a complete list, so the call must say so.
    """
    pages = [
        _page([_row()], total_count=250),
        _page([], total_count=250),
    ]
    client = _client(pages)

    with pytest.raises(StratixError, match="page 2 of 3"):
        client.results.get_all_by_id(evaluation_id="68a1f0c2d4e5b6a7c8d9e001")


def test_empty_evaluation_returns_an_empty_list_not_an_error() -> None:
    """An evaluation with no results at all is not a failure."""
    client = _client([_page([], total_count=0)])

    assert client.results.get_all_by_id(evaluation_id="68a1f0c2d4e5b6a7c8d9e001") == []


def test_results_null_is_read_as_an_empty_page() -> None:
    """The deployed API sends `"results": null` for a page with no matching rows.

    `Results []LLMResult` carries no omitempty and the SQL repositories build it
    with `var results []LLMResult` + sqlx Select, which only appends — so zero rows
    leaves it nil and it marshals to null. atlas-app now emits `[]` (see
    models.NewEvaluationResult), but every deployed build still sends null, so the
    SDK has to read it as "no rows" rather than reject the page.
    """
    client = _client([{"evaluation_id": "68a1f0c2d4e5b6a7c8d9e001", "results": None, "metrics": {"total_count": 0}}])

    response = client.results.get_by_id(evaluation_id="68a1f0c2d4e5b6a7c8d9e001")

    assert response is not None
    assert response.results == []


def test_async_get_all_raises_instead_of_truncating() -> None:
    """The async twin shares the defect line for line; assert it directly rather
    than by code identity."""
    good_page = _page([_row()], total_count=250)
    broken_page = {
        "evaluation_id": "68a1f0c2d4e5b6a7c8d9e001",
        "results": [_row(score="not-a-number")],
        "metrics": {"total_count": 250},
    }

    remaining: List[Any] = [good_page, broken_page]

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/results"):
            return httpx.Response(200, json=remaining.pop(0))
        return httpx.Response(404, json={"status": "error", "error": "unrouted"})

    transport = httpx.MockTransport(handle)

    from layerlens import AsyncStratix
    from layerlens.models import Project, Organization

    class _MockTransportAsyncStratix(AsyncStratix):
        def _get_organization(self) -> Organization:
            return Organization(
                id="68a1f0c2d4e5b6a7c8d9e100",
                name="Contract Fixture Org",
                projects=[Project(id="68a1f0c2d4e5b6a7c8d9e200", name="default")],
            )

        def _init_transport(self, *args: Any, **kwargs: Any) -> httpx.AsyncBaseTransport:
            return transport

    client = _MockTransportAsyncStratix(api_key="test-key", base_url=BASE_URL)

    async def drive() -> None:
        await client.results.get_all_by_id(evaluation_id="68a1f0c2d4e5b6a7c8d9e001")

    with pytest.raises(APIResponseValidationError):
        asyncio.run(drive())
