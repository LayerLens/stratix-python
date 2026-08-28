"""One unparseable row must not silently discard the page (LAY-2772).

``get_many`` built its rows with a bare list comprehension outside any try, so a
single bad row raised a raw ``pydantic_core.ValidationError`` — not a ``layerlens``
exception type, and with no indication of *which* row of up to 500 was at fault.

The tempting fix is to move that line inside the ``try`` seventeen lines below it.
Do not: ``pydantic.ValidationError`` is a subclass of ``ValueError``, so the
existing ``except (ValueError, KeyError): return None`` would swallow it and hand
back an empty page indistinguishable from "this project has no evaluations".
"""

from __future__ import annotations

import json
from typing import Any, Dict, Callable
from pathlib import Path

import httpx
import pytest

from layerlens import Stratix
from layerlens._exceptions import StratixError, APIResponseValidationError
from layerlens._public_client import PublicClient

CORPUS = Path(__file__).resolve().parents[1] / "contract" / "corpus"

BASE_URL = "https://api.test.invalid/api/v1"

_ORGANIZATION_BODY = {
    "data": {
        "id": "68a1f0c2d4e5b6a7c8d9e100",
        "name": "Contract Fixture Org",
        "projects": [{"id": "68a1f0c2d4e5b6a7c8d9e200", "name": "default"}],
    }
}


def _handler(evaluations_body: Any) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/organizations"):
            return httpx.Response(200, json=_ORGANIZATION_BODY)
        if request.url.path.endswith("/evaluations"):
            if isinstance(evaluations_body, httpx.Response):
                return evaluations_body
            return httpx.Response(200, json=evaluations_body)
        return httpx.Response(404, json={"status": "error", "error": "unrouted in test"})

    return handle


def _client(evaluations_body: Any) -> Stratix:
    transport = httpx.MockTransport(_handler(evaluations_body))

    class _MockTransportStratix(Stratix):
        def _init_transport(self, *args: Any, **kwargs: Any) -> httpx.BaseTransport:
            return transport

    return _MockTransportStratix(api_key="test-key", base_url=BASE_URL)


def _public(evaluations_body: Any) -> PublicClient:
    transport = httpx.MockTransport(_handler(evaluations_body))

    class _MockTransportPublicClient(PublicClient):
        def _init_transport(self, *args: Any, **kwargs: Any) -> httpx.BaseTransport:
            return transport

    return _MockTransportPublicClient(api_key="test-key", base_url=BASE_URL)


def _page_with_one_bad_row() -> Dict[str, Any]:
    """A page whose second row has an unparseable status.

    An unenumerated status is a realistic drift: the SDK's EvaluationStatus is a
    closed enum, so a value the server adds tomorrow fails the whole row.
    """
    good = json.loads((CORPUS / "evaluations_get_one_not_computed.json").read_text())
    bad = dict(good, id="68a1f0c2d4e5b6a7c8d9e003", status="quiesced")

    return {"evaluations": [good, bad], "count": 2, "total_count": 2}


def test_bad_row_raises_a_layerlens_error_naming_the_row_and_field() -> None:
    with pytest.raises(APIResponseValidationError) as caught:
        _client(_page_with_one_bad_row()).evaluations.get_many()

    message = str(caught.value)
    assert "evaluations[1]" in message, "the error must name which row failed"
    assert "status" in message, "the error must name the offending field"
    assert isinstance(caught.value, StratixError), "must be catchable as a layerlens error"


def test_bad_row_does_not_return_an_empty_page() -> None:
    """The failure mode this replaces: `return None`, which a caller reads as
    "no evaluations exist" and cannot tell apart from a genuinely empty project.

    Written as try/fail rather than `pytest.raises` so the *returned* value is
    named in the failure message if the loud path regresses to a quiet one.
    """
    client = _client(_page_with_one_bad_row())

    try:
        response = client.evaluations.get_many()
    except APIResponseValidationError:
        return

    pytest.fail(
        f"get_many() returned {response!r} instead of raising. A None or short page is "
        "indistinguishable from 'this project has no evaluations'."
    )


def test_public_bad_row_raises_too() -> None:
    with pytest.raises(APIResponseValidationError) as caught:
        _public(_page_with_one_bad_row()).evaluations.get_many()

    assert "evaluations[1]" in str(caught.value)


def test_body_that_is_not_an_object_raises() -> None:
    with pytest.raises(APIResponseValidationError, match="not a JSON object"):
        _client(["not", "an", "object"]).evaluations.get_many()


def test_empty_project_returns_an_empty_page_not_an_error() -> None:
    """A project with no evaluations is not a failure, and must stay
    distinguishable from the error cases above."""
    response = _client({"evaluations": [], "count": 0, "total_count": 0}).evaluations.get_many()

    assert response is not None
    assert response.evaluations == []
    assert response.pagination.total_count == 0


def test_null_evaluations_list_reads_as_empty() -> None:
    """Defensive: a nil Go slice with no omitempty would serialize as null. The
    handler initialises with make() so this should not occur, but reading it as
    "no rows" is cheaper than a customer outage if that ever changes."""
    response = _client({"evaluations": None, "count": 0, "total_count": 0}).evaluations.get_many()

    assert response is not None
    assert response.evaluations == []


def test_every_good_row_survives_and_is_client_attached() -> None:
    """The page still parses fully, and rows keep the client needed by
    `evaluation.get_results()`."""
    good = json.loads((CORPUS / "evaluations_get_one_not_computed.json").read_text())
    body = {"evaluations": [good, dict(good, id="68a1f0c2d4e5b6a7c8d9e004")], "count": 2, "total_count": 2}

    client = _client(body)
    response = client.evaluations.get_many()

    assert response is not None
    assert [e.id for e in response.evaluations] == [
        "68a1f0c2d4e5b6a7c8d9e001",
        "68a1f0c2d4e5b6a7c8d9e004",
    ]
    assert all(e._client is client for e in response.evaluations)
