"""LAY-3765 regression: the API sends `null` for scores it never computed.

A customer pinned to ``layerlens==1.6.0`` started failing on both the private and
the public evaluations endpoints without changing a line of their code, and worked
around it by bypassing SDK deserialization and parsing raw JSON.

Root cause, both halves ours:

* **API** — atlas-app ``f0a320f97`` (PR #2098, 2026-07-17) made
  ``readability_score``/``toxicity_score`` ``*float64`` with ``omitempty`` on the
  **bson tag only**, so a nil pointer marshals to a literal JSON ``null``.
* **SDK** — ``readability_score: float = 0.0``. A pydantic default covers a
  **missing** key and does nothing for an explicit ``null``. Byte-identical at
  v1.6.0 and at HEAD: the SDK was never in contract, it just was not exercised
  until the API started sending nulls.

These tests drive the **real client** over ``httpx.MockTransport``, so the
transport is the only seam: the genuine ``_request_cast`` → ``response.json()`` →
``cast_to(**data)`` / ``Evaluation(**e)`` chain runs. That matters because the
existing resource tests stub above the deserialization boundary — they hand a
bare ``Mock()`` pre-built ``Evaluation`` objects, so ``Evaluation(**e)``, the
exact line that raises, never runs against a raw JSON dict.

The bodies are not hand-written. They are read from the recorded corpus generated
by atlas-app's ``TestSDKCorpus`` from the production Go structs — the
``*_legacy_null_scores.json`` files record what the currently-deployed build
emits, which is what every pinned client is receiving right now.

Two different failure surfaces, so each path asserts the right thing:

============================  ==========================================
call                          raises before the fix
============================  ==========================================
``get_many`` (both clients)   raw ``pydantic_core.ValidationError``
``get_by_id`` (both clients)  ``APIResponseValidationError`` (wrapped by
                              ``_base_client`` at the ``cast_to`` site)
============================  ==========================================
"""

from __future__ import annotations

import json
import asyncio
from typing import Any, Dict, Callable
from pathlib import Path

import httpx
import pytest

from layerlens import Stratix, AsyncStratix
from layerlens.models import Evaluation
from layerlens._public_client import PublicClient, AsyncPublicClient

CORPUS = Path(__file__).resolve().parents[1] / "contract" / "corpus"

BASE_URL = "https://api.test.invalid/api/v1"

_ORGANIZATION_BODY = {
    "data": {
        "id": "68a1f0c2d4e5b6a7c8d9e100",
        "name": "Contract Fixture Org",
        "owner_id": "contract-fixture@layerlens.ai",
        "projects": [{"id": "68a1f0c2d4e5b6a7c8d9e200", "name": "default"}],
    }
}


def _corpus(name: str) -> Any:
    """Load a recorded body. Parsed from the committed bytes, never hand-built."""
    return json.loads((CORPUS / name).read_text())


def _handler(routes: Dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    """Serve `routes` keyed by URL path, plus the org bootstrap every client needs.

    Bodies go out through ``json.dumps`` so the SDK parses real wire bytes rather
    than a Python dict that happens to look similar.
    """

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        for suffix, body in routes.items():
            if path.endswith(suffix):
                return httpx.Response(
                    200,
                    content=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
        if path.endswith("/organizations"):
            return httpx.Response(
                200,
                content=json.dumps(_ORGANIZATION_BODY).encode(),
                headers={"Content-Type": "application/json"},
            )
        return httpx.Response(404, content=b'{"status":"error","error":"unrouted in test"}')

    return handle


def _sync_client(routes: Dict[str, Any]) -> Stratix:
    """A real Stratix whose only difference is the transport.

    ``_init_transport`` is httpx's own hook and is called from
    ``httpx.Client.__init__``, so the org bootstrap in ``Stratix.__init__`` is
    served by the mock too — nothing is patched.
    """
    transport = httpx.MockTransport(_handler(routes))

    class _MockTransportStratix(Stratix):
        def _init_transport(self, *args: Any, **kwargs: Any) -> httpx.BaseTransport:
            return transport

    return _MockTransportStratix(api_key="test-key", base_url=BASE_URL)


def _async_client(routes: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> AsyncStratix:
    """A real AsyncStratix over the mock transport.

    ``AsyncStratix._get_organization`` opens its own *synchronous* ``httpx.Client``
    rather than using the async client's transport, so it cannot be intercepted by
    ``_init_transport``. It is stubbed here because it is not on the path under
    test; every call the tests actually make goes through the mock transport.
    """
    transport = httpx.MockTransport(_handler(routes))

    from layerlens.models import Project, Organization

    monkeypatch.setattr(
        AsyncStratix,
        "_get_organization",
        lambda self: Organization(
            id="68a1f0c2d4e5b6a7c8d9e100",
            name="Contract Fixture Org",
            projects=[Project(id="68a1f0c2d4e5b6a7c8d9e200", name="default")],
        ),
    )

    class _MockTransportAsyncStratix(AsyncStratix):
        def _init_transport(self, *args: Any, **kwargs: Any) -> httpx.AsyncBaseTransport:
            return transport

    return _MockTransportAsyncStratix(api_key="test-key", base_url=BASE_URL)


def _public_client(routes: Dict[str, Any]) -> PublicClient:
    transport = httpx.MockTransport(_handler(routes))

    class _MockTransportPublicClient(PublicClient):
        def _init_transport(self, *args: Any, **kwargs: Any) -> httpx.BaseTransport:
            return transport

    return _MockTransportPublicClient(api_key="test-key", base_url=BASE_URL)


def _async_public_client(routes: Dict[str, Any]) -> AsyncPublicClient:
    transport = httpx.MockTransport(_handler(routes))

    class _MockTransportAsyncPublicClient(AsyncPublicClient):
        def _init_transport(self, *args: Any, **kwargs: Any) -> httpx.AsyncBaseTransport:
            return transport

    return _MockTransportAsyncPublicClient(api_key="test-key", base_url=BASE_URL)


def _assert_not_computed(evaluation: Evaluation) -> None:
    """The uncomputed metrics must read as None, and the row must survive intact.

    ``None``, not ``0.0``: a zero readability score is a fabricated measurement,
    and distinguishing "not computed" from a real zero is the entire reason the
    API made these fields nullable. The surrounding assertions prove the row was
    not silently blanked to defaults on the way through.
    """
    assert evaluation.readability_score is None
    assert evaluation.toxicity_score is None

    assert evaluation.id == "68a1f0c2d4e5b6a7c8d9e001"
    assert evaluation.model_name == "GPT-4o"
    assert evaluation.benchmark_name == "MMLU"
    assert evaluation.accuracy == 0.87
    assert evaluation.failed_prompt_count == 2


# --------------------------------------------------------------------------- #
# private client (API key) — list and get, sync and async
# --------------------------------------------------------------------------- #


def test_private_get_many_accepts_null_scores() -> None:
    client = _sync_client({"/evaluations": _corpus("evaluations_get_many_legacy_null_scores.json")})

    response = client.evaluations.get_many()

    assert response is not None
    assert len(response.evaluations) == 1
    _assert_not_computed(response.evaluations[0])


def test_private_get_by_id_accepts_null_scores() -> None:
    client = _sync_client(
        {"/evaluations/68a1f0c2d4e5b6a7c8d9e001": _corpus("evaluations_get_one_legacy_null_scores.json")}
    )

    evaluation = client.evaluations.get_by_id("68a1f0c2d4e5b6a7c8d9e001")

    assert evaluation is not None
    _assert_not_computed(evaluation)


def test_private_get_many_accepts_null_scores_async(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _async_client(
        {"/evaluations": _corpus("evaluations_get_many_legacy_null_scores.json")},
        monkeypatch,
    )

    response = asyncio.run(client.evaluations.get_many())

    assert response is not None
    assert len(response.evaluations) == 1
    _assert_not_computed(response.evaluations[0])


def test_private_get_by_id_accepts_null_scores_async(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _async_client(
        {"/evaluations/68a1f0c2d4e5b6a7c8d9e001": _corpus("evaluations_get_one_legacy_null_scores.json")},
        monkeypatch,
    )

    evaluation = asyncio.run(client.evaluations.get_by_id("68a1f0c2d4e5b6a7c8d9e001"))

    assert evaluation is not None
    _assert_not_computed(evaluation)


# --------------------------------------------------------------------------- #
# public client — list and get, sync and async
# --------------------------------------------------------------------------- #


def test_public_get_many_accepts_null_scores() -> None:
    client = _public_client({"/evaluations": _corpus("evaluations_get_many_legacy_null_scores.json")})

    response = client.evaluations.get_many()

    assert response is not None
    assert len(response.evaluations) == 1
    _assert_not_computed(response.evaluations[0])


def test_public_get_by_id_accepts_null_scores() -> None:
    client = _public_client(
        {"/evaluations/68a1f0c2d4e5b6a7c8d9e001": _corpus("evaluations_get_one_legacy_null_scores.json")}
    )

    evaluation = client.evaluations.get_by_id("68a1f0c2d4e5b6a7c8d9e001")

    assert evaluation is not None
    _assert_not_computed(evaluation)


def test_public_get_many_accepts_null_scores_async() -> None:
    client = _async_public_client({"/evaluations": _corpus("evaluations_get_many_legacy_null_scores.json")})

    response = asyncio.run(client.evaluations.get_many())

    assert response is not None
    assert len(response.evaluations) == 1
    _assert_not_computed(response.evaluations[0])


def test_public_get_by_id_accepts_null_scores_async() -> None:
    client = _async_public_client(
        {"/evaluations/68a1f0c2d4e5b6a7c8d9e001": _corpus("evaluations_get_one_legacy_null_scores.json")}
    )

    evaluation = asyncio.run(client.evaluations.get_by_id("68a1f0c2d4e5b6a7c8d9e001"))

    assert evaluation is not None
    _assert_not_computed(evaluation)


# --------------------------------------------------------------------------- #
# the polling path, and the shape of the fix
# --------------------------------------------------------------------------- #


def test_wait_for_completion_copies_null_scores() -> None:
    """``wait_for_completion`` copies both fields onto the caller's Evaluation.

    A long-running wait therefore died at the same point as a plain read, one poll
    after the evaluation finished.
    """
    body = _corpus("evaluations_get_one_legacy_null_scores.json")
    client = _sync_client({"/evaluations/68a1f0c2d4e5b6a7c8d9e001": body})

    pending = Evaluation(
        id="68a1f0c2d4e5b6a7c8d9e001",
        status="in-progress",
        submitted_at=1752710400,
        finished_at=0,
        model_id="68a1f0c2d4e5b6a7c8d9e010",
        dataset_id="68a1f0c2d4e5b6a7c8d9e020",
        average_duration=0,
        accuracy=0.0,
    ).attach_client(client)

    finished = pending.wait_for_completion(interval_seconds=0)

    assert finished is not None
    assert finished.readability_score is None
    assert finished.toxicity_score is None


def test_explicit_null_is_accepted_for_every_score_field() -> None:
    """All three score fields must accept an explicit null.

    ``ethics_score`` is not broken today — the Go side is still a non-pointer
    ``float64``, which ``encoding/json`` cannot emit as null. It is widened as
    prophylaxis: same shape, adjacent line, same metric family, one server change
    from the same fate. Excluding it would only mean a second ticket.
    """
    body = _corpus("evaluations_get_one_legacy_null_scores.json")
    body["ethics_score"] = None

    evaluation = Evaluation(**json.loads(json.dumps(body)))

    assert evaluation.readability_score is None
    assert evaluation.toxicity_score is None
    assert evaluation.ethics_score is None


def test_absent_score_keys_read_as_none_not_zero() -> None:
    """With the fix deployed the API omits the keys instead of sending null.

    Both must land on ``None``. Reading an omitted key as ``0.0`` would report a
    perfect toxicity score for every evaluation that was never scored.
    """
    body = _corpus("evaluations_get_one_not_computed.json")

    assert "readability_score" not in body, "corpus fixture no longer covers the omitted-key case"
    assert "toxicity_score" not in body

    evaluation = Evaluation(**body)

    assert evaluation.readability_score is None
    assert evaluation.toxicity_score is None


def test_computed_zero_is_preserved_not_conflated_with_missing() -> None:
    """A real 0.0 must stay 0.0 and stay distinguishable from "not computed"."""
    body = _corpus("evaluations_get_one_computed.json")

    assert body["readability_score"] == 0, "corpus fixture no longer covers the genuine-zero case"

    evaluation = Evaluation(**body)

    assert evaluation.readability_score == 0.0
    assert evaluation.readability_score is not None
    assert evaluation.toxicity_score == 0.014
