"""Prod-readiness regression: replay the recorded error corpus (LAY-3644).

These run in CI with NO live calls — they replay the real HTTP shapes captured
2026-06-26 (provider 4xx + atlas error responses + the SDK error-surface
invariants) so the production-readiness findings can't silently regress:

* every recorded atlas error status maps to the correct SDK taxonomy exception;
* the recorded ``*_sdk_surface`` results still encode the no-secret-leak
  invariants (a future fixture edit that allows a leak fails here);
* the manifest references files that exist.

The live probes (BOLA -> 401, delete -> GET 404, int64 exact, garbage -> 4xx)
remain live-only and are locked server-side by the atlas-app Go tests; this is
the corpus half of the regression (CI, ~$0).
"""

from __future__ import annotations

import os
import glob
import json
from http import HTTPStatus

import httpx
import pytest

from layerlens._client import Stratix
from layerlens._exceptions import (
    ConflictError,
    NotFoundError,
    APIStatusError,
    RateLimitError,
    BadRequestError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    UnprocessableEntityError,
)

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "prodready_errors")

_STATUS_EXC = {
    HTTPStatus.BAD_REQUEST: BadRequestError,
    HTTPStatus.UNAUTHORIZED: AuthenticationError,
    HTTPStatus.FORBIDDEN: PermissionDeniedError,
    HTTPStatus.NOT_FOUND: NotFoundError,
    HTTPStatus.CONFLICT: ConflictError,
    HTTPStatus.UNPROCESSABLE_ENTITY: UnprocessableEntityError,
    HTTPStatus.TOO_MANY_REQUESTS: RateLimitError,
}


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _atlas_error_fixtures():
    out = []
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "atlas_*.json"))):
        d = _load(path)
        status = d.get("status")
        if isinstance(status, int) and status >= 400:
            out.append((os.path.basename(path), d))
    return out


def _sdk_surface_fixtures():
    return [(os.path.basename(p), _load(p)) for p in sorted(glob.glob(os.path.join(CORPUS_DIR, "*_sdk_surface.json")))]


def _expected_exc(status: int):
    if status in _STATUS_EXC:
        return _STATUS_EXC[status]
    if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return InternalServerError
    return APIStatusError  # e.g. 413/415 fall through to the base status error


def test_corpus_present():
    assert os.path.isdir(CORPUS_DIR), "recorded prodready corpus is missing"
    assert _atlas_error_fixtures(), "no atlas error fixtures found in the corpus"


@pytest.mark.invariant
@pytest.mark.parametrize("name,fixture", _atlas_error_fixtures(), ids=lambda v: v if isinstance(v, str) else "")
def test_recorded_atlas_error_maps_to_taxonomy(name, fixture):
    status = fixture["status"]
    body = json.dumps(fixture.get("body", {})).encode("utf-8")
    response = httpx.Response(
        status,
        content=body,
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", "https://api.test.local/x"),
    )
    # _make_status_error[_from_response] uses no instance state, so bypass __init__
    # (which would hit the network) to exercise the real mapping.
    client = object.__new__(Stratix)
    exc = client._make_status_error_from_response(response)
    expected = _expected_exc(status)
    assert isinstance(exc, expected), f"{name}: status {status} -> {type(exc).__name__}, expected {expected.__name__}"


@pytest.mark.invariant
@pytest.mark.parametrize("name,fixture", _sdk_surface_fixtures(), ids=lambda v: v if isinstance(v, str) else "")
def test_recorded_sdk_surface_no_secret_leak(name, fixture):
    result = fixture["result"]
    assert result.get("raised_str_has_key") is False, f"{name}: api key leaked into str(exc)"
    assert result.get("key_in_stored_dump") is False, f"{name}: api key reached the stored telemetry dump"
    assert result.get("secret_patterns_in_dump") == [], f"{name}: secret patterns found in the stored dump"


def test_manifest_references_existing_files():
    manifest = _load(os.path.join(CORPUS_DIR, "_manifest.json"))
    for probe, rel in manifest["probes"].items():
        assert os.path.exists(os.path.join(CORPUS_DIR, os.path.basename(rel))), f"{probe}: {rel} missing"
