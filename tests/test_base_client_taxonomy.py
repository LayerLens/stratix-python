"""Exception-taxonomy enforcement + strict reads (LAY-3637, F-L9-002/003/004).

Raw httpx transport errors and response-decode failures must surface as SDK
taxonomy exceptions, and opt-in strict reads must raise instead of swallowing
to None."""

from __future__ import annotations

from unittest.mock import Mock, patch

import httpx
import pytest

from layerlens._exceptions import (
    StratixError,
    APITimeoutError,
    APIConnectionError,
    APIResponseValidationError,
)
from layerlens._base_client import BaseClient
from layerlens.resources.traces.traces import Traces


def _client() -> BaseClient:
    return BaseClient(base_url="https://api.test.com")


@pytest.mark.invariant
class TestTransportErrorTaxonomy:
    @patch("httpx.Client.request")
    def test_connect_error_becomes_api_connection_error(self, mock_request):
        mock_request.side_effect = httpx.ConnectError("connection refused")
        with pytest.raises(APIConnectionError):
            _client()._request_cast("GET", "/x")

    @patch("httpx.Client.request")
    def test_timeout_becomes_api_timeout_error(self, mock_request):
        mock_request.side_effect = httpx.ConnectTimeout("timed out")
        with pytest.raises(APITimeoutError):
            _client()._request_cast("GET", "/x")

    @patch("httpx.Client.request")
    def test_bad_json_becomes_response_validation_error(self, mock_request):
        resp = Mock(spec=httpx.Response)
        resp.status_code = 200
        resp.raise_for_status = Mock()
        resp.json = Mock(side_effect=ValueError("not json"))
        resp.request = httpx.Request("GET", "https://api.test.com/x")
        mock_request.return_value = resp
        with pytest.raises(APIResponseValidationError):
            _client()._request_cast("GET", "/x", cast_to=dict)


def _traces_resource() -> Traces:
    client = Mock()
    client.organization_id = "org-1"
    client.project_id = "proj-1"
    client.get_cast = Mock()
    client.post_cast = Mock()
    client.delete_cast = Mock()
    return Traces(client)


@pytest.mark.invariant
class TestStrictReads:
    # A real 404 already raises NotFoundError (mapped in _base_client). strict
    # surfaces CONTRACT DRIFT — a 200 whose body is empty or unparseable — which
    # the default swallows to None. The resource has no httpx.Response at this
    # layer (cast_to=dict), so drift surfaces as the SDK-base StratixError.
    def test_get_strict_raises_on_empty_response(self):
        r = _traces_resource()
        r._get.return_value = None
        with pytest.raises(StratixError):
            r.get("t1", strict=True)

    def test_get_strict_raises_on_unparseable_shape(self):
        r = _traces_resource()
        r._get.return_value = {"unexpected": object()}  # Trace(**data) fails to construct
        assert r.get("t1") is None  # default swallows
        with pytest.raises(StratixError):
            r.get("t1", strict=True)

    def test_get_default_is_non_strict_returns_none(self):
        r = _traces_resource()
        r._get.return_value = None
        assert r.get("t1") is None  # default behavior unchanged

    def test_get_many_strict_raises_on_empty_response(self):
        r = _traces_resource()
        r._get.return_value = None
        with pytest.raises(StratixError):
            r.get_many(strict=True)

    def test_get_many_default_returns_none(self):
        r = _traces_resource()
        r._get.return_value = None
        assert r.get_many() is None  # default behavior unchanged
