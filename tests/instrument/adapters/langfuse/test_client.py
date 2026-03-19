"""Tests for Langfuse API client."""

import base64
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError, URLError

from layerlens.instrument.adapters.langfuse.client import (
    LangfuseAPIClient,
    LangfuseAPIError,
)


def _make_response(data: dict, status: int = 200):
    """Create a mock urlopen response."""
    mock = MagicMock()
    mock.read.return_value = json.dumps(data).encode()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


class TestLangfuseAPIClient:
    """Tests for LangfuseAPIClient."""

    def test_init_sets_auth_header(self):
        client = LangfuseAPIClient(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://langfuse.example.com",
        )
        expected = base64.b64encode(b"pk-test:sk-test").decode()
        assert client._auth_header == f"Basic {expected}"

    def test_init_strips_trailing_slash(self):
        client = LangfuseAPIClient(
            public_key="pk", secret_key="sk", host="https://example.com/"
        )
        assert client._host == "https://example.com"

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_health_check(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({"status": "OK"})
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        result = client.health_check()
        assert result == {"status": "OK"}

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_list_traces(self, mock_urlopen):
        traces_data = {
            "data": [{"id": "trace-1"}, {"id": "trace-2"}],
            "meta": {"totalPages": 1, "page": 1},
        }
        mock_urlopen.return_value = _make_response(traces_data)
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        result = client.list_traces(page=1, limit=50)
        assert len(result["data"]) == 2

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_list_traces_with_tags(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({"data": [], "meta": {}})
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        client.list_traces(tags=["v2", "production"])
        # Verify tags were included in the URL
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert "tags=v2" in request.full_url
        assert "tags=production" in request.full_url

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_list_traces_with_from_timestamp(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({"data": [], "meta": {}})
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        client.list_traces(from_timestamp=ts)
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert "fromTimestamp=" in request.full_url

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_get_trace(self, mock_urlopen):
        trace_data = {
            "id": "trace-1",
            "input": "Hello",
            "output": "World",
            "observations": [],
        }
        mock_urlopen.return_value = _make_response(trace_data)
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        result = client.get_trace("trace-1")
        assert result["id"] == "trace-1"

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_create_trace(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({"successes": [{"id": "t1"}]})
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        result = client.create_trace({"id": "t1", "name": "test"})
        assert "successes" in result

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_create_generation(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({"successes": [{"id": "g1"}]})
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        result = client.create_generation({"id": "g1", "traceId": "t1"})
        assert "successes" in result

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_create_span(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({"successes": [{"id": "s1"}]})
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        result = client.create_span({"id": "s1", "traceId": "t1"})
        assert "successes" in result

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_ingestion_batch(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({"successes": []})
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        batch = [
            {"id": "1", "type": "trace-create", "timestamp": "now", "body": {}},
            {"id": "2", "type": "span-create", "timestamp": "now", "body": {}},
        ]
        result = client.ingestion_batch(batch)
        assert "successes" in result

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_get_all_traces_single_page(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({
            "data": [{"id": "t1"}, {"id": "t2"}],
            "meta": {"totalPages": 1, "page": 1},
        })
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        traces = client.get_all_traces()
        assert len(traces) == 2

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_get_all_traces_multi_page(self, mock_urlopen):
        responses = [
            _make_response({
                "data": [{"id": "t1"}],
                "meta": {"totalPages": 2, "page": 1},
            }),
            _make_response({
                "data": [{"id": "t2"}],
                "meta": {"totalPages": 2, "page": 2},
            }),
        ]
        mock_urlopen.side_effect = responses
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        traces = client.get_all_traces()
        assert len(traces) == 2
        assert traces[0]["id"] == "t1"
        assert traces[1]["id"] == "t2"

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_get_all_traces_empty(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({
            "data": [],
            "meta": {"totalPages": 0},
        })
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        traces = client.get_all_traces()
        assert traces == []

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_retry_on_429(self, mock_urlopen):
        """Client retries on 429 rate limit responses."""
        error_resp = MagicMock()
        error_resp.read.return_value = b'{"error": "rate limited"}'
        http_error = HTTPError(
            url="http://test", code=429, msg="Too Many Requests",
            hdrs=None, fp=error_resp,
        )
        success_resp = _make_response({"status": "OK"})
        mock_urlopen.side_effect = [http_error, success_resp]

        client = LangfuseAPIClient(public_key="pk", secret_key="sk", max_retries=2)
        with patch("layerlens.instrument.adapters.langfuse.client.time.sleep"):
            result = client.health_check()
        assert result == {"status": "OK"}
        assert mock_urlopen.call_count == 2

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_retry_on_500(self, mock_urlopen):
        """Client retries on 500 server errors."""
        error_resp = MagicMock()
        error_resp.read.return_value = b'{"error": "internal"}'
        http_error = HTTPError(
            url="http://test", code=500, msg="Internal Server Error",
            hdrs=None, fp=error_resp,
        )
        success_resp = _make_response({"status": "OK"})
        mock_urlopen.side_effect = [http_error, success_resp]

        client = LangfuseAPIClient(public_key="pk", secret_key="sk", max_retries=2)
        with patch("layerlens.instrument.adapters.langfuse.client.time.sleep"):
            result = client.health_check()
        assert result == {"status": "OK"}

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_no_retry_on_400(self, mock_urlopen):
        """Client does NOT retry on 400 client errors."""
        error_resp = MagicMock()
        error_resp.read.return_value = b'{"error": "bad request"}'
        http_error = HTTPError(
            url="http://test", code=400, msg="Bad Request",
            hdrs=None, fp=error_resp,
        )
        mock_urlopen.side_effect = http_error

        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        with pytest.raises(LangfuseAPIError) as exc_info:
            client.health_check()
        assert exc_info.value.status_code == 400
        assert mock_urlopen.call_count == 1

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_max_retries_exceeded(self, mock_urlopen):
        """Client raises after exhausting retries."""
        error_resp = MagicMock()
        error_resp.read.return_value = b'{"error": "server error"}'
        http_error = HTTPError(
            url="http://test", code=500, msg="Internal Server Error",
            hdrs=None, fp=error_resp,
        )
        mock_urlopen.side_effect = http_error

        client = LangfuseAPIClient(public_key="pk", secret_key="sk", max_retries=2)
        with patch("layerlens.instrument.adapters.langfuse.client.time.sleep"):
            with pytest.raises(LangfuseAPIError) as exc_info:
                client.health_check()
        assert exc_info.value.status_code == 500
        # 1 initial + 2 retries = 3 calls
        assert mock_urlopen.call_count == 3

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_connection_error_retry(self, mock_urlopen):
        """Client retries on connection errors."""
        url_error = URLError("Connection refused")
        success_resp = _make_response({"status": "OK"})
        mock_urlopen.side_effect = [url_error, success_resp]

        client = LangfuseAPIClient(public_key="pk", secret_key="sk", max_retries=2)
        with patch("layerlens.instrument.adapters.langfuse.client.time.sleep"):
            result = client.health_check()
        assert result == {"status": "OK"}

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_empty_response(self, mock_urlopen):
        """Client handles empty response body."""
        mock = MagicMock()
        mock.read.return_value = b""
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        result = client.health_check()
        assert result == {}

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_auth_header_sent(self, mock_urlopen):
        """Client sends correct Basic auth header."""
        mock_urlopen.return_value = _make_response({})
        client = LangfuseAPIClient(public_key="pk-abc", secret_key="sk-xyz")
        client.health_check()
        request = mock_urlopen.call_args[0][0]
        expected = base64.b64encode(b"pk-abc:sk-xyz").decode()
        assert request.get_header("Authorization") == f"Basic {expected}"

    @patch("layerlens.instrument.adapters.langfuse.client.urlopen")
    def test_list_observations(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({
            "data": [{"id": "obs-1"}],
            "meta": {},
        })
        client = LangfuseAPIClient(public_key="pk", secret_key="sk")
        result = client.list_observations(trace_id="t1", type="GENERATION")
        assert len(result["data"]) == 1
