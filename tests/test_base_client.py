from dataclasses import dataclass
from unittest.mock import Mock, patch

import httpx
import pytest

from atlas import _exceptions
from atlas._base_client import BaseClient


@dataclass
class ResponseModel:
    """Test model for response casting."""

    name: str
    value: int


class TestBaseClient:
    """Test BaseClient HTTP functionality."""

    @pytest.fixture
    def client(self):
        """Create a BaseClient instance for testing."""
        return BaseClient(base_url="https://api.test.com")

    @pytest.fixture
    def mock_response(self):
        """Mock httpx Response."""
        mock = Mock(spec=httpx.Response)
        mock.status_code = 200
        mock.raise_for_status.return_value = None
        mock.json.return_value = {"name": "test", "value": 42}
        return mock

    def test_init_sets_base_url(self):
        """BaseClient initializes with correct base URL."""
        client = BaseClient(base_url="https://custom.api.com")

        assert str(client.base_url) == "https://custom.api.com"

    def test_init_with_headers(self):
        """BaseClient accepts custom headers."""
        headers = {"X-Custom": "value"}
        client = BaseClient(base_url="https://api.test.com", headers=headers)

        assert client.headers["X-Custom"] == "value"

    def test_auth_headers_empty_by_default(self, client):
        """BaseClient auth_headers returns empty dict by default."""
        assert client.auth_headers == {}

    def test_default_headers_structure(self, client):
        """BaseClient default_headers includes required headers."""
        headers = client.default_headers

        assert headers["Accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"
        assert isinstance(headers, dict)

    def test_default_headers_includes_auth(self, client):
        """default_headers merges auth_headers."""
        with patch.object(
            type(client), "auth_headers", new_callable=lambda: property(lambda _: {"Authorization": "Bearer token"})
        ):
            headers = client.default_headers

            assert headers["Authorization"] == "Bearer token"
            assert headers["Accept"] == "application/json"

    @patch("httpx.Client.request")
    def test_request_cast_without_cast_to(self, mock_request, client, mock_response):
        """_request_cast returns raw response when cast_to is None."""
        mock_request.return_value = mock_response

        result = client._request_cast("GET", "/test")

        assert result is mock_response
        mock_request.assert_called_once_with(
            method="GET", url="/test", json=None, params=None, headers=client.default_headers
        )

    @patch("httpx.Client.request")
    def test_request_cast_with_cast_to(self, mock_request, client, mock_response):
        """_request_cast casts response to specified type."""
        mock_request.return_value = mock_response

        result = client._request_cast("GET", "/test", cast_to=ResponseModel)

        assert isinstance(result, ResponseModel)
        assert result.name == "test"
        assert result.value == 42
        mock_response.json.assert_called_once()

    @patch("httpx.Client.request")
    def test_request_cast_combines_headers(self, mock_request, client, mock_response):
        """_request_cast merges default and custom headers."""
        mock_request.return_value = mock_response
        custom_headers = {"X-Custom": "value"}

        client._request_cast("POST", "/test", headers=custom_headers)

        expected_headers = {**client.default_headers, **custom_headers}
        mock_request.assert_called_once_with(
            method="POST", url="/test", json=None, params=None, headers=expected_headers
        )

    @patch("httpx.Client.request")
    def test_request_cast_with_body_and_params(self, mock_request, client, mock_response):
        """_request_cast sends body and params correctly."""
        mock_request.return_value = mock_response
        body = {"key": "value"}
        params = {"filter": "active"}

        client._request_cast("POST", "/test", body=body, params=params)

        mock_request.assert_called_once_with(
            method="POST", url="/test", json=body, params=params, headers=client.default_headers
        )

    @patch("httpx.Client.request")
    def test_request_cast_handles_http_error(self, mock_request, client):
        """_request_cast converts HTTPStatusError to APIStatusError."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.headers = {}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=Mock(), response=mock_response
        )
        mock_request.return_value = mock_response

        with patch.object(client, "_make_status_error_from_response") as mock_make_error:
            mock_make_error.side_effect = _exceptions.APIStatusError("Test error", response=mock_response, body=None)

            with pytest.raises(_exceptions.APIStatusError):
                client._request_cast("GET", "/test")

            mock_make_error.assert_called_once_with(mock_response)

    @patch("httpx.Client.request")
    def test_get_cast_delegates_correctly(self, mock_request, client, mock_response):
        """get_cast delegates to _request_cast with GET method."""
        mock_request.return_value = mock_response
        params = {"page": 1}
        headers = {"X-Test": "value"}

        result = client.get_cast("/test", params=params, headers=headers, cast_to=ResponseModel)

        assert isinstance(result, ResponseModel)
        mock_request.assert_called_once_with(
            method="GET", url="/test", json=None, params=params, headers={**client.default_headers, **headers}
        )

    @patch("httpx.Client.request")
    def test_post_cast_delegates_correctly(self, mock_request, client, mock_response):
        """post_cast delegates to _request_cast with POST method."""
        mock_request.return_value = mock_response
        body = {"name": "test"}
        headers = {"X-Test": "value"}

        result = client.post_cast("/test", body=body, headers=headers, cast_to=ResponseModel)

        assert isinstance(result, ResponseModel)
        mock_request.assert_called_once_with(
            method="POST", url="/test", json=body, params=None, headers={**client.default_headers, **headers}
        )

    def test_make_status_error_from_response_with_json(self, client):
        """_make_status_error_from_response parses JSON error body."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.text = '{"error": "Bad Request", "code": 400}'

        with patch.object(client, "_make_status_error") as mock_make_error:
            client._make_status_error_from_response(mock_response)

            mock_make_error.assert_called_once()
            args, kwargs = mock_make_error.call_args
            assert "Error code: 400" in args[0]
            assert kwargs["body"] == {"error": "Bad Request", "code": 400}
            assert kwargs["response"] is mock_response

    def test_make_status_error_from_response_with_text(self, client):
        """_make_status_error_from_response handles plain text errors."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(client, "_make_status_error") as mock_make_error:
            client._make_status_error_from_response(mock_response)

            mock_make_error.assert_called_once()
            args, kwargs = mock_make_error.call_args
            assert args[0] == "Internal Server Error"
            assert kwargs["body"] == "Internal Server Error"

    def test_make_status_error_from_response_empty_text(self, client):
        """_make_status_error_from_response handles empty response text."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 503
        mock_response.text = ""

        with patch.object(client, "_make_status_error") as mock_make_error:
            client._make_status_error_from_response(mock_response)

            mock_make_error.assert_called_once()
            args, _ = mock_make_error.call_args
            assert args[0] == "Error code: 503"

    def test_make_status_error_not_implemented(self, client):
        """_make_status_error raises NotImplementedError."""
        mock_response = Mock(spec=httpx.Response)

        with pytest.raises(NotImplementedError):
            client._make_status_error("test", body=None, response=mock_response)
