from http import HTTPStatus
from unittest.mock import Mock

import httpx
import pytest

from layerlens._exceptions import (
    APIError,
    StratixError,
    ConflictError,
    NotFoundError,
    APIStatusError,
    RateLimitError,
    APITimeoutError,
    BadRequestError,
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    UnprocessableEntityError,
    APIResponseValidationError,
)


class TestExceptionHierarchy:
    """Test exception inheritance and basic functionality."""

    def test_stratix_error_is_base_exception(self):
        """StratixError inherits from Exception."""
        error = StratixError("test message")

        assert isinstance(error, Exception)
        assert str(error) == "test message"

    def test_api_error_inherits_from_stratix_error(self):
        """APIError inherits from StratixError."""
        mock_request = Mock(spec=httpx.Request)
        error = APIError("api error", mock_request, body=None)

        assert isinstance(error, StratixError)
        assert isinstance(error, Exception)

    def test_api_status_error_inherits_from_api_error(self):
        """APIStatusError inherits from APIError."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.request = Mock(spec=httpx.Request)
        mock_response.status_code = 400
        mock_response.headers = {}

        error = APIStatusError("status error", response=mock_response, body=None)

        assert isinstance(error, APIError)
        assert isinstance(error, StratixError)

    @pytest.mark.parametrize(
        "exception_class",
        [
            BadRequestError,
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
            ConflictError,
            UnprocessableEntityError,
            RateLimitError,
            InternalServerError,
        ],
    )
    def test_status_exceptions_inherit_from_api_status_error(self, exception_class):
        """All status-specific exceptions inherit from APIStatusError."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.request = Mock(spec=httpx.Request)
        mock_response.status_code = 400
        mock_response.headers = {}

        error = exception_class("test error", response=mock_response, body=None)

        assert isinstance(error, APIStatusError)
        assert isinstance(error, APIError)
        assert isinstance(error, StratixError)


class TestAPIError:
    """Test APIError functionality."""

    @pytest.fixture
    def mock_request(self):
        """Mock httpx.Request."""
        return Mock(spec=httpx.Request)

    def test_api_error_stores_message_and_request(self, mock_request):
        """APIError stores message, request, and body."""
        body = {"error": "test"}
        error = APIError("test message", mock_request, body=body)

        assert error.message == "test message"
        assert error.request is mock_request
        assert error.body == body
        assert str(error) == "test message"

    def test_api_error_with_none_body(self, mock_request):
        """APIError handles None body."""
        error = APIError("test message", mock_request, body=None)

        assert error.body is None
        assert error.message == "test message"

    def test_api_error_with_json_body(self, mock_request):
        """APIError stores JSON body correctly."""
        body = {"error": "validation failed", "code": 422}
        error = APIError("validation error", mock_request, body=body)

        assert error.body == body
        assert isinstance(error.body, dict)
        assert error.body["error"] == "validation failed"

    def test_api_error_with_string_body(self, mock_request):
        """APIError stores string body correctly."""
        body = "Plain text error message"
        error = APIError("server error", mock_request, body=body)

        assert error.body == body


class TestAPIResponseValidationError:
    """Test APIResponseValidationError functionality."""

    @pytest.fixture
    def mock_response(self):
        """Mock httpx.Response."""
        mock = Mock(spec=httpx.Response)
        mock.request = Mock(spec=httpx.Request)
        mock.status_code = 200
        return mock

    def test_validation_error_with_default_message(self, mock_response):
        """APIResponseValidationError uses default message when none provided."""
        error = APIResponseValidationError(mock_response, body=None)

        assert error.message == "Data returned by API invalid for expected schema."
        assert error.response is mock_response
        assert error.status_code == 200

    def test_validation_error_with_custom_message(self, mock_response):
        """APIResponseValidationError uses custom message when provided."""
        custom_message = "Custom validation error"
        error = APIResponseValidationError(mock_response, body=None, message=custom_message)

        assert error.message == custom_message
        assert str(error) == custom_message

    def test_validation_error_stores_response_data(self, mock_response):
        """APIResponseValidationError stores response and body."""
        body = {"invalid": "data"}
        error = APIResponseValidationError(mock_response, body=body)

        assert error.response is mock_response
        assert error.body == body
        assert error.request is mock_response.request


class TestAPIStatusError:
    """Test APIStatusError functionality."""

    @pytest.fixture
    def mock_response(self):
        """Mock httpx.Response with headers."""
        mock = Mock(spec=httpx.Response)
        mock.request = Mock(spec=httpx.Request)
        mock.status_code = 404
        mock.headers = {"x-request-id": "req-123"}
        return mock

    def test_status_error_stores_response_data(self, mock_response):
        """APIStatusError stores response, status code, and request ID."""
        error = APIStatusError("not found", response=mock_response, body=None)

        assert error.response is mock_response
        assert error.status_code == 404
        assert error.request_id == "req-123"
        assert error.request is mock_response.request

    def test_status_error_without_request_id(self, mock_response):
        """APIStatusError handles missing request ID header."""
        mock_response.headers = {}
        error = APIStatusError("error", response=mock_response, body=None)

        assert error.request_id is None

    def test_status_error_with_body(self, mock_response):
        """APIStatusError stores error body."""
        body = {"error": "Resource not found", "code": "NOT_FOUND"}
        error = APIStatusError("not found", response=mock_response, body=body)

        assert error.body == body


class TestConnectionErrors:
    """Test connection-related errors."""

    @pytest.fixture
    def mock_request(self):
        """Mock httpx.Request."""
        return Mock(spec=httpx.Request)

    def test_api_connection_error_default_message(self, mock_request):
        """APIConnectionError uses default message."""
        error = APIConnectionError(request=mock_request)

        assert error.message == "Connection error."
        assert error.request is mock_request
        assert error.body is None

    def test_api_connection_error_custom_message(self, mock_request):
        """APIConnectionError accepts custom message."""
        custom_message = "Failed to connect to server"
        error = APIConnectionError(message=custom_message, request=mock_request)

        assert error.message == custom_message

    def test_api_timeout_error_inherits_from_connection_error(self, mock_request):
        """APITimeoutError inherits from APIConnectionError."""
        error = APITimeoutError(mock_request)

        assert isinstance(error, APIConnectionError)
        assert isinstance(error, APIError)
        assert error.message == "Request timed out."
        assert error.request is mock_request


class TestStatusCodeExceptions:
    """Test HTTP status code specific exceptions."""

    @pytest.fixture
    def mock_response_factory(self):
        """Factory for creating mock responses with different status codes."""

        def _create_response(status_code: int) -> Mock:
            mock = Mock(spec=httpx.Response)
            mock.request = Mock(spec=httpx.Request)
            mock.status_code = status_code
            mock.headers = {}
            return mock

        return _create_response

    @pytest.mark.parametrize(
        "exception_class,expected_status",
        [
            (BadRequestError, HTTPStatus.BAD_REQUEST),
            (AuthenticationError, HTTPStatus.UNAUTHORIZED),
            (PermissionDeniedError, HTTPStatus.FORBIDDEN),
            (NotFoundError, HTTPStatus.NOT_FOUND),
            (ConflictError, HTTPStatus.CONFLICT),
            (UnprocessableEntityError, HTTPStatus.UNPROCESSABLE_ENTITY),
            (RateLimitError, HTTPStatus.TOO_MANY_REQUESTS),
        ],
    )
    def test_status_exception_has_correct_status_code(self, exception_class, expected_status, mock_response_factory):
        """Status-specific exceptions have correct status codes."""
        mock_response = mock_response_factory(expected_status.value)
        error = exception_class("test error", response=mock_response, body=None)

        assert error.status_code == expected_status.value
        assert hasattr(error.__class__, "status_code")
        assert error.__class__.status_code == expected_status

    def test_bad_request_error_properties(self, mock_response_factory):
        """BadRequestError has correct properties."""
        mock_response = mock_response_factory(400)
        body = {"error": "Invalid request", "field": "name"}
        error = BadRequestError("bad request", response=mock_response, body=body)

        assert error.status_code == 400
        assert error.body == body
        assert isinstance(error, APIStatusError)

    def test_authentication_error_properties(self, mock_response_factory):
        """AuthenticationError has correct properties."""
        mock_response = mock_response_factory(401)
        error = AuthenticationError("unauthorized", response=mock_response, body=None)

        assert error.status_code == 401
        assert error.__class__.status_code == HTTPStatus.UNAUTHORIZED

    def test_internal_server_error_no_fixed_status(self, mock_response_factory):
        """InternalServerError doesn't have a fixed status code."""
        mock_response = mock_response_factory(500)
        error = InternalServerError("server error", response=mock_response, body=None)

        assert error.status_code == 500
        assert not hasattr(error.__class__, "status_code") or error.__class__.status_code is None


class TestErrorMessages:
    """Test error message handling and formatting."""

    def test_exception_str_representation(self):
        """Exception string representation shows message."""
        mock_request = Mock(spec=httpx.Request)
        error = APIError("Test error message", mock_request, body=None)

        assert str(error) == "Test error message"

    def test_exception_with_complex_body(self):
        """Exception handles complex body structures."""
        mock_request = Mock(spec=httpx.Request)
        body = {
            "error": {
                "code": "VALIDATION_ERROR",
                "details": ["Field 'name' is required"],
            },
            "request_id": "req-456",
        }
        error = APIError("Validation failed", mock_request, body=body)

        assert isinstance(error.body, dict)
        assert error.body["error"]["code"] == "VALIDATION_ERROR"
        assert error.body["request_id"] == "req-456"
