from __future__ import annotations

from http import HTTPStatus
from typing import Literal

import httpx


class AtlasError(Exception):
    pass


class APIError(AtlasError):
    message: str
    request: httpx.Request

    """The API response body.

    If the API responded with a valid JSON structure then this property will be the
    decoded result.

    If it isn't a valid JSON structure then this will be the raw response.

    If there was no response associated with this error then it will be `None`.
    """
    body: object | None

    def __init__(self, message: str, request: httpx.Request, *, body: object | None) -> None:
        super().__init__(message)
        self.request = request
        self.message = message
        self.body = body


class APIResponseValidationError(APIError):
    response: httpx.Response
    status_code: int

    def __init__(
        self,
        response: httpx.Response,
        body: object | None,
        *,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message or "Data returned by API invalid for expected schema.",
            response.request,
            body=body,
        )
        self.response = response
        self.status_code = response.status_code


class APIStatusError(APIError):
    """Raised when an API response has a status code of 4xx or 5xx."""

    response: httpx.Response
    status_code: int
    request_id: str | None

    def __init__(self, message: str, *, response: httpx.Response, body: object | None) -> None:
        super().__init__(message, response.request, body=body)
        self.response = response
        self.status_code = response.status_code
        self.request_id = response.headers.get("x-request-id")


class APIConnectionError(APIError):
    def __init__(self, *, message: str = "Connection error.", request: httpx.Request) -> None:
        super().__init__(message, request, body=None)


class APITimeoutError(APIConnectionError):
    def __init__(self, request: httpx.Request) -> None:
        super().__init__(message="Request timed out.", request=request)


class BadRequestError(APIStatusError):
    status_code: Literal[HTTPStatus.BAD_REQUEST] = HTTPStatus.BAD_REQUEST


class AuthenticationError(APIStatusError):
    status_code: Literal[HTTPStatus.UNAUTHORIZED] = HTTPStatus.UNAUTHORIZED


class PermissionDeniedError(APIStatusError):
    status_code: Literal[HTTPStatus.FORBIDDEN] = HTTPStatus.FORBIDDEN


class NotFoundError(APIStatusError):
    status_code: Literal[HTTPStatus.NOT_FOUND] = HTTPStatus.NOT_FOUND


class ConflictError(APIStatusError):
    status_code: Literal[HTTPStatus.CONFLICT] = HTTPStatus.CONFLICT


class UnprocessableEntityError(APIStatusError):
    status_code: Literal[HTTPStatus.UNPROCESSABLE_ENTITY] = HTTPStatus.UNPROCESSABLE_ENTITY


class RateLimitError(APIStatusError):
    status_code: Literal[HTTPStatus.TOO_MANY_REQUESTS] = HTTPStatus.TOO_MANY_REQUESTS


class InternalServerError(APIStatusError):
    pass
