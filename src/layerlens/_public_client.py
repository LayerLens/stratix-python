from __future__ import annotations

import os
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Union, Mapping
from functools import cached_property
from typing_extensions import Self, override

import httpx

from . import _exceptions
from ._utils import is_mapping
from ._constants import DEFAULT_TIMEOUT
from ._exceptions import APIStatusError
from ._base_client import BaseClient, BaseAsyncClient

if TYPE_CHECKING:
    from .resources.comparisons import Comparisons, AsyncComparisons
    from .resources.public_models import PublicModelsResource, AsyncPublicModelsResource
    from .resources.public_benchmarks import PublicBenchmarksResource, AsyncPublicBenchmarksResource
    from .resources.public_evaluations import PublicEvaluationsResource, AsyncPublicEvaluationsResource


__all__ = ["PublicClient", "AsyncPublicClient"]


def _make_status_error(
    err_msg: str,
    *,
    body: object,
    response: httpx.Response,
) -> APIStatusError:
    data = body.get("error", body) if is_mapping(body) else body

    if response.status_code == HTTPStatus.BAD_REQUEST:
        return _exceptions.BadRequestError(err_msg, response=response, body=data)
    if response.status_code == HTTPStatus.UNAUTHORIZED:
        return _exceptions.AuthenticationError(err_msg, response=response, body=data)
    if response.status_code == HTTPStatus.FORBIDDEN:
        return _exceptions.PermissionDeniedError(err_msg, response=response, body=data)
    if response.status_code == HTTPStatus.NOT_FOUND:
        return _exceptions.NotFoundError(err_msg, response=response, body=data)
    if response.status_code == HTTPStatus.CONFLICT:
        return _exceptions.ConflictError(err_msg, response=response, body=data)
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        return _exceptions.UnprocessableEntityError(err_msg, response=response, body=data)
    if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        return _exceptions.RateLimitError(err_msg, response=response, body=data)
    if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return _exceptions.InternalServerError(err_msg, response=response, body=data)

    return APIStatusError(err_msg, response=response, body=data)


class PublicClient(BaseClient):
    """Client for accessing public LayerLens API endpoints."""

    api_key: str

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, httpx.Timeout, None] = DEFAULT_TIMEOUT,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("LAYERLENS_STRATIX_API_KEY")
        if api_key is None or api_key == "":
            raise _exceptions.StratixError(
                "The api_key client option must be set either by passing api_key to the client or by setting the LAYERLENS_STRATIX_API_KEY environment variable",
            )
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get("LAYERLENS_STRATIX_BASE_URL") or os.environ.get("LAYERLENS_ATLAS_BASE_URL")
        if base_url is None:
            base_url = "https://api.layerlens.ai/api/v1"

        super().__init__(base_url=base_url, timeout=timeout)

    @cached_property
    def models(self) -> PublicModelsResource:
        from .resources.public_models import PublicModelsResource

        return PublicModelsResource(self)

    @cached_property
    def benchmarks(self) -> PublicBenchmarksResource:
        from .resources.public_benchmarks import PublicBenchmarksResource

        return PublicBenchmarksResource(self)

    @cached_property
    def comparisons(self) -> Comparisons:
        from .resources.comparisons import Comparisons

        return Comparisons(self)

    @cached_property
    def evaluations(self) -> PublicEvaluationsResource:
        from .resources.public_evaluations import PublicEvaluationsResource

        return PublicEvaluationsResource(self)

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key}

    def copy(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        return self.__class__(
            api_key=api_key or self.api_key,
            base_url=base_url or self.base_url,
            timeout=self.timeout or timeout,
            **_extra_kwargs,
        )

    with_options = copy

    @override
    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
        return _make_status_error(err_msg, body=body, response=response)


class AsyncPublicClient(BaseAsyncClient):
    """Async client for accessing public LayerLens API endpoints."""

    api_key: str

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("LAYERLENS_STRATIX_API_KEY")
        if api_key is None or api_key == "":
            raise _exceptions.StratixError(
                "The api_key client option must be set either by passing api_key to the client or by setting the LAYERLENS_STRATIX_API_KEY environment variable",
            )
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get("LAYERLENS_STRATIX_BASE_URL") or os.environ.get("LAYERLENS_ATLAS_BASE_URL")
        if base_url is None:
            base_url = "https://api.layerlens.ai/api/v1"

        super().__init__(base_url=base_url, timeout=timeout)

    @cached_property
    def models(self) -> AsyncPublicModelsResource:
        from .resources.public_models import AsyncPublicModelsResource

        return AsyncPublicModelsResource(self)

    @cached_property
    def benchmarks(self) -> AsyncPublicBenchmarksResource:
        from .resources.public_benchmarks import AsyncPublicBenchmarksResource

        return AsyncPublicBenchmarksResource(self)

    @cached_property
    def comparisons(self) -> AsyncComparisons:
        from .resources.comparisons import AsyncComparisons

        return AsyncComparisons(self)

    @cached_property
    def evaluations(self) -> AsyncPublicEvaluationsResource:
        from .resources.public_evaluations import AsyncPublicEvaluationsResource

        return AsyncPublicEvaluationsResource(self)

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key}

    def copy(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        return self.__class__(
            api_key=api_key or self.api_key,
            base_url=base_url or self.base_url,
            timeout=self.timeout or timeout,
            **_extra_kwargs,
        )

    with_options = copy

    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
        return _make_status_error(err_msg, body=body, response=response)
