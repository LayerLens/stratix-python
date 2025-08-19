from __future__ import annotations

import os
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Union, Mapping, Optional
from functools import cached_property
from typing_extensions import Self, override

import httpx

from . import _exceptions
from ._utils import is_mapping
from .models import Organization, OrganizationResponse
from ._constants import DEFAULT_TIMEOUT
from ._exceptions import AtlasError, APIStatusError
from ._base_client import BaseClient, BaseAsyncClient

if TYPE_CHECKING:
    from .resources.models import Models, AsyncModels
    from .resources.results import Results, AsyncResults
    from .resources.benchmarks import Benchmarks, AsyncBenchmarks
    from .resources.evaluations import Evaluations, AsyncEvaluations


__all__ = ["Atlas", "Client"]


class Atlas(BaseClient):
    api_key: str
    organization_id: str | None
    project_id: str | None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, httpx.Timeout, None] = DEFAULT_TIMEOUT,
    ) -> None:
        """Construct a new synchronous Atlas client instance.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `LAYERLENS_ATLAS_API_KEY`
        """
        if api_key is None:
            api_key = os.environ.get("LAYERLENS_ATLAS_API_KEY")
        if api_key is None:
            raise AtlasError(
                "The api_key client option must be set either by passing api_key to the client or by setting the LAYERLENS_ATLAS_API_KEY environment variable"
            )
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get("LAYERLENS_ATLAS_BASE_URL")
        if base_url is None:
            base_url = "https://8bg48mbhyi.execute-api.us-east-1.amazonaws.com/prod/api/v1"

        super().__init__(
            base_url=base_url,
            timeout=timeout,
        )

        organization = self._get_organization()
        if organization is None:
            raise AtlasError(f"Organization could not be fetched. Please contact LayerLens Atlas support.")
        self.organization_id = organization.id

        if organization.projects is None or len(organization.projects) == 0:
            raise AtlasError(
                f"Organization {self.organization_id} is missing project. Please contact LayerLens Atlas support."
            )
        self.project_id = organization.projects[0].id

    @cached_property
    def benchmarks(self) -> Benchmarks:
        from .resources.benchmarks import Benchmarks

        return Benchmarks(self)

    @cached_property
    def evaluations(self) -> Evaluations:
        from .resources.evaluations import Evaluations

        return Evaluations(self)

    @cached_property
    def models(self) -> Models:
        from .resources.models import Models

        return Models(self)

    @cached_property
    def results(self) -> Results:
        from .resources.results import Results

        return Results(self)

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        api_key = self.api_key
        if not api_key:
            return {}
        return {"x-api-key": api_key}

    def copy(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        return self.__class__(
            api_key=api_key or self.api_key,
            base_url=base_url or self.base_url,
            timeout=self.timeout or timeout,
            **_extra_kwargs,
        )

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    @override
    def _make_status_error(
        self,
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

    def _get_organization(self) -> Optional[Organization]:
        organization = super().get_cast(
            f"/organizations",
            timeout=30,
            cast_to=OrganizationResponse,
        )
        if isinstance(organization, OrganizationResponse):
            return organization.data
        return None


class AsyncAtlas(BaseAsyncClient):
    api_key: str
    organization_id: str | None
    project_id: str | None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> None:
        """Construct a new asynchronous Atlas client instance.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `LAYERLENS_ATLAS_API_KEY`
        """
        if api_key is None:
            api_key = os.environ.get("LAYERLENS_ATLAS_API_KEY")
        if api_key is None:
            raise AtlasError(
                "The api_key client option must be set either by passing api_key to the client "
                "or by setting the LAYERLENS_ATLAS_API_KEY environment variable"
            )
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get("LAYERLENS_ATLAS_BASE_URL")
        if base_url is None:
            base_url = "https://8bg48mbhyi.execute-api.us-east-1.amazonaws.com/prod/api/v1"

        super().__init__(base_url=base_url, timeout=timeout)

        # org/project must be fetched asynchronously
        self.organization_id = None
        self.project_id = None

    @classmethod
    async def create(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> "AsyncAtlas":
        """Async factory that combines __init__ and ainit into one call."""
        self = cls(api_key=api_key, base_url=base_url, timeout=timeout)
        organization = await self._get_organization()
        if organization is None:
            raise AtlasError("Organization could not be fetched. Please contact LayerLens Atlas support.")
        self.organization_id = organization.id

        if not organization.projects:
            raise AtlasError(
                f"Organization {self.organization_id} is missing project. Please contact LayerLens Atlas support."
            )
        self.project_id = organization.projects[0].id
        return self

    @cached_property
    def benchmarks(self) -> AsyncBenchmarks:
        from .resources.benchmarks import AsyncBenchmarks

        return AsyncBenchmarks(self)

    @cached_property
    def evaluations(self) -> AsyncEvaluations:
        from .resources.evaluations import AsyncEvaluations

        return AsyncEvaluations(self)

    @cached_property
    def models(self) -> AsyncModels:
        from .resources.models import AsyncModels

        return AsyncModels(self)

    @cached_property
    def results(self) -> AsyncResults:
        from .resources.results import AsyncResults

        return AsyncResults(self)

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

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

    # Alias for nicer inline usage
    with_options = copy

    def _make_status_error(
        self,
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

    async def _get_organization(self) -> Optional[Organization]:
        organization = await super().get_cast(
            "/organizations",
            timeout=30,
            cast_to=OrganizationResponse,
        )
        if isinstance(organization, OrganizationResponse):
            return organization.data
        return None


Client = Atlas
AsyncClient = AsyncAtlas
