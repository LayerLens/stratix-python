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
from ._constants import DEFAULT_TIMEOUT, DEFAULT_BASE_URL
from ._exceptions import StratixError, APIStatusError
from ._base_client import BaseClient, BaseAsyncClient

if TYPE_CHECKING:
    from ._public_client import PublicClient, AsyncPublicClient
    from .resources.judges import Judges, AsyncJudges
    from .resources.models import Models, AsyncModels
    from .resources.traces import Traces, AsyncTraces
    from .resources.results import Results, AsyncResults
    from .resources.benchmarks import Benchmarks, AsyncBenchmarks
    from .resources.evaluations import Evaluations, AsyncEvaluations
    from .resources.trace_evaluations import TraceEvaluations, AsyncTraceEvaluations
    from .resources.judge_optimizations import JudgeOptimizations, AsyncJudgeOptimizations


__all__ = ["Stratix", "Client"]


class Stratix(BaseClient):
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
        """Construct a new synchronous Stratix client instance.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `LAYERLENS_STRATIX_API_KEY`
        """
        if api_key is None:
            api_key = os.environ.get("LAYERLENS_STRATIX_API_KEY") or os.environ.get("LAYERLENS_ATLAS_API_KEY")
        if api_key is None or api_key == "":
            raise StratixError(
                "The api_key client option must be set either by passing api_key to the client or by setting the LAYERLENS_STRATIX_API_KEY environment variable"
            )
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get("LAYERLENS_STRATIX_BASE_URL") or os.environ.get("LAYERLENS_ATLAS_BASE_URL")
        if base_url is None:
            base_url = DEFAULT_BASE_URL

        super().__init__(
            base_url=base_url,
            timeout=timeout,
        )

        organization = self._get_organization()
        if organization is None:
            raise StratixError(f"Organization could not be fetched. Please contact LayerLens Stratix support.")
        self.organization_id = organization.id

        if organization.projects is None or len(organization.projects) == 0:
            raise StratixError(
                f"Organization {self.organization_id} is missing project. Please contact LayerLens Stratix support."
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
    def judges(self) -> Judges:
        from .resources.judges import Judges

        return Judges(self)

    @cached_property
    def judge_optimizations(self) -> JudgeOptimizations:
        from .resources.judge_optimizations import JudgeOptimizations

        return JudgeOptimizations(self)

    @cached_property
    def models(self) -> Models:
        from .resources.models import Models

        return Models(self)

    @cached_property
    def results(self) -> Results:
        from .resources.results import Results

        return Results(self)

    @cached_property
    def traces(self) -> Traces:
        from .resources.traces import Traces

        return Traces(self)

    @cached_property
    def trace_evaluations(self) -> TraceEvaluations:
        from .resources.trace_evaluations import TraceEvaluations

        return TraceEvaluations(self)

    @cached_property
    def public(self) -> PublicClient:
        from ._public_client import PublicClient

        return PublicClient(api_key=self.api_key, base_url=str(self.base_url), timeout=self.timeout)

    @property
    @override
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
        return organization.data if isinstance(organization, OrganizationResponse) else None


class AsyncStratix(BaseAsyncClient):
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
        """Construct a new asynchronous Stratix client instance.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `LAYERLENS_STRATIX_API_KEY`
        """
        if api_key is None:
            api_key = os.environ.get("LAYERLENS_STRATIX_API_KEY") or os.environ.get("LAYERLENS_ATLAS_API_KEY")
        if api_key is None or api_key == "":
            raise StratixError(
                "The api_key client option must be set either by passing api_key to the client "
                "or by setting the LAYERLENS_STRATIX_API_KEY environment variable"
            )
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get("LAYERLENS_STRATIX_BASE_URL") or os.environ.get("LAYERLENS_ATLAS_BASE_URL")
        if base_url is None:
            base_url = DEFAULT_BASE_URL

        super().__init__(base_url=base_url, timeout=timeout)

        organization = self._get_organization()
        if organization is None:
            raise StratixError(f"Organization could not be fetched. Please contact LayerLens Stratix support.")
        self.organization_id = organization.id

        if organization.projects is None or len(organization.projects) == 0:
            raise StratixError(
                f"Organization {self.organization_id} is missing project. Please contact LayerLens Stratix support."
            )
        self.project_id = organization.projects[0].id

    @cached_property
    def benchmarks(self) -> AsyncBenchmarks:
        from .resources.benchmarks import AsyncBenchmarks

        return AsyncBenchmarks(self)

    @cached_property
    def evaluations(self) -> AsyncEvaluations:
        from .resources.evaluations import AsyncEvaluations

        return AsyncEvaluations(self)

    @cached_property
    def judges(self) -> AsyncJudges:
        from .resources.judges import AsyncJudges

        return AsyncJudges(self)

    @cached_property
    def judge_optimizations(self) -> AsyncJudgeOptimizations:
        from .resources.judge_optimizations import AsyncJudgeOptimizations

        return AsyncJudgeOptimizations(self)

    @cached_property
    def models(self) -> AsyncModels:
        from .resources.models import AsyncModels

        return AsyncModels(self)

    @cached_property
    def results(self) -> AsyncResults:
        from .resources.results import AsyncResults

        return AsyncResults(self)

    @cached_property
    def traces(self) -> AsyncTraces:
        from .resources.traces import AsyncTraces

        return AsyncTraces(self)

    @cached_property
    def trace_evaluations(self) -> AsyncTraceEvaluations:
        from .resources.trace_evaluations import AsyncTraceEvaluations

        return AsyncTraceEvaluations(self)

    @cached_property
    def public(self) -> AsyncPublicClient:
        from ._public_client import AsyncPublicClient

        return AsyncPublicClient(api_key=self.api_key, base_url=str(self.base_url), timeout=self.timeout)

    @property
    @override
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

    def _get_organization(self) -> Optional[Organization]:
        url = f"{self.base_url}organizations"

        with httpx.Client(timeout=30) as http:
            response = http.get(url, headers=self.default_headers)
            response.raise_for_status()

        data = response.json()

        organization = OrganizationResponse(**data)
        return organization.data if isinstance(organization, OrganizationResponse) else None


Client = Stratix
AsyncClient = AsyncStratix

# Backward-compatibility aliases
Atlas = Stratix
AsyncAtlas = AsyncStratix
