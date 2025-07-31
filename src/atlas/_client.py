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
from ._exceptions import AtlasError, APIStatusError
from ._base_client import BaseClient

if TYPE_CHECKING:
    from .resources.results import Results
    from .resources.evaluations import Evaluations


__all__ = ["Atlas", "Client"]


class Atlas(BaseClient):
    api_key: str
    organization_id: str | None
    project_id: str | None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization_id: str | None = None,
        project_id: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, httpx.Timeout, None] = DEFAULT_TIMEOUT,
    ) -> None:
        """Construct a new synchronous Atlas client instance.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `LAYERLENS_ATLAS_API_KEY`
        - `organization_id` from `LAYERLENS_ATLAS_ORG_ID`
        - `project_id` from `LAYERLENS_ATLAS_PROJECT_ID`
        """
        if api_key is None:
            api_key = os.environ.get("LAYERLENS_ATLAS_API_KEY")
        if api_key is None:
            raise AtlasError(
                "The api_key client option must be set either by passing api_key to the client or by setting the LAYERLENS_ATLAS_API_KEY environment variable"
            )
        self.api_key = api_key

        if organization_id is None:
            organization_id = os.environ.get("LAYERLENS_ATLAS_ORG_ID")
        self.organization_id = organization_id

        if project_id is None:
            project_id = os.environ.get("LAYERLENS_ATLAS_PROJECT_ID")
        self.project_id = project_id

        if base_url is None:
            base_url = os.environ.get("LAYERLENS_ATLAS_BASE_URL")
        if base_url is None:
            base_url = "https://8bg48mbhyi.execute-api.us-east-1.amazonaws.com/prod/api/v1/key"

        super().__init__(
            base_url=base_url,
            timeout=timeout,
        )

    @cached_property
    def evaluations(self) -> Evaluations:
        from .resources.evaluations import Evaluations

        return Evaluations(self)

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
        organization_id: str | None = None,
        project_id: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        return self.__class__(
            api_key=api_key or self.api_key,
            organization_id=organization_id or self.organization_id,
            project_id=project_id or self.project_id,
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


Client = Atlas
