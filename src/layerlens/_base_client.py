from __future__ import annotations

import json
import logging
from typing import Any, Dict, Type, Union, TypeVar, Optional

import httpx
from httpx import URL

from . import _exceptions
from ._utils import SensitiveHeadersFilter

ResponseT = TypeVar("ResponseT")


log: logging.Logger = logging.getLogger(__name__)
log.addFilter(SensitiveHeadersFilter())


class BaseClient(httpx.Client):
    def __init__(
        self,
        *,
        base_url: URL | str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        **kwargs: Any,
    ):
        super().__init__(base_url=base_url, headers=headers, timeout=timeout, **kwargs)

    @property
    def auth_headers(self) -> dict[str, str]:
        return {}

    @property
    def default_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **self.auth_headers,
        }

    def _request_cast(
        self,
        method: str,
        url: str,
        *,
        cast_to: Optional[Type[ResponseT]] = None,
        body: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[ResponseT, httpx.Response]:
        combined_headers = {**self.default_headers, **(headers or {})}

        response = super().request(
            method=method,
            url=url,
            json=body,
            params=params,
            headers=combined_headers,
            **kwargs,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            log.debug("Encountered httpx.HTTPStatusError", exc_info=True)
            log.debug("Re-raising status error")
            raise self._make_status_error_from_response(err.response) from None

        if cast_to:
            data = response.json()
            return cast_to(**data)
        return response

    def get_cast(
        self,
        url: str,
        *,
        cast_to: Optional[Type[ResponseT]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[ResponseT, httpx.Response]:
        return self._request_cast("GET", url, cast_to=cast_to, params=params, headers=headers, **kwargs)

    def post_cast(
        self,
        url: str,
        *,
        cast_to: Optional[Type[ResponseT]] = None,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[ResponseT, httpx.Response]:
        return self._request_cast("POST", url, cast_to=cast_to, body=body, headers=headers, **kwargs)

    def patch_cast(
        self,
        url: str,
        *,
        cast_to: Optional[Type[ResponseT]] = None,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[ResponseT, httpx.Response]:
        return self._request_cast("PATCH", url, cast_to=cast_to, body=body, headers=headers, **kwargs)

    def delete_cast(
        self,
        url: str,
        *,
        cast_to: Optional[Type[ResponseT]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[ResponseT, httpx.Response]:
        return self._request_cast("DELETE", url, cast_to=cast_to, params=params, headers=headers, **kwargs)

    def _make_status_error_from_response(
        self,
        response: httpx.Response,
    ) -> _exceptions.APIStatusError:
        err_text = response.text.strip()
        body = err_text

        try:
            body = json.loads(err_text)
            err_msg = f"Error code: {response.status_code} - {body}"
        except Exception:
            err_msg = err_text or f"Error code: {response.status_code}"

        return self._make_status_error(err_msg, body=body, response=response)

    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> _exceptions.APIStatusError:
        raise NotImplementedError()


class BaseAsyncClient(httpx.AsyncClient):
    def __init__(
        self,
        *,
        base_url: URL | str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        **kwargs: Any,
    ):
        super().__init__(base_url=base_url, headers=headers, timeout=timeout, **kwargs)

    @property
    def auth_headers(self) -> dict[str, str]:
        return {}

    @property
    def default_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **self.auth_headers,
        }

    async def _request_cast(
        self,
        method: str,
        url: str,
        *,
        cast_to: Optional[Type[ResponseT]] = None,
        body: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[ResponseT, httpx.Response]:
        combined_headers = {**self.default_headers, **(headers or {})}

        response = await super().request(
            method=method,
            url=url,
            json=body,
            params=params,
            headers=combined_headers,
            **kwargs,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            log.debug("Encountered httpx.HTTPStatusError", exc_info=True)
            log.debug("Re-raising status error")
            raise self._make_status_error_from_response(err.response) from None

        if cast_to:
            data = response.json()
            return cast_to(**data)
        return response

    async def get_cast(
        self,
        url: str,
        *,
        cast_to: Optional[Type[ResponseT]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[ResponseT, httpx.Response]:
        return await self._request_cast("GET", url, cast_to=cast_to, params=params, headers=headers, **kwargs)

    async def post_cast(
        self,
        url: str,
        *,
        cast_to: Optional[Type[ResponseT]] = None,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[ResponseT, httpx.Response]:
        return await self._request_cast("POST", url, cast_to=cast_to, body=body, headers=headers, **kwargs)

    async def patch_cast(
        self,
        url: str,
        *,
        cast_to: Optional[Type[ResponseT]] = None,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[ResponseT, httpx.Response]:
        return await self._request_cast("PATCH", url, cast_to=cast_to, body=body, headers=headers, **kwargs)

    async def delete_cast(
        self,
        url: str,
        *,
        cast_to: Optional[Type[ResponseT]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Union[ResponseT, httpx.Response]:
        return await self._request_cast("DELETE", url, cast_to=cast_to, params=params, headers=headers, **kwargs)

    def _make_status_error_from_response(
        self,
        response: httpx.Response,
    ) -> _exceptions.APIStatusError:
        err_text = response.text.strip()
        body = err_text

        try:
            body = json.loads(err_text)
            err_msg = f"Error code: {response.status_code} - {body}"
        except Exception:
            err_msg = err_text or f"Error code: {response.status_code}"

        return self._make_status_error(err_msg, body=body, response=response)

    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> _exceptions.APIStatusError:
        raise NotImplementedError()
