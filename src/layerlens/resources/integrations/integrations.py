from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from ...models import (
    Integration,
    IntegrationsResponse,
    TestIntegrationResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500


def _unwrap(resp: Any) -> Any:
    """Unwrap {"status": ..., "data": ...} envelope if present."""
    if isinstance(resp, dict) and "data" in resp and "status" in resp:
        return resp["data"]
    return resp


class Integrations(SyncAPIResource):
    def _base_url(self) -> str:
        return f"/organizations/{self._client.organization_id}/integrations"

    def get(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Integration]:
        resp = self._get(
            f"{self._base_url()}/{id}",
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return Integration(**data)
            except Exception:
                return None
        return None

    def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[IntegrationsResponse]:
        params: Dict[str, Any] = {}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)

        resp = self._get(
            self._base_url(),
            params=params,
            timeout=timeout,
            cast_to=dict,
        )
        if not resp or not isinstance(resp, dict):
            return None

        data = _unwrap(resp)

        # The API returns the integrations array directly (wrapped in the
        # standard {"status": ..., "data": [...]} envelope).  After unwrapping,
        # ``data`` is a list of integration dicts.
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("integrations", [])
        else:
            return None

        integrations = [i if isinstance(i, Integration) else Integration(**i) for i in items]
        count = len(integrations)
        total_count = count

        try:
            return IntegrationsResponse(integrations=integrations, count=count, total_count=total_count)
        except Exception:
            return None

    def test(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[TestIntegrationResponse]:
        resp = self._post(
            f"{self._base_url()}/{id}/test",
            body={},
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return TestIntegrationResponse(**data)
            except Exception:
                return None
        return None


class AsyncIntegrations(AsyncAPIResource):
    def _base_url(self) -> str:
        return f"/organizations/{self._client.organization_id}/integrations"

    async def get(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Integration]:
        resp = await self._get(
            f"{self._base_url()}/{id}",
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return Integration(**data)
            except Exception:
                return None
        return None

    async def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[IntegrationsResponse]:
        params: Dict[str, Any] = {}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)

        resp = await self._get(
            self._base_url(),
            params=params,
            timeout=timeout,
            cast_to=dict,
        )
        if not resp or not isinstance(resp, dict):
            return None

        data = _unwrap(resp)

        # The API returns the integrations array directly (wrapped in the
        # standard {"status": ..., "data": [...]} envelope).  After unwrapping,
        # ``data`` is a list of integration dicts.
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("integrations", [])
        else:
            return None

        integrations = [i if isinstance(i, Integration) else Integration(**i) for i in items]
        count = len(integrations)
        total_count = count

        try:
            return IntegrationsResponse(integrations=integrations, count=count, total_count=total_count)
        except Exception:
            return None

    async def test(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[TestIntegrationResponse]:
        resp = await self._post(
            f"{self._base_url()}/{id}/test",
            body={},
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return TestIntegrationResponse(**data)
            except Exception:
                return None
        return None
