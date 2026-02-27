from __future__ import annotations

from typing import List, Literal, Optional

import httpx

from ...models import PublicModelsListResponse
from ..._resource import SyncPublicAPIResource, AsyncPublicAPIResource
from ..._constants import DEFAULT_TIMEOUT


class PublicModelsResource(SyncPublicAPIResource):
    def get(
        self,
        *,
        query: Optional[str] = None,
        name: Optional[str] = None,
        key: Optional[str] = None,
        ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        companies: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        licenses: Optional[List[str]] = None,
        sizes: Optional[List[str]] = None,
        sort_by: Optional[
            Literal["name", "createdAt", "releasedAt", "architectureType", "contextLength", "license", "region"]
        ] = None,
        order: Optional[Literal["asc", "desc"]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        include_deprecated: Optional[bool] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[PublicModelsListResponse]:
        params = {}
        if query:
            params["query"] = query
        if name:
            params["name"] = name
        if key:
            params["key"] = key
        if ids:
            params["ids"] = ",".join(ids)
        if categories:
            params["categories"] = ",".join(categories)
        if companies:
            params["companies"] = ",".join(companies)
        if regions:
            params["regions"] = ",".join(regions)
        if licenses:
            params["licenses"] = ",".join(licenses)
        if sizes:
            params["sizes"] = ",".join(sizes)
        if sort_by:
            params["sortBy"] = sort_by
        if order:
            params["order"] = order
        if page is not None:
            params["page"] = str(page)
        if page_size is not None:
            params["pageSize"] = str(page_size)
        if include_deprecated is not None:
            params["include_deprecated"] = str(include_deprecated).lower()

        resp = self._get(
            "/models",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )

        if not isinstance(resp, dict):
            return None

        return PublicModelsListResponse.model_validate(resp)


class AsyncPublicModelsResource(AsyncPublicAPIResource):
    async def get(
        self,
        *,
        query: Optional[str] = None,
        name: Optional[str] = None,
        key: Optional[str] = None,
        ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        companies: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        licenses: Optional[List[str]] = None,
        sizes: Optional[List[str]] = None,
        sort_by: Optional[
            Literal["name", "createdAt", "releasedAt", "architectureType", "contextLength", "license", "region"]
        ] = None,
        order: Optional[Literal["asc", "desc"]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        include_deprecated: Optional[bool] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[PublicModelsListResponse]:
        params = {}
        if query:
            params["query"] = query
        if name:
            params["name"] = name
        if key:
            params["key"] = key
        if ids:
            params["ids"] = ",".join(ids)
        if categories:
            params["categories"] = ",".join(categories)
        if companies:
            params["companies"] = ",".join(companies)
        if regions:
            params["regions"] = ",".join(regions)
        if licenses:
            params["licenses"] = ",".join(licenses)
        if sizes:
            params["sizes"] = ",".join(sizes)
        if sort_by:
            params["sortBy"] = sort_by
        if order:
            params["order"] = order
        if page is not None:
            params["page"] = str(page)
        if page_size is not None:
            params["pageSize"] = str(page_size)
        if include_deprecated is not None:
            params["include_deprecated"] = str(include_deprecated).lower()

        resp = await self._get(
            "/models",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )

        if not isinstance(resp, dict):
            return None

        return PublicModelsListResponse.model_validate(resp)
