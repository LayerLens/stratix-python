from __future__ import annotations

import math
from typing import List, Literal, Optional

import httpx

from ...models import (
    BenchmarkPrompt,
    BenchmarkPromptsResponse,
    PublicBenchmarksListResponse,
)
from ..._resource import SyncPublicAPIResource, AsyncPublicAPIResource
from ..._constants import DEFAULT_TIMEOUT

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500

DEFAULT_PROMPTS_PAGE_SIZE = 100
MAX_PROMPTS_PAGE_SIZE = 500


class PublicBenchmarksResource(SyncPublicAPIResource):
    def get(
        self,
        *,
        query: Optional[str] = None,
        name: Optional[str] = None,
        key: Optional[str] = None,
        ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        sort_by: Optional[Literal["name"]] = None,
        order: Optional[Literal["asc", "desc"]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        include_deprecated: Optional[bool] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[PublicBenchmarksListResponse]:
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
        if languages:
            params["languages"] = ",".join(languages)
        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)

        if sort_by:
            params["sort_by"] = sort_by
        if order:
            params["order"] = order
        if include_deprecated is not None:
            params["include_deprecated"] = str(include_deprecated).lower()

        resp = self._get(
            "/datasets",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )

        if not isinstance(resp, dict):
            return None

        return PublicBenchmarksListResponse.model_validate(resp)

    def get_prompts(
        self,
        benchmark_id: str,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        search_field: Optional[Literal["id", "input", "truth"]] = None,
        search_value: Optional[str] = None,
        sort_by: Optional[Literal["id", "input", "truth"]] = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[BenchmarkPromptsResponse]:
        params = {}
        if page is not None:
            params["page"] = str(page)
        if page_size is not None:
            params["page_size"] = str(page_size)
        if search_field:
            params["search"] = search_field
        if search_value:
            params["search_value"] = search_value
        if sort_by:
            params["sort_by"] = sort_by
        if sort_order:
            params["sort_order"] = sort_order

        resp = self._get(
            f"/datasets/{benchmark_id}/prompts",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )

        if not isinstance(resp, dict):
            return None

        return BenchmarkPromptsResponse.model_validate(resp)

    def get_all_prompts(
        self,
        benchmark_id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> List[BenchmarkPrompt]:
        all_prompts: List[BenchmarkPrompt] = []
        page = 1
        page_size = DEFAULT_PROMPTS_PAGE_SIZE

        while True:
            resp = self.get_prompts(
                benchmark_id,
                page=page,
                page_size=page_size,
                timeout=timeout,
            )
            if resp is None or not resp.data.prompts:
                break

            all_prompts.extend(resp.data.prompts)

            total_count = resp.data.count
            total_pages = math.ceil(total_count / page_size) if total_count > 0 else 0
            if page >= total_pages:
                break

            page += 1

        return all_prompts


class AsyncPublicBenchmarksResource(AsyncPublicAPIResource):
    async def get(
        self,
        *,
        query: Optional[str] = None,
        name: Optional[str] = None,
        key: Optional[str] = None,
        ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        sort_by: Optional[Literal["name"]] = None,
        order: Optional[Literal["asc", "desc"]] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        include_deprecated: Optional[bool] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[PublicBenchmarksListResponse]:
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
        if languages:
            params["languages"] = ",".join(languages)
        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)

        if sort_by:
            params["sort_by"] = sort_by
        if order:
            params["order"] = order
        if include_deprecated is not None:
            params["include_deprecated"] = str(include_deprecated).lower()

        resp = await self._get(
            "/datasets",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )

        if not isinstance(resp, dict):
            return None

        return PublicBenchmarksListResponse.model_validate(resp)

    async def get_prompts(
        self,
        benchmark_id: str,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        search_field: Optional[Literal["id", "input", "truth"]] = None,
        search_value: Optional[str] = None,
        sort_by: Optional[Literal["id", "input", "truth"]] = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[BenchmarkPromptsResponse]:
        params = {}
        if page is not None:
            params["page"] = str(page)
        if page_size is not None:
            params["page_size"] = str(page_size)
        if search_field:
            params["search"] = search_field
        if search_value:
            params["search_value"] = search_value
        if sort_by:
            params["sort_by"] = sort_by
        if sort_order:
            params["sort_order"] = sort_order

        resp = await self._get(
            f"/datasets/{benchmark_id}/prompts",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )

        if not isinstance(resp, dict):
            return None

        return BenchmarkPromptsResponse.model_validate(resp)

    async def get_all_prompts(
        self,
        benchmark_id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> List[BenchmarkPrompt]:
        all_prompts: List[BenchmarkPrompt] = []
        page = 1
        page_size = DEFAULT_PROMPTS_PAGE_SIZE

        while True:
            resp = await self.get_prompts(
                benchmark_id,
                page=page,
                page_size=page_size,
                timeout=timeout,
            )
            if resp is None or not resp.data.prompts:
                break

            all_prompts.extend(resp.data.prompts)

            total_count = resp.data.count
            total_pages = math.ceil(total_count / page_size) if total_count > 0 else 0
            if page >= total_pages:
                break

            page += 1

        return all_prompts
