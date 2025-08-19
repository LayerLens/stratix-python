from __future__ import annotations

import math
from typing import Optional

import httpx

from ...models import Evaluation, ResultsResponse
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 100


class Results(SyncAPIResource):
    def get(
        self,
        *,
        evaluation: Evaluation,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> ResultsResponse | None:
        return self.get_by_id(evaluation_id=evaluation.id, page=page, page_size=page_size, timeout=timeout)

    def get_by_id(
        self,
        *,
        evaluation_id: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> ResultsResponse | None:
        """
        Get evaluation results with optional pagination.

        Args:
            evaluation: Evaluation to get the results for
            page: Page number for pagination (1-based, defaults to 1 if not provided)
            page_size: Number of results per page (default: 100, optional)
            timeout: Request timeout

        Returns:
            ResultsResponse object containing:
            - evaluation_id: The evaluation ID
            - results: List of Result objects for the current page
            - metrics: Contains total_count and score ranges
            - pagination: Calculated pagination info (total_count, page_size, total_pages)
            or None if the request fails
        """
        params = {"evaluation_id": evaluation_id}

        effective_page_size = page_size if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        if page_size is not None:
            params["pageSize"] = str(page_size)

        # Get the response with cast_to to get parsed data
        resp = self._get(
            f"/results",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )

        if not resp or not isinstance(resp, dict):
            return None

        # Calculate pagination info
        metrics = resp.get("metrics", {})
        total_count = metrics.get("total_count", 0)
        total_pages = math.ceil(total_count / effective_page_size) if total_count > 0 and effective_page_size > 0 else 0

        # Add pagination to the response
        resp_with_pagination = {
            **resp,
            "pagination": {
                "total_count": total_count,
                "page_size": effective_page_size,
                "total_pages": total_pages,
            },
        }

        try:
            return ResultsResponse.model_validate(resp_with_pagination)
        except Exception:
            return None


class AsyncResults(AsyncAPIResource):
    async def get(
        self,
        *,
        evaluation: Evaluation,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> ResultsResponse | None:
        return await self.get_by_id(evaluation_id=evaluation.id, page=page, page_size=page_size, timeout=timeout)

    async def get_by_id(
        self,
        *,
        evaluation_id: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> ResultsResponse | None:
        """
        Get evaluation results with optional pagination.

        Args:
            evaluation: Evaluation to get the results for
            page: Page number for pagination (1-based, defaults to 1 if not provided)
            page_size: Number of results per page (default: 100, optional)
            timeout: Request timeout

        Returns:
            ResultsResponse object containing:
            - evaluation_id: The evaluation ID
            - results: List of Result objects for the current page
            - metrics: Contains total_count and score ranges
            - pagination: Calculated pagination info (total_count, page_size, total_pages)
            or None if the request fails
        """
        params = {"evaluation_id": evaluation_id}

        effective_page_size = page_size if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        if page_size is not None:
            params["pageSize"] = str(page_size)

        # Get the response with cast_to to get parsed data
        resp = await self._get(
            f"/results",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )

        if not resp or not isinstance(resp, dict):
            return None

        # Calculate pagination info
        metrics = resp.get("metrics", {})
        total_count = metrics.get("total_count", 0)
        total_pages = math.ceil(total_count / effective_page_size) if total_count > 0 and effective_page_size > 0 else 0

        # Add pagination to the response
        resp_with_pagination = {
            **resp,
            "pagination": {
                "total_count": total_count,
                "page_size": effective_page_size,
                "total_pages": total_pages,
            },
        }

        try:
            return ResultsResponse.model_validate(resp_with_pagination)
        except Exception:
            return None
