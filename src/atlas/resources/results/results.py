from __future__ import annotations

import math
from typing import List, Optional

import httpx

from ...models import Result, Evaluation, ResultsResponse
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
        timeout: Optional[float | httpx.Timeout] = DEFAULT_TIMEOUT,
    ) -> ResultsResponse | None:
        """
        Get evaluation results with optional pagination.

        Args:
            evaluation: evaluation to get the results for
            page: Page number for pagination (1-based, defaults to 1 if not provided)
            page_size: Number of results per page (default: 100, optional)
            timeout: Request timeout

        Returns:
            ResultsResponse object containing:
            - evaluation_id: The evaluation ID
            - results: List of Result objects for the current page
            - metrics: Contains total_count and score ranges
            - pagination: Calculated pagination info
            or None if the request fails
        """
        return self.get_by_id(evaluation_id=evaluation.id, page=page, page_size=page_size, timeout=timeout)

    def get_by_id(
        self,
        *,
        evaluation_id: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: Optional[float | httpx.Timeout] = DEFAULT_TIMEOUT,
    ) -> ResultsResponse | None:
        """
        Get evaluation results with optional pagination.

        Args:
            evaluation_id: ID of evaluation to get the results for
            page: Page number for pagination (1-based, defaults to 1 if not provided)
            page_size: Number of results per page (default: 100, optional)
            timeout: Request timeout

        Returns:
            ResultsResponse object containing:
            - evaluation_id: The evaluation ID
            - results: List of Result objects for the current page
            - metrics: Contains total_count and score ranges
            - pagination: Calculated pagination info
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
                "page": effective_page,
                "page_size": effective_page_size,
                "total_pages": total_pages,
                "total_count": total_count,
            },
        }

        try:
            return ResultsResponse.model_validate(resp_with_pagination)
        except Exception:
            return None

    def get_all(
        self,
        *,
        evaluation: Evaluation,
        timeout: Optional[float | httpx.Timeout] = DEFAULT_TIMEOUT,
    ) -> List[Result]:
        """
        Fetch all results for the given evaluation by iterating over all pages.

        Args:
            evaluation: Evaluation to get the results for
            timeout: Request timeout

        Returns:
            List of all Result objects across all pages.
        """
        return self.get_all_by_id(evaluation_id=evaluation.id, timeout=timeout)

    def get_all_by_id(
        self,
        *,
        evaluation_id: str,
        timeout: Optional[float | httpx.Timeout] = DEFAULT_TIMEOUT,
    ) -> List[Result]:
        """
        Fetch all results for the given evaluation by iterating over all pages.

        Args:
            evaluation_id: ID of evaluation to get the results for
            timeout: Request timeout

        Returns:
            List of all Result objects across all pages.
        """
        all_results: List[Result] = []
        current_page = 1

        while True:
            resp = self.get_by_id(
                evaluation_id=evaluation_id,
                page=current_page,
                page_size=DEFAULT_PAGE_SIZE,
                timeout=timeout,
            )

            if resp is None or not resp.results:
                break

            all_results.extend(resp.results)

            # Stop if we reached the last page
            if resp.pagination.page >= resp.pagination.total_pages:
                break

            current_page += 1

        return all_results


class AsyncResults(AsyncAPIResource):
    async def get(
        self,
        *,
        evaluation: Evaluation,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: Optional[float | httpx.Timeout] = DEFAULT_TIMEOUT,
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
        return await self.get_by_id(evaluation_id=evaluation.id, page=page, page_size=page_size, timeout=timeout)

    async def get_by_id(
        self,
        *,
        evaluation_id: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: Optional[float | httpx.Timeout] = DEFAULT_TIMEOUT,
    ) -> ResultsResponse | None:
        """
        Get evaluation results with optional pagination.

        Args:
            evaluation_id: ID of evaluation to get the results for
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
        params["pageSize"] = str(effective_page_size)

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
                "page": effective_page,
                "page_size": effective_page_size,
                "total_pages": total_pages,
                "total_count": total_count,
            },
        }

        try:
            return ResultsResponse.model_validate(resp_with_pagination)
        except Exception:
            return None

    async def get_all(
        self,
        *,
        evaluation: Evaluation,
        timeout: Optional[float | httpx.Timeout] = DEFAULT_TIMEOUT,
    ) -> List[Result]:
        """
        Fetch all results for the given evaluation by iterating over all pages.

        Args:
            evaluation: Evaluation to get the results for
            timeout: Request timeout

        Returns:
            List of all Result objects across all pages.
        """
        return await self.get_all_by_id(evaluation_id=evaluation.id, timeout=timeout)

    async def get_all_by_id(
        self,
        *,
        evaluation_id: str,
        timeout: Optional[float | httpx.Timeout] = DEFAULT_TIMEOUT,
    ) -> List[Result]:
        """
        Fetch all results for the given evaluation by iterating over all pages.

        Args:
            evaluation_id: ID of evaluation to get the results for
            timeout: Request timeout

        Returns:
            List of all Result objects across all pages.
        """
        all_results: List[Result] = []
        current_page = 1

        while True:
            resp = await self.get_by_id(
                evaluation_id=evaluation_id,
                page=current_page,
                page_size=DEFAULT_PAGE_SIZE,
                timeout=timeout,
            )

            if resp is None or not resp.results:
                break

            all_results.extend(resp.results)

            # Stop if we reached the last page
            if resp.pagination.page >= resp.pagination.total_pages:
                break

            current_page += 1

        return all_results
