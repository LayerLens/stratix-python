from __future__ import annotations

import math
from typing import Optional

import httpx

from ..._resource import SyncAPIResource
from ..._constants import DEFAULT_TIMEOUT
from ...models.api import Results as ResultsData

DEFAULT_PAGE_SIZE = 100


class Results(SyncAPIResource):
    def get(
        self,
        *,
        evaluation_id: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> ResultsData | None:
        """
        Get evaluation results with optional pagination.

        Args:
            evaluation_id: The ID of the evaluation to get results for
            page: Page number for pagination (1-based, defaults to 1 if not provided)
            page_size: Number of results per page (default: 100, optional)
            timeout: Request timeout

        Returns:
            ResultsData object containing:
            - evaluation_id: The evaluation ID
            - results: List of Result objects for the current page
            - metrics: Contains total_count and score ranges
            - pagination: Calculated pagination info (total_count, page_size, total_pages)
            or None if the request fails
        """
        params = {"evaluation_id": evaluation_id}

        # Set default page_size if not provided
        effective_page_size = page_size if page_size is not None else DEFAULT_PAGE_SIZE

        # Set default page to 1 if not provided
        effective_page = page if page is not None else 1

        params["page"] = str(effective_page)
        if page_size is not None:
            params["pageSize"] = str(page_size)

        # Get the response with cast_to to get parsed data
        response_data = self._get(
            f"/results",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )

        if not response_data or not isinstance(response_data, dict):
            return None

        # Calculate pagination info
        metrics = response_data.get("metrics", {})
        total_count = metrics.get("total_count", 0)
        total_pages = math.ceil(total_count / effective_page_size) if total_count > 0 and effective_page_size > 0 else 0

        # Add pagination to the response
        response_with_pagination = {
            **response_data,
            "pagination": {
                "total_count": total_count,
                "page_size": effective_page_size,
                "total_pages": total_pages,
            },
        }

        try:
            return ResultsData.model_validate(response_with_pagination)
        except Exception:
            return None
