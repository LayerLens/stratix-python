from __future__ import annotations

import math
from typing import List, Optional

import httpx

from ..._wire import json_object, parse_model
from ...models import Result, Evaluation, ResultsResponse
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT
from ..._exceptions import StratixError

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500


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
    ) -> Optional[ResultsResponse]:
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

        Raises:
            APIResponseValidationError: the server returned 2xx but the body does
                not match the documented shape. Previously swallowed and reported
                as None, which `get_all` then read as "no more pages" — see
                `_wire`.
        """
        params = {"evaluation_id": evaluation_id}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)

        # The raw response, not cast_to=dict: pagination has to be derived from the
        # body before the envelope can be validated, and holding the response is
        # what lets a schema failure below raise with real context attached.
        response = self._get(f"/results", params=params, timeout=timeout)
        assert isinstance(response, httpx.Response), (
            "expected the raw response: this call passes no cast_to, so the transport must hand back an httpx.Response"
        )

        payload = json_object(response, endpoint="/results")

        metrics = payload.get("metrics") or {}
        total_count = metrics.get("total_count", 0) if isinstance(metrics, dict) else 0
        total_pages = math.ceil(total_count / effective_page_size) if total_count > 0 and effective_page_size > 0 else 0

        resp_with_pagination = {
            **payload,
            # A page with no matching rows arrives as `"results": null`, because the
            # API's `Results []LLMResult` has no omitempty and the SQL repositories
            # leave the slice nil when nothing matches. Read that as "no rows"
            # rather than rejecting the page.
            "results": payload.get("results") or [],
            "pagination": {
                "page": effective_page,
                "page_size": effective_page_size,
                "total_pages": total_pages,
                "total_count": total_count,
            },
        }

        return parse_model(
            ResultsResponse,
            resp_with_pagination,
            response=response,
            endpoint="/results",
            detail=f"page {effective_page} of evaluation {evaluation_id}",
        )

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

        Raises:
            APIResponseValidationError: a page did not match the documented shape.
            StratixError: the server's row count and its pages disagree, so no
                complete list can be returned.

        Never returns a short list silently. Both conditions above used to break
        the loop and hand back whatever had accumulated, which a caller could not
        distinguish from a complete result set.
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

            # Defensive: get_by_id raises rather than returning None, but its
            # declared type still permits it. Treating None as end-of-pages is the
            # exact bug being fixed, so refuse instead.
            if resp is None:
                raise StratixError(
                    f"/results page {current_page} for evaluation {evaluation_id} returned no "
                    f"usable body after {len(all_results)} results; refusing to return a "
                    "partial list as if it were complete"
                )

            all_results.extend(resp.results)

            if resp.pagination.page >= resp.pagination.total_pages:
                break

            # An empty page before the last one means total_count and the rows
            # disagree. Breaking here is what silently truncated the list; there is
            # no honest way to call the result complete.
            if not resp.results:
                raise StratixError(
                    f"/results page {current_page} of {resp.pagination.total_pages} for evaluation "
                    f"{evaluation_id} returned 0 rows, but the server reports "
                    f"{resp.pagination.total_count} results in total and only "
                    f"{len(all_results)} have been read; refusing to return a partial list as if "
                    "it were complete"
                )

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
    ) -> Optional[ResultsResponse]:
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

        Raises:
            APIResponseValidationError: the server returned 2xx but the body does
                not match the documented shape.
        """
        params = {"evaluation_id": evaluation_id}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)

        # See the sync twin: the raw response is what lets a schema failure below
        # raise with the payload and field paths attached.
        response = await self._get(f"/results", params=params, timeout=timeout)
        assert isinstance(response, httpx.Response), (
            "expected the raw response: this call passes no cast_to, so the transport must hand back an httpx.Response"
        )

        payload = json_object(response, endpoint="/results")

        metrics = payload.get("metrics") or {}
        total_count = metrics.get("total_count", 0) if isinstance(metrics, dict) else 0
        total_pages = math.ceil(total_count / effective_page_size) if total_count > 0 and effective_page_size > 0 else 0

        resp_with_pagination = {
            **payload,
            # `"results": null` is what an empty page looks like on the wire.
            "results": payload.get("results") or [],
            "pagination": {
                "page": effective_page,
                "page_size": effective_page_size,
                "total_pages": total_pages,
                "total_count": total_count,
            },
        }

        return parse_model(
            ResultsResponse,
            resp_with_pagination,
            response=response,
            endpoint="/results",
            detail=f"page {effective_page} of evaluation {evaluation_id}",
        )

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

        Raises:
            APIResponseValidationError: a page did not match the documented shape.
            StratixError: the server's row count and its pages disagree.

        Never returns a short list silently — see the sync twin.
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

            if resp is None:
                raise StratixError(
                    f"/results page {current_page} for evaluation {evaluation_id} returned no "
                    f"usable body after {len(all_results)} results; refusing to return a "
                    "partial list as if it were complete"
                )

            all_results.extend(resp.results)

            if resp.pagination.page >= resp.pagination.total_pages:
                break

            if not resp.results:
                raise StratixError(
                    f"/results page {current_page} of {resp.pagination.total_pages} for evaluation "
                    f"{evaluation_id} returned 0 rows, but the server reports "
                    f"{resp.pagination.total_count} results in total and only "
                    f"{len(all_results)} have been read; refusing to return a partial list as if "
                    "it were complete"
                )

            current_page += 1

        return all_results
