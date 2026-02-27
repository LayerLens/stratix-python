from __future__ import annotations

from typing import Literal, Optional

import httpx

from ...models import ComparisonResponse
from ..._resource import SyncPublicAPIResource, AsyncPublicAPIResource
from ..._constants import DEFAULT_TIMEOUT


class Comparisons(SyncPublicAPIResource):
    def compare(
        self,
        *,
        evaluation_id_1: str,
        evaluation_id_2: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        outcome_filter: Optional[
            Literal["all", "both_succeed", "both_fail", "reference_fails", "comparison_fails"]
        ] = None,
        search: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[ComparisonResponse]:
        params = {
            "evaluation_id_1": evaluation_id_1,
            "evaluation_id_2": evaluation_id_2,
        }
        if page is not None:
            params["page"] = str(page)
        if page_size is not None:
            params["pageSize"] = str(page_size)
        if outcome_filter:
            params["outcomeFilter"] = outcome_filter
        if search:
            params["search"] = search

        resp = self._get(
            "/results/comparison",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )

        if not isinstance(resp, dict):
            return None

        return ComparisonResponse.model_validate(resp)


class AsyncComparisons(AsyncPublicAPIResource):
    async def compare(
        self,
        *,
        evaluation_id_1: str,
        evaluation_id_2: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        outcome_filter: Optional[
            Literal["all", "both_succeed", "both_fail", "reference_fails", "comparison_fails"]
        ] = None,
        search: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[ComparisonResponse]:
        params = {
            "evaluation_id_1": evaluation_id_1,
            "evaluation_id_2": evaluation_id_2,
        }
        if page is not None:
            params["page"] = str(page)
        if page_size is not None:
            params["pageSize"] = str(page_size)
        if outcome_filter:
            params["outcomeFilter"] = outcome_filter
        if search:
            params["search"] = search

        resp = await self._get(
            "/results/comparison",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )

        if not isinstance(resp, dict):
            return None

        return ComparisonResponse.model_validate(resp)
