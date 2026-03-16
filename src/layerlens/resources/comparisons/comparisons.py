from __future__ import annotations

from typing import Literal, Optional

import httpx

from ...models import EvaluationStatus, ComparisonResponse, EvaluationsResponse
from ..._resource import SyncPublicAPIResource, AsyncPublicAPIResource
from ..._constants import DEFAULT_TIMEOUT

_OUTCOME_FILTER = Literal["all", "both_succeed", "both_fail", "reference_fails", "comparison_fails"]


def _find_evaluation_id(response: Optional[EvaluationsResponse], model_id: str, benchmark_id: str) -> str:
    """Extract the first evaluation ID from a response, or raise ValueError."""
    if not response or not response.evaluations:
        raise ValueError(f"No successful evaluation found for model '{model_id}' on benchmark '{benchmark_id}'")
    return str(response.evaluations[0].id)


class Comparisons(SyncPublicAPIResource):
    def compare(
        self,
        *,
        evaluation_id_1: str,
        evaluation_id_2: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        outcome_filter: Optional[_OUTCOME_FILTER] = None,
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
            params["page_size"] = str(page_size)
        if outcome_filter:
            params["outcome_filter"] = outcome_filter
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

    def compare_models(
        self,
        *,
        benchmark_id: str,
        model_id_1: str,
        model_id_2: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        outcome_filter: Optional[_OUTCOME_FILTER] = None,
        search: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[ComparisonResponse]:
        """Compare two models on a benchmark by automatically finding their evaluations.

        Finds the most recent successful evaluation for each model on the given
        benchmark, then compares the results side-by-side.

        Raises:
            ValueError: If no successful evaluation is found for either model.
        """
        resp1 = self._client.evaluations.get_many(
            model_ids=[model_id_1],
            benchmark_ids=[benchmark_id],
            status=EvaluationStatus.SUCCESS,
            sort_by="submitted_at",
            order="desc",
            page_size=1,
            timeout=timeout,
        )
        eval_id_1 = _find_evaluation_id(resp1, model_id_1, benchmark_id)

        resp2 = self._client.evaluations.get_many(
            model_ids=[model_id_2],
            benchmark_ids=[benchmark_id],
            status=EvaluationStatus.SUCCESS,
            sort_by="submitted_at",
            order="desc",
            page_size=1,
            timeout=timeout,
        )
        eval_id_2 = _find_evaluation_id(resp2, model_id_2, benchmark_id)

        return self.compare(
            evaluation_id_1=eval_id_1,
            evaluation_id_2=eval_id_2,
            page=page,
            page_size=page_size,
            outcome_filter=outcome_filter,
            search=search,
            timeout=timeout,
        )


class AsyncComparisons(AsyncPublicAPIResource):
    async def compare(
        self,
        *,
        evaluation_id_1: str,
        evaluation_id_2: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        outcome_filter: Optional[_OUTCOME_FILTER] = None,
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
            params["page_size"] = str(page_size)
        if outcome_filter:
            params["outcome_filter"] = outcome_filter
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

    async def compare_models(
        self,
        *,
        benchmark_id: str,
        model_id_1: str,
        model_id_2: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        outcome_filter: Optional[_OUTCOME_FILTER] = None,
        search: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[ComparisonResponse]:
        """Compare two models on a benchmark by automatically finding their evaluations.

        Finds the most recent successful evaluation for each model on the given
        benchmark, then compares the results side-by-side.

        Raises:
            ValueError: If no successful evaluation is found for either model.
        """
        resp1 = await self._client.evaluations.get_many(
            model_ids=[model_id_1],
            benchmark_ids=[benchmark_id],
            status=EvaluationStatus.SUCCESS,
            sort_by="submitted_at",
            order="desc",
            page_size=1,
            timeout=timeout,
        )
        eval_id_1 = _find_evaluation_id(resp1, model_id_1, benchmark_id)

        resp2 = await self._client.evaluations.get_many(
            model_ids=[model_id_2],
            benchmark_ids=[benchmark_id],
            status=EvaluationStatus.SUCCESS,
            sort_by="submitted_at",
            order="desc",
            page_size=1,
            timeout=timeout,
        )
        eval_id_2 = _find_evaluation_id(resp2, model_id_2, benchmark_id)

        return await self.compare(
            evaluation_id_1=eval_id_1,
            evaluation_id_2=eval_id_2,
            page=page,
            page_size=page_size,
            outcome_filter=outcome_filter,
            search=search,
            timeout=timeout,
        )
