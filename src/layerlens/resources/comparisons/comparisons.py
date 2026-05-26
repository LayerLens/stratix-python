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


def _require_one_of(id_value: Optional[str], key_value: Optional[str], id_name: str, key_name: str) -> None:
    if (id_value is None) == (key_value is None):
        raise ValueError(f"Exactly one of '{id_name}' or '{key_name}' must be provided.")


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
        benchmark_id: Optional[str] = None,
        model_id_1: Optional[str] = None,
        model_id_2: Optional[str] = None,
        benchmark_key: Optional[str] = None,
        model_key_1: Optional[str] = None,
        model_key_2: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        outcome_filter: Optional[_OUTCOME_FILTER] = None,
        search: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[ComparisonResponse]:
        """Compare two models on a benchmark by automatically finding their evaluations.

        Each of the benchmark and the two models can be addressed by either its
        ID or its unique key — provide one of ``benchmark_id``/``benchmark_key``,
        one of ``model_id_1``/``model_key_1``, and one of ``model_id_2``/``model_key_2``.

        Finds the most recent successful evaluation for each model on the given
        benchmark, then compares the results side-by-side.

        Raises:
            ValueError: If both ID and key are provided for the same entity,
                neither is provided, a key cannot be resolved, or no successful
                evaluation is found for either model.
        """
        _require_one_of(benchmark_id, benchmark_key, "benchmark_id", "benchmark_key")
        _require_one_of(model_id_1, model_key_1, "model_id_1", "model_key_1")
        _require_one_of(model_id_2, model_key_2, "model_id_2", "model_key_2")

        resolved_benchmark_id = benchmark_id or self._resolve_benchmark_key(benchmark_key, timeout=timeout)
        resolved_model_id_1 = model_id_1 or self._resolve_model_key(model_key_1, timeout=timeout)
        resolved_model_id_2 = model_id_2 or self._resolve_model_key(model_key_2, timeout=timeout)

        resp1 = self._client.evaluations.get_many(
            model_ids=[resolved_model_id_1],
            benchmark_ids=[resolved_benchmark_id],
            status=EvaluationStatus.SUCCESS,
            sort_by="submitted_at",
            order="desc",
            page_size=1,
            unique=True,
            timeout=timeout,
        )
        eval_id_1 = _find_evaluation_id(resp1, resolved_model_id_1, resolved_benchmark_id)

        resp2 = self._client.evaluations.get_many(
            model_ids=[resolved_model_id_2],
            benchmark_ids=[resolved_benchmark_id],
            status=EvaluationStatus.SUCCESS,
            sort_by="submitted_at",
            order="desc",
            page_size=1,
            unique=True,
            timeout=timeout,
        )
        eval_id_2 = _find_evaluation_id(resp2, resolved_model_id_2, resolved_benchmark_id)

        return self.compare(
            evaluation_id_1=eval_id_1,
            evaluation_id_2=eval_id_2,
            page=page,
            page_size=page_size,
            outcome_filter=outcome_filter,
            search=search,
            timeout=timeout,
        )

    def _resolve_model_key(self, key: Optional[str], *, timeout: float | httpx.Timeout | None) -> str:
        resp = self._client.models.get(key=key, timeout=timeout)
        if resp is not None:
            for model in resp.models:
                if model.key == key:
                    return str(model.id)
        raise ValueError(f"No model found for key '{key}'")

    def _resolve_benchmark_key(self, key: Optional[str], *, timeout: float | httpx.Timeout | None) -> str:
        resp = self._client.benchmarks.get(key=key, timeout=timeout)
        if resp is not None:
            for benchmark in resp.datasets:
                if benchmark.key == key:
                    return str(benchmark.id)
        raise ValueError(f"No benchmark found for key '{key}'")


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
        benchmark_id: Optional[str] = None,
        model_id_1: Optional[str] = None,
        model_id_2: Optional[str] = None,
        benchmark_key: Optional[str] = None,
        model_key_1: Optional[str] = None,
        model_key_2: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        outcome_filter: Optional[_OUTCOME_FILTER] = None,
        search: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[ComparisonResponse]:
        """Compare two models on a benchmark by automatically finding their evaluations.

        Each of the benchmark and the two models can be addressed by either its
        ID or its unique key — provide one of ``benchmark_id``/``benchmark_key``,
        one of ``model_id_1``/``model_key_1``, and one of ``model_id_2``/``model_key_2``.

        Finds the most recent successful evaluation for each model on the given
        benchmark, then compares the results side-by-side.

        Raises:
            ValueError: If both ID and key are provided for the same entity,
                neither is provided, a key cannot be resolved, or no successful
                evaluation is found for either model.
        """
        _require_one_of(benchmark_id, benchmark_key, "benchmark_id", "benchmark_key")
        _require_one_of(model_id_1, model_key_1, "model_id_1", "model_key_1")
        _require_one_of(model_id_2, model_key_2, "model_id_2", "model_key_2")

        resolved_benchmark_id = benchmark_id or await self._resolve_benchmark_key(benchmark_key, timeout=timeout)
        resolved_model_id_1 = model_id_1 or await self._resolve_model_key(model_key_1, timeout=timeout)
        resolved_model_id_2 = model_id_2 or await self._resolve_model_key(model_key_2, timeout=timeout)

        resp1 = await self._client.evaluations.get_many(
            model_ids=[resolved_model_id_1],
            benchmark_ids=[resolved_benchmark_id],
            status=EvaluationStatus.SUCCESS,
            sort_by="submitted_at",
            order="desc",
            page_size=1,
            unique=True,
            timeout=timeout,
        )
        eval_id_1 = _find_evaluation_id(resp1, resolved_model_id_1, resolved_benchmark_id)

        resp2 = await self._client.evaluations.get_many(
            model_ids=[resolved_model_id_2],
            benchmark_ids=[resolved_benchmark_id],
            status=EvaluationStatus.SUCCESS,
            sort_by="submitted_at",
            order="desc",
            page_size=1,
            unique=True,
            timeout=timeout,
        )
        eval_id_2 = _find_evaluation_id(resp2, resolved_model_id_2, resolved_benchmark_id)

        return await self.compare(
            evaluation_id_1=eval_id_1,
            evaluation_id_2=eval_id_2,
            page=page,
            page_size=page_size,
            outcome_filter=outcome_filter,
            search=search,
            timeout=timeout,
        )

    async def _resolve_model_key(self, key: Optional[str], *, timeout: float | httpx.Timeout | None) -> str:
        resp = await self._client.models.get(key=key, timeout=timeout)
        if resp is not None:
            for model in resp.models:
                if model.key == key:
                    return str(model.id)
        raise ValueError(f"No model found for key '{key}'")

    async def _resolve_benchmark_key(self, key: Optional[str], *, timeout: float | httpx.Timeout | None) -> str:
        resp = await self._client.benchmarks.get(key=key, timeout=timeout)
        if resp is not None:
            for benchmark in resp.datasets:
                if benchmark.key == key:
                    return str(benchmark.id)
        raise ValueError(f"No benchmark found for key '{key}'")
