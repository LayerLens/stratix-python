from __future__ import annotations

import math
import time
import asyncio
from typing import List, Literal, Optional

import httpx

from ...models import (
    Model,
    Benchmark,
    Evaluation,
    CustomModel,
    CustomBenchmark,
    EvaluationStatus,
    EvaluationsResponse,
    CreateEvaluationsResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500


class Evaluations(SyncAPIResource):
    def create(
        self,
        *,
        model: Model,
        benchmark: Benchmark,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Evaluation]:
        evaluations = self._post(
            f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/evaluations",
            body=[
                {
                    "model_id": model.id,
                    "dataset_id": benchmark.id,
                    "is_custom_model": isinstance(model, CustomModel),
                    "is_custom_dataset": isinstance(benchmark, CustomBenchmark),
                }
            ],
            timeout=timeout,
            cast_to=CreateEvaluationsResponse,
        )
        if isinstance(evaluations, CreateEvaluationsResponse) and len(evaluations.data) > 0:
            evaluation = evaluations.data[0]
            evaluation.attach_client(self._client)
            return evaluation
        return None

    def get(
        self,
        evaluation: Evaluation,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Evaluation]:
        return self.get_by_id(evaluation.id, timeout=timeout)

    def get_by_id(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Evaluation]:
        evaluation = self._get(
            f"/evaluations/{id}",
            timeout=timeout,
            cast_to=Evaluation,
        )
        if isinstance(evaluation, Evaluation):
            evaluation.attach_client(self._client)
            return evaluation
        return None

    def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort_by: Optional[Literal["submittedAt", "accuracy", "averageDuration"]] = None,
        order: Optional[Literal["asc", "desc"]] = None,
        model_ids: Optional[List[str]] = None,
        benchmark_ids: Optional[List[str]] = None,
        status: Optional[EvaluationStatus] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[EvaluationsResponse]:
        """
        Get evaluations with optional pagination, sorting, and filtering.

        Args:
            page: Page number for pagination (1-based, defaults to 1 if not provided)
            page_size: Number of evaluations per page (default: 100, optional)
            sort_by: Sort evaluations by field (submittedAt, accuracy, averageDuration)
            order: Sort order (asc or desc)
            model_ids: Filter by model IDs
            benchmark_ids: Filter by benchmark/dataset IDs
            status: Filter by evaluation status
            timeout: Request timeout

        Returns:
            EvaluationsResponse object or None
        """
        params = {
            "organizationID": self._client.organization_id,
            "projectID": self._client.project_id,
        }

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["pageSize"] = str(effective_page_size)

        if sort_by:
            params["sortBy"] = sort_by
        if order:
            params["order"] = order
        if model_ids:
            params["models"] = ",".join(model_ids)
        if benchmark_ids:
            params["datasets"] = ",".join(benchmark_ids)
        if status:
            params["status"] = status.value

        resp = self._get(
            f"/evaluations",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )
        if not resp or not isinstance(resp, dict):
            return None

        evaluations = [e if isinstance(e, Evaluation) else Evaluation(**e) for e in resp.get("evaluations", [])]
        for e in evaluations:
            e.attach_client(self._client)

        total_count = resp.get("total_count", 0)
        total_pages = math.ceil(total_count / effective_page_size) if total_count > 0 and effective_page_size > 0 else 0

        resp_with_pagination = {
            "evaluations": evaluations,
            "pagination": {
                "page": effective_page,
                "page_size": effective_page_size,
                "total_pages": total_pages,
                "total_count": total_count,
            },
        }

        try:
            return EvaluationsResponse.model_validate(resp_with_pagination)
        except Exception:
            return None

    def wait_for_completion(
        self,
        evaluation: Evaluation,
        *,
        interval_seconds: int = 30,
        timeout_seconds: Optional[int] = None,
    ) -> Optional[Evaluation]:
        """Poll until the evaluation finishes or timeout is reached."""
        start = time.time()

        updated_evaluation: Optional[Evaluation] = self.get(evaluation)
        while updated_evaluation and not updated_evaluation.is_finished:
            if timeout_seconds and (time.time() - start) > timeout_seconds:
                raise TimeoutError(
                    f"Evaluation {updated_evaluation.id} did not complete within {timeout_seconds} seconds"
                )

            time.sleep(interval_seconds)
            updated_evaluation = self.get(updated_evaluation)

        return updated_evaluation


class AsyncEvaluations(AsyncAPIResource):
    async def create(
        self,
        *,
        model: Model,
        benchmark: Benchmark,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Evaluation]:
        evaluations = await self._post(
            f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/evaluations",
            body=[
                {
                    "model_id": model.id,
                    "dataset_id": benchmark.id,
                    "is_custom_model": isinstance(model, CustomModel),
                    "is_custom_dataset": isinstance(benchmark, CustomBenchmark),
                }
            ],
            timeout=timeout,
            cast_to=CreateEvaluationsResponse,
        )
        if isinstance(evaluations, CreateEvaluationsResponse) and len(evaluations.data) > 0:
            evaluation = evaluations.data[0]
            evaluation.attach_client(self._client)
            return evaluation
        return None

    async def get(
        self,
        evaluation: Evaluation,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Evaluation]:
        return await self.get_by_id(evaluation.id, timeout=timeout)

    async def get_by_id(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Evaluation]:
        evaluation = await self._get(
            f"/evaluations/{id}",
            timeout=timeout,
            cast_to=Evaluation,
        )
        if isinstance(evaluation, Evaluation):
            evaluation.attach_client(self._client)
            return evaluation
        return None

    async def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort_by: Optional[Literal["submittedAt", "accuracy", "averageDuration"]] = None,
        order: Optional[Literal["asc", "desc"]] = None,
        model_ids: Optional[List[str]] = None,
        benchmark_ids: Optional[List[str]] = None,
        status: Optional[EvaluationStatus] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[EvaluationsResponse]:
        """
        Get evaluations with optional pagination, sorting, and filtering.

        Args:
            page: Page number for pagination (1-based, defaults to 1 if not provided)
            page_size: Number of evaluations per page (default: 100, optional)
            sort_by: Sort evaluations by field (submittedAt, accuracy, averageDuration)
            order: Sort order (asc or desc)
            model_ids: Filter by model IDs
            benchmark_ids: Filter by benchmark/dataset IDs
            status: Filter by evaluation status
            timeout: Request timeout

        Returns:
            EvaluationsResponse object or None
        """
        params = {
            "organizationID": self._client.organization_id,
            "projectID": self._client.project_id,
        }

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["pageSize"] = str(effective_page_size)

        if sort_by:
            params["sortBy"] = sort_by
        if order:
            params["order"] = order
        if model_ids:
            params["models"] = ",".join(model_ids)
        if benchmark_ids:
            params["datasets"] = ",".join(benchmark_ids)
        if status:
            params["status"] = status.value

        resp = await self._get(
            f"/evaluations",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )
        if not resp or not isinstance(resp, dict):
            return None

        evaluations = [e if isinstance(e, Evaluation) else Evaluation(**e) for e in resp.get("evaluations", [])]
        for e in evaluations:
            e.attach_client(self._client)

        total_count = resp.get("total_count", 0)
        total_pages = math.ceil(total_count / effective_page_size) if total_count > 0 and effective_page_size > 0 else 0

        resp_with_pagination = {
            "evaluations": evaluations,
            "pagination": {
                "page": effective_page,
                "page_size": effective_page_size,
                "total_pages": total_pages,
                "total_count": total_count,
            },
        }

        try:
            return EvaluationsResponse.model_validate(resp_with_pagination)
        except Exception:
            return None

    async def wait_for_completion(
        self,
        evaluation: Evaluation,
        *,
        interval_seconds: int = 30,
        timeout_seconds: Optional[int] = None,
    ) -> Optional[Evaluation]:
        """Poll asynchronously until the evaluation finishes or timeout is reached."""
        start = asyncio.get_event_loop().time()

        updated_evaluation: Optional[Evaluation] = await self.get(evaluation)
        while updated_evaluation and not updated_evaluation.is_finished:
            if timeout_seconds and (asyncio.get_event_loop().time() - start) > timeout_seconds:
                raise TimeoutError(
                    f"Evaluation {updated_evaluation.id} did not complete within {timeout_seconds} seconds"
                )

            await asyncio.sleep(interval_seconds)
            updated_evaluation = await self.get(updated_evaluation)

        return updated_evaluation
