from __future__ import annotations

import math
from typing import List, Literal, Optional

import httpx

from ...models import (
    Evaluation,
    EvaluationStatus,
    EvaluationsResponse,
)
from ..._resource import SyncPublicAPIResource, AsyncPublicAPIResource
from ..._constants import DEFAULT_TIMEOUT

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500


class PublicEvaluationsResource(SyncPublicAPIResource):
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
            return evaluation
        return None

    def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort_by: Optional[Literal["submitted_at", "accuracy", "average_duration"]] = None,
        order: Optional[Literal["asc", "desc"]] = None,
        model_ids: Optional[List[str]] = None,
        benchmark_ids: Optional[List[str]] = None,
        status: Optional[EvaluationStatus] = None,
        unique: bool = False,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[EvaluationsResponse]:
        """
        Get evaluations with optional pagination, sorting, and filtering.

        Args:
            page: Page number for pagination (1-based, defaults to 1 if not provided)
            page_size: Number of evaluations per page (default: 100, optional)
            sort_by: Sort evaluations by field (submitted_at, accuracy, average_duration)
            order: Sort order (asc or desc)
            model_ids: Filter by model IDs
            benchmark_ids: Filter by benchmark/dataset IDs
            status: Filter by evaluation status
            unique: If True, deduplicate by model+dataset keeping only the latest evaluation per pair
            timeout: Request timeout

        Returns:
            EvaluationsResponse object or None
        """
        params: dict[str, str] = {}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)

        if sort_by:
            params["sort_by"] = sort_by
        if order:
            params["order"] = order
        if model_ids:
            params["models"] = ",".join(model_ids)
        if benchmark_ids:
            params["datasets"] = ",".join(benchmark_ids)
        if status:
            params["status"] = status.value
        if unique:
            params["unique"] = "true"

        resp = self._get(
            "/evaluations",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )
        if not resp or not isinstance(resp, dict):
            return None

        evaluations = [e if isinstance(e, Evaluation) else Evaluation(**e) for e in resp.get("evaluations", [])]

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
        except (ValueError, KeyError):
            return None


class AsyncPublicEvaluationsResource(AsyncPublicAPIResource):
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
            return evaluation
        return None

    async def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort_by: Optional[Literal["submitted_at", "accuracy", "average_duration"]] = None,
        order: Optional[Literal["asc", "desc"]] = None,
        model_ids: Optional[List[str]] = None,
        benchmark_ids: Optional[List[str]] = None,
        status: Optional[EvaluationStatus] = None,
        unique: bool = False,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[EvaluationsResponse]:
        """
        Get evaluations with optional pagination, sorting, and filtering.

        Args:
            page: Page number for pagination (1-based, defaults to 1 if not provided)
            page_size: Number of evaluations per page (default: 100, optional)
            sort_by: Sort evaluations by field (submitted_at, accuracy, average_duration)
            order: Sort order (asc or desc)
            model_ids: Filter by model IDs
            benchmark_ids: Filter by benchmark/dataset IDs
            status: Filter by evaluation status
            unique: If True, deduplicate by model+dataset keeping only the latest evaluation per pair
            timeout: Request timeout

        Returns:
            EvaluationsResponse object or None
        """
        params: dict[str, str] = {}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)

        if sort_by:
            params["sort_by"] = sort_by
        if order:
            params["order"] = order
        if model_ids:
            params["models"] = ",".join(model_ids)
        if benchmark_ids:
            params["datasets"] = ",".join(benchmark_ids)
        if status:
            params["status"] = status.value
        if unique:
            params["unique"] = "true"

        resp = await self._get(
            "/evaluations",
            params=params,
            timeout=timeout,
            cast_to=dict,
        )
        if not resp or not isinstance(resp, dict):
            return None

        evaluations = [e if isinstance(e, Evaluation) else Evaluation(**e) for e in resp.get("evaluations", [])]

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
        except (ValueError, KeyError):
            return None
