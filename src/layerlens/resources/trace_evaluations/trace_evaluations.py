from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from ...models import (
    TraceEvaluation,
    CostEstimateResponse,
    TraceEvaluationResult,
    TraceEvaluationsResponse,
    TraceEvaluationResultsResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


def _unwrap(resp: Any) -> Any:
    """Unwrap {"status": ..., "data": ...} envelope if present."""
    if isinstance(resp, dict) and "data" in resp and "status" in resp:
        return resp["data"]
    return resp


class TraceEvaluations(SyncAPIResource):
    def _base_url(self) -> str:
        return f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/trace-evaluations"

    def create(
        self,
        *,
        trace_id: str,
        judge_id: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[TraceEvaluation]:
        resp = self._post(
            self._base_url(),
            body={"trace_id": trace_id, "judge_id": judge_id},
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return TraceEvaluation(**data)
            except Exception:
                return None
        return None

    def get(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[TraceEvaluation]:
        resp = self._get(
            f"{self._base_url()}/{id}",
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return TraceEvaluation(**data)
            except Exception:
                return None
        return None

    def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        judge_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        outcome: Optional[str] = None,
        time_range: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[TraceEvaluationsResponse]:
        params: Dict[str, Any] = {}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)

        if judge_id is not None:
            params["judgeId"] = judge_id
        if trace_id is not None:
            params["traceId"] = trace_id
        if outcome is not None:
            params["outcome"] = outcome
        if time_range is not None:
            params["timeRange"] = time_range
        if search is not None:
            params["search"] = search
        if sort_by is not None:
            params["sortBy"] = sort_by
        if sort_order is not None:
            params["sortOrder"] = sort_order

        resp = self._get(
            self._base_url(),
            params=params,
            timeout=timeout,
            cast_to=dict,
        )
        if not resp or not isinstance(resp, dict):
            return None

        data = _unwrap(resp)
        if not isinstance(data, dict):
            return None

        evaluations = [
            te if isinstance(te, TraceEvaluation) else TraceEvaluation(**te) for te in data.get("trace_evaluations", [])
        ]
        count: int = data.get("count", len(evaluations))
        total: int = data.get("total", count)

        try:
            return TraceEvaluationsResponse(trace_evaluations=evaluations, count=count, total=total)
        except Exception:
            return None

    def get_results(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[TraceEvaluationResultsResponse]:
        resp = self._get(
            f"{self._base_url()}/{id}/results",
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if not data or not isinstance(data, dict):
            return None

        results = [
            r if isinstance(r, TraceEvaluationResult) else TraceEvaluationResult(**r) for r in data.get("results", [])
        ]

        try:
            return TraceEvaluationResultsResponse(results=results)
        except Exception:
            return None

    def estimate_cost(
        self,
        *,
        trace_ids: List[str],
        judge_id: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[CostEstimateResponse]:
        resp = self._post(
            f"{self._base_url()}/estimate",
            body={"trace_ids": trace_ids, "judge_id": judge_id},
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return CostEstimateResponse(**data)
            except Exception:
                return None
        return None


class AsyncTraceEvaluations(AsyncAPIResource):
    def _base_url(self) -> str:
        return f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/trace-evaluations"

    async def create(
        self,
        *,
        trace_id: str,
        judge_id: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[TraceEvaluation]:
        resp = await self._post(
            self._base_url(),
            body={"trace_id": trace_id, "judge_id": judge_id},
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return TraceEvaluation(**data)
            except Exception:
                return None
        return None

    async def get(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[TraceEvaluation]:
        resp = await self._get(
            f"{self._base_url()}/{id}",
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return TraceEvaluation(**data)
            except Exception:
                return None
        return None

    async def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        judge_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        outcome: Optional[str] = None,
        time_range: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[TraceEvaluationsResponse]:
        params: Dict[str, Any] = {}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)

        if judge_id is not None:
            params["judgeId"] = judge_id
        if trace_id is not None:
            params["traceId"] = trace_id
        if outcome is not None:
            params["outcome"] = outcome
        if time_range is not None:
            params["timeRange"] = time_range
        if search is not None:
            params["search"] = search
        if sort_by is not None:
            params["sortBy"] = sort_by
        if sort_order is not None:
            params["sortOrder"] = sort_order

        resp = await self._get(
            self._base_url(),
            params=params,
            timeout=timeout,
            cast_to=dict,
        )
        if not resp or not isinstance(resp, dict):
            return None

        data = _unwrap(resp)
        if not isinstance(data, dict):
            return None

        evaluations = [
            te if isinstance(te, TraceEvaluation) else TraceEvaluation(**te) for te in data.get("trace_evaluations", [])
        ]
        count: int = data.get("count", len(evaluations))
        total: int = data.get("total", count)

        try:
            return TraceEvaluationsResponse(trace_evaluations=evaluations, count=count, total=total)
        except Exception:
            return None

    async def get_results(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[TraceEvaluationResultsResponse]:
        resp = await self._get(
            f"{self._base_url()}/{id}/results",
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if not data or not isinstance(data, dict):
            return None

        results = [
            r if isinstance(r, TraceEvaluationResult) else TraceEvaluationResult(**r) for r in data.get("results", [])
        ]

        try:
            return TraceEvaluationResultsResponse(results=results)
        except Exception:
            return None

    async def estimate_cost(
        self,
        *,
        trace_ids: List[str],
        judge_id: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[CostEstimateResponse]:
        resp = await self._post(
            f"{self._base_url()}/estimate",
            body={"trace_ids": trace_ids, "judge_id": judge_id},
            timeout=timeout,
            cast_to=dict,
        )
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return CostEstimateResponse(**data)
            except Exception:
                return None
        return None
