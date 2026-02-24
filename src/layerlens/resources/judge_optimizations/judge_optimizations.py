from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from ...models import (
    JudgeOptimizationRun,
    JudgeOptimizationRunsResponse,
    CreateJudgeOptimizationRunResponse,
    ApplyJudgeOptimizationResultResponse,
    EstimateJudgeOptimizationCostResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 500


class JudgeOptimizations(SyncAPIResource):
    def _base_url(self) -> str:
        return f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/judge-optimizations"

    def estimate(
        self,
        *,
        judge_id: str,
        budget: str = "medium",
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[EstimateJudgeOptimizationCostResponse]:
        body: Dict[str, Any] = {
            "judge_id": judge_id,
            "budget": budget,
        }

        resp = self._post(
            f"{self._base_url()}/estimate",
            body=body,
            timeout=timeout,
            cast_to=EstimateJudgeOptimizationCostResponse,
        )
        return resp if isinstance(resp, EstimateJudgeOptimizationCostResponse) else None

    def create(
        self,
        *,
        judge_id: str,
        budget: str = "medium",
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[CreateJudgeOptimizationRunResponse]:
        body: Dict[str, Any] = {
            "judge_id": judge_id,
            "budget": budget,
        }

        resp = self._post(
            self._base_url(),
            body=body,
            timeout=timeout,
            cast_to=CreateJudgeOptimizationRunResponse,
        )
        return resp if isinstance(resp, CreateJudgeOptimizationRunResponse) else None

    def get(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[JudgeOptimizationRun]:
        resp = self._get(
            f"{self._base_url()}/{id}",
            timeout=timeout,
            cast_to=JudgeOptimizationRun,
        )
        return resp if isinstance(resp, JudgeOptimizationRun) else None

    def get_many(
        self,
        *,
        judge_id: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[JudgeOptimizationRunsResponse]:
        params: Dict[str, str] = {}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)

        if judge_id is not None:
            params["judge_id"] = judge_id

        resp = self._get(
            self._base_url(),
            params=params,
            timeout=timeout,
            cast_to=dict,
        )
        if not resp or not isinstance(resp, dict):
            return None

        runs = [
            r if isinstance(r, JudgeOptimizationRun) else JudgeOptimizationRun(**r)
            for r in resp.get("optimization_runs", [])
        ]
        count: int = resp.get("count", len(runs))
        total: int = resp.get("total", count)

        try:
            return JudgeOptimizationRunsResponse(optimization_runs=runs, count=count, total=total)
        except Exception:
            return None

    def apply(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[ApplyJudgeOptimizationResultResponse]:
        resp = self._post(
            f"{self._base_url()}/{id}/apply",
            body={},
            timeout=timeout,
            cast_to=ApplyJudgeOptimizationResultResponse,
        )
        return resp if isinstance(resp, ApplyJudgeOptimizationResultResponse) else None


class AsyncJudgeOptimizations(AsyncAPIResource):
    def _base_url(self) -> str:
        return f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/judge-optimizations"

    async def estimate(
        self,
        *,
        judge_id: str,
        budget: str = "medium",
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[EstimateJudgeOptimizationCostResponse]:
        body: Dict[str, Any] = {
            "judge_id": judge_id,
            "budget": budget,
        }

        resp = await self._post(
            f"{self._base_url()}/estimate",
            body=body,
            timeout=timeout,
            cast_to=EstimateJudgeOptimizationCostResponse,
        )
        return resp if isinstance(resp, EstimateJudgeOptimizationCostResponse) else None

    async def create(
        self,
        *,
        judge_id: str,
        budget: str = "medium",
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[CreateJudgeOptimizationRunResponse]:
        body: Dict[str, Any] = {
            "judge_id": judge_id,
            "budget": budget,
        }

        resp = await self._post(
            self._base_url(),
            body=body,
            timeout=timeout,
            cast_to=CreateJudgeOptimizationRunResponse,
        )
        return resp if isinstance(resp, CreateJudgeOptimizationRunResponse) else None

    async def get(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[JudgeOptimizationRun]:
        resp = await self._get(
            f"{self._base_url()}/{id}",
            timeout=timeout,
            cast_to=JudgeOptimizationRun,
        )
        return resp if isinstance(resp, JudgeOptimizationRun) else None

    async def get_many(
        self,
        *,
        judge_id: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[JudgeOptimizationRunsResponse]:
        params: Dict[str, str] = {}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)

        if judge_id is not None:
            params["judge_id"] = judge_id

        resp = await self._get(
            self._base_url(),
            params=params,
            timeout=timeout,
            cast_to=dict,
        )
        if not resp or not isinstance(resp, dict):
            return None

        runs = [
            r if isinstance(r, JudgeOptimizationRun) else JudgeOptimizationRun(**r)
            for r in resp.get("optimization_runs", [])
        ]
        count: int = resp.get("count", len(runs))
        total: int = resp.get("total", count)

        try:
            return JudgeOptimizationRunsResponse(optimization_runs=runs, count=count, total=total)
        except Exception:
            return None

    async def apply(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[ApplyJudgeOptimizationResultResponse]:
        resp = await self._post(
            f"{self._base_url()}/{id}/apply",
            body={},
            timeout=timeout,
            cast_to=ApplyJudgeOptimizationResultResponse,
        )
        return resp if isinstance(resp, ApplyJudgeOptimizationResultResponse) else None
