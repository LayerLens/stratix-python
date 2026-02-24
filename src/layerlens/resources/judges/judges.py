from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from ...models import (
    Judge,
    JudgesResponse,
    CreateJudgeResponse,
    DeleteJudgeResponse,
    UpdateJudgeResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500


class Judges(SyncAPIResource):
    def _base_url(self) -> str:
        return f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/judges"

    def create(
        self,
        *,
        name: str,
        evaluation_goal: str,
        model_id: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Judge]:
        body: Dict[str, Any] = {
            "name": name,
            "evaluation_goal": evaluation_goal,
        }
        if model_id is not None:
            body["model_id"] = model_id

        resp = self._post(
            self._base_url(),
            body=body,
            timeout=timeout,
            cast_to=CreateJudgeResponse,
        )
        if isinstance(resp, CreateJudgeResponse):
            return self.get(resp.id, timeout=timeout)
        return None

    def get(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Judge]:
        resp = self._get(
            f"{self._base_url()}/{id}",
            timeout=timeout,
            cast_to=Judge,
        )
        return resp if isinstance(resp, Judge) else None

    def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[JudgesResponse]:
        params: dict[str, str] = {}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["pageSize"] = str(effective_page_size)

        resp = self._get(
            self._base_url(),
            params=params,
            timeout=timeout,
            cast_to=dict,
        )
        if not resp or not isinstance(resp, dict):
            return None

        judges = [j if isinstance(j, Judge) else Judge(**j) for j in resp.get("judges", [])]
        count: int = resp.get("count", len(judges))
        total_count: int = resp.get("total_count", count)

        try:
            return JudgesResponse(judges=judges, count=count, total_count=total_count)
        except Exception:
            return None

    def update(
        self,
        id: str,
        *,
        name: Optional[str] = None,
        evaluation_goal: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[UpdateJudgeResponse]:
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if evaluation_goal is not None:
            body["evaluation_goal"] = evaluation_goal
        if model_id is not None:
            body["model_id"] = model_id

        resp = self._patch(
            f"{self._base_url()}/{id}",
            body=body,
            timeout=timeout,
            cast_to=UpdateJudgeResponse,
        )
        return resp if isinstance(resp, UpdateJudgeResponse) else None

    def delete(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[DeleteJudgeResponse]:
        resp = self._delete(
            f"{self._base_url()}/{id}",
            timeout=timeout,
            cast_to=DeleteJudgeResponse,
        )
        return resp if isinstance(resp, DeleteJudgeResponse) else None


class AsyncJudges(AsyncAPIResource):
    def _base_url(self) -> str:
        return f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/judges"

    async def create(
        self,
        *,
        name: str,
        evaluation_goal: str,
        model_id: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Judge]:
        body: Dict[str, Any] = {
            "name": name,
            "evaluation_goal": evaluation_goal,
        }
        if model_id is not None:
            body["model_id"] = model_id

        resp = await self._post(
            self._base_url(),
            body=body,
            timeout=timeout,
            cast_to=CreateJudgeResponse,
        )
        if isinstance(resp, CreateJudgeResponse):
            return await self.get(resp.id, timeout=timeout)
        return None

    async def get(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Judge]:
        resp = await self._get(
            f"{self._base_url()}/{id}",
            timeout=timeout,
            cast_to=Judge,
        )
        return resp if isinstance(resp, Judge) else None

    async def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[JudgesResponse]:
        params: dict[str, str] = {}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["pageSize"] = str(effective_page_size)

        resp = await self._get(
            self._base_url(),
            params=params,
            timeout=timeout,
            cast_to=dict,
        )
        if not resp or not isinstance(resp, dict):
            return None

        judges = [j if isinstance(j, Judge) else Judge(**j) for j in resp.get("judges", [])]
        count: int = resp.get("count", len(judges))
        total_count: int = resp.get("total_count", count)

        try:
            return JudgesResponse(judges=judges, count=count, total_count=total_count)
        except Exception:
            return None

    async def update(
        self,
        id: str,
        *,
        name: Optional[str] = None,
        evaluation_goal: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[UpdateJudgeResponse]:
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if evaluation_goal is not None:
            body["evaluation_goal"] = evaluation_goal
        if model_id is not None:
            body["model_id"] = model_id

        resp = await self._patch(
            f"{self._base_url()}/{id}",
            body=body,
            timeout=timeout,
            cast_to=UpdateJudgeResponse,
        )
        return resp if isinstance(resp, UpdateJudgeResponse) else None

    async def delete(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[DeleteJudgeResponse]:
        resp = await self._delete(
            f"{self._base_url()}/{id}",
            timeout=timeout,
            cast_to=DeleteJudgeResponse,
        )
        return resp if isinstance(resp, DeleteJudgeResponse) else None
