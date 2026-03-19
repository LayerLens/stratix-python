from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from ...models import EvaluationSpace, EvaluationSpacesResponse
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500


def _unwrap(resp: Any) -> Any:
    if isinstance(resp, dict) and "data" in resp and "status" in resp:
        return resp["data"]
    return resp


class EvaluationSpaces(SyncAPIResource):
    def _base_url(self) -> str:
        return f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/evaluation-spaces"

    def get(self, id: str, *, timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT) -> Optional[EvaluationSpace]:
        resp = self._get(f"{self._base_url()}/{id}", timeout=timeout, cast_to=dict)
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return EvaluationSpace(**data)
            except Exception:
                return None
        return None

    def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort_by: Optional[str] = None,
        order: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[EvaluationSpacesResponse]:
        params: Dict[str, Any] = {}
        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE
        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)
        if sort_by:
            params["sort_by"] = sort_by
        if order:
            params["order"] = order

        resp = self._get(self._base_url(), params=params, timeout=timeout, cast_to=dict)
        if not resp or not isinstance(resp, dict):
            return None
        data = _unwrap(resp)
        if not isinstance(data, dict):
            return None

        spaces = [EvaluationSpace(**s) if isinstance(s, dict) else s for s in data.get("evaluation_spaces", [])]
        count: int = data.get("count", len(spaces))
        total_count: int = data.get("total_count", count)
        try:
            return EvaluationSpacesResponse(evaluation_spaces=spaces, count=count, total_count=total_count)
        except Exception:
            return None

    def create(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        visibility: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[EvaluationSpace]:
        body: Dict[str, Any] = {"name": name}
        if description:
            body["description"] = description
        if visibility:
            body["visibility"] = visibility
        resp = self._post(self._base_url(), body=body, timeout=timeout, cast_to=dict)
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return EvaluationSpace(**data)
            except Exception:
                return None
        return None

    def update(
        self,
        id: str,
        *,
        description: Optional[str] = None,
        visibility: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[EvaluationSpace]:
        body: Dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        if visibility is not None:
            body["visibility"] = visibility
        resp = self._patch(f"{self._base_url()}/{id}", body=body, timeout=timeout, cast_to=dict)
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return EvaluationSpace(**data)
            except Exception:
                return None
        return None

    def delete(self, id: str, *, timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT) -> bool:
        try:
            self._delete(f"{self._base_url()}/{id}", timeout=timeout, cast_to=dict)
            return True
        except Exception:
            return False


class AsyncEvaluationSpaces(AsyncAPIResource):
    def _base_url(self) -> str:
        return f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/evaluation-spaces"

    async def get(
        self, id: str, *, timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT
    ) -> Optional[EvaluationSpace]:
        resp = await self._get(f"{self._base_url()}/{id}", timeout=timeout, cast_to=dict)
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return EvaluationSpace(**data)
            except Exception:
                return None
        return None

    async def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort_by: Optional[str] = None,
        order: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[EvaluationSpacesResponse]:
        params: Dict[str, Any] = {}
        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE
        params["page"] = str(effective_page)
        params["page_size"] = str(effective_page_size)
        if sort_by:
            params["sort_by"] = sort_by
        if order:
            params["order"] = order
        resp = await self._get(self._base_url(), params=params, timeout=timeout, cast_to=dict)
        if not resp or not isinstance(resp, dict):
            return None
        data = _unwrap(resp)
        if not isinstance(data, dict):
            return None
        spaces = [EvaluationSpace(**s) if isinstance(s, dict) else s for s in data.get("evaluation_spaces", [])]
        count: int = data.get("count", len(spaces))
        total_count: int = data.get("total_count", count)
        try:
            return EvaluationSpacesResponse(evaluation_spaces=spaces, count=count, total_count=total_count)
        except Exception:
            return None

    async def create(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        visibility: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[EvaluationSpace]:
        body: Dict[str, Any] = {"name": name}
        if description:
            body["description"] = description
        if visibility:
            body["visibility"] = visibility
        resp = await self._post(self._base_url(), body=body, timeout=timeout, cast_to=dict)
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return EvaluationSpace(**data)
            except Exception:
                return None
        return None

    async def update(
        self,
        id: str,
        *,
        description: Optional[str] = None,
        visibility: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[EvaluationSpace]:
        body: Dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        if visibility is not None:
            body["visibility"] = visibility
        resp = await self._patch(f"{self._base_url()}/{id}", body=body, timeout=timeout, cast_to=dict)
        data = _unwrap(resp)
        if isinstance(data, dict):
            try:
                return EvaluationSpace(**data)
            except Exception:
                return None
        return None

    async def delete(self, id: str, *, timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT) -> bool:
        try:
            await self._delete(f"{self._base_url()}/{id}", timeout=timeout, cast_to=dict)
            return True
        except Exception:
            return False
