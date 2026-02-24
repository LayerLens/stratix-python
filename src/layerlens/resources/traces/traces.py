from __future__ import annotations

import os
import mimetypes
from typing import Any, Dict, List, Optional

import httpx

from ...models import (
    Trace,
    TracesResponse,
    UploadURLResponse,
    CreateTracesResponse,
    TraceWithEvaluations,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


class Traces(SyncAPIResource):
    def _base_url(self) -> str:
        return f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/traces"

    def upload(
        self,
        file_path: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[CreateTracesResponse]:
        """Upload a JSON or JSONL trace file.

        Three-step flow:
        1. Request a presigned upload URL from the API
        2. Upload the file to S3 via the presigned URL
        3. Create trace records from the uploaded file
        """
        file_path = os.path.abspath(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        if file_size > MAX_UPLOAD_SIZE:
            raise ValueError(f"File size {file_size} exceeds maximum of {MAX_UPLOAD_SIZE} bytes (50 MB)")

        content_type = _get_content_type(filename)

        # Step 1: Get presigned upload URL
        upload_resp = self._post(
            f"{self._base_url()}/upload",
            body={"filename": filename, "type": content_type, "size": file_size},
            timeout=timeout,
            cast_to=UploadURLResponse,
        )
        if not isinstance(upload_resp, UploadURLResponse):
            return None

        # Step 2: Upload file to S3
        with open(file_path, "rb") as f:
            put_resp = httpx.put(
                upload_resp.url,
                content=f.read(),
                headers={"Content-Type": content_type},
                timeout=timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout),
            )
            put_resp.raise_for_status()

        # Step 3: Create trace records
        create_resp = self._post(
            self._base_url(),
            body={"filename": filename},
            timeout=timeout,
            cast_to=CreateTracesResponse,
        )
        return create_resp if isinstance(create_resp, CreateTracesResponse) else None

    def get(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Trace]:
        resp = self._get(
            f"{self._base_url()}/{id}",
            timeout=timeout,
            cast_to=Trace,
        )
        return resp if isinstance(resp, Trace) else None

    def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        source: Optional[str] = None,
        judge_id: Optional[str] = None,
        status: Optional[str] = None,
        time_range: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[TracesResponse]:
        params: Dict[str, Any] = {}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["pageSize"] = str(effective_page_size)

        if source is not None:
            params["source"] = source
        if judge_id is not None:
            params["judgeId"] = judge_id
        if status is not None:
            params["status"] = status
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

        traces = [
            t if isinstance(t, TraceWithEvaluations) else TraceWithEvaluations(**t) for t in resp.get("traces", [])
        ]
        count: int = resp.get("count", len(traces))
        total_count: int = resp.get("total_count", count)

        try:
            return TracesResponse(traces=traces, count=count, total_count=total_count)
        except Exception:
            return None

    def delete(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> bool:
        resp = self._delete(
            f"{self._base_url()}/{id}",
            timeout=timeout,
        )
        return resp is not None

    def get_sources(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> List[str]:
        resp = self._get(
            f"{self._base_url()}/sources",
            timeout=timeout,
            cast_to=dict,
        )
        if isinstance(resp, dict):
            return list(resp.get("sources", []))
        return []


class AsyncTraces(AsyncAPIResource):
    def _base_url(self) -> str:
        return f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/traces"

    async def upload(
        self,
        file_path: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[CreateTracesResponse]:
        """Upload a JSON or JSONL trace file.

        Three-step flow:
        1. Request a presigned upload URL from the API
        2. Upload the file to S3 via the presigned URL
        3. Create trace records from the uploaded file
        """
        file_path = os.path.abspath(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        if file_size > MAX_UPLOAD_SIZE:
            raise ValueError(f"File size {file_size} exceeds maximum of {MAX_UPLOAD_SIZE} bytes (50 MB)")

        content_type = _get_content_type(filename)

        # Step 1: Get presigned upload URL
        upload_resp = await self._post(
            f"{self._base_url()}/upload",
            body={"filename": filename, "type": content_type, "size": file_size},
            timeout=timeout,
            cast_to=UploadURLResponse,
        )
        if not isinstance(upload_resp, UploadURLResponse):
            return None

        # Step 2: Upload file to S3
        async with httpx.AsyncClient() as upload_client:
            with open(file_path, "rb") as f:
                put_resp = await upload_client.put(
                    upload_resp.url,
                    content=f.read(),
                    headers={"Content-Type": content_type},
                    timeout=timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout),
                )
                put_resp.raise_for_status()

        # Step 3: Create trace records
        create_resp = await self._post(
            self._base_url(),
            body={"filename": filename},
            timeout=timeout,
            cast_to=CreateTracesResponse,
        )
        return create_resp if isinstance(create_resp, CreateTracesResponse) else None

    async def get(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Trace]:
        resp = await self._get(
            f"{self._base_url()}/{id}",
            timeout=timeout,
            cast_to=Trace,
        )
        return resp if isinstance(resp, Trace) else None

    async def get_many(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        source: Optional[str] = None,
        judge_id: Optional[str] = None,
        status: Optional[str] = None,
        time_range: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[TracesResponse]:
        params: Dict[str, Any] = {}

        effective_page_size = min(max(page_size, 1), MAX_PAGE_SIZE) if page_size is not None else DEFAULT_PAGE_SIZE
        effective_page = page if page is not None else DEFAULT_PAGE

        params["page"] = str(effective_page)
        params["pageSize"] = str(effective_page_size)

        if source is not None:
            params["source"] = source
        if judge_id is not None:
            params["judgeId"] = judge_id
        if status is not None:
            params["status"] = status
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

        traces = [
            t if isinstance(t, TraceWithEvaluations) else TraceWithEvaluations(**t) for t in resp.get("traces", [])
        ]
        count: int = resp.get("count", len(traces))
        total_count: int = resp.get("total_count", count)

        try:
            return TracesResponse(traces=traces, count=count, total_count=total_count)
        except Exception:
            return None

    async def delete(
        self,
        id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> bool:
        resp = await self._delete(
            f"{self._base_url()}/{id}",
            timeout=timeout,
        )
        return resp is not None

    async def get_sources(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> List[str]:
        resp = await self._get(
            f"{self._base_url()}/sources",
            timeout=timeout,
            cast_to=dict,
        )
        if isinstance(resp, dict):
            return list(resp.get("sources", []))
        return []


def _get_content_type(filename: str) -> str:
    if filename.endswith(".jsonl"):
        return "application/jsonl"
    if filename.endswith(".json"):
        return "application/json"
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/json"
