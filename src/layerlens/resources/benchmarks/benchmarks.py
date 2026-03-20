from __future__ import annotations

import os
import mimetypes
from typing import Any, Dict, List, Literal, Optional

import httpx

from ...models import (
    Benchmark,
    CustomBenchmark,
    PublicBenchmark,
    BenchmarksResponse,
    CreateBenchmarkResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


def _get_content_type(filename: str) -> str:
    ct, _ = mimetypes.guess_type(filename)
    if ct:
        return ct
    ext = os.path.splitext(filename)[1].lower()
    return {
        ".jsonl": "application/jsonl",
        ".json": "application/json",
        ".csv": "text/csv",
        ".parquet": "application/x-parquet",
    }.get(ext, "application/octet-stream")


class Benchmarks(SyncAPIResource):
    def get(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        type: Literal["custom", "public"] | None = None,
        name: Optional[str] = None,
        key: Optional[str] = None,
        categories: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
    ) -> Optional[List[Benchmark]]:
        base_url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/benchmarks"

        def fetch(bench_type: str) -> BenchmarksResponse | None:
            params = {"type": bench_type}
            if name:
                params["name"] = name
            if key:
                params["key"] = key
            if categories:
                params["categories"] = ",".join(categories)
            if languages:
                params["languages"] = ",".join(languages)

            resp = self._get(
                base_url,
                params=params,
                timeout=timeout,
                cast_to=BenchmarksResponse,
            )
            return resp if isinstance(resp, BenchmarksResponse) else None

        benchmarks: List[Benchmark] = []

        def cast_benchmark(b: Benchmark, bench_type: str) -> Benchmark:
            if bench_type == "custom":
                return CustomBenchmark(**b.model_dump())
            elif bench_type == "public":
                return PublicBenchmark(**b.model_dump())
            return b  # fallback, just base class

        if type is None:
            for t in ["custom", "public"]:
                resp = fetch(t)
                if resp:
                    benchmarks.extend([cast_benchmark(b, t) for b in resp.data.benchmarks])
        else:  # fetch only one type
            resp = fetch(type)
            if resp:
                benchmarks.extend([cast_benchmark(b, type) for b in resp.data.benchmarks])

        # Exclude custom benchmarks when filtering by fields they don't have
        if categories:
            cat_set = {c.lower() for c in categories}
            benchmarks = [
                b
                for b in benchmarks
                if isinstance(b, PublicBenchmark) and b.categories and any(c.lower() in cat_set for c in b.categories)
            ]

        if languages:
            lang_set = {l.lower() for l in languages}
            benchmarks = [
                b
                for b in benchmarks
                if isinstance(b, PublicBenchmark) and b.language and b.language.lower() in lang_set
            ]

        return benchmarks

    def get_by_id(self, id: str, *, timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT) -> Optional[Benchmark]:
        base_url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/benchmarks/{id}"

        resp = self._get(
            base_url,
            timeout=timeout,
            cast_to=dict,
        )

        if not isinstance(resp, dict):
            return None

        benchmark = resp.get("data")
        if not isinstance(benchmark, dict):
            return None

        # Detect type dynamically: presence of "organization_id" means custom
        if "organization_id" in benchmark:
            return CustomBenchmark(**benchmark)
        else:
            return PublicBenchmark(**benchmark)

    def get_by_key(
        self,
        key: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Benchmark]:
        """Fetch a single benchmark by its unique key."""
        benchmarks = self.get(timeout=timeout, key=key)

        if not benchmarks:
            return None

        for benchmark in benchmarks:
            if benchmark.key == key:
                return benchmark
        return None

    def add(
        self,
        *benchmark_ids: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> bool:
        """Add benchmarks to the project by their IDs."""
        current = self.get(timeout=timeout) or []
        current_ids = [b.id for b in current]
        new_ids = list(dict.fromkeys(current_ids + list(benchmark_ids)))
        return self._patch_project_benchmarks(new_ids, timeout)

    def remove(
        self,
        *benchmark_ids: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> bool:
        """Remove benchmarks from the project by their IDs."""
        current = self.get(timeout=timeout) or []
        remove_set = set(benchmark_ids)
        new_ids = [b.id for b in current if b.id not in remove_set]
        return self._patch_project_benchmarks(new_ids, timeout)

    def _patch_project_benchmarks(
        self,
        dataset_ids: List[str],
        timeout: float | httpx.Timeout | None,
    ) -> bool:
        url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}"
        resp = self._patch(
            url,
            body={"datasets": dataset_ids},
            timeout=timeout,
            cast_to=dict,
        )
        return isinstance(resp, dict) and "id" in resp

    def _upload_file(
        self,
        file_path: str,
        benchmark_name: str,
        timeout: float | httpx.Timeout | None,
    ) -> str:
        """Upload a file and return the filename for use in benchmark creation."""
        file_path = os.path.abspath(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        if file_size > MAX_UPLOAD_SIZE:
            raise ValueError(f"File size {file_size} exceeds maximum of {MAX_UPLOAD_SIZE} bytes (50 MB)")

        content_type = _get_content_type(filename)
        base = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}"

        raw_resp = self._post(
            f"{base}/upload",
            body={"key": benchmark_name, "filename": filename, "type": content_type, "size": file_size},
            timeout=timeout,
            cast_to=dict,
        )
        # Unwrap {"status": ..., "data": {...}} envelope if present
        resp = raw_resp
        if isinstance(resp, dict) and "data" in resp and "status" in resp:
            resp = resp["data"]
        if not isinstance(resp, dict) or "url" not in resp:
            raise ValueError("Failed to get upload URL")

        with open(file_path, "rb") as f:
            put_resp = httpx.put(
                resp["url"],
                content=f.read(),
                headers={"Content-Type": content_type},
                timeout=timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout),
            )
            put_resp.raise_for_status()

        return filename

    def create_custom(
        self,
        *,
        name: str,
        description: str,
        file_path: str,
        additional_metrics: Optional[List[str]] = None,
        custom_scorer_ids: Optional[List[str]] = None,
        input_type: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[CreateBenchmarkResponse]:
        """Create a custom benchmark by uploading a JSONL file.

        Args:
            name: Benchmark name (max 64 characters).
            description: Benchmark description (max 280 characters).
            file_path: Path to a JSONL file with benchmark prompts.
            additional_metrics: Optional metrics: "readability", "toxicity", "hallucination".
            custom_scorer_ids: Optional list of custom scorer IDs.
            input_type: Optional input type: "messages" or "json_payload".
            timeout: Request timeout override.

        Returns:
            CreateBenchmarkResponse with benchmark_id, or None on failure.
        """
        filename = self._upload_file(file_path, name, timeout)

        base = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}"
        body: Dict[str, Any] = {"name": name, "description": description, "file": filename}
        if additional_metrics:
            body["additional_metrics"] = additional_metrics
        if custom_scorer_ids:
            body["custom_scorers"] = custom_scorer_ids
        if input_type:
            body["input_type"] = input_type

        resp = self._post(
            f"{base}/custom-benchmarks",
            body=body,
            timeout=timeout,
            cast_to=dict,
        )
        if isinstance(resp, dict) and "data" in resp and "status" in resp:
            resp = resp["data"]
        if isinstance(resp, dict) and "benchmark_id" in resp:
            return CreateBenchmarkResponse(**resp)
        return None

    def create_smart(
        self,
        *,
        name: str,
        description: str,
        system_prompt: str,
        file_paths: List[str],
        metrics: Optional[List[str]] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[CreateBenchmarkResponse]:
        """Create a smart benchmark from uploaded files.

        The platform will use AI to generate benchmark prompts from the provided files.

        Args:
            name: Benchmark name (max 256 characters).
            description: Benchmark description (max 500 characters).
            system_prompt: System prompt for benchmark generation (max 4000 characters).
            file_paths: List of file paths to upload (1-20 files).
            metrics: Optional metrics: "readability", "toxicity", "hallucination".
            timeout: Request timeout override.

        Returns:
            CreateBenchmarkResponse with benchmark_id, or None on failure.
        """
        filenames = []
        for fp in file_paths:
            filename = self._upload_file(fp, name, timeout)
            filenames.append(filename)

        base = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}"
        body: Dict[str, Any] = {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "files": filenames,
        }
        if metrics:
            body["metrics"] = metrics

        resp = self._post(
            f"{base}/smart-benchmarks",
            body=body,
            timeout=timeout,
            cast_to=dict,
        )
        if isinstance(resp, dict) and "data" in resp and "status" in resp:
            resp = resp["data"]
        if isinstance(resp, dict) and "benchmark_id" in resp:
            return CreateBenchmarkResponse(**resp)
        return None


class AsyncBenchmarks(AsyncAPIResource):
    async def get(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        type: Literal["custom", "public"] | None = None,
        name: Optional[str] = None,
        key: Optional[str] = None,
        categories: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
    ) -> Optional[List[Benchmark]]:
        base_url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/benchmarks"

        async def fetch(bench_type: str) -> Optional[BenchmarksResponse]:
            params = {"type": bench_type}
            if name:
                params["name"] = name
            if key:
                params["key"] = key
            if categories:
                params["categories"] = ",".join(categories)
            if languages:
                params["languages"] = ",".join(languages)

            resp = await self._get(
                base_url,
                params=params,
                timeout=timeout,
                cast_to=BenchmarksResponse,
            )
            return resp if isinstance(resp, BenchmarksResponse) else None

        def cast_benchmark(b: Benchmark, bench_type: str) -> Benchmark:
            if bench_type == "custom":
                return CustomBenchmark(**b.model_dump())
            elif bench_type == "public":
                return PublicBenchmark(**b.model_dump())
            return b  # fallback to base class

        benchmarks: List[Benchmark] = []

        if type is None:  # fetch both custom + public
            for t in ["custom", "public"]:
                resp = await fetch(t)
                if resp:
                    benchmarks.extend([cast_benchmark(b, t) for b in resp.data.benchmarks])
        else:  # fetch only one type
            resp = await fetch(type)
            if resp:
                benchmarks.extend([cast_benchmark(b, type) for b in resp.data.benchmarks])

        # Exclude custom benchmarks when filtering by fields they don't have
        if categories:
            cat_set = {c.lower() for c in categories}
            benchmarks = [
                b
                for b in benchmarks
                if isinstance(b, PublicBenchmark) and b.categories and any(c.lower() in cat_set for c in b.categories)
            ]

        if languages:
            lang_set = {l.lower() for l in languages}
            benchmarks = [
                b
                for b in benchmarks
                if isinstance(b, PublicBenchmark) and b.language and b.language.lower() in lang_set
            ]

        return benchmarks

    async def get_by_id(
        self, id: str, *, timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT
    ) -> Optional[Benchmark]:
        base_url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/benchmarks/{id}"

        resp = await self._get(
            base_url,
            timeout=timeout,
            cast_to=dict,
        )

        if not isinstance(resp, dict):
            return None

        benchmark = resp.get("data")
        if not isinstance(benchmark, dict):
            return None

        # Detect type dynamically: presence of "organization_id" means custom
        if "organization_id" in benchmark:
            return CustomBenchmark(**benchmark)
        else:
            return PublicBenchmark(**benchmark)

    async def get_by_key(
        self,
        key: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Benchmark]:
        """Fetch a single benchmark by its unique key."""
        benchmarks = await self.get(timeout=timeout, key=key)

        if not benchmarks:
            return None

        for benchmark in benchmarks:
            if benchmark.key == key:
                return benchmark
        return None

    async def add(
        self,
        *benchmark_ids: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> bool:
        """Add benchmarks to the project by their IDs."""
        current = await self.get(timeout=timeout) or []
        current_ids = [b.id for b in current]
        new_ids = list(dict.fromkeys(current_ids + list(benchmark_ids)))
        return await self._patch_project_benchmarks(new_ids, timeout)

    async def remove(
        self,
        *benchmark_ids: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> bool:
        """Remove benchmarks from the project by their IDs."""
        current = await self.get(timeout=timeout) or []
        remove_set = set(benchmark_ids)
        new_ids = [b.id for b in current if b.id not in remove_set]
        return await self._patch_project_benchmarks(new_ids, timeout)

    async def _patch_project_benchmarks(
        self,
        dataset_ids: List[str],
        timeout: float | httpx.Timeout | None,
    ) -> bool:
        url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}"
        resp = await self._patch(
            url,
            body={"datasets": dataset_ids},
            timeout=timeout,
            cast_to=dict,
        )
        return isinstance(resp, dict) and "id" in resp

    async def _upload_file(
        self,
        file_path: str,
        benchmark_name: str,
        timeout: float | httpx.Timeout | None,
    ) -> str:
        """Upload a file and return the filename for use in benchmark creation."""
        file_path = os.path.abspath(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        if file_size > MAX_UPLOAD_SIZE:
            raise ValueError(f"File size {file_size} exceeds maximum of {MAX_UPLOAD_SIZE} bytes (50 MB)")

        content_type = _get_content_type(filename)
        base = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}"

        raw_resp = await self._post(
            f"{base}/upload",
            body={"key": benchmark_name, "filename": filename, "type": content_type, "size": file_size},
            timeout=timeout,
            cast_to=dict,
        )
        # Unwrap {"status": ..., "data": {...}} envelope if present
        resp = raw_resp
        if isinstance(resp, dict) and "data" in resp and "status" in resp:
            resp = resp["data"]
        if not isinstance(resp, dict) or "url" not in resp:
            raise ValueError("Failed to get upload URL")

        async with httpx.AsyncClient() as http:
            with open(file_path, "rb") as f:
                put_resp = await http.put(
                    resp["url"],
                    content=f.read(),
                    headers={"Content-Type": content_type},
                    timeout=timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout),
                )
                put_resp.raise_for_status()

        return filename

    async def create_custom(
        self,
        *,
        name: str,
        description: str,
        file_path: str,
        additional_metrics: Optional[List[str]] = None,
        custom_scorer_ids: Optional[List[str]] = None,
        input_type: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[CreateBenchmarkResponse]:
        """Create a custom benchmark by uploading a JSONL file.

        Args:
            name: Benchmark name (max 64 characters).
            description: Benchmark description (max 280 characters).
            file_path: Path to a JSONL file with benchmark prompts.
            additional_metrics: Optional metrics: "readability", "toxicity", "hallucination".
            custom_scorer_ids: Optional list of custom scorer IDs.
            input_type: Optional input type: "messages" or "json_payload".
            timeout: Request timeout override.

        Returns:
            CreateBenchmarkResponse with benchmark_id, or None on failure.
        """
        filename = await self._upload_file(file_path, name, timeout)

        base = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}"
        body: Dict[str, Any] = {"name": name, "description": description, "file": filename}
        if additional_metrics:
            body["additional_metrics"] = additional_metrics
        if custom_scorer_ids:
            body["custom_scorers"] = custom_scorer_ids
        if input_type:
            body["input_type"] = input_type

        resp = await self._post(
            f"{base}/custom-benchmarks",
            body=body,
            timeout=timeout,
            cast_to=dict,
        )
        if isinstance(resp, dict) and "data" in resp and "status" in resp:
            resp = resp["data"]
        if isinstance(resp, dict) and "benchmark_id" in resp:
            return CreateBenchmarkResponse(**resp)
        return None

    async def create_smart(
        self,
        *,
        name: str,
        description: str,
        system_prompt: str,
        file_paths: List[str],
        metrics: Optional[List[str]] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[CreateBenchmarkResponse]:
        """Create a smart benchmark from uploaded files.

        The platform will use AI to generate benchmark prompts from the provided files.

        Args:
            name: Benchmark name (max 256 characters).
            description: Benchmark description (max 500 characters).
            system_prompt: System prompt for benchmark generation (max 4000 characters).
            file_paths: List of file paths to upload (1-20 files).
            metrics: Optional metrics: "readability", "toxicity", "hallucination".
            timeout: Request timeout override.

        Returns:
            CreateBenchmarkResponse with benchmark_id, or None on failure.
        """
        filenames = []
        for fp in file_paths:
            filename = await self._upload_file(fp, name, timeout)
            filenames.append(filename)

        base = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}"
        body: Dict[str, Any] = {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "files": filenames,
        }
        if metrics:
            body["metrics"] = metrics

        resp = await self._post(
            f"{base}/smart-benchmarks",
            body=body,
            timeout=timeout,
            cast_to=dict,
        )
        if isinstance(resp, dict) and "data" in resp and "status" in resp:
            resp = resp["data"]
        if isinstance(resp, dict) and "benchmark_id" in resp:
            return CreateBenchmarkResponse(**resp)
        return None
