from __future__ import annotations

from typing import List, Literal, Optional

import httpx

from ...models import (
    Benchmark,
    CustomBenchmark,
    PublicBenchmark,
    BenchmarksResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT


class Benchmarks(SyncAPIResource):
    def get(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        type: Literal["custom", "public"] | None = None,
        name: Optional[str] = None,
    ) -> Optional[List[Benchmark]]:
        base_url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/benchmarks"

        def fetch(bench_type: str) -> BenchmarksResponse | None:
            params = {"type": bench_type}
            if name:
                params["query"] = name

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


class AsyncBenchmarks(AsyncAPIResource):
    async def get(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        type: Literal["custom", "public"] | None = None,
        name: Optional[str] = None,
    ) -> Optional[List[Benchmark]]:
        base_url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/benchmarks"

        async def fetch(bench_type: str) -> Optional[BenchmarksResponse]:
            params = {"type": bench_type}
            if name:
                params["query"] = name

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
