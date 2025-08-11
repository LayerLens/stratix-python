from __future__ import annotations

from typing import List, Literal, Optional

import httpx

from ...models import Benchmark, Benchmarks as BenchmarksResponse
from ..._resource import SyncAPIResource
from ..._constants import DEFAULT_TIMEOUT


class Benchmarks(SyncAPIResource):
    def get(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        type: Literal["custom", "public"] | None = None,
        name: Optional[str] = None,
    ) -> List[Benchmark] | None:
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

        if type is None:
            for t in ["custom", "public"]:
                resp = fetch(t)
                if resp:
                    benchmarks.extend(resp.benchmarks)
        else:  # fetch only one type
            resp = fetch(type)
            if resp:
                benchmarks.extend(resp.benchmarks)

        return benchmarks
