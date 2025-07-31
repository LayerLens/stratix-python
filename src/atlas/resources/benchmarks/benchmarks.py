from __future__ import annotations

from typing import List, Literal

import httpx

from ..._models import Benchmark, Benchmarks as BenchmarksData, CustomBenchmark
from ..._resource import SyncAPIResource
from ..._constants import DEFAULT_TIMEOUT


class Benchmarks(SyncAPIResource):
    def get(
        self,
        *,
        type: Literal["public"] | Literal["custom"],
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> List[Benchmark | CustomBenchmark] | None:
        benchmarks = self._get(
            f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/benchmarks",
            params={
                "type": type,
            },
            timeout=timeout,
            cast_to=BenchmarksData,
        )
        if isinstance(benchmarks, BenchmarksData):
            return benchmarks.benchmarks
        return None
