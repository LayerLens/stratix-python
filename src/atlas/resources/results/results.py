from __future__ import annotations

from typing import List

import httpx

from ..._models import Result, Results as ResultsData
from ..._resource import SyncAPIResource
from ..._constants import DEFAULT_TIMEOUT


class Results(SyncAPIResource):
    def get(
        self,
        *,
        evaluation_id: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> List[Result] | None:
        results = self._get(
            f"/results",
            params={
                "evaluation_id": evaluation_id,
            },
            timeout=timeout,
            cast_to=ResultsData,
        )
        if isinstance(results, ResultsData):
            return results.results
        return None
