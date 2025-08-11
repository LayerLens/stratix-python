from __future__ import annotations

import httpx

from ...models import Model, Benchmark, Evaluation, Evaluations as EvaluationsResponse
from ..._resource import SyncAPIResource
from ..._constants import DEFAULT_TIMEOUT


class Evaluations(SyncAPIResource):
    def create(
        self,
        *,
        model: Model,
        benchmark: Benchmark,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Evaluation | None:
        evaluations = self._post(
            f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/evaluations",
            body=[
                {
                    "model_id": model.id,
                    "dataset_id": benchmark.id,
                    "is_custom_model": False,
                    "is_custom_dataset": False,
                }
            ],
            timeout=timeout,
            cast_to=EvaluationsResponse,
        )
        if isinstance(evaluations, EvaluationsResponse) and len(evaluations.data) > 0:
            return evaluations.data[0]
        return None
