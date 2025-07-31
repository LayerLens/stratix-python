from __future__ import annotations

import httpx

from ..._models import Evaluation, Evaluations as EvaluationsData
from ..._resource import SyncAPIResource
from ..._constants import DEFAULT_TIMEOUT


class Evaluations(SyncAPIResource):
    def create(
        self,
        *,
        model: str,
        benchmark: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Evaluation | None:
        evaluations = self._post(
            f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/evaluations",
            body=[
                {
                    "model_id": model,
                    "dataset_id": benchmark,
                    "is_custom_model": False,
                    "is_custom_dataset": False,
                }
            ],
            timeout=timeout,
            cast_to=EvaluationsData,
        )
        if isinstance(evaluations, EvaluationsData) and len(evaluations.data) > 0:
            return evaluations.data[0]
        return None
