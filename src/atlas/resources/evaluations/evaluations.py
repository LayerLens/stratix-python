from __future__ import annotations

import time
import asyncio
from typing import Optional

import httpx

from ...models import (
    Model,
    Benchmark,
    Evaluation,
    CustomModel,
    CustomBenchmark,
    EvaluationsResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT


class Evaluations(SyncAPIResource):
    def create(
        self,
        *,
        model: Model,
        benchmark: Benchmark,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Evaluation]:
        evaluations = self._post(
            f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/evaluations",
            body=[
                {
                    "model_id": model.id,
                    "dataset_id": benchmark.id,
                    "is_custom_model": isinstance(model, CustomModel),
                    "is_custom_dataset": isinstance(benchmark, CustomBenchmark),
                }
            ],
            timeout=timeout,
            cast_to=EvaluationsResponse,
        )
        if isinstance(evaluations, EvaluationsResponse) and len(evaluations.data) > 0:
            evaluation = evaluations.data[0]
            evaluation.attach_client(self._client)
            return evaluation
        return None

    def get(
        self,
        evaluation: Evaluation,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Evaluation]:
        return self.get_by_id(evaluation.id, timeout=timeout)

    def get_by_id(
        self,
        evaluation_id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Evaluation]:
        evaluation = self._get(
            f"/evaluations/{evaluation_id}",
            timeout=timeout,
            cast_to=Evaluation,
        )
        if isinstance(evaluation, Evaluation):
            evaluation.attach_client(self._client)
            return evaluation
        return None

    def wait_for_completion(
        self,
        evaluation: Evaluation,
        *,
        interval_seconds: int = 30,
        timeout_seconds: int | None = None,
    ) -> Optional[Evaluation]:
        """Poll until the evaluation finishes or timeout is reached."""
        start = time.time()

        updated_evaluation: Optional[Evaluation] = self.get(evaluation)
        while updated_evaluation and not updated_evaluation.is_finished:
            if timeout_seconds and (time.time() - start) > timeout_seconds:
                raise TimeoutError(
                    f"Evaluation {updated_evaluation.id} did not complete within {timeout_seconds} seconds"
                )

            time.sleep(interval_seconds)
            updated_evaluation = self.get(updated_evaluation)

        return updated_evaluation


class AsyncEvaluations(AsyncAPIResource):
    async def create(
        self,
        *,
        model: Model,
        benchmark: Benchmark,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Evaluation]:
        evaluations = await self._post(
            f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/evaluations",
            body=[
                {
                    "model_id": model.id,
                    "dataset_id": benchmark.id,
                    "is_custom_model": isinstance(model, CustomModel),
                    "is_custom_dataset": isinstance(benchmark, CustomBenchmark),
                }
            ],
            timeout=timeout,
            cast_to=EvaluationsResponse,
        )
        if isinstance(evaluations, EvaluationsResponse) and len(evaluations.data) > 0:
            evaluation = evaluations.data[0]
            evaluation.attach_client(self._client)
            return evaluation
        return None

    async def get(
        self,
        evaluation: Evaluation,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Evaluation]:
        return await self.get_by_id(evaluation.id, timeout=timeout)

    async def get_by_id(
        self,
        evaluation_id: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Evaluation]:
        evaluation = await self._get(
            f"/evaluations/{evaluation_id}",
            timeout=timeout,
            cast_to=Evaluation,
        )
        if isinstance(evaluation, Evaluation):
            evaluation.attach_client(self._client)
            return evaluation
        return None

    async def wait_for_completion(
        self,
        evaluation: Evaluation,
        *,
        interval_seconds: int = 30,
        timeout_seconds: Optional[int] = None,
    ) -> Optional[Evaluation]:
        """Poll asynchronously until the evaluation finishes or timeout is reached."""
        start = asyncio.get_event_loop().time()

        updated_evaluation: Optional[Evaluation] = await self.get(evaluation)
        while updated_evaluation and not updated_evaluation.is_finished:
            if timeout_seconds and (asyncio.get_event_loop().time() - start) > timeout_seconds:
                raise TimeoutError(
                    f"Evaluation {updated_evaluation.id} did not complete within {timeout_seconds} seconds"
                )

            await asyncio.sleep(interval_seconds)
            updated_evaluation = await self.get(updated_evaluation)

        return updated_evaluation
