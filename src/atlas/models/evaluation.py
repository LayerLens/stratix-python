from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional
from datetime import timedelta

import httpx
from pydantic import Field, BaseModel, ConfigDict

if TYPE_CHECKING:
    from .api import ResultsResponse
    from .._client import Atlas, AsyncAtlas


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    FAILURE = "failure"
    IN_PROGRESS = "in-progress"
    PAUSED = "paused"
    SUCCESS = "success"
    TIMEOUT = "timeout"


class Evaluation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    status: EvaluationStatus
    submitted_at: int
    finished_at: int
    model_id: str
    benchmark_id: str = Field(..., alias="dataset_id")
    average_duration: int
    accuracy: float

    _client: "Optional[Atlas | AsyncAtlas]" = None

    def attach_client(self, client: "Atlas | AsyncAtlas") -> "Evaluation":
        self._client = client
        return self

    @property
    def is_finished(self) -> bool:
        """Return True if evaluation is done (success, failure, or timeout)."""
        return self.status in {
            EvaluationStatus.SUCCESS,
            EvaluationStatus.FAILURE,
            EvaluationStatus.TIMEOUT,
        }

    @property
    def is_success(self) -> bool:
        """Return True if evaluation completed successfully."""
        return self.status == EvaluationStatus.SUCCESS

    def get_results(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Optional[ResultsResponse]:
        """Fetch results synchronously if a sync client is attached."""
        from .._client import AsyncAtlas

        if self._client is None:
            raise ValueError("No client attached")
        if isinstance(self._client, AsyncAtlas):
            raise RuntimeError("Use `await get_results_async()` with an async client")

        return self._client.results.get(evaluation=self, page=page, page_size=page_size, timeout=timeout)

    async def get_results_async(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Optional[ResultsResponse]:
        """Fetch results asynchronously if an async client is attached."""
        from .._client import AsyncAtlas

        if self._client is None:
            raise ValueError("No client attached")
        if not isinstance(self._client, AsyncAtlas):
            raise RuntimeError("Use `get_results()` with a sync client")

        return await self._client.results.get(evaluation=self, page=page, page_size=page_size, timeout=timeout)

    def wait_for_completion(
        self, *, interval_seconds: int = 30, timeout_seconds: Optional[int] = None
    ) -> Optional["Evaluation"]:
        """Sync polling using a sync client."""
        from .._client import AsyncAtlas

        if self._client is None:
            raise ValueError("No client attached")
        if isinstance(self._client, AsyncAtlas):
            raise RuntimeError("Use `wait_for_completion_async()` with an async client")

        evaluation = self._client.evaluations.wait_for_completion(
            self, interval_seconds=interval_seconds, timeout_seconds=timeout_seconds
        )
        if evaluation:
            self.status = evaluation.status
            self.finished_at = evaluation.finished_at
            self.average_duration = evaluation.average_duration
            self.accuracy = evaluation.accuracy

        return self

    async def wait_for_completion_async(
        self, *, interval_seconds: int = 30, timeout_seconds: Optional[int] = None
    ) -> Optional["Evaluation"]:
        """Async polling using an async client."""
        from .._client import AsyncAtlas

        if self._client is None:
            raise ValueError("No client attached")
        if not isinstance(self._client, AsyncAtlas):
            raise RuntimeError("Use `wait_for_completion()` with a sync client")

        evaluation = await self._client.evaluations.wait_for_completion(
            self, interval_seconds=interval_seconds, timeout_seconds=timeout_seconds
        )
        if evaluation:
            self.status = evaluation.status
            self.finished_at = evaluation.finished_at
            self.average_duration = evaluation.average_duration
            self.accuracy = evaluation.accuracy

        return self


class Result(BaseModel):
    subset: str
    prompt: str
    result: str
    truth: str
    duration: timedelta
    score: float
    metrics: Dict[str, Optional[float]]
