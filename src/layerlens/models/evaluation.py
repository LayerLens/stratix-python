from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from datetime import timedelta

import httpx
from pydantic import Field, BaseModel, ConfigDict

if TYPE_CHECKING:
    from .api import ResultsResponse
    from .._client import Stratix, AsyncStratix


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    FAILURE = "failure"
    IN_PROGRESS = "in-progress"
    PAUSED = "paused"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class EvaluationMetric(BaseModel):
    name: str
    description: str = ""


class EvaluationTaskType(BaseModel):
    name: str
    description: str = ""


class EvaluationDataset(BaseModel):
    total_size: int = 0
    training_size: int = 0
    test_size: int = 0
    characteristics: List[str] = []


class EvaluationModelInfo(BaseModel):
    model_name: str = ""
    performance: Any = None


class PerformanceDetails(BaseModel):
    strengths: List[str] = []
    challenges: List[str] = []


class ErrorAnalysis(BaseModel):
    common_failure_modes: List[str] = []
    example: str = ""


class AnalysisSummary(BaseModel):
    key_takeaways: List[str] = []


class EvaluationSummary(BaseModel):
    name: str = ""
    goal: str = ""
    metrics: List[EvaluationMetric] = []
    task_types: List[EvaluationTaskType] = []
    dataset: Optional[EvaluationDataset] = None
    model: Optional[EvaluationModelInfo] = None
    performance_details: Optional[PerformanceDetails] = None
    error_analysis: Optional[ErrorAnalysis] = None
    analysis_summary: Optional[AnalysisSummary] = None


class Evaluation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    status: EvaluationStatus
    status_description: str = ""
    submitted_at: int
    finished_at: int
    model_id: str
    model_name: str = ""
    model_key: str = ""
    model_company: str = ""
    benchmark_id: str = Field(..., alias="dataset_id")
    benchmark_name: str = Field("", alias="dataset_name")
    average_duration: int
    accuracy: float
    readability_score: float = 0.0
    toxicity_score: float = 0.0
    ethics_score: float = 0.0
    failed_prompt_count: int = 0
    queue_id: int = 0
    summary: Optional[EvaluationSummary] = None

    _client: "Optional[Stratix | AsyncStratix]" = None

    def attach_client(self, client: "Stratix | AsyncStratix") -> "Evaluation":
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
        from .._client import AsyncStratix

        if self._client is None:
            raise ValueError("No client attached")
        if isinstance(self._client, AsyncStratix):
            raise RuntimeError("Use `await get_results_async()` with an async client")

        return self._client.results.get(evaluation=self, page=page, page_size=page_size, timeout=timeout)

    def get_all_results(
        self,
        *,
        timeout: float | httpx.Timeout | None = None,
    ) -> List[Result]:
        """Fetch results synchronously if a sync client is attached."""
        from .._client import AsyncStratix

        if self._client is None:
            raise ValueError("No client attached")
        if isinstance(self._client, AsyncStratix):
            raise RuntimeError("Use `await get_results_async()` with an async client")

        return self._client.results.get_all(evaluation=self, timeout=timeout)

    async def get_results_async(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Optional[ResultsResponse]:
        """Fetch results asynchronously if an async client is attached."""
        from .._client import AsyncStratix

        if self._client is None:
            raise ValueError("No client attached")
        if not isinstance(self._client, AsyncStratix):
            raise RuntimeError("Use `get_results()` with a sync client")

        return await self._client.results.get(evaluation=self, page=page, page_size=page_size, timeout=timeout)

    async def get_all_results_async(
        self,
        *,
        timeout: float | httpx.Timeout | None = None,
    ) -> List[Result]:
        """Fetch results asynchronously if an async client is attached."""
        from .._client import AsyncStratix

        if self._client is None:
            raise ValueError("No client attached")
        if not isinstance(self._client, AsyncStratix):
            raise RuntimeError("Use `get_results()` with a sync client")

        return await self._client.results.get_all(evaluation=self, timeout=timeout)

    def wait_for_completion(
        self, *, interval_seconds: int = 30, timeout_seconds: Optional[int] = None
    ) -> Optional["Evaluation"]:
        """Sync polling using a sync client."""
        from .._client import AsyncStratix

        if self._client is None:
            raise ValueError("No client attached")
        if isinstance(self._client, AsyncStratix):
            raise RuntimeError("Use `wait_for_completion_async()` with an async client")

        evaluation = self._client.evaluations.wait_for_completion(
            self, interval_seconds=interval_seconds, timeout_seconds=timeout_seconds
        )
        if evaluation:
            self.status = evaluation.status
            self.status_description = evaluation.status_description
            self.finished_at = evaluation.finished_at
            self.average_duration = evaluation.average_duration
            self.accuracy = evaluation.accuracy
            self.readability_score = evaluation.readability_score
            self.toxicity_score = evaluation.toxicity_score
            self.ethics_score = evaluation.ethics_score
            self.failed_prompt_count = evaluation.failed_prompt_count
            self.summary = evaluation.summary

        return self

    async def wait_for_completion_async(
        self, *, interval_seconds: int = 30, timeout_seconds: Optional[int] = None
    ) -> Optional["Evaluation"]:
        """Async polling using an async client."""
        from .._client import AsyncStratix

        if self._client is None:
            raise ValueError("No client attached")
        if not isinstance(self._client, AsyncStratix):
            raise RuntimeError("Use `wait_for_completion()` with a sync client")

        evaluation = await self._client.evaluations.wait_for_completion(
            self, interval_seconds=interval_seconds, timeout_seconds=timeout_seconds
        )
        if evaluation:
            self.status = evaluation.status
            self.status_description = evaluation.status_description
            self.finished_at = evaluation.finished_at
            self.average_duration = evaluation.average_duration
            self.accuracy = evaluation.accuracy
            self.readability_score = evaluation.readability_score
            self.toxicity_score = evaluation.toxicity_score
            self.ethics_score = evaluation.ethics_score
            self.failed_prompt_count = evaluation.failed_prompt_count
            self.summary = evaluation.summary

        return self


class Result(BaseModel):
    subset: str
    prompt: str
    result: str
    truth: str
    duration: timedelta
    score: float
    metrics: Dict[str, Optional[float]]
