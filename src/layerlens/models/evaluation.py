from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Union, Optional
from datetime import timedelta
from typing_extensions import Annotated

import httpx
from pydantic import Field, BaseModel, ConfigDict, BeforeValidator

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
    characteristics: Optional[List[str]] = []


class EvaluationModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = ""
    performance: Any = None


class PerformanceDetails(BaseModel):
    strengths: Optional[List[str]] = []
    challenges: Optional[List[str]] = []


class ErrorAnalysis(BaseModel):
    common_failure_modes: Optional[List[str]] = []
    example: str = ""


class AnalysisSummary(BaseModel):
    key_takeaways: Optional[List[str]] = []


class EvaluationSummary(BaseModel):
    name: str = ""
    goal: str = ""
    metrics: Optional[List[EvaluationMetric]] = []
    task_types: Optional[List[EvaluationTaskType]] = []
    dataset: Optional[EvaluationDataset] = None
    model: Optional[EvaluationModelInfo] = None
    performance_details: Optional[PerformanceDetails] = None
    error_analysis: Optional[ErrorAnalysis] = None
    analysis_summary: Optional[AnalysisSummary] = None


class Evaluation(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

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
    # Quality metrics. None means the metric was NOT COMPUTED for this evaluation,
    # which for readability/toxicity is the steady state rather than an edge case:
    # the pipeline stopped collecting them, so the API sends null (and, once
    # atlas-app ships the matching `omitempty`, omits the key). Do not render None
    # as 0 — a zero readability or toxicity score is a fabricated measurement, and
    # keeping "not computed" distinguishable from a real zero is exactly why the
    # API made these nullable.
    #
    # `Optional[float]` accepts a number, an explicit null AND a missing key. A
    # plain default does not: it covers a missing key and does nothing at all for
    # an explicit null, which was LAY-3765.
    #
    # ethics_score is nullable as prophylaxis only — the Go side is still a
    # non-pointer float64 that cannot serialize as null today.
    readability_score: Optional[float] = None
    toxicity_score: Optional[float] = None
    ethics_score: Optional[float] = None
    failed_prompt_count: int = 0
    queue_id: int = 0
    # Token consumption aggregates: sums over every prompt of the
    # provider-reported usage for that prompt's successful attempt. They
    # exclude tokens burned by failed retry attempts, LLM-judge/grader calls,
    # and prompt-cache read/write tokens. None means the evaluation never
    # recorded usage (runs predating token capture) — treat it as "not
    # recorded", not as zero.
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    avg_input_tokens_per_prompt: Optional[float] = None
    avg_output_tokens_per_prompt: Optional[float] = None
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
            self.total_input_tokens = evaluation.total_input_tokens
            self.total_output_tokens = evaluation.total_output_tokens
            self.avg_input_tokens_per_prompt = evaluation.avg_input_tokens_per_prompt
            self.avg_output_tokens_per_prompt = evaluation.avg_output_tokens_per_prompt
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
            self.total_input_tokens = evaluation.total_input_tokens
            self.total_output_tokens = evaluation.total_output_tokens
            self.avg_input_tokens_per_prompt = evaluation.avg_input_tokens_per_prompt
            self.avg_output_tokens_per_prompt = evaluation.avg_output_tokens_per_prompt
            self.summary = evaluation.summary

        return self


class ScorerResult(BaseModel):
    """The outcome of one custom scorer for one prompt.

    Mirrors the API's ``models.ScorerResult``. Both success and failure use this
    shape; ``score`` is None when the scorer failed to run, in which case
    ``error`` explains why. A failed scorer has no score — do not read None as 0.
    """

    score: Optional[float] = None
    status: str = ""
    error: str = ""


def _nanoseconds_to_timedelta(value: Any) -> Any:
    """Rescale the API's int64 nanosecond duration into a ``timedelta``.

    The API's field is a Go ``time.Duration``, which has no ``MarshalJSON``, so
    ``encoding/json`` emits the underlying int64 — a **nanosecond** count. pydantic
    reads a bare int into ``timedelta`` as **seconds**, so 2.5 seconds arrived as
    roughly 79 years and no exception was raised anywhere. The atlas frontend
    divides the same field by 1e9, so the wire unit is unambiguous and the SDK was
    the side that was wrong.

    Anything that is not a bare number is passed through to pydantic's own
    coercion, so constructing a ``Result`` with a real ``timedelta`` in Python
    still behaves as it always did.

    ``timedelta`` resolves to microseconds, so sub-microsecond precision is lost.
    That is inherent to the type and irrelevant at the scale of an LLM call.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value

    return timedelta(microseconds=value / 1000)


class Result(BaseModel):
    subset: str
    prompt: str
    result: str
    truth: str
    duration: Annotated[timedelta, BeforeValidator(_nanoseconds_to_timedelta)]
    score: float
    # Two shapes, because the API sends two. Built-in metrics are a flat map of
    # floats; an evaluation run with custom scorers instead gets one ScorerResult
    # object per scorer, keyed by scorer ID. The field is nullable and optional
    # because `Metrics json.RawMessage` carries no omitempty, so a nil value
    # serializes as null rather than being omitted.
    metrics: Optional[Dict[str, Union[float, ScorerResult, None]]] = None
    # Provider-reported usage for this prompt's successful attempt. Unlike the
    # evaluation-level aggregates, the API cannot distinguish "not recorded"
    # per prompt: runs predating token capture report 0. None only means the
    # backend did not send the field at all.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
