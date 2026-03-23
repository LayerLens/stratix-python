from __future__ import annotations

from typing import List, Optional

from pydantic import Field, BaseModel, ConfigDict

from .judge import Judge
from .model import Model
from .trace import TraceWithEvaluations
from .scorer import Scorer
from .benchmark import Benchmark
from .evaluation import Result, Evaluation
from .integration import Integration
from .organization import Organization
from .evaluation_space import EvaluationSpace
from .trace_evaluation import TraceEvaluation, TraceEvaluationResult
from .judge_optimization import JudgeOptimizationRun


class BenchmarksResponse(BaseModel):
    class Data(BaseModel):
        model_config = ConfigDict(populate_by_name=True)

        benchmarks: List[Benchmark] = Field(..., alias="datasets")

    data: Data


class CreateEvaluationsResponse(BaseModel):
    data: List[Evaluation]


class EvaluationsResponse(BaseModel):
    evaluations: List[Evaluation]
    pagination: Pagination


class ModelsResponse(BaseModel):
    class Data(BaseModel):
        models: List[Model]

    data: Data


class OrganizationResponse(BaseModel):
    data: Organization


class ResultMetrics(BaseModel):
    total_count: int


class Pagination(BaseModel):
    page: int
    page_size: int
    total_pages: int
    total_count: int


class ResultsResponse(BaseModel):
    evaluation_id: str
    results: List[Result]
    metrics: ResultMetrics
    pagination: Pagination


class JudgesResponse(BaseModel):
    judges: List[Judge]
    count: int
    total_count: int


class CreateJudgeResponse(BaseModel):
    id: str


class UpdateJudgeResponse(BaseModel):
    organization_id: str
    project_id: str
    id: str


class DeleteJudgeResponse(BaseModel):
    organization_id: str
    project_id: str
    id: str


class TracesResponse(BaseModel):
    traces: List[TraceWithEvaluations]
    count: int
    total_count: int


class UploadURLResponse(BaseModel):
    organization_id: str
    project_id: str
    url: str


class CreateBenchmarkResponse(BaseModel):
    organization_id: str
    project_id: str
    benchmark_id: str


class CreateModelResponse(BaseModel):
    organization_id: str
    project_id: str
    model_id: str


class CreateTracesResponse(BaseModel):
    trace_ids: List[str]


class TraceEvaluationsResponse(BaseModel):
    trace_evaluations: List[TraceEvaluation]
    count: int
    total: int


class TraceEvaluationResultsResponse(TraceEvaluationResult):
    pass


class CostEstimateResponse(BaseModel):
    estimated_cost: float
    input_tokens: int
    output_tokens: int
    model: str
    trace_count: int


class JudgeOptimizationRunsResponse(BaseModel):
    optimization_runs: List[JudgeOptimizationRun]
    count: int
    total: int


class CreateJudgeOptimizationRunResponse(BaseModel):
    id: str
    judge_id: str
    budget: str
    status: str


class EstimateJudgeOptimizationCostResponse(BaseModel):
    estimated_cost: float
    annotation_count: int
    budget: str


class ApplyJudgeOptimizationResultResponse(BaseModel):
    judge_id: str
    new_version: int
    message: str


class IntegrationsResponse(BaseModel):
    integrations: List[Integration]
    count: int
    total_count: int


class TestIntegrationResponse(BaseModel):
    success: bool
    message: Optional[str] = None


class ScorersResponse(BaseModel):
    scorers: List[Scorer]
    count: int
    total_count: int


class EvaluationSpacesResponse(BaseModel):
    evaluation_spaces: List[EvaluationSpace]
    count: int
    total_count: int
