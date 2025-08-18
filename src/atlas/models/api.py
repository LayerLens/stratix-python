from __future__ import annotations

from typing import List

from pydantic import Field, BaseModel, ConfigDict

from .model import Model
from .benchmark import Benchmark
from .evaluation import Result, Evaluation
from .organization import Organization


class BenchmarksResponse(BaseModel):
    class Data(BaseModel):
        model_config = ConfigDict(populate_by_name=True)

        benchmarks: List[Benchmark] = Field(..., alias="datasets")

    data: Data


class EvaluationsResponse(BaseModel):
    data: List[Evaluation]


class ModelsResponse(BaseModel):
    class Data(BaseModel):
        models: List[Model]

    data: Data


class OrganizationResponse(BaseModel):
    data: Organization


class ResultMetrics(BaseModel):
    total_count: int


class Pagination(BaseModel):
    total_count: int
    page_size: int
    total_pages: int


class ResultsResponse(BaseModel):
    evaluation_id: str
    results: List[Result]
    metrics: ResultMetrics
    pagination: Pagination
