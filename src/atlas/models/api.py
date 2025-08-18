from __future__ import annotations

from typing import List

from pydantic import Field, BaseModel, ConfigDict

from .model import Model
from .benchmark import Benchmark
from .evaluation import Result, Evaluation


class Benchmarks(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    benchmarks: List[Benchmark] = Field(..., alias="datasets")


class Evaluations(BaseModel):
    data: List[Evaluation]


class Models(BaseModel):
    models: List[Model]


class ResultMetrics(BaseModel):
    total_count: int


class Pagination(BaseModel):
    total_count: int
    page_size: int
    total_pages: int


class Results(BaseModel):
    evaluation_id: str
    results: List[Result]
    metrics: ResultMetrics
    pagination: Pagination
