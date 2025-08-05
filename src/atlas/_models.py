from __future__ import annotations

from typing import Dict, List, Union, Optional
from datetime import timedelta

from pydantic import Field, BaseModel, ConfigDict


class Evaluation(BaseModel):
    id: str
    status: str
    status_description: str
    submitted_at: int
    finished_at: int
    model_id: str
    model_name: str
    model_key: str
    model_company: str
    dataset_id: str
    dataset_name: str
    average_duration: int
    readability_score: float
    toxicity_score: float
    ethics_score: float
    accuracy: float


class Evaluations(BaseModel):
    data: List[Evaluation]


class Result(BaseModel):
    subset: str
    prompt: str
    result: str
    truth: str
    duration: timedelta
    score: float
    metrics: Dict[str, float]


class Results(BaseModel):
    results: List[Result]


class Model(BaseModel):
    id: str
    key: str
    name: str
    company: str
    description: str
    released_at: int
    parameters: float
    modality: str
    context_length: int
    architecture_type: str
    license: str
    open_weights: bool
    region: str
    deprecated: bool


class CustomModel(BaseModel):
    id: str
    key: str
    name: str
    description: str
    max_tokens: int
    api_url: str
    disabled: bool


class Models(BaseModel):
    models: List[Union[Model, CustomModel]]


class Benchmark(BaseModel):
    id: str
    key: str
    name: str
    full_description: str
    language: str
    categories: List[str]
    subsets: List[str]
    prompt_count: int
    deprecated: bool


class CustomBenchmark(BaseModel):
    id: str
    key: str
    name: str
    description: str
    system_prompt: Optional[str]
    subsets: List[str]
    prompt_count: int
    version_count: int
    regex_pattern: Optional[str]
    llm_judge_model_id: str
    custom_instructions: str
    scoring_metric: Optional[str]
    metrics: List[str]
    files: List[str]
    disabled: bool


class Benchmarks(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    benchmarks: List[Union[Benchmark, CustomBenchmark]] = Field(..., alias="datasets")
